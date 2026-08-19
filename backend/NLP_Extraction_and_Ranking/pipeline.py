"""
In-memory pipeline: GLiNER extract -> BGE-M3 rank -> reranker -> deduplicate -> cluster.
Used by main.py search flow. Models are served via hosted HTTP APIs (see nlp_serving_urls.py).
"""
from typing import List, Dict, Optional, Tuple
import logging
import time
import os

# Import Gliner client and extraction
from .Gliner_ import GLiNERClient, extract_entities_sliding_window

import numpy as np
import math

# Reuse the exact anchor clustering helpers from the original architecture.
from .deduplicate_nlps import _build_anchor_seed_texts, _l2_normalize, _spherical_kmeans
from .nlp_serving_urls import (
    BGE_ENCODE_URL,
    BIENCODER_API_URL,
    GLINER_PREDICT_URL,
    GLINER_THRESHOLD,
    RERANK_URL,
    USE_RERANKER,
)
from .bge_client import BGETritonClient
from .reranker_client import RerankerClient
from nlp_tier_utils import exceeds_max_nlp_words, is_exempt_nlp_text

log = logging.getLogger(__name__)

bge_client = BGETritonClient()
reranker_client = RerankerClient()
GLINER_RUNTIME_URL = GLINER_PREDICT_URL
BGE_RUNTIME_URL = BGE_ENCODE_URL


def extract_nlps_gliner(text: str, context_size: int = 600, step_size: int = 300) -> List[Dict]:
    """
    Extract entities from text using hosted GLiNER.
    Returns list of {"text": str, "count": int}.
    """
    if not (text or text.strip()):
        log.debug("[GLiNER] Skip: empty text")
        return []
    log.info("[Step 1/3] GLiNER extraction — API=%s | threshold=%.2f | context_size=%d | step_size=%d | text_len=%d",
             GLINER_RUNTIME_URL, GLINER_THRESHOLD, context_size, step_size, len(text))
    client = GLiNERClient()
    entities = extract_entities_sliding_window(
        text, client, step_size=step_size, context_size=context_size
    )
    log.info("[Step 1/3] GLiNER done — unique_entities=%d | total_instances=%d",
             len(entities), sum(e.get("count", 1) for e in entities))
    return entities


def rank_entities_in_memory(
    entities: List[Dict],
    title: str,
    biencoder_url: str = BIENCODER_API_URL,
    precomputed_entity_embeddings: Optional[np.ndarray] = None,
    precomputed_title_embedding: Optional[np.ndarray] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Rank entities by cosine similarity to title using BGE-M3 embeddings.
    entities: list of {"text": str, "count": int}
    Returns list with "biencoder_score" (and optional "crossencoder_score" after rerank step).
    """
    if not entities or not title:
        log.debug("[Rank] Skip: no entities or title")
        return list(entities), np.empty((0, 0), dtype=np.float32)

    log.info("[Step 2/3] Ranking — BiEncoder=%s | title=%r | entities=%d",
             BGE_RUNTIME_URL, title[:50] + ("..." if len(title) > 50 else ""), len(entities))
    if precomputed_entity_embeddings is not None and precomputed_title_embedding is not None:
        entity_emb = precomputed_entity_embeddings
        title_emb = precomputed_title_embedding.reshape(1, -1)
    else:
        texts = [title] + [e["text"] for e in entities]
        try:
            embeddings = bge_client.encode(texts, is_query=False)
        except Exception as e:
            log.warning("[Step 2/3] BiEncoder request failed: %s — returning unranked", e)
            return list(entities), np.empty((0, 0), dtype=np.float32)

        title_emb = embeddings[0:1]
        entity_emb = embeddings[1:]
    norms_title = np.linalg.norm(title_emb, axis=1, keepdims=True)
    norms_ent = np.linalg.norm(entity_emb, axis=1, keepdims=True)
    norms_ent = np.maximum(norms_ent, 1e-9)
    sims = (entity_emb / norms_ent) @ (title_emb / np.maximum(norms_title, 1e-9)).T
    similarity_scores = sims.ravel()

    ranked = []
    for i, entity in enumerate(entities):
        ranked.append({
            "text": entity["text"],
            "count": entity["count"],
            "label": entity.get("label") or "Other",
            "biencoder_score": round(float(similarity_scores[i]), 4),
        })

    # Sort by BiEncoder and preserve embedding alignment.
    sort_idx = sorted(range(len(ranked)), key=lambda i: ranked[i]["biencoder_score"], reverse=True)
    ranked = [ranked[i] for i in sort_idx]
    entity_emb = entity_emb[np.array(sort_idx, dtype=np.int32)]
    norms_ent = np.linalg.norm(entity_emb, axis=1, keepdims=True)
    norms_ent = np.maximum(norms_ent, 1e-9)
    entity_emb_unit = (entity_emb / norms_ent).astype(np.float32, copy=False)

    log.info("[Step 2/3] Ranking done — method=BiEncoder | top_score=%.4f", ranked[0]["biencoder_score"])
    return ranked, entity_emb_unit


def rerank_entities_in_memory(
    ranked_entities: List[Dict],
    title: str,
) -> List[Dict]:
    """Apply BGE reranker scores and re-sort by crossencoder_score."""
    if not ranked_entities or not title or not USE_RERANKER:
        return ranked_entities

    log.info(
        "[Step 2b/3] Reranking — reranker=%s | title=%r | entities=%d",
        RERANK_URL,
        title[:50] + ("..." if len(title) > 50 else ""),
        len(ranked_entities),
    )
    try:
        documents = [e["text"] for e in ranked_entities]
        scores = reranker_client.rerank(title, documents)
        for i, entity in enumerate(ranked_entities):
            entity["crossencoder_score"] = round(float(scores[i]), 4)
        ranked_entities = sorted(
            ranked_entities,
            key=lambda x: x.get("crossencoder_score", 0.0),
            reverse=True,
        )
        log.info(
            "[Step 2b/3] Reranking done — top_score=%.4f",
            ranked_entities[0].get("crossencoder_score", 0.0),
        )
    except Exception as e:
        log.warning("[Step 2b/3] Reranker failed: %s — keeping BiEncoder order", e)
    return ranked_entities


def deduplicate_in_memory(
    ranked_entities: List[Dict],
    similarity_threshold: float = 0.85,
    biencoder_url: str = BIENCODER_API_URL,
    precomputed_unit_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Deduplicate by semantic similarity using BGE-M3 embeddings.
    Merges entities with similarity > threshold (keeps higher count).
    """
    if len(ranked_entities) <= 1:
        log.debug("[Dedup] Skip: 1 or 0 entities")
        if precomputed_unit_embeddings is None:
            return list(ranked_entities), np.empty((0, 0), dtype=np.float32)
        return list(ranked_entities), precomputed_unit_embeddings

    log.info("[Step 3/3] Deduplication — BiEncoder=%s | similarity_threshold=%.2f | input_entities=%d",
             BGE_RUNTIME_URL, similarity_threshold, len(ranked_entities))
    current = list(ranked_entities)
    max_iterations = 50
    iteration = 0
    # Encode once and then only slice embeddings when entities are removed.
    # This keeps the *exact same* similarity math while avoiding repeated
    # expensive embedding API calls per iteration.
    if precomputed_unit_embeddings is not None and len(precomputed_unit_embeddings) == len(current):
        base_embeddings_unit = precomputed_unit_embeddings
    else:
        try:
            base_embeddings = bge_client.encode([e["text"] for e in current], is_query=False)
        except Exception:
            return current, np.empty((0, 0), dtype=np.float32)
        base_norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
        base_norms = np.maximum(base_norms, 1e-9)
        base_embeddings_unit = (base_embeddings / base_norms).astype(np.float32, copy=False)

    # Track which original rows are currently kept.
    keep_idx = np.arange(len(current), dtype=np.int32)

    for iteration in range(max_iterations):
        # Similarity matrix for current subset.
        emb_u = base_embeddings_unit[keep_idx]
        sim_matrix = emb_u @ emb_u.T

        to_keep = set(range(len(current)))
        merged_any = False
        for i in range(len(current)):
            if i not in to_keep:
                continue
            for j in range(i + 1, len(current)):
                if j not in to_keep:
                    continue
                if sim_matrix[i, j] <= similarity_threshold:
                    continue
                ci, cj = current[i]["count"], current[j]["count"]
                if ci >= cj:
                    to_keep.discard(j)
                else:
                    to_keep.discard(i)
                    break
                merged_any = True
                break

        kept_local = sorted(to_keep)
        current = [current[i] for i in kept_local]
        keep_idx = keep_idx[np.array(kept_local, dtype=np.int32)]
        if not merged_any:
            break

    log.info("[Step 3/3] Deduplication done — iterations=%d | output_entities=%d | removed=%d",
             iteration + 1, len(current), len(ranked_entities) - len(current))
    return current, base_embeddings_unit[keep_idx]


def cluster_with_anchor_title(
    entities: List[Dict],
    anchor_title: str,
    biencoder_url: str = BIENCODER_API_URL,
    precomputed_unit_embeddings: Optional[np.ndarray] = None,
) -> tuple[dict, dict]:
    """
    Copy of the "anchor title -> cluster_1 relevance bucket" architecture used in
    `deduplicate_nlps.py`, adapted to run in-memory and return clusters for API.

    Returns (clusters, cluster_scores).
    """
    entities_text = [e["text"] for e in (entities or []) if e.get("text")]
    anchor_title = (anchor_title or "").strip()

    clusters: dict[str, list[str]] = {"cluster_1": []}
    cluster_scores: dict[str, dict[str, float]] = {}

    if not anchor_title:
        # No anchor: put everything into cluster_1 without the title.
        clusters["cluster_1"] = entities_text
        cluster_scores["cluster_1"] = {
            "sim_to_cluster_1_centroid": 1.0,
            "sim_to_anchor_title": 1.0,
        }
        return clusters, cluster_scores

    if not entities_text:
        # Still include the anchor title as required.
        clusters["cluster_1"] = [anchor_title]
        cluster_scores["cluster_1"] = {
            "sim_to_cluster_1_centroid": 1.0,
            "sim_to_anchor_title": 1.0,
        }
        return clusters, cluster_scores

    anchor_seed_texts = _build_anchor_seed_texts(anchor_title) or [anchor_title]

    if precomputed_unit_embeddings is not None and len(precomputed_unit_embeddings) == len(entities_text):
        embeddings_unit = precomputed_unit_embeddings
        seed_embs_raw = bge_client.encode(anchor_seed_texts, is_query=False)
    else:
        # Fallback path if precomputed embeddings are unavailable.
        all_embeddings = bge_client.encode(anchor_seed_texts + entities_text, is_query=False)
        seed_count = len(anchor_seed_texts)
        seed_embs_raw = all_embeddings[:seed_count]
        embeddings = all_embeddings[seed_count:]
        embeddings_unit = _l2_normalize(embeddings)

    seed_embs = _l2_normalize(seed_embs_raw)
    title_emb_unit = seed_embs[0]

    # Relevance rule (copied defaults from `deduplicate_nlps.py`)
    per_seed_threshold = 0.55
    min_seed_hits = 2
    max_sim_override = 0.70

    # Prefer chunk seeds only (exclude full-title seed at index 0)
    seed_unit = seed_embs[1:] if seed_embs.shape[0] > 1 else seed_embs
    sims_matrix = embeddings_unit @ seed_unit.T
    sims_max = np.max(sims_matrix, axis=1).astype(np.float32)
    seed_hits = (sims_matrix >= per_seed_threshold).sum(axis=1).astype(np.int32)
    is_relevant = (seed_hits >= min_seed_hits) | (sims_max >= max_sim_override)

    idx_all = np.arange(len(entities_text))
    idx_relevant = idx_all[is_relevant]
    idx_rest = idx_all[~is_relevant]

    # cluster_1: relevant items
    clusters["cluster_1"] = [entities_text[i] for i in idx_relevant]

    # Score cluster_1 centroid (entity-only centroid, not including seeds)
    if len(idx_relevant) > 0:
        c1 = _l2_normalize(embeddings_unit[idx_relevant].mean(axis=0, keepdims=True))[0]
    else:
        # Fallback to title embedding
        c1 = title_emb_unit

    cluster_scores["cluster_1"] = {
        "sim_to_cluster_1_centroid": 1.0,
        "sim_to_anchor_title": float(title_emb_unit @ c1),
    }

    # Reclustering non-cluster_1 items (~<=5 per cluster), then order by similarity to cluster_1 centroid
    if len(idx_rest) > 0:
        rest_embeddings = embeddings_unit[idx_rest]
        rest_texts = [entities_text[i] for i in idx_rest]

        k_rest = max(1, int(math.ceil(len(rest_texts) / 5.0)))
        rng = np.random.default_rng(123)
        if rest_embeddings.shape[0] >= k_rest:
            init_idx = rng.choice(rest_embeddings.shape[0], size=k_rest, replace=False)
        else:
            init_idx = np.arange(rest_embeddings.shape[0])
        init_centroids_rest = rest_embeddings[init_idx]

        labels_rest, centroids_rest = _spherical_kmeans(
            rest_embeddings,
            init_centroids_rest,
            max_iter=100,
            tol=1e-6,
        )

        # Group texts by label, compute centroid similarities
        rest_clusters_text: dict[int, list[str]] = {}
        rest_centers: dict[int, np.ndarray] = {}
        for i_local, lbl in enumerate(labels_rest):
            rest_clusters_text.setdefault(int(lbl), []).append(rest_texts[i_local])
        for lbl, items in rest_clusters_text.items():
            mask = labels_rest == lbl
            center = _l2_normalize(rest_embeddings[mask].mean(axis=0, keepdims=True))[0]
            rest_centers[int(lbl)] = center

        scored = []
        for lbl, center in rest_centers.items():
            sim_c1 = float(c1 @ center)
            sim_anchor = float(title_emb_unit @ center)
            scored.append((lbl, sim_c1, sim_anchor))
        scored.sort(key=lambda x: x[1], reverse=True)

        next_cluster_idx = 2
        for lbl, sim_c1, sim_anchor in scored:
            key = f"cluster_{next_cluster_idx}"
            clusters[key] = rest_clusters_text[lbl]
            cluster_scores[key] = {
                "sim_to_cluster_1_centroid": sim_c1,
                "sim_to_anchor_title": sim_anchor,
            }
            next_cluster_idx += 1

    # Final cleanup: ensure the full anchor title is included exactly once at the top of cluster_1.
    full_title = anchor_seed_texts[0] if anchor_seed_texts else anchor_title
    full_title_lc = full_title.lower()
    sub_seeds = {s.lower() for s in anchor_seed_texts[1:]} if len(anchor_seed_texts) > 1 else set()

    for key, items in list(clusters.items()):
        filtered: list[str] = []
        for t in items:
            tl = (t or "").lower()
            if tl == full_title_lc:
                continue
            if tl in sub_seeds:
                continue
            filtered.append(t)
        clusters[key] = filtered

    clusters.setdefault("cluster_1", [])
    clusters["cluster_1"].insert(0, full_title)

    return clusters, cluster_scores


def run_pipeline(
    text: str,
    title: str,
    max_nlps: int = 100,
    dedup_threshold: float = 0.85,
    gliner_context_size: int = 600,
    gliner_step_size: int = 400,
) -> Dict:
    """
    Full pipeline: GLiNER extract -> rank -> deduplicate.
    Returns {"entities": [...], "ranking_method": "crossencoder"|"biencoder"}.
    """
    if not (text or text.strip()):
        return {
            "entities": [],
            "ranking_method": "biencoder",
            "timing_seconds": {
                "preprocess_seconds": 0.0,
                "gliner_seconds": 0.0,
                "ranking_seconds": 0.0,
                "dedup_seconds": 0.0,
                "clustering_seconds": 0.0,
                "embedding_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }

    log.info("[Pipeline] Start — title=%r | max_nlps=%d | dedup_threshold=%.2f | gliner_context=%d | gliner_step=%d",
             title[:60] + ("..." if len(title) > 60 else ""), max_nlps, dedup_threshold, gliner_context_size, gliner_step_size)
    t0 = time.perf_counter()

    # Cap text size for huge pages to keep NLP latency bounded.
    max_input_chars = int(os.getenv("MAX_NLP_INPUT_CHARS", "18000"))
    if max_input_chars > 0 and len(text) > max_input_chars:
        text = text[:max_input_chars]

    preprocess_seconds = 0.0
    t_gliner = time.perf_counter()
    entities = extract_nlps_gliner(text, context_size=gliner_context_size, step_size=gliner_step_size)
    gliner_seconds = time.perf_counter() - t_gliner
    if not entities:
        log.info("[Pipeline] No entities extracted — returning empty")
        total_seconds = time.perf_counter() - t0
        return {
            "entities": [],
            "ranking_method": "biencoder",
            "timing_seconds": {
                "preprocess_seconds": round(preprocess_seconds, 4),
                "gliner_seconds": round(gliner_seconds, 4),
                "ranking_seconds": 0.0,
                "dedup_seconds": 0.0,
                "clustering_seconds": 0.0,
                "embedding_seconds": 0.0,
                "total_seconds": round(total_seconds, 4),
            },
        }

    t_embed = time.perf_counter()
    try:
        raw_entity_embeddings = bge_client.encode([e["text"] for e in entities], is_query=False)
        title_embedding = bge_client.encode([title], is_query=True)[0]
    except Exception:
        log.exception("[Pipeline] Pre-embedding failed; falling back to in-step encoding")
        raw_entity_embeddings = None
        title_embedding = None
    embedding_seconds = time.perf_counter() - t_embed

    t_rank = time.perf_counter()
    ranked, ranked_embeddings_unit = rank_entities_in_memory(
        entities,
        title,
        precomputed_entity_embeddings=raw_entity_embeddings,
        precomputed_title_embedding=title_embedding,
    )
    ranked = rerank_entities_in_memory(ranked, title)
    method = "crossencoder" if USE_RERANKER and any(
        e.get("crossencoder_score") is not None for e in ranked
    ) else "biencoder"
    ranking_seconds = time.perf_counter() - t_rank
    t_dedup = time.perf_counter()
    ranked, dedup_embeddings_unit = deduplicate_in_memory(
        ranked,
        similarity_threshold=dedup_threshold,
        precomputed_unit_embeddings=ranked_embeddings_unit,
    )
    dedup_seconds = time.perf_counter() - t_dedup
    exempt_texts = [title] if (title or "").strip() else ()
    if ranked:
        kept_ranked: List[Dict] = []
        kept_emb_rows: List[np.ndarray] = []
        has_emb = (
            dedup_embeddings_unit is not None
            and len(dedup_embeddings_unit) == len(ranked)
        )
        for i, entity in enumerate(ranked):
            text = entity.get("text", "")
            if exceeds_max_nlp_words(text) and not is_exempt_nlp_text(text, exempt_texts):
                continue
            kept_ranked.append(entity)
            if has_emb:
                kept_emb_rows.append(dedup_embeddings_unit[i])
        ranked = kept_ranked
        if has_emb:
            dedup_embeddings_unit = (
                np.stack(kept_emb_rows, axis=0)
                if kept_emb_rows
                else dedup_embeddings_unit[:0]
            )
    ranked = ranked[:max_nlps]
    dedup_embeddings_unit = dedup_embeddings_unit[:max_nlps]
    if dedup_embeddings_unit is not None and len(dedup_embeddings_unit) == len(ranked):
        for i, entity in enumerate(ranked):
            entity["embedding_unit"] = dedup_embeddings_unit[i].astype(np.float32).tolist()
    t_cluster = time.perf_counter()
    clusters, cluster_scores = cluster_with_anchor_title(
        ranked,
        anchor_title=title,
        precomputed_unit_embeddings=dedup_embeddings_unit,
    )
    clustering_seconds = time.perf_counter() - t_cluster
    total_seconds = time.perf_counter() - t0
    log.info("[Pipeline] Done — ranking_method=%s | output_terms=%d | clusters=%d", method, len(ranked), len(clusters))
    return {
        "entities": ranked,
        "ranking_method": method,
        "clusters": clusters,
        "cluster_scores": cluster_scores,
        "timing_seconds": {
            "preprocess_seconds": round(preprocess_seconds, 4),
            "gliner_seconds": round(gliner_seconds, 4),
            "ranking_seconds": round(ranking_seconds, 4),
            "dedup_seconds": round(dedup_seconds, 4),
            "clustering_seconds": round(clustering_seconds, 4),
            "embedding_seconds": round(embedding_seconds, 4),
            "total_seconds": round(total_seconds, 4),
        },
    }
