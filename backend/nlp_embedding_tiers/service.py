"""In-memory embedding consensus tiering for merge / API use."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from merge_entities import normalize_entity_text
from nlp_tier_utils import build_anchor_exempt_texts, filter_entities_by_max_words
from NLP_Extraction_and_Ranking.bge_client import BGETritonClient
from NLP_Extraction_and_Ranking.deduplicate_nlps import _l2_normalize

from .anchored_clustering import assign_anchor_tiers_by_sim_quota
from .anchor_order import ordered_anchor_texts
from .config import ClusterConfig
from .consensus import build_final_clusters_from_instances
from .query_generator import generate_query_variants
from .validation import ensure_minimum_anchors

log = logging.getLogger(__name__)

bge_client = BGETritonClient()


def _entity_key(text: str) -> str:
    return (text or "").strip().casefold()


@dataclass
class TieringPrepCache:
    """Reusable anchor + entity embeddings for fast merge-tier recomputation."""

    keyword: str
    anchor_texts: list[str]
    query_unit: np.ndarray
    entity_embedding_map: dict[str, np.ndarray] = field(default_factory=dict)
    query_generation_method: str = ""
    raw_gliner: list[dict[str, Any]] = field(default_factory=list)
    gliner_variants: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.entity_embedding_map.keys())
        embeddings = (
            np.stack([self.entity_embedding_map[k] for k in keys], axis=0)
            if keys
            else np.zeros((0, self.query_unit.shape[1] if self.query_unit.ndim == 2 else 0), dtype=np.float32)
        )
        np.savez_compressed(
            path,
            query_unit=self.query_unit.astype(np.float32),
            anchor_texts=np.asarray(self.anchor_texts, dtype=object),
            entity_keys=np.asarray(keys, dtype=object),
            entity_embeddings=embeddings.astype(np.float32),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        keyword: str = "",
        query_generation_method: str = "",
        raw_gliner: list[dict[str, Any]] | None = None,
        gliner_variants: list[str] | None = None,
    ) -> TieringPrepCache:
        data = np.load(path, allow_pickle=True)
        query_unit = np.asarray(data["query_unit"], dtype=np.float32)
        anchor_texts = [str(x) for x in data["anchor_texts"].tolist()]
        keys = [str(x) for x in data["entity_keys"].tolist()]
        embeddings = np.asarray(data["entity_embeddings"], dtype=np.float32)
        emb_map: dict[str, np.ndarray] = {}
        for i, key in enumerate(keys):
            if i < len(embeddings):
                emb_map[key] = embeddings[i]
        return cls(
            keyword=keyword,
            anchor_texts=anchor_texts,
            query_unit=query_unit,
            entity_embedding_map=emb_map,
            query_generation_method=query_generation_method,
            raw_gliner=list(raw_gliner or []),
            gliner_variants=list(gliner_variants or []),
        )


def _gliner_span_meta(raw_gliner: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Map casefolded GLiNER span text to label and score metadata."""
    meta: dict[str, dict[str, Any]] = {}
    for ent in raw_gliner or []:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text", "")).strip()
        if not text:
            continue
        key = text.casefold()
        score = ent.get("score")
        label = ent.get("label")
        existing = meta.get(key)
        if existing is None or (
            score is not None
            and (existing.get("gliner_score") is None or score > existing.get("gliner_score", -1))
        ):
            meta[key] = {"label": label, "gliner_score": score}
    return meta


def build_keyword_instances(
    keyword: str,
    per_query_results: list[dict[str, Any]],
    entities_by_key: dict[str, dict[str, Any]],
    *,
    gliner_variants: set[str] | list[str] | None = None,
    raw_gliner: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build pre-consensus keyword instance payloads for the dashboard.

    Includes the original keyword and each accepted GLiNER sub-entity anchor.
    """
    keyword = (keyword or "").strip()
    gliner_meta = _gliner_span_meta(raw_gliner)
    results_by_text = {
        (r.get("query_text") or "").strip(): r for r in per_query_results
    }

    ordered_anchors = ordered_anchor_texts(
        per_query_results,
        keyword,
        gliner_variants=gliner_variants,
    )
    exempt_texts = build_anchor_exempt_texts(
        keyword,
        anchor_texts=ordered_anchors,
        gliner_variants=gliner_variants or [],
    )

    instances: list[dict[str, Any]] = []
    for anchor_text in ordered_anchors:
        anchor_key = anchor_text.casefold()
        is_original = bool(keyword) and anchor_key == keyword.casefold()
        role = "original" if is_original else "gliner_entity"
        result = results_by_text.get(anchor_text)
        if not result:
            continue
        span_meta = gliner_meta.get(anchor_text.casefold(), {})
        nlps: list[dict[str, Any]] = []
        for item in result.get("nlp_assignments") or []:
            text = item.get("text", "")
            entity = entities_by_key.get(_entity_key(text), {})
            nlps.append(
                {
                    "text": text,
                    "tier": item.get("tier", "orange"),
                    "sim_to_query": item.get("sim_to_query"),
                    "combined_count": entity.get("combined_count", 0),
                    "average_weightage": entity.get("average_weightage"),
                    "competitor_count": entity.get("competitor_count"),
                    "found_in_files": entity.get("found_in_files") or [],
                    "sources": entity.get("sources") or [],
                }
            )
        nlps = filter_entities_by_max_words(nlps, exempt_texts=exempt_texts)
        nlps.sort(
            key=lambda row: (
                -(row.get("sim_to_query") or 0),
                -(row.get("combined_count") or 0),
            )
        )
        tier_counts = result.get("tier_counts") or {"green": 0, "orange": 0, "white": 0}
        instances.append(
            {
                "text": anchor_text,
                "role": role,
                "label": span_meta.get("label") if role == "gliner_entity" else None,
                "gliner_score": span_meta.get("gliner_score") if role == "gliner_entity" else None,
                "tier_counts": tier_counts,
                "nlp_count": len(nlps),
                "nlps": nlps,
            }
        )
    return instances


def hydrate_tiering_prep_gliner_meta(
    prep: TieringPrepCache,
    config: ClusterConfig | None = None,
) -> TieringPrepCache:
    """Fill missing GLiNER metadata on a cached tiering prep (for older sessions)."""
    if prep.raw_gliner and prep.gliner_variants:
        return prep
    config = config or ClusterConfig.from_env()
    keyword = (prep.keyword or "").strip()
    if not keyword and prep.anchor_texts:
        keyword = prep.anchor_texts[0]
    if not keyword:
        return prep
    query_gen = generate_query_variants(keyword, config)
    if not prep.raw_gliner:
        prep.raw_gliner = list(query_gen.get("raw_gliner") or [])
    if not prep.gliner_variants:
        prep.gliner_variants = list(query_gen.get("gliner_variants") or [])
    return prep


def prepare_anchor_bundle(keyword: str, config: ClusterConfig | None = None) -> dict[str, Any]:
    """Generate query anchors and BGE query embeddings (GLiNER entities + validation)."""
    config = config or ClusterConfig.from_env()
    keyword = (keyword or "").strip()
    query_gen = generate_query_variants(keyword, config)
    gliner_variants = set(query_gen.get("gliner_variants") or [])
    anchor_texts, validation = ensure_minimum_anchors(
        keyword,
        query_gen.get("variants") or [],
        config,
        gliner_variants=gliner_variants,
    )
    query_embeddings = bge_client.encode(anchor_texts, is_query=True)
    query_unit = _l2_normalize(query_embeddings.astype(np.float32))
    return {
        "anchor_texts": anchor_texts,
        "query_unit": query_unit,
        "query_generation_method": query_gen.get("generation_method") or "",
        "raw_gliner": query_gen.get("raw_gliner") or [],
        "gliner_variants": sorted(gliner_variants),
        "anchor_validation": validation,
    }


def build_tiering_prep_cache(
    keyword: str,
    entity_texts: list[str],
    *,
    preloaded_embeddings: dict[str, np.ndarray] | None = None,
    anchor_texts: list[str] | None = None,
    query_unit: np.ndarray | None = None,
    query_generation_method: str = "",
    raw_gliner: list[dict[str, Any]] | None = None,
    gliner_variants: list[str] | None = None,
    config: ClusterConfig | None = None,
) -> TieringPrepCache:
    """Build or extend a session-level tiering prep cache."""
    config = config or ClusterConfig.from_env()
    keyword = (keyword or "").strip()
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in entity_texts:
        txt = (text or "").strip()
        if not txt:
            continue
        key = txt.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_texts.append(txt)

    emb_map: dict[str, np.ndarray] = dict(preloaded_embeddings or {})
    missing: list[str] = []
    for text in unique_texts:
        norm_key = normalize_entity_text(text)
        if norm_key not in emb_map:
            missing.append(text)

    if missing:
        encoded = bge_client.encode(missing, is_query=False)
        encoded_unit = _l2_normalize(encoded.astype(np.float32))
        for text, emb in zip(missing, encoded_unit):
            emb_map[normalize_entity_text(text)] = emb

    bundle_raw_gliner: list[dict[str, Any]] = list(raw_gliner or [])
    bundle_gliner_variants: list[str] = list(gliner_variants or [])

    if anchor_texts is None or query_unit is None:
        bundle = prepare_anchor_bundle(keyword, config)
        anchor_texts = bundle["anchor_texts"]
        query_unit = bundle["query_unit"]
        if not query_generation_method:
            query_generation_method = bundle.get("query_generation_method") or ""
        if not bundle_raw_gliner:
            bundle_raw_gliner = list(bundle.get("raw_gliner") or [])
        if not bundle_gliner_variants:
            bundle_gliner_variants = list(bundle.get("gliner_variants") or [])

    return TieringPrepCache(
        keyword=keyword,
        anchor_texts=list(anchor_texts or []),
        query_unit=np.asarray(query_unit, dtype=np.float32),
        entity_embedding_map=emb_map,
        query_generation_method=query_generation_method,
        raw_gliner=bundle_raw_gliner,
        gliner_variants=bundle_gliner_variants,
    )


def _lookup_entity_embeddings(
    nlp_texts: list[str],
    prep: TieringPrepCache,
) -> tuple[np.ndarray, list[str]]:
    """Resolve unit embeddings for entity texts; batch-encode any cache misses."""
    rows: list[np.ndarray] = []
    missing_texts: list[str] = []
    missing_indices: list[int] = []

    for i, text in enumerate(nlp_texts):
        norm_key = normalize_entity_text(text)
        emb = prep.entity_embedding_map.get(norm_key)
        if emb is not None:
            rows.append(np.asarray(emb, dtype=np.float32))
        else:
            rows.append(np.zeros((0,), dtype=np.float32))
            missing_texts.append(text)
            missing_indices.append(i)

    if missing_texts:
        encoded = bge_client.encode(missing_texts, is_query=False)
        encoded_unit = _l2_normalize(encoded.astype(np.float32))
        for idx, text, emb in zip(missing_indices, missing_texts, encoded_unit):
            norm_key = normalize_entity_text(text)
            prep.entity_embedding_map[norm_key] = emb
            rows[idx] = emb

    if not rows:
        dim = prep.query_unit.shape[1] if prep.query_unit.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), nlp_texts
    return np.stack(rows, axis=0).astype(np.float32), nlp_texts


def _run_consensus_from_embeddings(
    keyword: str,
    entities: list[dict[str, Any]],
    *,
    anchor_texts: list[str],
    query_unit: np.ndarray,
    nlp_unit: np.ndarray,
    nlp_texts: list[str],
    entities_by_key: dict[str, dict[str, Any]],
    avg_word_count: float,
    config: ClusterConfig,
    query_generation_method: str = "",
    competitor_domains: list[str] | None = None,
    raw_gliner: list[dict[str, Any]] | None = None,
    gliner_variants: list[str] | None = None,
) -> dict[str, Any]:
    def _cluster_one(idx: int) -> dict[str, Any]:
        return assign_anchor_tiers_by_sim_quota(
            nlp_unit,
            nlp_texts,
            query_unit[idx],
            anchor_texts[idx],
            config,
        )

    max_workers = min(8, max(1, len(anchor_texts)))
    if len(anchor_texts) <= 1:
        per_query_results = [_cluster_one(0)] if anchor_texts else []
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            per_query_results = list(executor.map(_cluster_one, range(len(anchor_texts))))

    keyword_instances = build_keyword_instances(
        keyword,
        per_query_results,
        entities_by_key,
        gliner_variants=gliner_variants,
        raw_gliner=raw_gliner,
    )

    final_clusters = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        keyword,
        config,
        avg_word_count=avg_word_count,
        competitor_domains=competitor_domains,
        gliner_variants=gliner_variants,
    )
    final_clusters["consensus_metadata"] = {
        **(final_clusters.get("consensus_metadata") or {}),
        "anchors_used": len(anchor_texts),
        "anchor_texts": anchor_texts,
        "query_generation_method": query_generation_method,
    }
    final_clusters["keyword_instances"] = keyword_instances
    return final_clusters


def compute_embedding_consensus_tiers_cached(
    keyword: str,
    entities: list[dict[str, Any]],
    *,
    prep: TieringPrepCache,
    avg_word_count: float = 0.0,
    config: ClusterConfig | None = None,
    competitor_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Fast path: reuse cached anchor + entity embeddings."""
    config = config or ClusterConfig.from_env()
    keyword = (keyword or prep.keyword or "").strip()

    entities_by_key: dict[str, dict[str, Any]] = {}
    for entity in entities:
        key = _entity_key(entity.get("text", ""))
        if key:
            entities_by_key[key] = entity

    nlp_texts = [e.get("text", "") for e in entities if e.get("text")]
    if not nlp_texts:
        return {
            "green_nlps": [],
            "orange_nlps": [],
            "white_nlps": [],
            "keyword_instances": [],
            "consensus_metadata": {
                "anchors_used": 0,
                "method": "proportional_instance_v1",
            },
        }

    nlp_unit, nlp_texts = _lookup_entity_embeddings(nlp_texts, prep)
    return _run_consensus_from_embeddings(
        keyword,
        entities,
        anchor_texts=prep.anchor_texts,
        query_unit=prep.query_unit,
        nlp_unit=nlp_unit,
        nlp_texts=nlp_texts,
        entities_by_key=entities_by_key,
        avg_word_count=avg_word_count,
        config=config,
        query_generation_method=prep.query_generation_method,
        competitor_domains=competitor_domains,
        raw_gliner=prep.raw_gliner,
        gliner_variants=prep.gliner_variants,
    )


def compute_embedding_consensus_tiers(
    keyword: str,
    entities: list[dict[str, Any]],
    *,
    avg_word_count: float = 0.0,
    config: ClusterConfig | None = None,
    competitor_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run embedding consensus tiering in memory (no artifact files).

    Returns dict with green_nlps, orange_nlps, white_nlps, consensus_metadata.
    """
    config = config or ClusterConfig.from_env()
    keyword = (keyword or "").strip()

    entities_by_key: dict[str, dict[str, Any]] = {}
    for entity in entities:
        key = _entity_key(entity.get("text", ""))
        if key:
            entities_by_key[key] = entity

    nlp_texts = [e.get("text", "") for e in entities if e.get("text")]
    if not nlp_texts:
        return {
            "green_nlps": [],
            "orange_nlps": [],
            "white_nlps": [],
            "keyword_instances": [],
            "consensus_metadata": {
                "anchors_used": 0,
                "method": "proportional_instance_v1",
            },
        }

    bundle = prepare_anchor_bundle(keyword, config)
    anchor_texts = bundle["anchor_texts"]
    query_unit = bundle["query_unit"]

    nlp_embeddings = bge_client.encode(nlp_texts, is_query=False)
    nlp_unit = _l2_normalize(nlp_embeddings.astype(np.float32))

    return _run_consensus_from_embeddings(
        keyword,
        entities,
        anchor_texts=anchor_texts,
        query_unit=query_unit,
        nlp_unit=nlp_unit,
        nlp_texts=nlp_texts,
        entities_by_key=entities_by_key,
        avg_word_count=avg_word_count,
        config=config,
        query_generation_method=bundle.get("query_generation_method") or "",
        competitor_domains=competitor_domains,
        raw_gliner=bundle.get("raw_gliner") or [],
        gliner_variants=bundle.get("gliner_variants") or [],
    )
