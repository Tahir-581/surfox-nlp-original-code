"""Shared merge aggregation, tiering, and session cache helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from merge_entities import get_best_display_form, normalize_entity_text
from nlp_embedding_tiers.merge_tiering import apply_merge_tiering, merge_output_has_embedding_tiers
from nlp_embedding_tiers.fuzzy_dedup import normalize_competitor_hostname
from nlp_embedding_tiers.service import (
    TieringPrepCache,
    build_tiering_prep_cache,
    compute_embedding_consensus_tiers,
    compute_embedding_consensus_tiers_cached,
    hydrate_tiering_prep_gliner_meta,
)
from nlp_word_buckets import build_tier_word_buckets
from NLP_Extraction_and_Ranking.Gliner_ import ENTITY_LABELS, GLiNERClient
from NLP_Extraction_and_Ranking.nlp_serving_urls import GLINER_THRESHOLD

log = logging.getLogger(__name__)

TIERING_PREP_FILENAME = "tiering_prep.npz"
TIERING_PREP_META_FILENAME = "tiering_prep_meta.json"
MERGE_CACHE_DIRNAME = "merge_cache"


def selection_cache_key(selected_urls: List[str]) -> str:
    normalized = sorted({(u or "").strip() for u in selected_urls if (u or "").strip()})
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:16]


def tiering_prep_path(session_dir: Path) -> Path:
    return session_dir / TIERING_PREP_FILENAME


def tiering_prep_meta_path(session_dir: Path) -> Path:
    return session_dir / TIERING_PREP_META_FILENAME


def merge_cache_dir(session_dir: Path) -> Path:
    return session_dir / MERGE_CACHE_DIRNAME


def merge_cache_file(session_dir: Path, cache_key: str) -> Path:
    return merge_cache_dir(session_dir) / f"{cache_key}.json"


def save_tiering_prep(session_dir: Path, prep: TieringPrepCache) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    prep.save(tiering_prep_path(session_dir))
    with open(tiering_prep_meta_path(session_dir), "w", encoding="utf-8") as f:
        json.dump(
            {
                "keyword": prep.keyword,
                "query_generation_method": prep.query_generation_method,
                "raw_gliner": prep.raw_gliner,
                "gliner_variants": prep.gliner_variants,
            },
            f,
            indent=2,
        )


def load_tiering_prep(session_dir: Path) -> Optional[TieringPrepCache]:
    path = tiering_prep_path(session_dir)
    if not path.exists():
        return None
    meta: dict[str, Any] = {}
    meta_path = tiering_prep_meta_path(session_dir)
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            log.exception("Failed to read tiering prep meta from %s", meta_path)
    try:
        return TieringPrepCache.load(
            path,
            keyword=meta.get("keyword", ""),
            query_generation_method=meta.get("query_generation_method", ""),
            raw_gliner=meta.get("raw_gliner") or [],
            gliner_variants=meta.get("gliner_variants") or [],
        )
    except Exception:
        log.exception("Failed to load tiering prep from %s", path)
        return None


def save_merge_cache(session_dir: Path, cache_key: str, merge_output: dict[str, Any]) -> None:
    cache_dir = merge_cache_dir(session_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(merge_cache_file(session_dir, cache_key), "w", encoding="utf-8") as f:
        json.dump(merge_output, f, ensure_ascii=False, indent=2)


def load_merge_cache(session_dir: Path, cache_key: str) -> Optional[dict[str, Any]]:
    path = merge_cache_file(session_dir, cache_key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        log.exception("Failed to load merge cache %s", path)
        return None


def _domain_tokens(domain: str) -> set[str]:
    d = (domain or "").strip().lower()
    if not d:
        return set()
    d = d.split(":")[0]
    parts = [p for p in d.split(".") if p and p not in {"www", "m", "amp"}]
    toks = set(parts)
    if len(parts) >= 2:
        toks.add(parts[-2])
    return toks


def _is_pure_number(txt: str) -> bool:
    t = (txt or "").strip()
    return bool(t) and t.isdigit()


def _is_banned_term(txt: str, banned_domain_tokens: set[str]) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    if t.casefold() in banned_domain_tokens:
        return True
    if _is_pure_number(t):
        return True
    return False


def collect_competitor_domains_from_items(items: List[Dict]) -> List[str]:
    """Collect unique normalized competitor hostnames from merge source items."""
    seen: set[str] = set()
    out: list[str] = []
    for data in items:
        normalized = normalize_competitor_hostname(data.get("domain", ""))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def collect_entity_texts_from_items(items: List[Dict]) -> List[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for data in items:
        terms = data.get("nlp_terms") or data.get("entities") or []
        for term in terms:
            txt = (term.get("text") or "").strip()
            if not txt or txt.lower() == "n/a":
                continue
            key = txt.casefold()
            if key in seen:
                continue
            seen.add(key)
            texts.append(txt)
    return texts


def collect_preloaded_embeddings_from_items(items: List[Dict]) -> Dict[str, np.ndarray]:
    """Map normalized entity text -> unit embedding from per-page NLP output."""
    emb_map: dict[str, np.ndarray] = {}
    for data in items:
        terms = data.get("nlp_terms") or data.get("entities") or []
        for term in terms:
            txt = (term.get("text") or "").strip()
            raw_emb = term.get("embedding_unit")
            if not txt or not raw_emb:
                continue
            key = normalize_entity_text(txt)
            try:
                emb_map[key] = np.asarray(raw_emb, dtype=np.float32)
            except Exception:
                continue
    return emb_map


def aggregate_entities_from_items(
    items: List[Dict],
    *,
    keep_ratio: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Aggregate NLP terms across selected page result dicts.

    Returns (merged_entities, stats, ranking_method).
    """
    entity_map = defaultdict(
        lambda: {
            "total_count": 0,
            "source_counts": defaultdict(int),
            "sources": set(),
            "relevance_sum": 0.0,
            "relevance_weight": 0,
            "weightage_sum": 0.0,
            "weightage_weight": 0,
            "keybert_score_sum": 0.0,
            "keybert_weight": 0,
            "label_counts": Counter(),
            "files": [],
            "original_forms": [],
        }
    )

    stats = {
        "total_files": 0,
        "avg_word_count": 0,
        "avg_heading_count": 0,
        "avg_para_count": 0,
        "avg_images_count": 0,
        "competitor_domains": collect_competitor_domains_from_items(items),
    }

    word_count_sum = 0
    heading_count_sum = 0
    para_count_sum = 0
    images_count_sum = 0
    file_count = 0
    ranking_method = "biencoder"
    ratio = min(1.0, max(0.0, float(keep_ratio or 1.0)))

    for idx, data in enumerate(items):
        try:
            if file_count == 0:
                ranking_method = data.get("ranking_method", "biencoder")

            terms = data.get("nlp_terms")
            if not terms:
                terms = data.get("entities", [])
                for t in terms:
                    t.setdefault("source", "gliner")

            if terms and ratio < 1.0:
                ranked_terms = sorted(
                    terms,
                    key=lambda t: (
                        float(
                            t.get("weightage", 0)
                            or t.get("relevance", 0)
                            or t.get("keybert_score", 0)
                            or 0
                        ),
                        int(t.get("count", 1) or 1),
                    ),
                    reverse=True,
                )
                keep_count = max(1, int(math.ceil(len(ranked_terms) * ratio)))
                terms = ranked_terms[:keep_count]

            domain = data.get("domain", "")
            domain_lc = (domain or "").strip().lower()
            banned_domain_tokens = _domain_tokens(domain_lc)

            word_count_sum += data.get("word_count", 0)
            heading_count_sum += data.get("heading_count", 0)
            para_count_sum += data.get("para_count", 0)
            images_count_sum += data.get("images_count", 0)
            file_count += 1

            for term in terms:
                text = term.get("text")
                label = term.get("label")
                source = (term.get("source") or "gliner").lower()
                count = term.get("count", 1) or 1
                try:
                    count = int(count)
                except Exception:
                    count = 1
                count = max(1, count)

                relevance = term.get("relevance", 0) or 0
                weightage = term.get("weightage", 0) or 0
                keybert_score = term.get("keybert_score", 0) or 0

                if text and text.lower() != "n/a" and not _is_banned_term(text, banned_domain_tokens):
                    key = normalize_entity_text(text)

                    entity_map[key]["sources"].add(source)
                    entity_map[key]["source_counts"][source] += count
                    entity_map[key]["total_count"] += count

                    if label:
                        entity_map[key]["label_counts"][label] += count

                    try:
                        entity_map[key]["relevance_sum"] += float(relevance) * count
                        entity_map[key]["relevance_weight"] += count
                    except Exception:
                        pass
                    try:
                        entity_map[key]["weightage_sum"] += float(weightage) * count
                        entity_map[key]["weightage_weight"] += count
                    except Exception:
                        pass
                    if source == "keybert":
                        try:
                            entity_map[key]["keybert_score_sum"] += float(keybert_score) * count
                            entity_map[key]["keybert_weight"] += count
                        except Exception:
                            pass

                    entity_map[key]["files"].append(domain)
                    entity_map[key]["original_forms"].append(text.lower().strip())

        except Exception:
            log.exception("Error processing merge source item index=%s", idx)
            continue

    if file_count > 0:
        stats["total_files"] = file_count
        stats["avg_word_count"] = round(word_count_sum / file_count, 2)
        stats["avg_heading_count"] = round(heading_count_sum / file_count, 2)
        stats["avg_para_count"] = round(para_count_sum / file_count, 2)
        stats["avg_images_count"] = round(images_count_sum / file_count, 2)

    merged_entities: list[dict[str, Any]] = []
    for _text_lower, data in entity_map.items():
        avg_relevance = (
            data["relevance_sum"] / data["relevance_weight"] if data["relevance_weight"] > 0 else 0.0
        )
        avg_weightage = (
            data["weightage_sum"] / data["weightage_weight"] if data["weightage_weight"] > 0 else 0.0
        )
        avg_keybert = (
            data["keybert_score_sum"] / data["keybert_weight"] if data["keybert_weight"] > 0 else 0.0
        )

        display_text = get_best_display_form(data["original_forms"])
        if data["label_counts"]:
            label_val = data["label_counts"].most_common(1)[0][0]
        elif "keybert" in data["sources"]:
            label_val = "Keyphrase"
        else:
            label_val = "Other"
        merged_entities.append(
            {
                "text": display_text.title() if len(display_text.split()) > 1 else display_text,
                "label": label_val,
                "combined_count": data["total_count"],
                "sources": sorted(list(data["sources"])),
                "source_counts": {k: int(v) for k, v in data["source_counts"].items()},
                "average_relevance": round(avg_relevance, 4),
                "average_weightage": round(avg_weightage, 4),
                "average_keybert_score": round(avg_keybert, 4),
                "competitor_count": len(set(data["files"])),
                "found_in_files": list(set(data["files"])),
            }
        )

    for entity in merged_entities:
        competitor_count = entity["competitor_count"]
        if competitor_count >= 3:
            multiplier = 3
        elif competitor_count == 2:
            multiplier = 2
        else:
            multiplier = 1
        entity["competitor_multiplier"] = multiplier
        entity["adjusted_weightage"] = entity["average_weightage"] * multiplier

    merged_entities.sort(key=lambda x: x["average_weightage"], reverse=True)

    x_value = stats["avg_word_count"] * 0.60
    total_adjusted_weightage = sum(e["adjusted_weightage"] for e in merged_entities)
    for entity in merged_entities:
        probability = (
            entity["adjusted_weightage"] / total_adjusted_weightage if total_adjusted_weightage > 0 else 0
        )
        entity["word_range"] = math.ceil(probability * x_value)

    stats["word_range_60_percent_value"] = round(x_value, 2)
    stats["total_adjusted_weightage"] = round(total_adjusted_weightage, 2)

    return merged_entities, stats, ranking_method


def _tier_texts(items: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        txt = (it.get("text") or "").strip()
        key = txt.casefold()
        if not txt or key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def _entities_from_merge_output(merge_output: dict[str, Any]) -> list[dict[str, Any]]:
    entities = merge_output.get("entities")
    if entities:
        return list(entities)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for tier_key in ("green_nlps", "orange_nlps", "white_nlps"):
        for entity in merge_output.get(tier_key) or []:
            text = (entity.get("text") or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(entity)
    return out


def _expected_keyword_instance_count(
    keyword: str,
    gliner_variants: list[str] | None,
) -> int:
    """Original keyword row plus one row per GLiNER sub-entity anchor."""
    keyword = (keyword or "").strip()
    gliner_count = len(gliner_variants or [])
    if keyword:
        return 1 + gliner_count
    return gliner_count


def merge_output_needs_keyword_instances(
    merge_output: dict[str, Any],
    *,
    tiering_prep: Optional[TieringPrepCache] = None,
    keyword: str = "",
) -> bool:
    if not merge_output or not merge_output_has_embedding_tiers(merge_output):
        return False
    instances = merge_output.get("keyword_instances")
    if instances is None:
        return True
    if len(instances) == 0:
        return True

    prep = tiering_prep
    if prep is not None:
        prep = hydrate_tiering_prep_gliner_meta(prep)
        kw = (keyword or prep.keyword or "").strip()
        expected = _expected_keyword_instance_count(kw, prep.gliner_variants)
        if expected > 0 and len(instances) < expected:
            return True

    return False


def _merge_tiering_method(merge_output: dict[str, Any]) -> str:
    meta = merge_output.get("tiering_metadata") or merge_output.get("consensus_metadata") or {}
    return str(merge_output.get("tiering_method") or meta.get("method") or "").strip()


def merge_output_needs_proportional_tier_refresh(merge_output: dict[str, Any]) -> bool:
    """True when saved tiers should be recomputed with proportional_instance_v1."""
    if not merge_output or not merge_output_has_embedding_tiers(merge_output):
        return False
    if _merge_tiering_method(merge_output) != "proportional_instance_v1":
        return True
    for tier_key in ("green_nlps", "orange_nlps", "white_nlps"):
        for entity in merge_output.get(tier_key) or []:
            consensus = entity.get("consensus") or {}
            if consensus.get("instance_rank") is None or consensus.get("instance_index") is None:
                return True
    return False


def ensure_proportional_tiers_in_merge_output(
    merge_output: dict[str, Any],
    *,
    keyword: str = "",
    tiering_prep: Optional[TieringPrepCache] = None,
    max_numerical_per_tier: int = 2,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    Recompute green / orange / white tiers (and keyword_instances) for older sessions.

    Returns (merge_output, changed).
    """
    if not merge_output:
        return merge_output, False

    if not force and not merge_output_needs_proportional_tier_refresh(merge_output):
        return merge_output, False

    entities = merge_output.get("entities") or _entities_from_merge_output(merge_output)
    if not entities:
        return merge_output, False

    keyword = (keyword or "").strip()
    avg_word_count = float(
        (merge_output.get("average_statistics") or {}).get("avg_word_count") or 0.0
    )
    competitor_domains = merge_output.get("competitor_domains") or []

    prep = tiering_prep
    if prep is not None:
        prep = hydrate_tiering_prep_gliner_meta(prep)
        if keyword and not prep.keyword:
            prep.keyword = keyword

    try:
        tiering = apply_merge_tiering(
            entities,
            keyword or (prep.keyword if prep else ""),
            avg_word_count=avg_word_count,
            max_numerical_per_tier=max_numerical_per_tier,
            tiering_prep=prep,
            competitor_domains=competitor_domains,
        )
    except Exception:
        log.exception(
            "[Backfill] proportional tier refresh failed for keyword=%r",
            keyword,
        )
        return merge_output, False

    green = tiering.get("green_nlps") or []
    orange = tiering.get("orange_nlps") or []
    white = tiering.get("white_nlps") or []
    if not (green or orange or white) and not force:
        return merge_output, False

    updated = dict(merge_output)
    updated["green_nlps"] = green
    updated["orange_nlps"] = orange
    updated["white_nlps"] = white
    updated["keyword_instances"] = tiering.get("keyword_instances") or []
    updated["tiering_method"] = tiering.get("tiering_method") or "proportional_instance_v1"
    updated["tiering_metadata"] = tiering.get("tiering_metadata") or {}
    updated["consensus_metadata"] = tiering.get("tiering_metadata") or {}
    updated["word_buckets"] = build_tier_word_buckets(green, white, orange)
    updated["keyword_buckets"] = {
        "Green": _tier_texts(green),
        "Orange": _tier_texts(orange),
        "White": _tier_texts(white),
    }
    return updated, True


def ensure_keyword_instances_in_merge_output(
    merge_output: dict[str, Any],
    *,
    keyword: str = "",
    tiering_prep: Optional[TieringPrepCache] = None,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    Add or refresh keyword_instances in merge_output for older saved sessions.

    Returns (merge_output, changed).
    """
    if not merge_output:
        return merge_output, False

    prep = tiering_prep
    if prep is not None:
        prep = hydrate_tiering_prep_gliner_meta(prep)
        if keyword and not prep.keyword:
            prep.keyword = keyword

    if not force and not merge_output_needs_keyword_instances(
        merge_output,
        tiering_prep=prep,
        keyword=keyword,
    ):
        return merge_output, False

    entities = _entities_from_merge_output(merge_output)
    if not entities:
        return merge_output, False

    keyword = (keyword or "").strip()
    avg_word_count = float(
        (merge_output.get("average_statistics") or {}).get("avg_word_count") or 0.0
    )
    competitor_domains = merge_output.get("competitor_domains") or []

    try:
        if prep is not None:
            result = compute_embedding_consensus_tiers_cached(
                keyword or prep.keyword,
                entities,
                prep=prep,
                avg_word_count=avg_word_count,
                competitor_domains=competitor_domains,
            )
        else:
            result = compute_embedding_consensus_tiers(
                keyword,
                entities,
                avg_word_count=avg_word_count,
                competitor_domains=competitor_domains,
            )
    except Exception:
        log.exception(
            "[Backfill] keyword_instances failed for keyword=%r",
            keyword,
        )
        return merge_output, False

    instances = result.get("keyword_instances") or []
    if not instances and not force:
        return merge_output, False

    updated = dict(merge_output)
    updated["keyword_instances"] = instances
    return updated, True


def _resolve_gliner_label(entity_label: str | None, catalog: list[str]) -> str:
    label = (entity_label or "").strip()
    if label in catalog:
        return label
    return "Other"


def build_nlps_by_gliner_label(
    tiered_entities: list[dict[str, Any]],
    catalog: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group tiered NLP entities by GLiNER label (unknown labels -> Other)."""
    labels = list(catalog or ENTITY_LABELS)
    buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    seen_by_label: dict[str, set[str]] = {label: set() for label in labels}

    for entity in tiered_entities:
        text = (entity.get("text") or "").strip()
        if not text:
            continue
        bucket_label = _resolve_gliner_label(entity.get("label"), labels)
        key = text.casefold()
        if key in seen_by_label[bucket_label]:
            continue
        seen_by_label[bucket_label].add(key)
        buckets[bucket_label].append(entity)

    return buckets


LEGACY_ENTITY_LABELS = frozenset({"NLP", ""})


def _needs_gliner_relabel(label: str | None) -> bool:
    cleaned = (label or "").strip()
    if not cleaned or cleaned in LEGACY_ENTITY_LABELS:
        return True
    return cleaned not in ENTITY_LABELS


def _pick_label_from_gliner_response(text: str, entities: list[dict[str, Any]]) -> str:
    if not entities:
        return "Other"
    target = text.casefold().strip()
    exact = [
        ent
        for ent in entities
        if (ent.get("text") or "").strip().casefold() == target
    ]
    pool = exact or entities
    best = max(pool, key=lambda ent: float(ent.get("score") or 0))
    label = (best.get("label") or "Other").strip() or "Other"
    return label if label in ENTITY_LABELS else "Other"


def infer_gliner_labels_for_texts(
    texts: list[str],
    *,
    keyword: str = "",
    client: GLiNERClient | None = None,
) -> dict[str, str]:
    """Infer GLiNER labels for standalone NLP phrases (used for session backfill)."""
    unique: list[str] = []
    seen: set[str] = set()
    keyword_cf = (keyword or "").strip().casefold()
    label_by_text: dict[str, str] = {}

    for text in texts:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if keyword_cf and key == keyword_cf:
            label_by_text[key] = "Phrase"

    to_infer = [text for text in unique if text.casefold() not in label_by_text]
    if not to_infer:
        return label_by_text

    gliner = client or GLiNERClient()
    batch_size = max(1, int(os.getenv("GLINER_LABEL_BACKFILL_BATCH_SIZE", "32")))

    for start in range(0, len(to_infer), batch_size):
        batch = to_infer[start : start + batch_size]
        try:
            if hasattr(gliner, "predict_entities_batch"):
                responses = gliner.predict_entities_batch(
                    batch, ENTITY_LABELS, threshold=GLINER_THRESHOLD
                )
            else:
                responses = [
                    gliner.predict_entities(item, ENTITY_LABELS, threshold=GLINER_THRESHOLD) or []
                    for item in batch
                ]
        except Exception:
            log.exception("[Backfill] GLiNER label inference failed for batch")
            for text in batch:
                label_by_text.setdefault(text.casefold(), "Other")
            continue

        if len(responses) != len(batch):
            responses = [
                gliner.predict_entities(item, ENTITY_LABELS, threshold=GLINER_THRESHOLD) or []
                for item in batch
            ]

        for text, ents in zip(batch, responses):
            label_by_text[text.casefold()] = _pick_label_from_gliner_response(text, ents or [])

    return label_by_text


def _iter_merge_entity_lists(merge_output: dict[str, Any]):
    for key in ("entities", "green_nlps", "orange_nlps", "white_nlps"):
        for entity in merge_output.get(key) or []:
            yield key, entity


def _unique_entities_needing_relabel(merge_output: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for _key, entity in _iter_merge_entity_lists(merge_output):
        text = (entity.get("text") or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        if not _needs_gliner_relabel(entity.get("label")):
            continue
        seen.add(key)
        out.append(entity)
    return out


def merge_output_needs_gliner_labels(merge_output: dict[str, Any]) -> bool:
    if not merge_output:
        return False
    if not merge_output.get("gliner_labels") or not merge_output.get("nlps_by_gliner_label"):
        return True
    return bool(_unique_entities_needing_relabel(merge_output))


def _refresh_gliner_label_fields(merge_output: dict[str, Any]) -> None:
    tiered_entities = (
        list(merge_output.get("green_nlps") or [])
        + list(merge_output.get("orange_nlps") or [])
        + list(merge_output.get("white_nlps") or [])
    )
    if not tiered_entities:
        tiered_entities = list(merge_output.get("entities") or [])
    merge_output["gliner_labels"] = list(ENTITY_LABELS)
    merge_output["nlps_by_gliner_label"] = build_nlps_by_gliner_label(tiered_entities)


def ensure_gliner_labels_in_merge_output(
    merge_output: dict[str, Any],
    *,
    keyword: str = "",
    client: GLiNERClient | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    Backfill GLiNER labels for older saved merge_output sessions.

    Returns (merge_output, changed).
    """
    if not merge_output:
        return merge_output, False

    if not force and not merge_output_needs_gliner_labels(merge_output):
        return merge_output, False

    entities_to_relabel = _unique_entities_needing_relabel(merge_output)
    had_catalog = bool(merge_output.get("gliner_labels") and merge_output.get("nlps_by_gliner_label"))

    if not entities_to_relabel and had_catalog:
        return merge_output, False

    updated = dict(merge_output)
    changed = False

    if entities_to_relabel:
        try:
            label_by_text = infer_gliner_labels_for_texts(
                [(entity.get("text") or "").strip() for entity in entities_to_relabel],
                keyword=keyword,
                client=client,
            )
        except Exception:
            log.exception("[Backfill] GLiNER label inference failed for keyword=%r", keyword)
            if had_catalog:
                return merge_output, False
            _refresh_gliner_label_fields(updated)
            return updated, True

        for _key, entity in _iter_merge_entity_lists(updated):
            text = (entity.get("text") or "").strip()
            if not text:
                continue
            if not force and not _needs_gliner_relabel(entity.get("label")):
                continue
            new_label = label_by_text.get(text.casefold())
            if not new_label:
                continue
            if entity.get("label") != new_label:
                entity["label"] = new_label
                changed = True

    _refresh_gliner_label_fields(updated)
    if not had_catalog:
        changed = True

    return updated, changed


def build_merge_response(
    entities: List[Dict[str, Any]],
    stats: Dict[str, Any],
    ranking_method: str,
    keyword: str,
    *,
    tiering_prep: Optional[TieringPrepCache] = None,
    max_numerical_per_tier: int = 2,
    json_outputs_dir: Optional[Path] = None,
    persist_keyword_json: bool = True,
    upsert_keyword_json: Optional[Callable[..., Any]] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run tiering, word buckets, and optional keyword JSON export."""
    keyword_for_file = (keyword or "").strip()
    tiering = apply_merge_tiering(
        entities,
        keyword_for_file,
        avg_word_count=float(stats.get("avg_word_count") or 0.0),
        max_numerical_per_tier=max_numerical_per_tier,
        tiering_prep=tiering_prep,
        competitor_domains=stats.get("competitor_domains") or [],
    )

    green_nlps = tiering["green_nlps"]
    orange_nlps = tiering["orange_nlps"]
    white_nlps = tiering["white_nlps"]
    dropped_numerical = tiering["dropped_numerical"]
    dropped_long_words = tiering.get("dropped_long_words") or {
        "green": 0,
        "white": 0,
        "orange": 0,
    }

    log.info(
        "[Merge] Tiering method=%s — Green=%d | Orange=%d | White=%d",
        tiering["tiering_method"],
        len(green_nlps),
        len(orange_nlps),
        len(white_nlps),
    )
    if any(dropped_numerical.values()):
        log.info(
            "[Merge] Dropped numerical NLPs — Green=%d | White=%d | Orange=%d",
            dropped_numerical["green"],
            dropped_numerical["white"],
            dropped_numerical["orange"],
        )
    if any(dropped_long_words.values()):
        log.info(
            "[Merge] Dropped long-word NLPs — Green=%d | White=%d | Orange=%d",
            dropped_long_words["green"],
            dropped_long_words["white"],
            dropped_long_words["orange"],
        )

    payload: dict[str, list[str]] = {}
    if keyword_for_file and persist_keyword_json and json_outputs_dir is not None:
        safe_name = re.sub(r"[^\w\s-]", "", keyword_for_file)
        safe_name = re.sub(r"[-\s]+", "_", safe_name).strip() or "merge"
        json_outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = json_outputs_dir / f"{safe_name}.json"
        payload = {
            "Green": _tier_texts(green_nlps),
            "Orange": _tier_texts(orange_nlps),
            "White": _tier_texts(white_nlps),
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if upsert_keyword_json and user_id is not None:
                try:
                    upsert_keyword_json(keyword_for_file, out_path.name, payload, user_id)
                except Exception:
                    log.exception("[Merge] Could not store NLP keyword JSON in PostgreSQL")
            log.info(
                "[Merge] Saved NLPs by keyword to %s (Green=%d | Orange=%d | White=%d)",
                out_path,
                len(green_nlps),
                len(orange_nlps),
                len(white_nlps),
            )
        except Exception:
            log.exception("[Merge] Could not save json outputs")

    word_buckets = build_tier_word_buckets(green_nlps, white_nlps, orange_nlps)
    x_value = stats.get("word_range_60_percent_value", 0.0)
    total_adjusted_weightage = stats.get("total_adjusted_weightage", 0.0)
    tiered_entities = list(green_nlps) + list(orange_nlps) + list(white_nlps)

    return {
        "merge_date": datetime.now().isoformat(),
        "ranking_method": ranking_method,
        "total_files_processed": stats.get("total_files", 0),
        "average_statistics": {
            "avg_word_count": stats.get("avg_word_count", 0),
            "avg_heading_count": stats.get("avg_heading_count", 0),
            "avg_paragraph_count": stats.get("avg_para_count", 0),
            "avg_images_count": stats.get("avg_images_count", 0),
            "word_range_60_percent_value": x_value,
            "total_adjusted_weightage": total_adjusted_weightage,
        },
        "total_unique_entities": len(entities),
        "total_entity_occurrences": sum(e.get("combined_count", 0) for e in entities),
        "competitor_domains": stats.get("competitor_domains") or [],
        "entities": entities,
        "green_nlps": green_nlps,
        "white_nlps": white_nlps,
        "orange_nlps": orange_nlps,
        "tiering_method": tiering["tiering_method"],
        "tiering_metadata": tiering["tiering_metadata"],
        "consensus_metadata": tiering["tiering_metadata"],
        "word_buckets": word_buckets,
        "keyword_buckets": payload if keyword_for_file else {},
        "keyword_instances": tiering.get("keyword_instances") or [],
        "gliner_labels": list(ENTITY_LABELS),
        "nlps_by_gliner_label": build_nlps_by_gliner_label(tiered_entities),
    }


def build_session_tiering_prep(
    keyword: str,
    all_items: List[Dict],
    *,
    anchor_texts: Optional[List[str]] = None,
    query_unit: Optional[np.ndarray] = None,
    query_generation_method: str = "",
    raw_gliner: Optional[List[Dict[str, Any]]] = None,
    gliner_variants: Optional[List[str]] = None,
) -> TieringPrepCache:
    entity_texts = collect_entity_texts_from_items(all_items)
    preloaded = collect_preloaded_embeddings_from_items(all_items)
    return build_tiering_prep_cache(
        keyword,
        entity_texts,
        preloaded_embeddings=preloaded,
        anchor_texts=anchor_texts,
        query_unit=query_unit,
        query_generation_method=query_generation_method,
        raw_gliner=raw_gliner,
        gliner_variants=gliner_variants,
    )
