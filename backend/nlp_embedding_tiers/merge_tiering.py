"""Apply merge-tier assignment (embedding consensus or percentile fallback)."""
from __future__ import annotations

import logging
import os
from typing import Any

from nlp_tier_utils import (
    build_anchor_exempt_texts,
    count_dropped_long_words,
    filter_entities_by_max_words,
    split_entities_into_tiers,
)

from .config import ClusterConfig
from .service import (
    TieringPrepCache,
    compute_embedding_consensus_tiers,
    compute_embedding_consensus_tiers_cached,
    hydrate_tiering_prep_gliner_meta,
)

log = logging.getLogger(__name__)

DEFAULT_MERGE_TIER_METHOD = "embedding_consensus"


def merge_tier_method() -> str:
    return (
        os.getenv("MERGE_TIER_METHOD", DEFAULT_MERGE_TIER_METHOD).strip().lower()
        or DEFAULT_MERGE_TIER_METHOD
    )


def _tiering_exempt_texts(
    keyword: str,
    tiering_prep: TieringPrepCache | None,
) -> list[str]:
    anchor_texts: list[str] = []
    gliner_variants: list[str] = []
    if tiering_prep is not None:
        anchor_texts = list(tiering_prep.anchor_texts or [])
        gliner_variants = list(tiering_prep.gliner_variants or [])
    return build_anchor_exempt_texts(
        keyword,
        anchor_texts=anchor_texts,
        gliner_variants=gliner_variants,
    )


def apply_merge_tiering(
    entities: list[dict[str, Any]],
    keyword: str,
    *,
    avg_word_count: float = 0.0,
    max_numerical_per_tier: int = 2,
    config: ClusterConfig | None = None,
    tiering_prep: TieringPrepCache | None = None,
    competitor_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    Assign green / orange / white NLP tiers for merge output.

    Returns green_nlps, orange_nlps, white_nlps, tiering_method, tiering_metadata,
    and dropped_numerical counts.
    """
    keyword = (keyword or "").strip()
    exempt = _tiering_exempt_texts(keyword, tiering_prep)
    entities = filter_entities_by_max_words(entities, exempt_texts=exempt)
    method = merge_tier_method()

    if method == "embedding_consensus":
        try:
            if tiering_prep is not None:
                config = config or ClusterConfig.from_env()
                tiering_prep = hydrate_tiering_prep_gliner_meta(tiering_prep, config)
                if keyword and not tiering_prep.keyword:
                    tiering_prep.keyword = keyword
                result = compute_embedding_consensus_tiers_cached(
                    keyword,
                    entities,
                    prep=tiering_prep,
                    avg_word_count=avg_word_count,
                    config=config,
                    competitor_domains=competitor_domains,
                )
            else:
                result = compute_embedding_consensus_tiers(
                    keyword,
                    entities,
                    avg_word_count=avg_word_count,
                    config=config,
                    competitor_domains=competitor_domains,
                )
            return {
                "green_nlps": result.get("green_nlps") or [],
                "orange_nlps": result.get("orange_nlps") or [],
                "white_nlps": result.get("white_nlps") or [],
                "keyword_instances": result.get("keyword_instances") or [],
                "tiering_method": "proportional_instance_v1",
                "tiering_metadata": result.get("consensus_metadata") or {},
                "dropped_numerical": {"green": 0, "white": 0, "orange": 0},
                "dropped_long_words": (
                    (result.get("consensus_metadata") or {}).get("long_word_filter")
                    or {"green": 0, "white": 0, "orange": 0}
                ),
            }
        except Exception:
            log.exception(
                "[Tiering] embedding_consensus failed for keyword=%r; using percentile fallback",
                keyword,
            )

    green, white, orange, dropped = split_entities_into_tiers(
        entities,
        max_numerical_per_tier=max_numerical_per_tier,
        exempt_texts=exempt,
    )
    green_before, white_before, orange_before = green, white, orange
    green = filter_entities_by_max_words(green, exempt_texts=exempt)
    white = filter_entities_by_max_words(white, exempt_texts=exempt)
    orange = filter_entities_by_max_words(orange, exempt_texts=exempt)
    dropped_long_words = {
        "green": count_dropped_long_words(green_before, green, exempt_texts=exempt),
        "white": count_dropped_long_words(white_before, white, exempt_texts=exempt),
        "orange": count_dropped_long_words(orange_before, orange, exempt_texts=exempt),
    }
    return {
        "green_nlps": green,
        "orange_nlps": orange,
        "white_nlps": white,
        "keyword_instances": [],
        "tiering_method": "percentile",
        "tiering_metadata": {"method": "percentile", "long_word_filter": dropped_long_words},
        "dropped_numerical": dropped,
        "dropped_long_words": dropped_long_words,
    }


def merge_output_has_embedding_tiers(merge_output: dict[str, Any]) -> bool:
    meta = merge_output.get("tiering_metadata") or merge_output.get("consensus_metadata") or {}
    method = merge_output.get("tiering_method") or meta.get("method") or ""
    return method in {
        "embedding_consensus_v1",
        "embedding_consensus",
        "proportional_instance_v1",
    }
