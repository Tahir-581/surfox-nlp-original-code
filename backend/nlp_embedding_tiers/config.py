"""Environment-driven configuration for embedding consensus tiering."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ClusterConfig:
    gliner_keyword_threshold: float
    min_query_sim: float
    green_sim_floor: float
    white_sim_ceiling: float
    green_vote_ratio: float
    min_anchors: int
    duplicate_query_sim: float
    all_relevant_p10: float
    kmeans_seed: int
    max_iter: int
    max_numerical_per_tier: int
    variant_count: int
    middle_strength: float
    middle_side: str
    fuzzy_dedup_enabled: bool
    fuzzy_dedup_threshold: float
    fuzzy_domain_filter_enabled: bool
    fuzzy_domain_threshold: float

    @classmethod
    def from_env(cls) -> ClusterConfig:
        return cls(
            gliner_keyword_threshold=float(
                os.getenv("GLINER_KEYWORD_ANCHOR_THRESHOLD", "0.05")
            ),
            min_query_sim=float(os.getenv("CLUSTER_MIN_QUERY_SIM", "0.75")),
            green_sim_floor=float(os.getenv("CLUSTER_GREEN_SIM_FLOOR", "0.55")),
            white_sim_ceiling=float(os.getenv("CLUSTER_WHITE_SIM_CEILING", "0.35")),
            green_vote_ratio=float(os.getenv("CLUSTER_GREEN_VOTE_RATIO", "0.6")),
            min_anchors=int(os.getenv("CLUSTER_MIN_ANCHORS", "4")),
            duplicate_query_sim=float(os.getenv("CLUSTER_DUPLICATE_QUERY_SIM", "0.95")),
            all_relevant_p10=float(os.getenv("CLUSTER_ALL_RELEVANT_P10", "0.80")),
            kmeans_seed=int(os.getenv("CLUSTER_KMEANS_SEED", "42")),
            max_iter=int(os.getenv("CLUSTER_KMEANS_MAX_ITER", "100")),
            max_numerical_per_tier=int(os.getenv("MAX_NUMERICAL_NLPS_PER_TIER", "2")),
            # Legacy; no longer caps GLiNER keyword anchor count.
            variant_count=int(os.getenv("CLUSTER_VARIANT_COUNT", "5")),
            middle_strength=float(os.getenv("CLUSTER_MIDDLE_STRENGTH", "0.1")),
            middle_side=os.getenv("CLUSTER_MIDDLE_SIDE", "original").strip().lower()
            or "original",
            fuzzy_dedup_enabled=_env_bool("CLUSTER_FUZZY_DEDUP_ENABLED", True),
            fuzzy_dedup_threshold=float(os.getenv("CLUSTER_FUZZY_DEDUP_THRESHOLD", "75")),
            fuzzy_domain_filter_enabled=_env_bool("CLUSTER_FUZZY_DOMAIN_FILTER_ENABLED", True),
            fuzzy_domain_threshold=float(os.getenv("CLUSTER_FUZZY_DOMAIN_THRESHOLD", "60")),
        )

    def public_dict(self) -> dict:
        """Config safe to write to JSON artifacts."""
        return asdict(self)
