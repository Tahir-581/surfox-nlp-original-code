"""Query-anchored spherical k-means tier assignment per query anchor."""
from __future__ import annotations

from typing import Any

import numpy as np

from nlp_tier_utils import assign_tiers_by_sim_quota

from NLP_Extraction_and_Ranking.deduplicate_nlps import _l2_normalize, _spherical_kmeans

from .config import ClusterConfig
from .embedding_triplet import triplet_init_centroids, triplet_sim_to_query_metadata

TIER_NAMES = ("green", "orange", "white")


def _init_centroids_from_triplet(
    query_embedding_unit: np.ndarray,
    config: ClusterConfig,
    *,
    k: int = 3,
) -> np.ndarray:
    """Initialize centroids from query embedding triplet (original / middle / opposite)."""
    return triplet_init_centroids(
        query_embedding_unit,
        middle_strength=config.middle_strength,
        middle_side=config.middle_side,
        k=k,
    )


def _relabel_clusters_by_query_distance(
    centroids_unit: np.ndarray,
    query_embedding_unit: np.ndarray,
) -> dict[int, str]:
    k = centroids_unit.shape[0]
    sims = [float(centroids_unit[j] @ query_embedding_unit) for j in range(k)]
    order = sorted(range(k), key=lambda j: sims[j], reverse=True)
    tier_order = ["green", "orange", "white"]
    mapping: dict[int, str] = {}
    for rank, cluster_id in enumerate(order):
        tier = tier_order[rank] if rank < len(tier_order) else "orange"
        mapping[cluster_id] = tier
    return mapping


def _apply_sim_thresholds(
    tier: str,
    sim_to_query: float,
    *,
    green_sim_floor: float,
    white_sim_ceiling: float,
) -> tuple[str, bool]:
    adjusted = False
    if tier == "green" and sim_to_query < green_sim_floor:
        tier = "orange" if sim_to_query >= white_sim_ceiling else "white"
        adjusted = True
    elif tier == "white" and sim_to_query > white_sim_ceiling:
        tier = "orange"
        adjusted = True
    return tier, adjusted


def assign_anchor_tiers_by_sim_quota(
    nlp_embeddings_unit: np.ndarray,
    nlp_texts: list[str],
    query_embedding_unit: np.ndarray,
    query_text: str,
    config: ClusterConfig,
) -> dict[str, Any]:
    """
    Assign per-anchor tiers by sim_to_query quota: top 35% green, bottom 10% white.
    """
    if len(nlp_texts) != nlp_embeddings_unit.shape[0]:
        raise ValueError("nlp_texts length must match embedding rows")

    query_unit = _l2_normalize(query_embedding_unit.reshape(1, -1))[0]
    emb_unit = _l2_normalize(nlp_embeddings_unit.astype(np.float32, copy=True))
    n = emb_unit.shape[0]
    if n == 0:
        return {
            "query_text": query_text,
            "clustering_method": "sim_quota",
            "nlp_assignments": [],
            "tier_counts": {"green": 0, "orange": 0, "white": 0},
            "sim_stats": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
        }

    sims_to_query = (emb_unit @ query_unit).astype(np.float32)
    raw_assignments = [
        {
            "text": text,
            "sim_to_query": round(float(sims_to_query[i]), 4),
        }
        for i, text in enumerate(nlp_texts)
    ]
    tiered = assign_tiers_by_sim_quota(raw_assignments)
    tier_counts = {"green": 0, "orange": 0, "white": 0}
    assignments: list[dict[str, Any]] = []
    for item in tiered:
        tier = item.get("tier", "orange")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        assignments.append(
            {
                "text": item["text"],
                "tier": tier,
                "sim_to_query": item.get("sim_to_query"),
            }
        )

    return {
        "query_text": query_text,
        "clustering_method": "sim_quota",
        "nlp_assignments": assignments,
        "tier_counts": tier_counts,
        "sim_stats": {
            "p10": round(float(np.percentile(sims_to_query, 10)), 4),
            "p50": round(float(np.percentile(sims_to_query, 50)), 4),
            "p90": round(float(np.percentile(sims_to_query, 90)), 4),
        },
    }


def cluster_nlps_for_anchor(
    nlp_embeddings_unit: np.ndarray,
    nlp_texts: list[str],
    query_embedding_unit: np.ndarray,
    query_text: str,
    config: ClusterConfig,
) -> dict[str, Any]:
    """
    Run query-anchored spherical k-means (k=3) and return per-NLP tier assignments.
    """
    if len(nlp_texts) != nlp_embeddings_unit.shape[0]:
        raise ValueError("nlp_texts length must match embedding rows")

    query_unit = _l2_normalize(query_embedding_unit.reshape(1, -1))[0]
    emb_unit = _l2_normalize(nlp_embeddings_unit.astype(np.float32, copy=True))

    centroid_init_meta = {
        "method": "embedding_triplet",
        "middle_strength": config.middle_strength,
        "middle_side": config.middle_side,
        "init_centroids_sim_to_query": triplet_sim_to_query_metadata(
            query_unit,
            middle_strength=config.middle_strength,
            middle_side=config.middle_side,
        ),
    }

    n = emb_unit.shape[0]
    if n == 0:
        return {
            "query_text": query_text,
            "clustering_method": "anchored_spherical_kmeans",
            "centroid_init": centroid_init_meta,
            "centroids": {},
            "nlp_assignments": [],
            "tier_counts": {"green": 0, "orange": 0, "white": 0},
            "all_relevant_mode": False,
        }

    sims_to_query = (emb_unit @ query_unit).astype(np.float32)
    all_relevant_mode = False
    if n >= 10:
        p10 = float(np.percentile(sims_to_query, 10))
        all_relevant_mode = p10 >= config.all_relevant_p10

    k = min(3, n) if n < 3 else 3
    init_centroids = _init_centroids_from_triplet(query_unit, config, k=k)
    labels, centroids = _spherical_kmeans(
        emb_unit,
        init_centroids,
        max_iter=config.max_iter,
        pinned_indices=[0],
    )

    cluster_to_tier = _relabel_clusters_by_query_distance(centroids, query_unit)

    # Build centroid metadata keyed by tier name
    centroids_meta: dict[str, Any] = {}
    for cluster_id, tier_name in cluster_to_tier.items():
        c = centroids[cluster_id]
        centroids_meta[tier_name] = {
            "cluster_id": int(cluster_id),
            "pinned_to_query": cluster_id == 0,
            "sim_to_query": round(float(c @ query_unit), 4),
        }

    assignments: list[dict[str, Any]] = []
    tier_counts = {"green": 0, "orange": 0, "white": 0}

    for i, text in enumerate(nlp_texts):
        sim = float(sims_to_query[i])
        cluster_id = int(labels[i])
        tier = cluster_to_tier.get(cluster_id, "orange")
        tier, threshold_adjusted = _apply_sim_thresholds(
            tier,
            sim,
            green_sim_floor=config.green_sim_floor,
            white_sim_ceiling=config.white_sim_ceiling,
        )
        if all_relevant_mode and tier == "white":
            tier = "orange"
            threshold_adjusted = True
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        assignments.append(
            {
                "text": text,
                "cluster_id": cluster_id,
                "tier": tier,
                "sim_to_query": round(sim, 4),
                "threshold_adjusted": threshold_adjusted,
            }
        )

    return {
        "query_text": query_text,
        "clustering_method": "anchored_spherical_kmeans",
        "centroid_init": centroid_init_meta,
        "centroids": centroids_meta,
        "nlp_assignments": assignments,
        "tier_counts": tier_counts,
        "all_relevant_mode": all_relevant_mode,
        "sim_stats": {
            "p10": round(float(np.percentile(sims_to_query, 10)), 4) if n else 0.0,
            "p50": round(float(np.percentile(sims_to_query, 50)), 4) if n else 0.0,
            "p90": round(float(np.percentile(sims_to_query, 90)), 4) if n else 0.0,
        },
    }
