"""Tests for proportional keyword-instance tiering."""
from __future__ import annotations

from nlp_embedding_tiers.anchored_clustering import assign_anchor_tiers_by_sim_quota
from nlp_embedding_tiers.config import ClusterConfig
from nlp_embedding_tiers.consensus import (
    _apply_final_post_processing,
    build_final_clusters_from_instances,
)
from nlp_embedding_tiers.service import build_keyword_instances
from nlp_tier_utils import (
    assign_tiers_by_sim_quota,
    compute_bottom_quota_slot_counts,
    distribute_slots_evenly,
    split_indices_by_sim_quotas,
)

import numpy as np


def _per_query_result(query_text: str, assignments: list[dict]) -> dict:
    tier_counts = {"green": 0, "orange": 0, "white": 0}
    for item in assignments:
        tier = item.get("tier", "orange")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    return {
        "query_text": query_text,
        "nlp_assignments": assignments,
        "tier_counts": tier_counts,
    }


def test_split_indices_by_sim_quotas_ten_items():
    green, white, orange = split_indices_by_sim_quotas(10)
    assert green == 3
    assert white == 1
    assert orange == 6


def test_assign_tiers_by_sim_quota_orders_green_white_correctly():
    assignments = [
        {"text": f"nlp-{i}", "sim_to_query": i / 10}
        for i in range(10)
    ]
    tiered = assign_tiers_by_sim_quota(assignments)
    by_text = {item["text"]: item["tier"] for item in tiered}
    assert by_text["nlp-9"] == "green"
    assert by_text["nlp-8"] == "green"
    assert by_text["nlp-7"] == "green"
    assert by_text["nlp-0"] == "white"
    assert by_text["nlp-5"] == "orange"


def test_small_n_has_no_green_white_overlap():
    for n in (1, 2, 3):
        assignments = [{"text": f"nlp-{i}", "sim_to_query": float(i)} for i in range(n)]
        tiered = assign_tiers_by_sim_quota(assignments)
        greens = {item["text"] for item in tiered if item["tier"] == "green"}
        whites = {item["text"] for item in tiered if item["tier"] == "white"}
        assert greens.isdisjoint(whites)


def test_distribute_slots_evenly_with_remainder():
    assert distribute_slots_evenly(7, 3) == [3, 2, 2]
    assert sum(distribute_slots_evenly(30, 3)) == 30


def test_compute_bottom_quota_slot_counts_hundred_entities():
    green, white, orange = compute_bottom_quota_slot_counts(100)
    assert green == 35
    assert white == 10
    assert orange == 55


def test_build_final_clusters_three_anchors_hundred_entities():
    entities_by_key = {
        f"entity-{i}": {
            "text": f"entity-{i}",
            "average_weightage": float(100 - i),
            "combined_count": 1,
        }
        for i in range(100)
    }
    per_query_results = []
    for anchor_idx in range(3):
        assignments = []
        for i in range(100):
            sim = 1.0 - (i / 100) + (anchor_idx * 0.001)
            tier = "green" if i < 35 else ("white" if i >= 90 else "orange")
            assignments.append(
                {"text": f"entity-{i}", "tier": tier, "sim_to_query": round(sim, 4)}
            )
        per_query_results.append(_per_query_result(f"anchor-{anchor_idx}", assignments))

    config = ClusterConfig.from_env()
    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        "test keyword",
        config,
        avg_word_count=0.0,
        competitor_domains=[],
    )

    assert len(result["green_nlps"]) <= 35
    assert len(result["white_nlps"]) <= 10
    assert len(result["orange_nlps"]) <= 55
    assert result["consensus_metadata"]["method"] == "proportional_instance_v1"
    assert result["consensus_metadata"]["per_anchor_shares"]["green"] == [12, 12, 11]


def test_redistribution_fills_green_when_anchor_bucket_is_small():
    names = [
        "term life coverage",
        "whole life policy",
        "universal life plan",
        "variable life option",
        "final expense cover",
        "burial insurance plan",
        "senior life policy",
        "group life benefit",
        "sparse green pick",
        "low relevance term",
    ]
    entities_by_key = {name: {"text": name} for name in names}
    rich_assignments = [
        {"text": name, "tier": "green", "sim_to_query": 1.0 - i * 0.01}
        for i, name in enumerate(names[:8])
    ]
    poor_assignments = [
        {"text": "sparse green pick", "tier": "green", "sim_to_query": 0.5},
    ]
    per_query_results = [
        _per_query_result("rich", rich_assignments),
        _per_query_result("poor", poor_assignments),
    ]
    config = ClusterConfig(
        gliner_keyword_threshold=0.05,
        min_query_sim=0.75,
        green_sim_floor=0.55,
        white_sim_ceiling=0.35,
        green_vote_ratio=0.6,
        min_anchors=4,
        duplicate_query_sim=0.95,
        all_relevant_p10=0.80,
        kmeans_seed=42,
        max_iter=100,
        max_numerical_per_tier=2,
        variant_count=5,
        middle_strength=0.1,
        middle_side="original",
        fuzzy_dedup_enabled=False,
        fuzzy_dedup_threshold=75,
        fuzzy_domain_filter_enabled=False,
        fuzzy_domain_threshold=60,
    )
    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        "",
        config,
    )
    green_texts = {e["text"] for e in result["green_nlps"]}
    assert len(green_texts) == 3
    assert "sparse green pick" in green_texts or "term life coverage" in green_texts


def test_dedup_keeps_tier_from_highest_sim_anchor():
    entities_by_key = {
        "shared term": {"text": "shared term", "average_weightage": 1.0},
        "other term": {"text": "other term", "average_weightage": 0.5},
    }
    per_query_results = [
        _per_query_result(
            "anchor-a",
            [
                {"text": "shared term", "tier": "green", "sim_to_query": 0.95},
                {"text": "other term", "tier": "green", "sim_to_query": 0.8},
            ],
        ),
        _per_query_result(
            "anchor-b",
            [
                {"text": "shared term", "tier": "orange", "sim_to_query": 0.4},
            ],
        ),
    ]
    config = ClusterConfig.from_env()
    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        "",
        config,
    )
    all_texts = {
        tier: {e["text"] for e in result[f"{tier}_nlps"]}
        for tier in ("green", "orange", "white")
    }
    assert "shared term" in all_texts["green"]
    assert sum("shared term" in texts for texts in all_texts.values()) == 1
    green_entity = next(e for e in result["green_nlps"] if e["text"] == "shared term")
    assert green_entity["consensus"]["winning_sim_to_query"] == 0.95


def test_assign_anchor_tiers_by_sim_quota_unit_embeddings():
    config = ClusterConfig.from_env()
    nlp_texts = [f"nlp-{i}" for i in range(5)]
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    nlp_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = assign_anchor_tiers_by_sim_quota(
        nlp_embeddings,
        nlp_texts,
        query,
        "anchor",
        config,
    )
    assert result["clustering_method"] == "sim_quota"
    assert result["tier_counts"]["green"] == 1
    assert result["tier_counts"]["white"] == 1
    assert result["tier_counts"]["orange"] == 3
    by_text = {a["text"]: a["tier"] for a in result["nlp_assignments"]}
    assert by_text["nlp-0"] == "green"
    assert by_text["nlp-4"] == "white"


def test_build_keyword_instances_with_sim_quota_assignments():
    keyword = "best life insurance"
    entities_by_key = {
        "term life": {"text": "term life", "combined_count": 4, "sources": ["gliner"]},
        "whole life": {"text": "whole life", "combined_count": 2, "sources": ["keybert"]},
    }
    per_query_results = [
        _per_query_result(
            keyword,
            [
                {"text": "term life", "tier": "green", "sim_to_query": 0.9},
                {"text": "whole life", "tier": "orange", "sim_to_query": 0.4},
            ],
        ),
        _per_query_result(
            "life insurance",
            [{"text": "term life", "tier": "green", "sim_to_query": 0.95}],
        ),
    ]
    instances = build_keyword_instances(
        keyword,
        per_query_results,
        entities_by_key,
        gliner_variants=["life insurance"],
        raw_gliner=[{"text": "life insurance", "label": "product", "score": 0.88}],
    )
    assert len(instances) == 2
    assert instances[0]["role"] == "original"
    assert instances[1]["role"] == "gliner_entity"
    assert instances[0]["tier_counts"]["green"] == 1


def _no_fuzzy_config() -> ClusterConfig:
    return ClusterConfig(
        gliner_keyword_threshold=0.05,
        min_query_sim=0.75,
        green_sim_floor=0.55,
        white_sim_ceiling=0.35,
        green_vote_ratio=0.6,
        min_anchors=4,
        duplicate_query_sim=0.95,
        all_relevant_p10=0.80,
        kmeans_seed=42,
        max_iter=100,
        max_numerical_per_tier=2,
        variant_count=5,
        middle_strength=0.1,
        middle_side="original",
        fuzzy_dedup_enabled=False,
        fuzzy_dedup_threshold=75,
        fuzzy_domain_filter_enabled=False,
        fuzzy_domain_threshold=60,
    )


def test_final_green_order_interleaves_three_instances_two_picks_each():
    green_names = [
        ("anchor-0", ["alpha-one", "alpha-two"]),
        ("anchor-1", ["beta-one", "beta-two"]),
        ("anchor-2", ["gamma-one", "gamma-two"]),
    ]
    entities_by_key = {
        name: {"text": name}
        for _, picks in green_names
        for name in picks
    }
    for i in range(14):
        key = f"filler-{i}"
        entities_by_key[key] = {"text": key}

    per_query_results = []
    for anchor, picks in green_names:
        assignments = [
            {"text": name, "tier": "green", "sim_to_query": 1.0 - idx * 0.01}
            for idx, name in enumerate(picks)
        ]
        assignments.extend(
            {"text": f"filler-{i}", "tier": "orange", "sim_to_query": 0.2}
            for i in range(14)
        )
        per_query_results.append(_per_query_result(anchor, assignments))

    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        "",
        _no_fuzzy_config(),
    )
    assert [e["text"] for e in result["green_nlps"]] == [
        "alpha-one",
        "beta-one",
        "gamma-one",
        "alpha-two",
        "beta-two",
        "gamma-two",
    ]


def test_final_green_order_interleaves_uneven_instance_picks():
    entities_by_key = {
        "alpha-one": {"text": "alpha-one"},
        "alpha-two": {"text": "alpha-two"},
        "beta-one": {"text": "beta-one"},
    }
    for i in range(7):
        entities_by_key[f"filler-{i}"] = {"text": f"filler-{i}"}

    per_query_results = [
        _per_query_result(
            "anchor-0",
            [
                {"text": "alpha-one", "tier": "green", "sim_to_query": 0.99},
                {"text": "alpha-two", "tier": "green", "sim_to_query": 0.98},
                *[
                    {"text": f"filler-{i}", "tier": "orange", "sim_to_query": 0.2}
                    for i in range(7)
                ],
            ],
        ),
        _per_query_result(
            "anchor-1",
            [
                {"text": "beta-one", "tier": "green", "sim_to_query": 0.97},
                *[
                    {"text": f"filler-{i}", "tier": "orange", "sim_to_query": 0.2}
                    for i in range(7)
                ],
            ],
        ),
    ]
    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        "",
        _no_fuzzy_config(),
    )
    assert [e["text"] for e in result["green_nlps"]] == [
        "alpha-one",
        "beta-one",
        "alpha-two",
    ]


def test_instance_index_matches_keyword_instances_order():
    keyword = "best life insurance"
    entities_by_key = {
        "term life": {"text": "term life"},
        "whole life": {"text": "whole life"},
        "life coverage": {"text": "life coverage"},
        "padding alpha": {"text": "padding alpha"},
        "padding beta": {"text": "padding beta"},
        "padding gamma": {"text": "padding gamma"},
        "padding delta": {"text": "padding delta"},
        "padding epsilon": {"text": "padding epsilon"},
        "padding zeta": {"text": "padding zeta"},
    }
    per_query_results = [
        _per_query_result(
            "life insurance",
            [
                {"text": "life coverage", "tier": "green", "sim_to_query": 0.9},
                {"text": "padding gamma", "tier": "orange", "sim_to_query": 0.2},
                {"text": "padding zeta", "tier": "orange", "sim_to_query": 0.19},
            ],
        ),
        _per_query_result(
            keyword,
            [
                {"text": "term life", "tier": "green", "sim_to_query": 0.95},
                {"text": "padding alpha", "tier": "orange", "sim_to_query": 0.2},
                {"text": "padding epsilon", "tier": "orange", "sim_to_query": 0.19},
            ],
        ),
        _per_query_result(
            "life coverage product",
            [
                {"text": "whole life", "tier": "green", "sim_to_query": 0.85},
                {"text": "padding beta", "tier": "orange", "sim_to_query": 0.2},
                {"text": "padding delta", "tier": "orange", "sim_to_query": 0.19},
            ],
        ),
    ]
    config = _no_fuzzy_config()
    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        keyword,
        config,
        gliner_variants=["life insurance", "life coverage product"],
    )
    green = result["green_nlps"]
    assert len(green) >= 2
    by_text = {e["text"]: e for e in green}
    assert by_text["term life"]["consensus"]["instance_index"] == 1
    assert by_text["life coverage"]["consensus"]["instance_index"] == 2
    assert by_text["term life"]["consensus"]["instance_rank"] == 1
    assert by_text["life coverage"]["consensus"]["instance_rank"] == 1


def test_interleaved_order_metadata_on_entities():
    entities_by_key = {
        "alpha-one": {"text": "alpha-one"},
        "beta-one": {"text": "beta-one"},
        "filler-a": {"text": "filler-a"},
        "filler-b": {"text": "filler-b"},
    }
    per_query_results = [
        _per_query_result(
            "anchor-0",
            [
                {"text": "alpha-one", "tier": "green", "sim_to_query": 0.99},
                {"text": "filler-a", "tier": "orange", "sim_to_query": 0.2},
            ],
        ),
        _per_query_result(
            "anchor-1",
            [
                {"text": "beta-one", "tier": "green", "sim_to_query": 0.98},
                {"text": "filler-b", "tier": "orange", "sim_to_query": 0.2},
            ],
        ),
    ]
    result = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        "",
        _no_fuzzy_config(),
    )
    for entity in result["green_nlps"]:
        assert entity["consensus"]["instance_rank"] is not None
        assert entity["consensus"]["instance_index"] is not None


def test_apply_final_post_processing_drops_long_word_nlps():
    tiers = {
        "green": [
            {"text": "short phrase"},
            {"text": "one two three four"},
            {"text": "best dog breeds for apartments"},
        ],
        "orange": [],
        "white": [],
    }
    keyword = "best dog breeds for apartments"
    processed = _apply_final_post_processing(
        tiers,
        keyword=keyword,
        config=_no_fuzzy_config(),
        avg_word_count=0.0,
        competitor_domains=[],
        exempt_texts=[keyword],
    )
    texts = {entity["text"] for entity in processed["green_nlps"]}
    assert "short phrase" in texts
    assert keyword in texts
    assert "one two three four" not in texts
    assert processed["long_word_filter"]["green"] == 1

