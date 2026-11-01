"""Unit tests for within-tier fuzzy deduplication."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nlp_embedding_tiers.config import ClusterConfig
from nlp_embedding_tiers.fuzzy_dedup import (
    fuzzy_dedup_all_tiers,
    fuzzy_dedup_within_tier,
    fuzzy_similarity,
)


def _config(**overrides) -> ClusterConfig:
    base = ClusterConfig.from_env()
    data = base.__dict__.copy()
    data.update(overrides)
    return ClusterConfig(**data)


def test_canine_canines_keeps_longer():
    items = [
        {"text": "canine", "consensus": {"final_rank_score": 10}},
        {"text": "canines", "consensus": {"final_rank_score": 9}},
    ]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(items, "", _config())
    texts = [e["text"] for e in kept]
    assert "canines" in texts
    assert "canine" not in texts
    assert len(removed) == 1
    assert removed[0]["keep"] == "canines"
    assert domain_removed == []


def test_unrelated_terms_not_merged():
    sim = fuzzy_similarity("mastiff", "loyalty")
    assert sim < 60
    items = [
        {"text": "mastiff", "consensus": {"final_rank_score": 5}},
        {"text": "loyalty", "consensus": {"final_rank_score": 4}},
    ]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(items, "", _config())
    assert len(kept) == 2
    assert removed == []
    assert domain_removed == []


def test_keyword_never_removed():
    items = [
        {"text": "best dog breeds", "consensus": {"final_rank_score": 20}},
        {"text": "best dog breed", "consensus": {"final_rank_score": 19}},
    ]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(
        items,
        "best dog breeds",
        _config(),
    )
    texts = [e["text"] for e in kept]
    assert "best dog breeds" in texts


def test_within_tier_only():
    tiers = {
        "green": [{"text": "canine", "consensus": {"final_rank_score": 1}}],
        "orange": [{"text": "canines", "consensus": {"final_rank_score": 1}}],
        "white": [],
    }
    deduped, meta = fuzzy_dedup_all_tiers(tiers, "", _config())
    assert len(deduped["green"]) == 1
    assert len(deduped["orange"]) == 1
    assert meta["removed_counts"]["green"] == 0
    assert meta["removed_counts"]["orange"] == 0


def test_domain_match_removes_nlp():
    items = [
        {"text": "petmd", "consensus": {"final_rank_score": 10}},
        {"text": "mastiff", "consensus": {"final_rank_score": 9}},
    ]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(
        items,
        "",
        _config(),
        competitor_domains=["petmd.com"],
    )
    texts = [e["text"] for e in kept]
    assert "petmd" not in texts
    assert "mastiff" in texts
    assert removed == []
    assert len(domain_removed) == 1
    assert domain_removed[0]["domain"] == "petmd.com"


def test_unrelated_nlp_kept_against_domain():
    items = [{"text": "mastiff", "consensus": {"final_rank_score": 5}}]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(
        items,
        "",
        _config(),
        competitor_domains=["petmd.com"],
    )
    assert len(kept) == 1
    assert kept[0]["text"] == "mastiff"
    assert domain_removed == []


def test_keyword_exempt_from_domain_filter():
    items = [{"text": "petmd", "consensus": {"final_rank_score": 10}}]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(
        items,
        "petmd",
        _config(),
        competitor_domains=["petmd.com"],
    )
    assert len(kept) == 1
    assert kept[0]["text"] == "petmd"
    assert domain_removed == []


def test_domain_filter_runs_before_nlp_dedup():
    items = [
        {"text": "petmd", "consensus": {"final_rank_score": 10}},
        {"text": "pet md", "consensus": {"final_rank_score": 9}},
        {"text": "mastiff", "consensus": {"final_rank_score": 8}},
    ]
    kept, removed, domain_removed = fuzzy_dedup_within_tier(
        items,
        "",
        _config(fuzzy_dedup_threshold=75),
        competitor_domains=["petmd.com"],
    )
    texts = [e["text"] for e in kept]
    assert "petmd" not in texts
    assert "pet md" in texts
    assert "mastiff" in texts
    assert len(domain_removed) == 1
    assert domain_removed[0]["nlp"] == "petmd"
    assert removed == []


def test_fuzzy_dedup_all_tiers_domain_metadata():
    tiers = {
        "green": [{"text": "petmd", "consensus": {"final_rank_score": 1}}],
        "orange": [],
        "white": [],
    }
    deduped, meta = fuzzy_dedup_all_tiers(
        tiers,
        "",
        _config(),
        competitor_domains=["www.petmd.com"],
    )
    assert deduped["green"] == []
    domain_filter = meta["domain_filter"]
    assert domain_filter["enabled"] is True
    assert domain_filter["threshold"] == 60
    assert "petmd.com" in domain_filter["domains_checked"]
    assert domain_filter["removed_counts"]["green"] == 1


if __name__ == "__main__":
    test_canine_canines_keeps_longer()
    test_unrelated_terms_not_merged()
    test_keyword_never_removed()
    test_within_tier_only()
    test_domain_match_removes_nlp()
    test_unrelated_nlp_kept_against_domain()
    test_keyword_exempt_from_domain_filter()
    test_domain_filter_runs_before_nlp_dedup()
    test_fuzzy_dedup_all_tiers_domain_metadata()
    print("All tests passed.")
