"""Tests for keyword instance payload builder."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from merge_service import (
    _entities_from_merge_output,
    build_merge_response,
    ensure_keyword_instances_in_merge_output,
    ensure_proportional_tiers_in_merge_output,
    merge_output_needs_keyword_instances,
    merge_output_needs_proportional_tier_refresh,
)
from nlp_embedding_tiers.merge_tiering import apply_merge_tiering
from nlp_embedding_tiers.service import TieringPrepCache, build_keyword_instances


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


def test_build_keyword_instances_includes_original_and_gliner_only():
    keyword = "best life insurance"
    entities_by_key = {
        "term life": {
            "text": "term life",
            "combined_count": 4,
            "average_weightage": 0.5,
            "competitor_count": 2,
            "sources": ["gliner"],
        },
        "whole life": {
            "text": "whole life",
            "combined_count": 2,
            "average_weightage": 0.3,
            "competitor_count": 1,
            "sources": ["keybert"],
        },
    }
    per_query_results = [
        _per_query_result(
            keyword,
            [
                {"text": "term life", "tier": "green", "sim_to_query": 0.81},
                {"text": "whole life", "tier": "orange", "sim_to_query": 0.55},
            ],
        ),
        _per_query_result(
            "life insurance",
            [{"text": "term life", "tier": "green", "sim_to_query": 0.9}],
        ),
        _per_query_result(
            "rejected variant",
            [{"text": "whole life", "tier": "white", "sim_to_query": 0.2}],
        ),
    ]

    instances = build_keyword_instances(
        keyword,
        per_query_results,
        entities_by_key,
        gliner_variants=["life insurance"],
        raw_gliner=[
            {"text": "life insurance", "label": "product", "score": 0.88},
        ],
    )

    assert len(instances) == 2
    assert instances[0]["text"] == keyword
    assert instances[0]["role"] == "original"
    assert instances[0]["label"] is None
    assert instances[0]["gliner_score"] is None

    gliner_instance = instances[1]
    assert gliner_instance["text"] == "life insurance"
    assert gliner_instance["role"] == "gliner_entity"
    assert gliner_instance["label"] == "product"
    assert gliner_instance["gliner_score"] == 0.88
    assert gliner_instance["nlp_count"] == 1
    assert gliner_instance["nlps"][0]["text"] == "term life"
    assert gliner_instance["nlps"][0]["combined_count"] == 4
    assert gliner_instance["nlps"][0]["sources"] == ["gliner"]


def test_build_keyword_instances_sorts_nlps_by_similarity():
    keyword = "car insurance"
    entities_by_key = {
        "auto coverage": {"text": "auto coverage", "combined_count": 1, "sources": []},
        "liability": {"text": "liability", "combined_count": 3, "sources": []},
    }
    per_query_results = [
        _per_query_result(
            keyword,
            [
                {"text": "auto coverage", "tier": "orange", "sim_to_query": 0.4},
                {"text": "liability", "tier": "green", "sim_to_query": 0.9},
            ],
        ),
    ]

    instances = build_keyword_instances(
        keyword,
        per_query_results,
        entities_by_key,
        gliner_variants=[],
        raw_gliner=[],
    )

    assert len(instances) == 1
    nlps = instances[0]["nlps"]
    assert [n["text"] for n in nlps] == ["liability", "auto coverage"]
    assert nlps[0]["sim_to_query"] == 0.9


def test_merge_output_needs_keyword_instances():
    assert merge_output_needs_keyword_instances({}) is False
    assert merge_output_needs_keyword_instances({"tiering_method": "percentile"}) is False
    assert merge_output_needs_keyword_instances(
        {"tiering_method": "proportional_instance_v1", "entities": [{"text": "a"}]}
    ) is True
    assert merge_output_needs_keyword_instances(
        {
            "tiering_method": "proportional_instance_v1",
            "keyword_instances": [{"text": "kw", "nlps": []}],
        }
    ) is False


def test_merge_output_needs_keyword_instances_under_counted():
    keyword = "best dog breeds for apartments"
    prep = TieringPrepCache(
        keyword=keyword,
        anchor_texts=[keyword, "dog breeds", "apartments"],
        query_unit=np.zeros((3, 4), dtype=np.float32),
        gliner_variants=["dog breeds", "apartments"],
        raw_gliner=[
            {"text": "dog breeds", "label": "animal", "score": 0.9},
            {"text": "apartments", "label": "location", "score": 0.85},
        ],
    )
    merge_output = {
        "tiering_method": "proportional_instance_v1",
        "keyword_instances": [{"text": keyword, "role": "original", "nlps": []}],
    }
    assert merge_output_needs_keyword_instances(
        merge_output,
        tiering_prep=prep,
        keyword=keyword,
    ) is True
    assert merge_output_needs_keyword_instances(
        {
            **merge_output,
            "keyword_instances": [
                {"text": keyword, "nlps": []},
                {"text": "dog breeds", "nlps": []},
                {"text": "apartments", "nlps": []},
            ],
        },
        tiering_prep=prep,
        keyword=keyword,
    ) is False


@patch("nlp_embedding_tiers.merge_tiering.compute_embedding_consensus_tiers_cached")
@patch("nlp_embedding_tiers.service.generate_query_variants")
def test_apply_merge_tiering_hydrates_empty_gliner_meta(mock_gen, mock_cached):
    keyword = "best dog breeds for apartments"
    gliner_variants = ["dog breeds", "apartments"]
    mock_gen.return_value = {
        "gliner_variants": gliner_variants,
        "raw_gliner": [
            {"text": "dog breeds", "label": "animal", "score": 0.9},
            {"text": "apartments", "label": "location", "score": 0.85},
        ],
        "variants": [keyword, *gliner_variants],
        "generation_method": "gliner",
    }
    instances = [
        {"text": keyword, "role": "original", "nlps": []},
        {"text": "dog breeds", "role": "gliner_entity", "nlps": []},
        {"text": "apartments", "role": "gliner_entity", "nlps": []},
    ]
    mock_cached.return_value = {
        "green_nlps": [],
        "orange_nlps": [],
        "white_nlps": [],
        "keyword_instances": instances,
        "consensus_metadata": {},
    }

    prep = TieringPrepCache(
        keyword=keyword,
        anchor_texts=[keyword, *gliner_variants],
        query_unit=np.zeros((3, 4), dtype=np.float32),
        gliner_variants=[],
        raw_gliner=[],
    )
    entities = [{"text": "bulldogs", "combined_count": 2}]

    result = apply_merge_tiering(entities, keyword, tiering_prep=prep)

    assert len(result["keyword_instances"]) == 3
    assert prep.gliner_variants == gliner_variants
    mock_cached.assert_called_once()
    called_prep = mock_cached.call_args.kwargs["prep"]
    assert called_prep.gliner_variants == gliner_variants


@patch("merge_service.compute_embedding_consensus_tiers_cached")
@patch("nlp_embedding_tiers.service.generate_query_variants")
def test_ensure_keyword_instances_backfills_under_counted(mock_gen, mock_cached):
    keyword = "best dog breeds for apartments"
    gliner_variants = ["dog breeds", "apartments"]
    mock_gen.return_value = {
        "gliner_variants": gliner_variants,
        "raw_gliner": [
            {"text": "dog breeds", "label": "animal", "score": 0.9},
            {"text": "apartments", "label": "location", "score": 0.85},
        ],
        "variants": [keyword, *gliner_variants],
        "generation_method": "gliner",
    }
    mock_cached.return_value = {
        "keyword_instances": [
            {"text": keyword, "role": "original", "nlps": []},
            {"text": "dog breeds", "role": "gliner_entity", "nlps": []},
            {"text": "apartments", "role": "gliner_entity", "nlps": []},
        ],
    }
    prep = TieringPrepCache(
        keyword=keyword,
        anchor_texts=[keyword, *gliner_variants],
        query_unit=np.zeros((3, 4), dtype=np.float32),
        gliner_variants=[],
        raw_gliner=[],
    )
    merge_output = {
        "tiering_method": "proportional_instance_v1",
        "entities": [{"text": "bulldogs"}],
        "keyword_instances": [{"text": keyword, "role": "original", "nlps": []}],
        "average_statistics": {"avg_word_count": 100},
    }

    updated, changed = ensure_keyword_instances_in_merge_output(
        merge_output,
        keyword=keyword,
        tiering_prep=prep,
    )

    assert changed is True
    assert len(updated["keyword_instances"]) == 3
    mock_cached.assert_called_once()


@patch("nlp_embedding_tiers.merge_tiering.compute_embedding_consensus_tiers_cached")
@patch("nlp_embedding_tiers.service.generate_query_variants")
def test_build_merge_response_subset_preserves_keyword_instance_count(mock_gen, mock_cached):
    keyword = "best dog breeds for apartments"
    gliner_variants = ["dog breeds", "apartments"]
    mock_gen.return_value = {
        "gliner_variants": gliner_variants,
        "raw_gliner": [
            {"text": "dog breeds", "label": "animal", "score": 0.9},
            {"text": "apartments", "label": "location", "score": 0.85},
        ],
        "variants": [keyword, *gliner_variants],
        "generation_method": "gliner",
    }
    instances = [
        {"text": keyword, "role": "original", "nlps": []},
        {"text": "dog breeds", "role": "gliner_entity", "nlps": []},
        {"text": "apartments", "role": "gliner_entity", "nlps": []},
    ]
    mock_cached.return_value = {
        "green_nlps": [{"text": "bulldogs"}],
        "orange_nlps": [],
        "white_nlps": [],
        "keyword_instances": instances,
        "consensus_metadata": {"anchors_used": 3},
    }
    prep = TieringPrepCache(
        keyword=keyword,
        anchor_texts=[keyword, *gliner_variants],
        query_unit=np.zeros((3, 4), dtype=np.float32),
        gliner_variants=[],
        raw_gliner=[],
    )
    entities = [{"text": "bulldogs", "combined_count": 2}]
    stats = {
        "total_files": 3,
        "avg_word_count": 1500.0,
        "avg_heading_count": 50.0,
        "avg_para_count": 20.0,
        "avg_images_count": 5.0,
        "competitor_domains": ["example.com"],
    }

    response = build_merge_response(
        entities,
        stats,
        "weightage",
        keyword,
        tiering_prep=prep,
        persist_keyword_json=False,
    )

    assert len(response["keyword_instances"]) == 3
    mock_cached.assert_called_once()
    called_prep = mock_cached.call_args.kwargs["prep"]
    assert called_prep.gliner_variants == gliner_variants


def test_entities_from_merge_output_falls_back_to_tier_lists():
    entities = _entities_from_merge_output(
        {
            "green_nlps": [{"text": "Alpha"}],
            "orange_nlps": [{"text": "Beta"}],
            "white_nlps": [{"text": "Alpha"}],
        }
    )
    assert [e["text"] for e in entities] == ["Alpha", "Beta"]


@patch("merge_service.compute_embedding_consensus_tiers")
def test_ensure_keyword_instances_backfills_missing(mock_compute):
    mock_compute.return_value = {
        "keyword_instances": [{"text": "car insurance", "role": "original", "nlps": []}],
    }
    merge_output = {
        "tiering_method": "embedding_consensus_v1",
        "entities": [{"text": "coverage"}],
        "average_statistics": {"avg_word_count": 100},
    }
    updated, changed = ensure_keyword_instances_in_merge_output(
        merge_output,
        keyword="car insurance",
    )
    assert changed is True
    assert len(updated["keyword_instances"]) == 1
    mock_compute.assert_called_once()


def test_merge_output_needs_proportional_tier_refresh_old_method():
    assert merge_output_needs_proportional_tier_refresh(
        {"tiering_method": "embedding_consensus_v1", "green_nlps": [{"text": "a"}]}
    )


def test_merge_output_needs_proportional_tier_refresh_missing_instance_rank():
    assert merge_output_needs_proportional_tier_refresh(
        {
            "tiering_method": "proportional_instance_v1",
            "green_nlps": [{"text": "a", "consensus": {"winning_sim_to_query": 0.9}}],
        }
    )


def test_merge_output_needs_proportional_tier_refresh_up_to_date():
    assert not merge_output_needs_proportional_tier_refresh(
        {
            "tiering_method": "proportional_instance_v1",
            "green_nlps": [
                {
                    "text": "a",
                    "consensus": {
                        "instance_index": 1,
                        "instance_rank": 1,
                        "winning_sim_to_query": 0.9,
                    },
                }
            ],
        }
    )


@patch("merge_service.apply_merge_tiering")
def test_ensure_proportional_tiers_updates_saved_merge(mock_tiering):
    mock_tiering.return_value = {
        "green_nlps": [
            {
                "text": "alpha",
                "consensus": {"instance_index": 1, "instance_rank": 1},
            }
        ],
        "orange_nlps": [],
        "white_nlps": [],
        "keyword_instances": [{"text": "kw", "role": "original", "nlps": []}],
        "tiering_method": "proportional_instance_v1",
        "tiering_metadata": {"method": "proportional_instance_v1"},
    }
    merge_output = {
        "tiering_method": "embedding_consensus_v1",
        "entities": [{"text": "alpha"}],
        "green_nlps": [{"text": "alpha", "consensus": {"final_rank_score": 1.0}}],
        "orange_nlps": [],
        "white_nlps": [],
        "average_statistics": {"avg_word_count": 100},
    }
    updated, changed = ensure_proportional_tiers_in_merge_output(
        merge_output,
        keyword="test keyword",
    )
    assert changed is True
    assert updated["tiering_method"] == "proportional_instance_v1"
    assert updated["green_nlps"][0]["consensus"]["instance_rank"] == 1
    assert updated["keyword_instances"]
    mock_tiering.assert_called_once()
