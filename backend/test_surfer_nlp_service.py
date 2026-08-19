"""Tests for Surfer keyword NLP export loading and transform."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from surfer_nlp_service import (
    build_merge_view,
    build_viewer_payload,
    list_keyword_nlp_outputs,
    load_keyword_nlp,
    term_to_entity,
)


SAMPLE_RAW = {
    "keyword": "best dog breeds for apartments",
    "surfer_id": "16219356",
    "surfer_link": "https://app.surferseo.com/drafts/s/example",
    "nlp_terms": {
        "terms": [
            {
                "term": "dog breeds",
                "ignored": False,
                "included": True,
                "is_nlp": True,
                "target_range": {"min": 11, "max": 19},
                "use_in_heading": True,
            },
            {
                "term": "high energy",
                "ignored": False,
                "included": False,
                "is_nlp": True,
                "target_range": {"min": 2, "max": 4},
                "use_in_heading": False,
            },
        ]
    },
}


def test_term_to_entity_midpoint():
    entity = term_to_entity(SAMPLE_RAW["nlp_terms"]["terms"][0])
    assert entity["text"] == "dog breeds"
    assert entity["combined_count"] == 15
    assert entity["included"] is True
    assert entity["is_nlp"] is True
    assert entity["sources"] == ["surfer"]


def test_build_merge_view_tier_split():
    merge_view = build_merge_view(SAMPLE_RAW)
    assert len(merge_view["entities"]) == 2
    assert len(merge_view["green_nlps"]) == 1
    assert len(merge_view["white_nlps"]) == 1
    assert merge_view["orange_nlps"] == []
    assert merge_view["green_nlps"][0]["text"] == "dog breeds"
    assert merge_view["white_nlps"][0]["text"] == "high energy"
    assert merge_view["total_entity_occurrences"] == 18


def test_build_viewer_payload():
    payload = build_viewer_payload(SAMPLE_RAW, "best_dog_breeds_for_apartments")
    assert payload["slug"] == "best_dog_breeds_for_apartments"
    assert payload["keyword"] == "best dog breeds for apartments"
    assert payload["merge_view"]["ranking_method"] == "surfer"


def test_list_and_load_keyword_nlp_outputs(tmp_path, monkeypatch):
    out_dir = tmp_path / "keyword-nlp-output"
    out_dir.mkdir()
    json_name = "sample_keyword.json"
    with open(out_dir / json_name, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_RAW, f)
    with open(out_dir / "_batch_summary.csv", "w", encoding="utf-8", newline="") as f:
        f.write("keyword,surfer_id,surfer_link,status,json_file\n")
        f.write(
            "best dog breeds for apartments,16219356,https://example.com,success,sample_keyword.json\n"
        )

    monkeypatch.setenv("KEYWORD_NLP_OUTPUT_DIR", str(out_dir))

    items = list_keyword_nlp_outputs()
    assert len(items) == 1
    assert items[0]["slug"] == "sample_keyword"
    assert items[0]["keyword"] == "best dog breeds for apartments"

    payload = load_keyword_nlp("sample_keyword")
    assert payload["merge_view"]["total_unique_entities"] == 2


def test_load_keyword_nlp_invalid_slug(tmp_path, monkeypatch):
    out_dir = tmp_path / "keyword-nlp-output"
    out_dir.mkdir()
    monkeypatch.setenv("KEYWORD_NLP_OUTPUT_DIR", str(out_dir))
    with pytest.raises(ValueError):
        load_keyword_nlp("../escape")
