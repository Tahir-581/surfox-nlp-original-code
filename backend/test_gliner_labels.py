"""Tests for GLiNER label preservation and merge grouping."""
from __future__ import annotations

from NLP_Extraction_and_Ranking.Gliner_ import ENTITY_LABELS, extract_entities_sliding_window
from merge_service import (
    aggregate_entities_from_items,
    build_nlps_by_gliner_label,
    ensure_gliner_labels_in_merge_output,
    merge_output_needs_gliner_labels,
)


class _MockGlinerModel:
  def __init__(self, responses):
    self._responses = responses

  def predict_entities(self, _chunk, _labels, threshold=0.5):
    return list(self._responses)


def test_extract_entities_sliding_window_preserves_most_common_label():
  model = _MockGlinerModel(
    [
      {"text": "Golden Retriever", "label": "Dog Breed", "score": 0.9},
      {"text": "Golden Retriever", "label": "Animal Breed", "score": 0.8},
      {"text": "Golden Retriever", "label": "Dog Breed", "score": 0.85},
      {"text": "Seattle", "label": "City", "score": 0.88},
    ]
  )
  text = "Golden Retriever dogs in Seattle " * 3
  entities = extract_entities_sliding_window(text, model, step_size=500, context_size=800)

  by_text = {e["text"]: e for e in entities}
  assert by_text["Golden Retriever"]["label"] == "Dog Breed"
  assert by_text["Golden Retriever"]["count"] >= 2
  assert by_text["Seattle"]["label"] == "City"


def test_aggregate_entities_picks_dominant_label_across_domains():
  items = [
    {
      "domain": "a.com",
      "nlp_terms": [
        {"text": "labrador", "label": "Dog Breed", "count": 3, "source": "gliner", "weightage": 0.9},
      ],
    },
    {
      "domain": "b.com",
      "nlp_terms": [
        {"text": "labrador", "label": "Animal Breed", "count": 1, "source": "gliner", "weightage": 0.8},
      ],
    },
  ]
  merged, _stats, _method = aggregate_entities_from_items(items)
  labrador = next(e for e in merged if e["text"].lower() == "labrador")
  assert labrador["label"] == "Dog Breed"


def test_build_nlps_by_gliner_label_groups_and_maps_unknown():
  catalog = ["Dog Breed", "City", "Other"]
  tiered = [
    {"text": "Poodle", "label": "Dog Breed"},
    {"text": "Austin", "label": "City"},
    {"text": "legacy term", "label": "NLP"},
  ]
  grouped = build_nlps_by_gliner_label(tiered, catalog=catalog)
  assert [e["text"] for e in grouped["Dog Breed"]] == ["Poodle"]
  assert [e["text"] for e in grouped["City"]] == ["Austin"]
  assert [e["text"] for e in grouped["Other"]] == ["legacy term"]
  assert grouped["Dog Breed"] is not grouped["City"]


def test_entity_labels_catalog_is_non_empty():
  assert "Dog Breed" in ENTITY_LABELS
  assert "Other" in ENTITY_LABELS


class _MockGlinerBackfillClient:
  def predict_entities_batch(self, texts, _labels, threshold=0.5):
    out = []
    for text in texts:
      tl = text.lower()
      if "labrador" in tl:
        out.append([{"text": text, "label": "Dog Breed", "score": 0.92}])
      elif "seattle" in tl:
        out.append([{"text": text, "label": "City", "score": 0.9}])
      else:
        out.append([])
    return out


def test_merge_output_needs_gliner_labels_for_legacy_sessions():
  legacy = {
    "entities": [{"text": "Labrador", "label": "NLP"}],
    "green_nlps": [{"text": "Labrador", "label": "NLP"}],
  }
  assert merge_output_needs_gliner_labels(legacy) is True


def test_ensure_gliner_labels_backfills_legacy_merge_output():
  merge_output = {
    "entities": [{"text": "Labrador Retriever", "label": "NLP"}],
    "green_nlps": [{"text": "Labrador Retriever", "label": "NLP"}],
    "orange_nlps": [],
    "white_nlps": [],
  }
  updated, changed = ensure_gliner_labels_in_merge_output(
    merge_output,
    keyword="best dog breeds",
    client=_MockGlinerBackfillClient(),
  )
  assert changed is True
  assert updated["green_nlps"][0]["label"] == "Dog Breed"
  assert "gliner_labels" in updated
  assert updated["nlps_by_gliner_label"]["Dog Breed"][0]["text"] == "Labrador Retriever"
