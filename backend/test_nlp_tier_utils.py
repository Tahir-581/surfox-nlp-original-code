"""Tests for green / white / orange tier quota assignment."""
from __future__ import annotations

from nlp_tier_utils import (
    GREEN_TIER_RATIO,
    WHITE_TIER_RATIO_OF_REMAINDER,
    assign_tier_quotas,
    build_anchor_exempt_texts,
    count_dropped_long_words,
    count_nlp_words,
    filter_entities_by_max_words,
    split_entities_into_tiers,
)


def _entities(n: int) -> list[dict]:
    return [
        {
            "text": f"entity-{chr(ord('a') + (i % 26))}-slot",
            "average_weightage": float(n - i),
        }
        for i in range(n)
    ]


def _expected_counts(total: int) -> tuple[int, int, int]:
    if total == 0:
        return 0, 0, 0
    green = max(1, int(total * GREEN_TIER_RATIO))
    remaining = total - green
    white = max(1, int(remaining * WHITE_TIER_RATIO_OF_REMAINDER)) if remaining else 0
    orange = remaining - white
    return green, white, orange


def test_split_entities_into_tiers_green_quota():
    for total in (10, 11, 3, 100, 50):
        entities = _entities(total)
        green, white, orange, _ = split_entities_into_tiers(entities)
        exp_green, exp_white, exp_orange = _expected_counts(total)
        assert len(green) == exp_green
        assert len(white) == exp_white
        assert len(orange) == exp_orange
        assert len(green) + len(white) + len(orange) == total


def test_assign_tier_quotas_preserves_input_order():
    """Pre-sorted list must not be re-sorted by average_weightage."""
    entities = [
        {"text": "high-consensus", "average_weightage": 1.0},
        {"text": "mid-consensus", "average_weightage": 99.0},
        {"text": "low-consensus", "average_weightage": 50.0},
        {"text": "tail-alpha", "average_weightage": 10.0},
        {"text": "tail-beta", "average_weightage": 5.0},
        {"text": "tail-gamma", "average_weightage": 2.0},
        {"text": "tail-delta", "average_weightage": 1.0},
        {"text": "tail-epsilon", "average_weightage": 0.5},
        {"text": "tail-zeta", "average_weightage": 0.1},
        {"text": "tail-eta", "average_weightage": 0.0},
    ]
    green, white, orange, _ = assign_tier_quotas(
        entities,
        apply_numerical_cap=False,
    )
    assert [e["text"] for e in green] == ["high-consensus", "mid-consensus", "low-consensus"]
    assert len(white) == 1
    assert white[0]["text"] == "tail-alpha"
    assert [e["text"] for e in orange] == [
        "tail-beta",
        "tail-gamma",
        "tail-delta",
        "tail-epsilon",
        "tail-zeta",
        "tail-eta",
    ]


def test_white_is_ten_percent_of_remainder():
    entities = _entities(100)
    green, white, orange, _ = split_entities_into_tiers(entities)
    assert len(green) == 35
    assert len(white) == 6
    assert len(orange) == 59
    assert len(green) + len(white) + len(orange) == 100


def test_count_nlp_words_examples():
    assert count_nlp_words("best dog breeds") == 3
    assert count_nlp_words("cost-effective health plans") == 4
    assert count_nlp_words("U.S. healthcare") == 3
    assert count_nlp_words("cost-effective plans") == 3


def test_filter_entities_by_max_words():
    items = [
        {"text": "short phrase"},
        {"text": "one two three four"},
        {"text": "best dog breeds"},
    ]
    kept = filter_entities_by_max_words(items)
    texts = {item["text"] for item in kept}
    assert texts == {"short phrase", "best dog breeds"}


def test_filter_entities_exempt_keyword():
    keyword = "best dog breeds for apartments"
    items = [
        {"text": keyword},
        {"text": "one two three four"},
    ]
    kept = filter_entities_by_max_words(items, exempt_texts=[keyword])
    assert [item["text"] for item in kept] == [keyword]


def test_build_anchor_exempt_texts_dedupes():
    exempt = build_anchor_exempt_texts(
        "Dog Food",
        anchor_texts=["Dog Food", "Treats"],
        gliner_variants=["Treats", "Puppy Chow"],
    )
    assert exempt == ["Dog Food", "Treats", "Puppy Chow"]


def test_count_dropped_long_words():
    before = [
        {"text": "short phrase"},
        {"text": "one two three four"},
    ]
    after = filter_entities_by_max_words(before)
    assert count_dropped_long_words(before, after) == 1
