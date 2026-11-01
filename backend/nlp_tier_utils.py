"""Tier splitting and numerical NLP capping for Green / White / Orange buckets."""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

NUMERICAL_NLP_PATTERN = re.compile(r"\d")
NLP_WORD_SPLIT_PATTERN = re.compile(r"[\s\-/_.]+|\W+")

MAX_NLP_WORDS = 3

GREEN_TIER_RATIO = 0.35
WHITE_TIER_RATIO_OF_REMAINDER = 0.1
WHITE_TIER_RATIO_BOTTOM = 0.10


def is_numerical_nlp(text: str) -> bool:
    """Return True if the term contains at least one digit."""
    return bool(NUMERICAL_NLP_PATTERN.search(text or ""))


def is_exempt_nlp_text(text: str, exempt_texts: Iterable[str]) -> bool:
    """Return True when text matches an exempt anchor phrase (case-insensitive)."""
    normalized = (text or "").strip().casefold()
    if not normalized:
        return False
    exempt = {(t or "").strip().casefold() for t in exempt_texts if (t or "").strip()}
    return normalized in exempt


def is_exempt_numerical_nlp(text: str, exempt_texts: Iterable[str]) -> bool:
    """Return True when text matches a keyword anchor (case-insensitive)."""
    return is_exempt_nlp_text(text, exempt_texts)


def build_anchor_exempt_texts(
    keyword: str = "",
    *,
    anchor_texts: Iterable[str] = (),
    gliner_variants: Iterable[str] = (),
) -> list[str]:
    """Build deduplicated exempt anchor phrases for word-length filtering."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in [keyword, *anchor_texts, *gliner_variants]:
        text = (raw or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def count_nlp_words(text: str) -> int:
    """Count words using normalized split on whitespace and punctuation boundaries."""
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return len([token for token in NLP_WORD_SPLIT_PATTERN.split(normalized) if token])


def exceeds_max_nlp_words(
    text: str,
    *,
    max_words: int = MAX_NLP_WORDS,
) -> bool:
    """Return True when text has more than max_words tokens."""
    return count_nlp_words(text) > max_words


def filter_entities_by_max_words(
    items: Sequence[Any],
    *,
    max_words: int = MAX_NLP_WORDS,
    exempt_texts: Iterable[str] = (),
) -> list[Any]:
    """Drop entity items whose text exceeds max_words unless exempt."""
    kept: list[Any] = []
    for item in items:
        text = _item_text(item)
        if not text:
            continue
        if exceeds_max_nlp_words(text, max_words=max_words) and not is_exempt_nlp_text(
            text, exempt_texts
        ):
            continue
        kept.append(item)
    return kept


def count_dropped_long_words(
    before: Sequence[Any],
    after: Sequence[Any],
    *,
    exempt_texts: Iterable[str] = (),
    max_words: int = MAX_NLP_WORDS,
) -> int:
    """Count non-exempt items removed by filter_entities_by_max_words."""
    after_keys = {_item_text(i).casefold() for i in after if _item_text(i)}
    dropped = 0
    for item in before:
        text = _item_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in after_keys:
            continue
        if exceeds_max_nlp_words(text, max_words=max_words) and not is_exempt_nlp_text(
            text, exempt_texts
        ):
            dropped += 1
    return dropped


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        return (item.get("text") or "").strip()
    return str(item or "").strip()


def cap_numerical_nlps(
    items: Sequence[Any],
    *,
    max_numerical: int = 2,
    exempt_texts: Iterable[str] = (),
) -> list[Any]:
    """Keep at most max_numerical non-exempt digit-containing terms; drop the rest."""
    kept: list[Any] = []
    numerical_count = 0
    for item in items:
        text = _item_text(item)
        if not text:
            continue
        if is_numerical_nlp(text) and not is_exempt_numerical_nlp(text, exempt_texts):
            if numerical_count >= max_numerical:
                continue
            numerical_count += 1
        kept.append(item)
    return kept


def count_dropped_numerical(
    before: Sequence[Any],
    after: Sequence[Any],
    *,
    exempt_texts: Iterable[str] = (),
) -> int:
    """Count non-exempt numerical items removed by cap_numerical_nlps."""
    before_keys = {_item_text(i).casefold() for i in before if _item_text(i)}
    after_keys = {_item_text(i).casefold() for i in after if _item_text(i)}
    dropped = 0
    for item in before:
        text = _item_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in after_keys:
            continue
        if is_numerical_nlp(text) and not is_exempt_numerical_nlp(text, exempt_texts):
            dropped += 1
    return dropped


def split_indices_by_sim_quotas(n: int) -> tuple[int, int, int]:
    """Return green / white / orange counts for top-35% / bottom-10% / middle split."""
    if n <= 0:
        return 0, 0, 0
    green_count = max(1, int(n * GREEN_TIER_RATIO))
    white_count = max(1, int(n * WHITE_TIER_RATIO_BOTTOM))
    if green_count + white_count > n:
        white_count = max(0, n - green_count)
    orange_count = max(0, n - green_count - white_count)
    return green_count, white_count, orange_count


def compute_bottom_quota_slot_counts(total: int) -> tuple[int, int, int]:
    """Global slot counts: 35% green, 10% white (bottom), rest orange."""
    if total <= 0:
        return 0, 0, 0
    green_slots = max(1, int(total * GREEN_TIER_RATIO))
    white_slots = max(1, int(total * WHITE_TIER_RATIO_BOTTOM))
    if green_slots + white_slots > total:
        white_slots = max(0, total - green_slots)
    orange_slots = max(0, total - green_slots - white_slots)
    return green_slots, white_slots, orange_slots


def distribute_slots_evenly(total_slots: int, n_anchors: int) -> list[int]:
    """Split total_slots across n_anchors with integer remainder distribution."""
    if n_anchors <= 0:
        return []
    if total_slots <= 0:
        return [0] * n_anchors
    base = total_slots // n_anchors
    remainder = total_slots % n_anchors
    return [base + (1 if i < remainder else 0) for i in range(n_anchors)]


def assign_tiers_by_sim_quota(
    assignments: list[dict[str, Any]],
    *,
    sim_key: str = "sim_to_query",
) -> list[dict[str, Any]]:
    """
    Label assignments green / orange / white by sim rank:
    top 35% green, bottom 10% white, middle orange.
    """
    if not assignments:
        return []
    sorted_items = sorted(
        assignments,
        key=lambda x: -(float(x.get(sim_key) or 0.0)),
    )
    n = len(sorted_items)
    green_count, white_count, _ = split_indices_by_sim_quotas(n)
    orange_end = n - white_count if white_count else n
    tier_by_key: dict[str, str] = {}
    for i, item in enumerate(sorted_items):
        text_key = (item.get("text") or "").strip().casefold()
        if not text_key:
            continue
        if i < green_count:
            tier = "green"
        elif white_count and i >= orange_end:
            tier = "white"
        else:
            tier = "orange"
        tier_by_key[text_key] = tier

    result: list[dict[str, Any]] = []
    for item in assignments:
        text_key = (item.get("text") or "").strip().casefold()
        out = dict(item)
        out["tier"] = tier_by_key.get(text_key, "orange")
        result.append(out)
    return result


def _split_sorted_entities(
    sorted_entities: Sequence[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Split a pre-sorted list into green / white / orange raw slices."""
    total = len(sorted_entities)
    if total == 0:
        return [], [], []

    green_count = max(1, int(total * GREEN_TIER_RATIO))
    green_raw = list(sorted_entities[:green_count])
    remaining = list(sorted_entities[green_count:])
    white_count = (
        max(1, int(len(remaining) * WHITE_TIER_RATIO_OF_REMAINDER)) if remaining else 0
    )
    white_raw = remaining[:white_count]
    orange_raw = remaining[white_count:]
    return green_raw, white_raw, orange_raw


def assign_tier_quotas(
    sorted_entities: Sequence[Any],
    *,
    max_numerical_per_tier: int = 2,
    exempt_texts: Iterable[str] = (),
    apply_numerical_cap: bool = True,
) -> tuple[list[Any], list[Any], list[Any], dict[str, int]]:
    """
    Split a pre-sorted entity list into Green (top 35%), White (top 10% of remainder),
    Orange (rest), then optionally cap numerical NLPs per tier.

    Returns (green, white, orange, dropped_counts_by_tier).
    """
    green_raw, white_raw, orange_raw = _split_sorted_entities(sorted_entities)
    if not green_raw and not white_raw and not orange_raw:
        return [], [], [], {"green": 0, "white": 0, "orange": 0}

    if not apply_numerical_cap:
        return green_raw, white_raw, orange_raw, {"green": 0, "white": 0, "orange": 0}

    green = cap_numerical_nlps(
        green_raw,
        max_numerical=max_numerical_per_tier,
        exempt_texts=exempt_texts,
    )
    white = cap_numerical_nlps(
        white_raw,
        max_numerical=max_numerical_per_tier,
        exempt_texts=exempt_texts,
    )
    orange = cap_numerical_nlps(
        orange_raw,
        max_numerical=max_numerical_per_tier,
        exempt_texts=exempt_texts,
    )

    dropped = {
        "green": count_dropped_numerical(green_raw, green, exempt_texts=exempt_texts),
        "white": count_dropped_numerical(white_raw, white, exempt_texts=exempt_texts),
        "orange": count_dropped_numerical(orange_raw, orange, exempt_texts=exempt_texts),
    }
    return green, white, orange, dropped


def split_entities_into_tiers(
    entities: Sequence[Any],
    *,
    max_numerical_per_tier: int = 2,
    exempt_texts: Iterable[str] = (),
) -> tuple[list[Any], list[Any], list[Any], dict[str, int]]:
    """
    Split entities into Green (top 35%), White (top 10% of remainder), Orange (rest),
    then cap numerical NLPs per tier.

    Returns (green, white, orange, dropped_counts_by_tier).
    """
    sorted_entities = sorted(
        list(entities),
        key=lambda x: (x.get("average_weightage") if isinstance(x, dict) else 0) or 0,
        reverse=True,
    )
    return assign_tier_quotas(
        sorted_entities,
        max_numerical_per_tier=max_numerical_per_tier,
        exempt_texts=exempt_texts,
    )
