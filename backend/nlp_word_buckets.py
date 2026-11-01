"""Build per-tier word buckets from NLP entity lists."""
from __future__ import annotations

import os
import re
from typing import Any, Sequence

from merge_entities import singularize_word
from nlp_tier_utils import split_entities_into_tiers

DEFAULT_MAX_NUMERICAL_PER_TIER = int(os.getenv("MAX_NUMERICAL_NLPS_PER_TIER", "2"))

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "her", "his", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "them", "they", "this", "to", "was", "were", "with", "you", "your",
})

MIN_SUBSTRING_WORD_LEN = 3
MIN_NLPS_PER_BUCKET = 2


def _entity_text(entity: Any) -> str:
    if isinstance(entity, dict):
        return (entity.get("text") or "").strip()
    return str(entity or "").strip()


def tokenize_nlp(text: str) -> list[str]:
    """Extract alphabetic tokens from NLP text."""
    return TOKEN_PATTERN.findall((text or "").lower())


def normalize_token(token: str) -> str:
    return singularize_word((token or "").lower().strip())


def collect_bucket_words(entities: Sequence[Any]) -> set[str]:
    """Collect all normalized bucket keys from entities in a tier."""
    words: set[str] = set()
    for entity in entities:
        text = _entity_text(entity)
        if not text:
            continue
        tokens = tokenize_nlp(text)
        if not tokens:
            continue
        if len(tokens) == 1:
            normalized = normalize_token(tokens[0])
            if normalized and normalized not in STOP_WORDS:
                words.add(normalized)
        for token in tokens:
            normalized = normalize_token(token)
            if normalized and normalized not in STOP_WORDS:
                words.add(normalized)
    return words


def nlp_matches_word(nlp_text: str, word: str) -> bool:
    """Return True if the NLP contains the bucket word."""
    if not nlp_text or not word:
        return False

    tokens = tokenize_nlp(nlp_text)
    if not tokens:
        return False

    if len(tokens) == 1:
        single = normalize_token(tokens[0])
        if single == word:
            return True

    for token in tokens:
        normalized = normalize_token(token)
        if normalized == word:
            return True
        if len(word) >= MIN_SUBSTRING_WORD_LEN and word in normalized:
            return True

    return False


def build_word_buckets(entities: Sequence[Any]) -> list[dict[str, Any]]:
    """
    Group tier NLPs by shared words.

    Returns buckets sorted by nlp_count desc, then word asc.
    """
    entity_list = [e for e in entities if _entity_text(e)]
    if not entity_list:
        return []

    bucket_words = collect_bucket_words(entity_list)
    buckets: list[dict[str, Any]] = []

    for word in bucket_words:
        matching: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for entity in entity_list:
            text = _entity_text(entity)
            key = text.casefold()
            if key in seen_texts:
                continue
            if not nlp_matches_word(text, word):
                continue
            seen_texts.add(key)
            combined = 0
            if isinstance(entity, dict):
                try:
                    combined = int(entity.get("combined_count") or 0)
                except (TypeError, ValueError):
                    combined = 0
            matching.append({"text": text, "combined_count": combined})

        if len(matching) < MIN_NLPS_PER_BUCKET:
            continue

        buckets.append({
            "word": word,
            "nlp_count": len(matching),
            "nlps": matching,
        })

    buckets.sort(key=lambda b: (-b["nlp_count"], b["word"]))
    return buckets


def build_tier_word_buckets(
    green: Sequence[Any],
    white: Sequence[Any],
    orange: Sequence[Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build word buckets for Green, White, and Orange tiers."""
    return {
        "green": build_word_buckets(green),
        "white": build_word_buckets(white),
        "orange": build_word_buckets(orange),
    }


def resolve_tier_nlps_from_merge_output(
    merge_output: dict[str, Any],
    *,
    keyword: str = "",
    max_numerical_per_tier: int = DEFAULT_MAX_NUMERICAL_PER_TIER,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Resolve Green / White / Orange NLP lists from stored merge output."""
    if not merge_output:
        return [], [], []

    if isinstance(merge_output.get("green_nlps"), list):
        return (
            merge_output.get("green_nlps") or [],
            merge_output.get("white_nlps") or [],
            merge_output.get("orange_nlps") or [],
        )

    entities = merge_output.get("entities") or []
    if not entities:
        return [], [], []

    exempt_texts = [keyword] if keyword else ()
    green, white, orange, _ = split_entities_into_tiers(
        entities,
        max_numerical_per_tier=max_numerical_per_tier,
        exempt_texts=exempt_texts,
    )
    return green, white, orange


def _word_buckets_are_stale(word_buckets: Any) -> bool:
    """Return True when stored buckets include single-NLP entries."""
    if not isinstance(word_buckets, dict):
        return False
    for tier in ("green", "white", "orange"):
        for bucket in word_buckets.get(tier) or []:
            if not isinstance(bucket, dict):
                continue
            nlp_count = bucket.get("nlp_count")
            if nlp_count is None:
                nlp_count = len(bucket.get("nlps") or [])
            if int(nlp_count or 0) < MIN_NLPS_PER_BUCKET:
                return True
    return False


def merge_output_needs_word_buckets(merge_output: dict[str, Any]) -> bool:
    """Return True when merge output needs word_buckets built or refreshed."""
    if not merge_output:
        return False

    green, white, orange = resolve_tier_nlps_from_merge_output(merge_output)
    if not (green or white or orange):
        return False

    word_buckets = merge_output.get("word_buckets")
    if word_buckets is None:
        return True
    return _word_buckets_are_stale(word_buckets)


def ensure_word_buckets_in_merge_output(
    merge_output: dict[str, Any],
    *,
    keyword: str = "",
    max_numerical_per_tier: int = DEFAULT_MAX_NUMERICAL_PER_TIER,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    Add or refresh word_buckets in merge_output.

    Returns (merge_output, changed).
    """
    if not merge_output:
        return merge_output, False

    green, white, orange = resolve_tier_nlps_from_merge_output(
        merge_output,
        keyword=keyword,
        max_numerical_per_tier=max_numerical_per_tier,
    )
    if not (green or white or orange):
        return merge_output, False

    if not force and not merge_output_needs_word_buckets(merge_output):
        return merge_output, False

    updated = dict(merge_output)
    updated["word_buckets"] = build_tier_word_buckets(green, white, orange)
    return updated, True
