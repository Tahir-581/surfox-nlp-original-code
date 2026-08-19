"""Generate query anchor variants via GLiNER entity extraction on the keyword."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from NLP_Extraction_and_Ranking.Gliner_ import (
    ENTITY_LABELS,
    GLiNERClient,
)

from .config import ClusterConfig

log = logging.getLogger(__name__)

_QUERY_VARIANT_CACHE: dict[str, dict[str, Any]] = {}

_PRONOUNS = {
    "i", "me", "you", "he", "him", "she", "her", "it",
    "we", "us", "they", "them", "my", "your", "his", "hers",
    "our", "their", "mine", "yours", "ours", "theirs",
    "who", "whom", "whose", "that", "this", "these", "those",
}
_EXCLUDED_SINGLE_TOKENS = _PRONOUNS | {"the", "a"}

_FRAGMENTS = {
    "tion", "ude", "ive", "nat", "iven", "attit", "ident", "itude",
    "ment", "ness", "ence", "ance", "ally", "cal", "ful", "ous",
    "ent", "ant", "est", "ity", "ly", "er", "ed",
    "al", "ic", "an", "or", "ar", "en", "on", "in", "at",
}


def _is_valid_entity_span(text_span: str) -> bool:
    text_span = (text_span or "").strip()
    if not text_span:
        return False

    if " " not in text_span:
        token = text_span.replace("'", "'").strip("\"'()[]{}<>.,;:!?-")
        base = token.split("'")[0].lower()
        if base in _EXCLUDED_SINGLE_TOKENS:
            return False

    cleaned = "".join(ch for ch in text_span if ch.isalnum())
    if len(cleaned) <= 2:
        return False

    base_lower = cleaned.lower()
    if base_lower in _FRAGMENTS:
        return False
    if len(base_lower) <= 5 and (
        base_lower.endswith(("tion", "ude", "ive", "ment", "ness", "ence", "ance"))
        or base_lower.startswith(("nat", "ident", "iven"))
    ):
        return False
    return True


def generate_gliner_keyword_entities(
    keyword: str,
    *,
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Run GLiNER on the keyword and return all deduplicated entity texts plus raw spans.
    """
    original = (keyword or "").strip()
    if not original:
        return [], []

    client = GLiNERClient()
    raw_entities = client.predict_entities(original, ENTITY_LABELS, threshold=threshold) or []

    entity_counts: Counter[str] = Counter()
    original_forms: dict[str, Counter[str]] = {}

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        text_span = str(ent.get("text", "")).strip()
        if not _is_valid_entity_span(text_span):
            continue
        norm_text = text_span.casefold()
        entity_counts[norm_text] += 1
        original_forms.setdefault(norm_text, Counter())[text_span] += 1

    variants: list[str] = []
    seen: set[str] = {original.casefold()}
    for norm_text in entity_counts:
        if norm_text in seen:
            continue
        display = original_forms[norm_text].most_common(1)[0][0]
        if display.casefold() == original.casefold():
            continue
        seen.add(norm_text)
        variants.append(display)

    variants.sort(key=lambda t: (-entity_counts[t.casefold()], t.casefold()))
    return variants, raw_entities


def generate_query_variants(
    query: str,
    config: ClusterConfig,
) -> dict[str, Any]:
    """
    Generate anchor variants for the original query using GLiNER entity extraction.

    Returns dict with original_query, variants, gliner_variants, generation_method, raw_gliner.
    All GLiNER entities are returned with no cap; no template or paraphrase fallbacks.
    """
    original = (query or "").strip()
    if not original:
        return {
            "original_query": "",
            "variants": [],
            "gliner_variants": [],
            "generation_method": "none",
            "raw_gliner": None,
            "error": "empty query",
        }

    cache_key = original.casefold()
    cached = _QUERY_VARIANT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    result = _generate_query_variants_uncached(query, config)
    _QUERY_VARIANT_CACHE[cache_key] = result
    if len(_QUERY_VARIANT_CACHE) > 128:
        oldest = next(iter(_QUERY_VARIANT_CACHE))
        _QUERY_VARIANT_CACHE.pop(oldest, None)
    return result


def _generate_query_variants_uncached(
    query: str,
    config: ClusterConfig,
) -> dict[str, Any]:
    original = (query or "").strip()
    if not original:
        return {
            "original_query": "",
            "variants": [],
            "gliner_variants": [],
            "generation_method": "none",
            "raw_gliner": None,
            "error": "empty query",
        }

    raw_gliner: list[dict[str, Any]] | None = None
    variants: list[str] = []
    method = "gliner"
    error: str | None = None

    try:
        variants, raw_gliner = generate_gliner_keyword_entities(
            original,
            threshold=config.gliner_keyword_threshold,
        )
    except Exception as exc:
        log.warning("GLiNER keyword entity extraction failed: %s", exc)
        method = "none"
        raw_gliner = []
        variants = []
        error = str(exc)

    gliner_variants = {v.casefold() for v in variants}

    result: dict[str, Any] = {
        "original_query": original,
        "variants": variants,
        "gliner_variants": sorted(gliner_variants),
        "generation_method": method,
        "raw_gliner": raw_gliner,
        "gliner_keyword_threshold": config.gliner_keyword_threshold,
    }
    if error:
        result["error"] = error
    return result
