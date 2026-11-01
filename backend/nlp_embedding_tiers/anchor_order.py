"""Shared keyword-instance anchor ordering helpers."""
from __future__ import annotations

from typing import Any


def ordered_anchor_texts(
    per_query_results: list[dict[str, Any]],
    keyword: str,
    *,
    gliner_variants: set[str] | list[str] | None = None,
) -> list[str]:
    """Return keyword-instance anchor texts: original first, then GLiNER entities."""
    keyword = (keyword or "").strip()
    gliner_keys = {str(v).casefold() for v in (gliner_variants or [])}
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    for result in per_query_results:
        anchor_text = (result.get("query_text") or "").strip()
        if not anchor_text:
            continue
        anchor_key = anchor_text.casefold()
        if anchor_key in seen:
            continue
        is_original = bool(keyword) and anchor_key == keyword.casefold()
        is_gliner = anchor_key in gliner_keys
        if not is_original and not is_gliner:
            continue
        ordered.append((anchor_text, "original" if is_original else "gliner_entity"))
        seen.add(anchor_key)

    ordered.sort(key=lambda item: 0 if item[1] == "original" else 1)
    return [text for text, _ in ordered]


def anchor_instance_index_map(ordered_anchors: list[str]) -> dict[str, int]:
    """Map casefolded anchor text to 1-based keyword-instance index."""
    return {
        (text or "").strip().casefold(): idx + 1
        for idx, text in enumerate(ordered_anchors)
    }


def reorder_per_query_results(
    per_query_results: list[dict[str, Any]],
    ordered_anchors: list[str],
) -> list[dict[str, Any]]:
    """Reorder per-query results to match keyword-instance anchor order."""
    by_text = {(r.get("query_text") or "").strip(): r for r in per_query_results}
    return [by_text[anchor] for anchor in ordered_anchors if anchor in by_text]
