"""Within-tier fuzzy string deduplication for final NLP clusters."""
from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from .config import ClusterConfig

MAX_REMOVED_PAIRS_PER_TIER = 50
MAX_DOMAIN_REMOVED_PER_TIER = 50


def fuzzy_similarity(a: str, b: str) -> float:
    """Case-insensitive token-sort fuzzy similarity (0–100)."""
    return float(fuzz.token_sort_ratio((a or "").strip(), (b or "").strip()))


def normalize_competitor_hostname(domain: str) -> str:
    """Normalize a competitor hostname for fuzzy matching."""
    d = (domain or "").strip().lower()
    if not d:
        return ""
    d = d.split(":")[0]
    for prefix in ("www.", "m.", "amp."):
        if d.startswith(prefix):
            d = d[len(prefix):]
            break
    return d


def normalize_competitor_hostnames(domains: list[str] | None) -> list[str]:
    """Dedupe normalized hostnames while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in domains or []:
        normalized = normalize_competitor_hostname(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        return (item.get("text") or "").strip()
    return str(item or "").strip()


def _rank_score(item: dict[str, Any]) -> float:
    consensus = item.get("consensus") or {}
    if consensus.get("winning_sim_to_query") is not None:
        return float(consensus["winning_sim_to_query"])
    return float(consensus.get("final_rank_score") or item.get("average_weightage") or 0.0)


def _is_keyword_exempt(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    return text.strip().casefold() == keyword.strip().casefold()


def _matches_any_competitor_domain(
    nlp_text: str,
    domains: list[str],
    threshold: float,
) -> tuple[bool, str, float]:
    """Return whether nlp_text fuzzy-matches any domain at or above threshold."""
    best_sim = 0.0
    matched_domain = ""
    for domain in domains:
        sim = fuzzy_similarity(nlp_text, domain)
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            matched_domain = domain
    if matched_domain:
        return True, matched_domain, best_sim
    return False, "", 0.0


def _filter_domain_matching_candidates(
    candidates: list[dict[str, Any]],
    keyword: str,
    domains: list[str],
    config: ClusterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove NLPs that fuzzy-match any competitor hostname."""
    if not candidates or not domains or not config.fuzzy_domain_filter_enabled:
        return candidates, []

    threshold = config.fuzzy_domain_threshold
    keyword = (keyword or "").strip()
    survivors: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for candidate in candidates:
        cand_text = _item_text(candidate)
        if _is_keyword_exempt(cand_text, keyword):
            survivors.append(candidate)
            continue

        matched, matched_domain, sim = _matches_any_competitor_domain(
            cand_text,
            domains,
            threshold,
        )
        if matched:
            removed.append(
                {
                    "nlp": cand_text,
                    "domain": matched_domain,
                    "similarity": round(sim, 1),
                }
            )
            continue

        survivors.append(candidate)

    return survivors, removed


def fuzzy_dedup_within_tier(
    items: list[dict[str, Any]],
    keyword: str,
    config: ClusterConfig,
    *,
    competitor_domains: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Remove domain-matching and fuzzy-duplicate NLPs within one tier.

    Domain matches are removed first; within-tier duplicates keep the longer phrase.

    Returns (kept_entities, removed_pairs, domain_removed).
    """
    if not items or not config.fuzzy_dedup_enabled:
        return list(items), [], []

    threshold = config.fuzzy_dedup_threshold
    keyword = (keyword or "").strip()
    domains = normalize_competitor_hostnames(competitor_domains)

    candidates = [dict(item) for item in items if _item_text(item)]
    if not candidates:
        return [], [], []

    candidates, domain_removed = _filter_domain_matching_candidates(
        candidates,
        keyword,
        domains,
        config,
    )

    candidates.sort(
        key=lambda e: (len(_item_text(e)), _rank_score(e)),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    removed_pairs: list[dict[str, Any]] = []

    for candidate in candidates:
        cand_text = _item_text(candidate)
        if _is_keyword_exempt(cand_text, keyword):
            kept.append(candidate)
            continue

        matched_keep: str | None = None
        best_sim = 0.0
        for kept_item in kept:
            kept_text = _item_text(kept_item)
            sim = fuzzy_similarity(cand_text, kept_text)
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                matched_keep = kept_text

        if matched_keep is not None:
            removed_pairs.append(
                {
                    "keep": matched_keep,
                    "remove": cand_text,
                    "similarity": round(best_sim, 1),
                }
            )
            continue

        kept.append(candidate)

    kept.sort(key=lambda e: (-_rank_score(e), -len(_item_text(e))))
    return kept, removed_pairs, domain_removed


def fuzzy_dedup_all_tiers(
    tiers: dict[str, list[dict[str, Any]]],
    keyword: str,
    config: ClusterConfig,
    *,
    competitor_domains: list[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run fuzzy dedup on green, orange, white lists; return tiers + metadata."""
    if not config.fuzzy_dedup_enabled:
        return tiers, {"enabled": False, "threshold": config.fuzzy_dedup_threshold}

    domains = normalize_competitor_hostnames(competitor_domains)
    deduped: dict[str, list[dict[str, Any]]] = {}
    all_removed: dict[str, list[dict[str, Any]]] = {}
    removed_counts: dict[str, int] = {}
    domain_removed_items: dict[str, list[dict[str, Any]]] = {}
    domain_removed_counts: dict[str, int] = {}

    for tier_name in ("green", "orange", "white"):
        kept, removed, domain_removed = fuzzy_dedup_within_tier(
            tiers.get(tier_name) or [],
            keyword,
            config,
            competitor_domains=domains,
        )
        deduped[tier_name] = kept
        all_removed[tier_name] = removed[:MAX_REMOVED_PAIRS_PER_TIER]
        removed_counts[tier_name] = len(removed)
        domain_removed_items[tier_name] = domain_removed[:MAX_DOMAIN_REMOVED_PER_TIER]
        domain_removed_counts[tier_name] = len(domain_removed)

    domain_filter_meta: dict[str, Any] = {
        "enabled": bool(domains and config.fuzzy_domain_filter_enabled),
        "threshold": config.fuzzy_domain_threshold,
        "domains_checked": domains,
        "removed_counts": domain_removed_counts,
        "removed_items": domain_removed_items,
        "items_truncated_per_tier": MAX_DOMAIN_REMOVED_PER_TIER,
    }

    metadata = {
        "enabled": True,
        "threshold": config.fuzzy_dedup_threshold,
        "removed_counts": removed_counts,
        "removed_pairs": all_removed,
        "pairs_truncated_per_tier": MAX_REMOVED_PAIRS_PER_TIER,
        "domain_filter": domain_filter_meta,
    }
    return deduped, metadata
