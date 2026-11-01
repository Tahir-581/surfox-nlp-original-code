"""Compare embedding consensus tiers against baseline percentile tiers."""
from __future__ import annotations

from typing import Any

from nlp_tier_utils import split_entities_into_tiers


def _tier_map_from_lists(
    green: list[Any],
    white: list[Any],
    orange: list[Any],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in green:
        key = (item.get("text") or "").strip().casefold()
        if key:
            mapping[key] = "green"
    for item in white:
        key = (item.get("text") or "").strip().casefold()
        if key:
            mapping[key] = "white"
    for item in orange:
        key = (item.get("text") or "").strip().casefold()
        if key:
            mapping[key] = "orange"
    return mapping


def _avg_mean_sim(tier_entities: list[dict[str, Any]]) -> float:
    sims = []
    for e in tier_entities:
        c = e.get("consensus") or {}
        if c.get("winning_sim_to_query") is not None:
            sims.append(float(c["winning_sim_to_query"]))
        elif "mean_sim" in c:
            sims.append(float(c["mean_sim"]))
    return round(sum(sims) / len(sims), 4) if sims else 0.0


def build_evaluation_report(
    *,
    keyword: str,
    entities: list[dict[str, Any]],
    final_clusters: dict[str, Any],
    consensus: dict[str, Any],
    merge_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce 09_evaluation_report.json content."""
    baseline_green, baseline_white, baseline_orange, _ = split_entities_into_tiers(
        entities,
        exempt_texts=[keyword] if keyword else (),
    )

    if merge_output and merge_output.get("green_nlps"):
        baseline_green = merge_output.get("green_nlps") or baseline_green
        baseline_white = merge_output.get("white_nlps") or baseline_white
        baseline_orange = merge_output.get("orange_nlps") or baseline_orange

    baseline_map = _tier_map_from_lists(baseline_green, baseline_white, baseline_orange)

    new_green = final_clusters.get("green_nlps") or []
    new_orange = final_clusters.get("orange_nlps") or []
    new_white = final_clusters.get("white_nlps") or []
    new_map = _tier_map_from_lists(new_green, new_white, new_orange)

    moved: list[dict[str, Any]] = []
    for key, new_tier in new_map.items():
        old_tier = baseline_map.get(key)
        if old_tier and old_tier != new_tier:
            text = next(
                (e.get("text") for e in entities if (e.get("text") or "").casefold() == key),
                key,
            )
            moved.append({"text": text, "baseline_tier": old_tier, "new_tier": new_tier})

    # Stability histogram from consensus scores
    stability_hist = {"high": 0, "medium": 0, "low": 0}
    for item in consensus.get("scores", []):
        conf = item.get("confidence", "medium")
        stability_hist[conf] = stability_hist.get(conf, 0) + 1

    keyword_key = (keyword or "").strip().casefold()
    keyword_in_green = keyword_key in {
        (e.get("text") or "").strip().casefold() for e in new_green
    }

    # Duplicate across tiers check
    all_texts = []
    dup_across_tiers: list[str] = []
    for tier_name, tier_list in (
        ("green", new_green),
        ("orange", new_orange),
        ("white", new_white),
    ):
        for e in tier_list:
            t = (e.get("text") or "").strip().casefold()
            if t in all_texts:
                dup_across_tiers.append(t)
            all_texts.append(t)

    avg_sim_by_tier = {
        "green": _avg_mean_sim(new_green),
        "orange": _avg_mean_sim(new_orange),
        "white": _avg_mean_sim(new_white),
    }

    fuzzy_dedup = (final_clusters.get("consensus_metadata") or {}).get("fuzzy_dedup") or {}

    return {
        "keyword": keyword,
        "tier_size_baseline": {
            "green": len(baseline_green),
            "white": len(baseline_white),
            "orange": len(baseline_orange),
        },
        "tier_size_new": {
            "green": len(new_green),
            "orange": len(new_orange),
            "white": len(new_white),
        },
        "tier_size_delta": {
            "green": len(new_green) - len(baseline_green),
            "orange": len(new_orange) - len(baseline_orange),
            "white": len(new_white) - len(baseline_white),
        },
        "avg_mean_sim_by_tier": avg_sim_by_tier,
        "sanity_green_gt_orange": avg_sim_by_tier["green"] >= avg_sim_by_tier["orange"],
        "sanity_orange_gt_white": avg_sim_by_tier["orange"] >= avg_sim_by_tier["white"],
        "keyword_in_green": keyword_in_green,
        "duplicate_across_tiers": dup_across_tiers,
        "fuzzy_dedup": fuzzy_dedup,
        "fuzzy_dedup_removed_total": sum(
            (fuzzy_dedup.get("removed_counts") or {}).values()
        ) if fuzzy_dedup.get("enabled") else 0,
        "terms_moved_tier": moved,
        "terms_moved_count": len(moved),
        "stability_histogram": stability_hist,
        "green_to_white_moves": [
            m for m in moved if m["baseline_tier"] == "green" and m["new_tier"] == "white"
        ],
        "white_to_green_moves": [
            m for m in moved if m["baseline_tier"] == "white" and m["new_tier"] == "green"
        ],
    }
