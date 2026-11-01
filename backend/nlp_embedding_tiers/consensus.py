"""Aggregate per-anchor tier assignments into final green / orange / white tiers."""
from __future__ import annotations

import math
from typing import Any

from nlp_tier_utils import (
    assign_tier_quotas,
    build_anchor_exempt_texts,
    cap_numerical_nlps,
    compute_bottom_quota_slot_counts,
    count_dropped_long_words,
    distribute_slots_evenly,
    filter_entities_by_max_words,
)

from .config import ClusterConfig
from .fuzzy_dedup import fuzzy_dedup_all_tiers
from .anchor_order import (
    anchor_instance_index_map,
    ordered_anchor_texts,
    reorder_per_query_results,
)

TIER_POINT_VALUES = {"green": 3, "orange": 2, "white": 1}

def _entity_key(text: str) -> str:
    return (text or "").strip().casefold()


def _vote_threshold(n_anchors: int, ratio: float) -> int:
    return max(1, math.ceil(n_anchors * ratio))


def build_consensus_scores(
    per_query_results: list[dict[str, Any]],
    similarity_matrix: dict[str, dict[str, float]],
    entities_by_key: dict[str, dict[str, Any]],
    config: ClusterConfig,
) -> dict[str, Any]:
    """
    Aggregate votes across anchors and compute consensus scores per NLP.
    """
    n_anchors = len(per_query_results)
    vote_records: dict[str, dict[str, Any]] = {}

    for result in per_query_results:
        query_text = result.get("query_text", "")
        for item in result.get("nlp_assignments", []):
            text = item.get("text", "")
            key = _entity_key(text)
            if not key:
                continue
            rec = vote_records.setdefault(
                key,
                {
                    "text": text,
                    "green_votes": 0,
                    "orange_votes": 0,
                    "white_votes": 0,
                    "per_anchor": {},
                },
            )
            tier = item.get("tier", "orange")
            rec[f"{tier}_votes"] = rec.get(f"{tier}_votes", 0) + 1
            rec["per_anchor"][query_text] = {
                "tier": tier,
                "sim_to_query": item.get("sim_to_query"),
            }

    green_min = _vote_threshold(n_anchors, config.green_vote_ratio)
    white_min = _vote_threshold(n_anchors, config.green_vote_ratio)

    scores_list: list[dict[str, Any]] = []
    for key, rec in vote_records.items():
        sim_row = similarity_matrix.get(rec["text"], {})
        mean_sim = (
            sum(sim_row.values()) / len(sim_row) if sim_row else 0.0
        )
        gv = rec["green_votes"]
        ov = rec["orange_votes"]
        wv = rec["white_votes"]
        tier_points = (
            gv * TIER_POINT_VALUES["green"]
            + ov * TIER_POINT_VALUES["orange"]
            + wv * TIER_POINT_VALUES["white"]
        )
        stability = gv / n_anchors if n_anchors else 0.0
        entity = entities_by_key.get(key, {"text": rec["text"]})
        avg_weightage = float(entity.get("average_weightage") or 0.0)
        final_rank_score = tier_points + mean_sim

        if gv >= green_min and mean_sim >= config.green_sim_floor:
            final_tier = "green"
            confidence = "high" if gv == n_anchors else "medium"
        elif wv >= white_min and mean_sim <= config.white_sim_ceiling:
            final_tier = "white"
            confidence = "high" if wv == n_anchors else "medium"
        else:
            final_tier = "orange"
            confidence = "low" if max(gv, ov, wv) <= n_anchors // 2 else "medium"

        scores_list.append(
            {
                "text": rec["text"],
                "green_votes": gv,
                "orange_votes": ov,
                "white_votes": wv,
                "stability": round(stability, 4),
                "mean_sim": round(mean_sim, 4),
                "tier_points": tier_points,
                "final_rank_score": round(final_rank_score, 4),
                "final_tier": final_tier,
                "confidence": confidence,
                "average_weightage": avg_weightage,
                "per_anchor": rec["per_anchor"],
            }
        )

    scores_list.sort(
        key=lambda x: (
            -x["final_rank_score"],
            -x["mean_sim"],
            -x["average_weightage"],
        )
    )

    return {
        "anchors_used": n_anchors,
        "green_vote_min": green_min,
        "white_vote_min": white_min,
        "scores": scores_list,
    }


def _sort_bucket_for_tier(items: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    if tier == "white":
        return sorted(items, key=lambda x: float(x.get("sim_to_query") or 0.0))
    return sorted(items, key=lambda x: -(float(x.get("sim_to_query") or 0.0)))


def _fill_tier_slots(
    per_query_results: list[dict[str, Any]],
    tier: str,
    shares_per_anchor: list[int],
    instance_index_by_anchor: dict[str, int],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Fill tier slots equally across anchors with redistribution on shortfall.
    Returns (selections_by_anchor, redistribution_log).
    """
    n_anchors = len(per_query_results)
    buckets: list[list[dict[str, Any]]] = []
    anchors: list[str] = []
    for result in per_query_results:
        anchor = result.get("query_text", "")
        anchors.append(anchor)
        bucket = [
            item
            for item in (result.get("nlp_assignments") or [])
            if item.get("tier") == tier
        ]
        buckets.append(_sort_bucket_for_tier(bucket, tier))

    cursors = [0] * n_anchors
    selected_keys: set[str] = set()
    selections_by_anchor: list[list[dict[str, Any]]] = [[] for _ in range(n_anchors)]
    redistribution_log: list[dict[str, Any]] = []

    def _take_from_anchor(anchor_idx: int, count: int, phase: str) -> int:
        taken = 0
        anchor = anchors[anchor_idx]
        instance_index = instance_index_by_anchor.get(anchor.casefold(), anchor_idx + 1)
        while taken < count and cursors[anchor_idx] < len(buckets[anchor_idx]):
            item = buckets[anchor_idx][cursors[anchor_idx]]
            cursors[anchor_idx] += 1
            key = _entity_key(item.get("text", ""))
            if not key or key in selected_keys:
                continue
            selected_keys.add(key)
            sim = float(item.get("sim_to_query") or 0.0)
            rank_in_anchor = len(selections_by_anchor[anchor_idx]) + 1
            selections_by_anchor[anchor_idx].append(
                {
                    "text": item.get("text", ""),
                    "tier": tier,
                    "anchor": anchor,
                    "instance_index": instance_index,
                    "instance_rank": rank_in_anchor,
                    "sim_to_query": sim,
                }
            )
            if phase == "redistribute":
                redistribution_log.append(
                    {
                        "tier": tier,
                        "anchor": anchor,
                        "instance_index": instance_index,
                        "instance_rank": rank_in_anchor,
                        "text": item.get("text", ""),
                        "sim_to_query": sim,
                    }
                )
            taken += 1
        return taken

    unfilled = 0
    for anchor_idx in range(n_anchors):
        share = shares_per_anchor[anchor_idx] if anchor_idx < len(shares_per_anchor) else 0
        taken = _take_from_anchor(anchor_idx, share, "round1")
        unfilled += max(0, share - taken)

    while unfilled > 0:
        spare_order = sorted(
            range(n_anchors),
            key=lambda idx: len(buckets[idx]) - cursors[idx],
            reverse=True,
        )
        progressed = False
        for anchor_idx in spare_order:
            if unfilled <= 0:
                break
            spare = len(buckets[anchor_idx]) - cursors[anchor_idx]
            if spare <= 0:
                continue
            taken = _take_from_anchor(anchor_idx, 1, "redistribute")
            if taken:
                unfilled -= 1
                progressed = True
        if not progressed:
            break

    return selections_by_anchor, redistribution_log


def _interleave_selections_by_instance_rank(
    selections_by_anchor: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Round-robin interleave: all instance rank-1 picks, then rank-2, etc."""
    if not selections_by_anchor:
        return []
    max_rank = max(len(lst) for lst in selections_by_anchor)
    interleaved: list[dict[str, Any]] = []
    for rank in range(max_rank):
        for anchor_idx in range(len(selections_by_anchor)):
            picks = selections_by_anchor[anchor_idx]
            if rank < len(picks):
                interleaved.append(picks[rank])
    return interleaved


def _flatten_grouped_selections(
    selections_by_anchor: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for picks in selections_by_anchor:
        flat.extend(picks)
    return flat


def _dedupe_grouped_tier_selections(
    tier_grouped: dict[str, list[list[dict[str, Any]]]],
) -> dict[str, list[list[dict[str, Any]]]]:
    """If an NLP appears in multiple tiers, keep the tier from the highest sim_to_query."""
    best_by_key: dict[str, dict[str, Any]] = {}
    for tier, grouped in tier_grouped.items():
        for picks in grouped:
            for pick in picks:
                key = _entity_key(pick.get("text", ""))
                if not key:
                    continue
                sim = float(pick.get("sim_to_query") or 0.0)
                existing = best_by_key.get(key)
                if existing is None or sim > float(existing.get("sim_to_query") or 0.0):
                    best_by_key[key] = {**pick, "tier": tier}

    deduped: dict[str, list[list[dict[str, Any]]]] = {}
    for tier, grouped in tier_grouped.items():
        tier_grouped_out: list[list[dict[str, Any]]] = []
        for picks in grouped:
            tier_picks: list[dict[str, Any]] = []
            for pick in picks:
                key = _entity_key(pick.get("text", ""))
                best = best_by_key.get(key)
                if best and best.get("tier") == tier:
                    tier_picks.append(best)
            tier_grouped_out.append(tier_picks)
        deduped[tier] = tier_grouped_out
    return deduped


def _filter_grouped_to_keys(
    selections_by_anchor: list[list[dict[str, Any]]],
    allowed_keys: set[str],
) -> list[list[dict[str, Any]]]:
    return [
        [pick for pick in picks if _entity_key(pick.get("text", "")) in allowed_keys]
        for picks in selections_by_anchor
    ]


def _sort_tier_by_instance_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(entity: dict[str, Any]) -> tuple[int, int, float]:
        consensus = entity.get("consensus") or {}
        rank = consensus.get("instance_rank")
        index = consensus.get("instance_index")
        if rank is not None and index is not None:
            return (int(rank), int(index), 0.0)
        sim = float(consensus.get("winning_sim_to_query") or 0.0)
        return (9999, 9999, -sim)

    return sorted(items, key=sort_key)


def _dedupe_selections_by_best_sim(
    tier_selections: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """If an NLP appears in multiple tiers, keep the tier from the highest sim_to_query."""
    best_by_key: dict[str, dict[str, Any]] = {}
    for tier in ("green", "orange", "white"):
        for item in tier_selections.get(tier, []):
            key = _entity_key(item.get("text", ""))
            if not key:
                continue
            sim = float(item.get("sim_to_query") or 0.0)
            existing = best_by_key.get(key)
            if existing is None or sim > float(existing.get("sim_to_query") or 0.0):
                best_by_key[key] = {**item, "tier": tier}

    deduped: dict[str, list[dict[str, Any]]] = {
        "green": [],
        "orange": [],
        "white": [],
    }
    for item in best_by_key.values():
        deduped[item["tier"]].append(item)
    return deduped


def _trim_tier_to_cap(items: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if len(items) <= cap:
        return items
    return sorted(
        items,
        key=lambda x: -(float(x.get("sim_to_query") or 0.0)),
    )[:cap]


def _apply_final_post_processing(
    tiers: dict[str, list[dict[str, Any]]],
    *,
    keyword: str,
    config: ClusterConfig,
    avg_word_count: float,
    competitor_domains: list[str] | None,
    exempt_texts: list[str] | None = None,
) -> dict[str, Any]:
    tiers, fuzzy_meta = fuzzy_dedup_all_tiers(
        tiers,
        keyword,
        config,
        competitor_domains=competitor_domains,
    )

    exempt = list(exempt_texts or ([keyword] if keyword else []))
    green_before = tiers["green"]
    orange_before = tiers["orange"]
    white_before = tiers["white"]
    tiers["green"] = filter_entities_by_max_words(tiers["green"], exempt_texts=exempt)
    tiers["orange"] = filter_entities_by_max_words(tiers["orange"], exempt_texts=exempt)
    tiers["white"] = filter_entities_by_max_words(tiers["white"], exempt_texts=exempt)
    long_word_filter = {
        "green": count_dropped_long_words(green_before, tiers["green"], exempt_texts=exempt),
        "orange": count_dropped_long_words(orange_before, tiers["orange"], exempt_texts=exempt),
        "white": count_dropped_long_words(white_before, tiers["white"], exempt_texts=exempt),
    }

    green = cap_numerical_nlps(
        tiers["green"],
        max_numerical=config.max_numerical_per_tier,
        exempt_texts=exempt,
    )
    orange = cap_numerical_nlps(
        tiers["orange"],
        max_numerical=config.max_numerical_per_tier,
        exempt_texts=exempt,
    )
    white = cap_numerical_nlps(
        tiers["white"],
        max_numerical=config.max_numerical_per_tier,
        exempt_texts=exempt,
    )

    green = _sort_tier_by_instance_order(green)
    orange = _sort_tier_by_instance_order(orange)
    white = _sort_tier_by_instance_order(white)

    all_entities = green + orange + white
    for entity in all_entities:
        cc = entity.get("competitor_count") or 1
        if cc >= 3:
            mult = 3
        elif cc == 2:
            mult = 2
        else:
            mult = 1
        entity.setdefault("competitor_multiplier", mult)
        aw = float(entity.get("average_weightage") or 0.0)
        entity.setdefault("adjusted_weightage", aw * mult)

    x_value = avg_word_count * 0.60
    total_adj = sum(float(e.get("adjusted_weightage") or 0.0) for e in all_entities)
    for entity in all_entities:
        adj = float(entity.get("adjusted_weightage") or 0.0)
        prob = adj / total_adj if total_adj > 0 else 0.0
        entity["word_range"] = math.ceil(prob * x_value) if x_value > 0 else 0

    return {
        "green_nlps": green,
        "orange_nlps": orange,
        "white_nlps": white,
        "fuzzy_meta": fuzzy_meta,
        "long_word_filter": long_word_filter,
    }


def build_final_clusters_from_instances(
    per_query_results: list[dict[str, Any]],
    entities_by_key: dict[str, dict[str, Any]],
    keyword: str,
    config: ClusterConfig,
    *,
    avg_word_count: float = 0.0,
    competitor_domains: list[str] | None = None,
    gliner_variants: list[str] | None = None,
) -> dict[str, Any]:
    """Build final tiers via equal proportional slot fill across keyword instances."""
    ordered_anchors = ordered_anchor_texts(
        per_query_results,
        keyword,
        gliner_variants=gliner_variants,
    )
    if not ordered_anchors:
        ordered_anchors = [
            (r.get("query_text") or "").strip()
            for r in per_query_results
            if (r.get("query_text") or "").strip()
        ]
    per_query_results = reorder_per_query_results(per_query_results, ordered_anchors)
    instance_index_by_anchor = anchor_instance_index_map(ordered_anchors)

    n_anchors = len(per_query_results)
    total = len(entities_by_key)
    green_slots, white_slots, orange_slots = compute_bottom_quota_slot_counts(total)

    per_anchor_shares = {
        "green": distribute_slots_evenly(green_slots, n_anchors),
        "white": distribute_slots_evenly(white_slots, n_anchors),
        "orange": distribute_slots_evenly(orange_slots, n_anchors),
    }

    tier_grouped: dict[str, list[list[dict[str, Any]]]] = {
        "green": [],
        "orange": [],
        "white": [],
    }
    redistribution_log: list[dict[str, Any]] = []
    for tier in ("green", "white", "orange"):
        grouped, redistributed = _fill_tier_slots(
            per_query_results,
            tier,
            per_anchor_shares[tier],
            instance_index_by_anchor,
        )
        tier_grouped[tier] = grouped
        redistribution_log.extend(redistributed)

    deduped_grouped = _dedupe_grouped_tier_selections(tier_grouped)
    tier_caps = {
        "green": green_slots,
        "white": white_slots,
        "orange": orange_slots,
    }
    interleaved_by_tier: dict[str, list[dict[str, Any]]] = {}
    for tier_name, grouped in deduped_grouped.items():
        flat = _flatten_grouped_selections(grouped)
        trimmed = _trim_tier_to_cap(flat, tier_caps[tier_name])
        trimmed_keys = {_entity_key(pick.get("text", "")) for pick in trimmed}
        filtered_grouped = _filter_grouped_to_keys(grouped, trimmed_keys)
        interleaved_by_tier[tier_name] = _interleave_selections_by_instance_rank(
            filtered_grouped
        )

    per_anchor_meta: dict[str, dict[str, dict[str, Any]]] = {}
    for result in per_query_results:
        anchor = result.get("query_text", "")
        for item in result.get("nlp_assignments") or []:
            key = _entity_key(item.get("text", ""))
            if not key:
                continue
            per_anchor_meta.setdefault(key, {})[anchor] = {
                "tier": item.get("tier"),
                "sim_to_query": item.get("sim_to_query"),
            }

    tiers: dict[str, list[dict[str, Any]]] = {"green": [], "orange": [], "white": []}
    for tier_name, picks in interleaved_by_tier.items():
        for pick in picks:
            key = _entity_key(pick.get("text", ""))
            entity = dict(entities_by_key.get(key, {"text": pick.get("text", "")}))
            winning_sim = float(pick.get("sim_to_query") or 0.0)
            entity["consensus"] = {
                "per_anchor": per_anchor_meta.get(key, {}),
                "instance_index": pick.get("instance_index"),
                "instance_rank": pick.get("instance_rank"),
                "winning_sim_to_query": round(winning_sim, 4),
                "winning_anchor": pick.get("anchor"),
                "method": "proportional_instance_v1",
            }
            tiers[tier_name].append(entity)

    processed = _apply_final_post_processing(
        tiers,
        keyword=keyword,
        config=config,
        avg_word_count=avg_word_count,
        competitor_domains=competitor_domains,
        exempt_texts=build_anchor_exempt_texts(
            keyword,
            gliner_variants=gliner_variants or [],
        ),
    )

    return {
        "green_nlps": processed["green_nlps"],
        "orange_nlps": processed["orange_nlps"],
        "white_nlps": processed["white_nlps"],
        "consensus_metadata": {
            "anchors_used": n_anchors,
            "method": "proportional_instance_v1",
            "slot_counts": {
                "green": green_slots,
                "white": white_slots,
                "orange": orange_slots,
            },
            "per_anchor_shares": per_anchor_shares,
            "redistribution_log": redistribution_log,
            "fuzzy_dedup": processed["fuzzy_meta"],
            "long_word_filter": processed["long_word_filter"],
        },
    }


def build_final_clusters(
    consensus: dict[str, Any],
    entities_by_key: dict[str, dict[str, Any]],
    keyword: str,
    config: ClusterConfig,
    *,
    avg_word_count: float = 0.0,
    competitor_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Build final tier lists with quota assignment, numerical capping, and word_range."""
    sorted_entities: list[dict[str, Any]] = []
    for item in consensus.get("scores", []):
        key = _entity_key(item.get("text", ""))
        entity = dict(entities_by_key.get(key, {"text": item.get("text", "")}))
        entity["consensus"] = {
            "green_votes": item.get("green_votes"),
            "orange_votes": item.get("orange_votes"),
            "white_votes": item.get("white_votes"),
            "mean_sim": item.get("mean_sim"),
            "final_rank_score": item.get("final_rank_score"),
            "confidence": item.get("confidence"),
            "semantic_tier": item.get("final_tier"),
        }
        sorted_entities.append(entity)

    green_raw, white_raw, orange_raw, _ = assign_tier_quotas(
        sorted_entities,
        apply_numerical_cap=False,
    )
    tiers: dict[str, list[dict[str, Any]]] = {
        "green": green_raw,
        "orange": orange_raw,
        "white": white_raw,
    }

    tiers, fuzzy_meta = fuzzy_dedup_all_tiers(
        tiers,
        keyword,
        config,
        competitor_domains=competitor_domains,
    )

    exempt = [keyword] if keyword else ()
    tiers["green"] = filter_entities_by_max_words(tiers["green"], exempt_texts=exempt)
    tiers["orange"] = filter_entities_by_max_words(tiers["orange"], exempt_texts=exempt)
    tiers["white"] = filter_entities_by_max_words(tiers["white"], exempt_texts=exempt)

    green = cap_numerical_nlps(
        tiers["green"],
        max_numerical=config.max_numerical_per_tier,
        exempt_texts=exempt,
    )
    orange = cap_numerical_nlps(
        tiers["orange"],
        max_numerical=config.max_numerical_per_tier,
        exempt_texts=exempt,
    )
    white = cap_numerical_nlps(
        tiers["white"],
        max_numerical=config.max_numerical_per_tier,
        exempt_texts=exempt,
    )

    # word_range from adjusted_weightage (same as merge endpoint)
    all_entities = green + orange + white
    for entity in all_entities:
        cc = entity.get("competitor_count") or 1
        if cc >= 3:
            mult = 3
        elif cc == 2:
            mult = 2
        else:
            mult = 1
        entity.setdefault("competitor_multiplier", mult)
        aw = float(entity.get("average_weightage") or 0.0)
        entity.setdefault("adjusted_weightage", aw * mult)

    x_value = avg_word_count * 0.60
    total_adj = sum(float(e.get("adjusted_weightage") or 0.0) for e in all_entities)
    for entity in all_entities:
        adj = float(entity.get("adjusted_weightage") or 0.0)
        prob = adj / total_adj if total_adj > 0 else 0.0
        entity["word_range"] = math.ceil(prob * x_value) if x_value > 0 else 0

    return {
        "green_nlps": green,
        "orange_nlps": orange,
        "white_nlps": white,
        "consensus_metadata": {
            "anchors_used": consensus.get("anchors_used"),
            "method": "embedding_consensus_v1",
            "fuzzy_dedup": fuzzy_meta,
        },
    }
