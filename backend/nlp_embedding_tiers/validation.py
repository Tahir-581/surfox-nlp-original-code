"""Validate and filter GLiNER-generated query anchor variants."""
from __future__ import annotations

from typing import Any

import numpy as np

from NLP_Extraction_and_Ranking.bge_client import BGETritonClient
from NLP_Extraction_and_Ranking.deduplicate_nlps import _l2_normalize

from .config import ClusterConfig

bge_client = BGETritonClient()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(_l2_normalize(a.reshape(1, -1))[0] @ _l2_normalize(b.reshape(1, -1))[0])


def validate_query_anchors(
    original_query: str,
    variants: list[str],
    config: ClusterConfig,
    *,
    gliner_variants: set[str] | None = None,
) -> dict[str, Any]:
    """Embed and validate query anchors; return accepted list."""
    original = (original_query or "").strip()
    candidates = [original] + [v.strip() for v in variants if v.strip()]
    gliner_keys = {v.casefold() for v in (gliner_variants or set())}

    if not original:
        return {
            "accepted_anchors": [],
            "rejected": [],
            "original_embedding_index": 0,
        }

    embeddings = bge_client.encode(candidates, is_query=True)
    original_emb = embeddings[0]

    accepted: list[dict[str, Any]] = [
        {
            "text": original,
            "role": "original",
            "sim_to_original": 1.0,
            "embedding_index": 0,
        }
    ]
    rejected: list[dict[str, Any]] = []
    seen_texts = {original.casefold()}

    for i, variant in enumerate(variants, start=1):
        if i >= len(embeddings):
            break
        v = variant.strip()
        if not v:
            continue
        sim = _cosine_sim(embeddings[i], original_emb)
        is_gliner_entity = v.casefold() in gliner_keys

        dup = False
        for acc in accepted:
            acc_idx = acc["embedding_index"]
            if _cosine_sim(embeddings[i], embeddings[acc_idx]) >= config.duplicate_query_sim:
                rejected.append(
                    {
                        "text": v,
                        "reason": "duplicate",
                        "sim_to_original": round(sim, 4),
                        "gliner_entity": is_gliner_entity,
                    }
                )
                dup = True
                break
        if dup:
            continue

        passes_sim = sim >= config.min_query_sim
        if not passes_sim and not is_gliner_entity:
            rejected.append(
                {
                    "text": v,
                    "reason": "below_min_sim",
                    "sim_to_original": round(sim, 4),
                    "gliner_entity": False,
                }
            )
            continue

        if v.casefold() in seen_texts:
            continue
        seen_texts.add(v.casefold())
        role = f"gliner_entity_{len(accepted)}" if is_gliner_entity else f"variant_{len(accepted)}"
        accepted.append(
            {
                "text": v,
                "role": role,
                "sim_to_original": round(sim, 4),
                "embedding_index": i,
                "gliner_entity": is_gliner_entity,
            }
        )

    return {
        "accepted_anchors": accepted,
        "rejected": rejected,
        "original_embedding_index": 0,
        "min_anchors_required": config.min_anchors,
    }


def ensure_minimum_anchors(
    original_query: str,
    variants: list[str],
    config: ClusterConfig,
    *,
    gliner_variants: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Validate GLiNER entity anchors against the original query.
    Returns (accepted texts including original first, validation_payload).
    """
    validation = validate_query_anchors(
        original_query,
        variants,
        config,
        gliner_variants=gliner_variants,
    )
    anchor_texts = [a["text"] for a in validation["accepted_anchors"]]
    validation["final_anchor_count"] = len(anchor_texts)
    return anchor_texts, validation
