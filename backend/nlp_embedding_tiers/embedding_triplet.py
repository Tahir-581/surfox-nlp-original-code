"""Query embedding triplet for k-means centroid initialization on the unit sphere."""
from __future__ import annotations

import math

import numpy as np

from NLP_Extraction_and_Ranking.deduplicate_nlps import _l2_normalize


def _orthogonal_unit(e_unit: np.ndarray) -> np.ndarray:
    """Deterministic unit vector orthogonal to e_unit (Gram-Schmidt)."""
    dim = e_unit.shape[0]
    basis = np.zeros(dim, dtype=np.float64)
    # Pick a basis vector least aligned with e to avoid near-zero projection.
    idx = int(np.argmin(np.abs(e_unit)))
    basis[idx] = 1.0
    u = basis - float(basis @ e_unit) * e_unit
    norm = np.linalg.norm(u)
    if norm < 1e-9:
        basis = np.zeros(dim, dtype=np.float64)
        basis[(idx + 1) % dim] = 1.0
        u = basis - float(basis @ e_unit) * e_unit
        norm = np.linalg.norm(u)
    if norm < 1e-9:
        raise ValueError("Could not construct orthogonal vector for embedding triplet.")
    return (u / norm).astype(np.float32)


def _middle_angle(middle_strength: float, middle_side: str) -> float:
    strength = float(np.clip(middle_strength, 0.0, 1.0))
    half_pi = math.pi / 2.0
    if middle_side == "original":
        return half_pi * (1.0 - strength)
    if middle_side == "opposite":
        return half_pi * (1.0 + strength)
    raise ValueError("middle_side must be either 'original' or 'opposite'.")


def generate_embedding_triplet(
    embedding: np.ndarray | list[float],
    middle_strength: float = 0.1,
    middle_side: str = "original",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build three unit-sphere centroids from one query embedding:

    1. original — query direction (green anchor)
    2. opposite — antipodal direction (white anchor init)
    3. middle — on the great circle between original and opposite,
       biased by middle_strength toward original or opposite

    Parameters
    ----------
    embedding:
        Raw or L2-normalized query embedding.
    middle_strength:
        0.0 = equidistant from original/opposite (sim ~ 0 to query)
        1.0 = collapses toward original or opposite (per middle_side)
    middle_side:
        "original" — middle biased toward query embedding
        "opposite" — middle biased toward opposite embedding
    """
    e = np.asarray(embedding, dtype=np.float64).ravel()
    norm = np.linalg.norm(e)
    if norm < 1e-9:
        raise ValueError("Input embedding cannot be the zero vector.")

    original = _l2_normalize(e.reshape(1, -1))[0].astype(np.float32)
    opposite = (-original).astype(np.float32)

    u = _orthogonal_unit(original)
    t = _middle_angle(middle_strength, middle_side)
    middle_raw = math.cos(t) * original + math.sin(t) * u
    middle = _l2_normalize(middle_raw.reshape(1, -1))[0].astype(np.float32)

    return original, opposite, middle


def triplet_init_centroids(
    query_embedding_unit: np.ndarray,
    *,
    middle_strength: float = 0.1,
    middle_side: str = "original",
    k: int = 3,
) -> np.ndarray:
    """Stack [original, middle, opposite] for k-means init (pinned index 0)."""
    original, opposite, middle = generate_embedding_triplet(
        query_embedding_unit,
        middle_strength=middle_strength,
        middle_side=middle_side,
    )
    centroids = np.stack([original, middle, opposite], axis=0)
    return centroids[: max(1, min(k, 3))]


def triplet_sim_to_query_metadata(
    query_embedding_unit: np.ndarray,
    *,
    middle_strength: float = 0.1,
    middle_side: str = "original",
) -> dict[str, float]:
    """Cosine similarities of init centroids to the query embedding."""
    original, opposite, middle = generate_embedding_triplet(
        query_embedding_unit,
        middle_strength=middle_strength,
        middle_side=middle_side,
    )
    q = _l2_normalize(np.asarray(query_embedding_unit, dtype=np.float32).reshape(1, -1))[0]
    return {
        "original": round(float(original @ q), 4),
        "middle": round(float(middle @ q), 4),
        "opposite": round(float(opposite @ q), 4),
    }
