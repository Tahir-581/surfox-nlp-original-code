"""Unit tests for query embedding triplet centroid initialization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nlp_embedding_tiers.embedding_triplet import generate_embedding_triplet


def _unit_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def test_triplet_unit_vectors_and_opposite():
    e = np.random.default_rng(0).standard_normal(64)
    original, opposite, middle = generate_embedding_triplet(e, middle_strength=0.1)

    assert abs(_unit_norm(original) - 1.0) < 1e-5
    assert abs(_unit_norm(opposite) - 1.0) < 1e-5
    assert abs(_unit_norm(middle) - 1.0) < 1e-5
    assert float(original @ opposite) < -0.99


def test_middle_distinct_from_poles():
    e = np.random.default_rng(1).standard_normal(128)
    original, opposite, middle = generate_embedding_triplet(
        e, middle_strength=0.1, middle_side="original"
    )
    assert float(original @ middle) < 0.99
    assert float(opposite @ middle) < 0.99


def test_middle_strength_toward_original():
    e = np.random.default_rng(2).standard_normal(128)
    e = e / np.linalg.norm(e)
    _, _, mid_low = generate_embedding_triplet(e, middle_strength=0.0, middle_side="original")
    _, _, mid_high = generate_embedding_triplet(e, middle_strength=0.5, middle_side="original")
    assert float(e @ mid_low) < float(e @ mid_high)


if __name__ == "__main__":
    test_triplet_unit_vectors_and_opposite()
    test_middle_distinct_from_poles()
    test_middle_strength_toward_original()
    print("All tests passed.")
