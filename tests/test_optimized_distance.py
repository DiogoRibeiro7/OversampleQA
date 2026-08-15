from __future__ import annotations

import numpy as np
import pytest

from oversampleqa.caching import ValidationCache
from oversampleqa.distance import _METRICS, distance_matrix
from oversampleqa.optimized_distance import OptimizedDistanceMatrix


def _naive_matrix(X1: np.ndarray, X2: np.ndarray, metric: str) -> np.ndarray:
    func = _METRICS[metric]
    matrix = np.empty((len(X1), len(X2)), dtype=float)
    for i, u in enumerate(X1):
        for j, v in enumerate(X2):
            matrix[i, j] = func(u, v)
    return matrix


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine", "canberra", "chebyshev", "hassanat"]
)
def test_distance_matrix_matches_naive(metric):
    rng = np.random.default_rng(123)
    X1 = rng.random((12, 6))
    X2 = rng.random((8, 6))
    optimized = distance_matrix(X1, X2, metric=metric)
    baseline = _naive_matrix(X1, X2, metric)
    assert np.allclose(optimized, baseline, atol=1e-12)


def test_validation_cache_avoids_recompute(tmp_path, monkeypatch):
    cache = ValidationCache(cache_dir=tmp_path / "cache")
    optimizer = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=cache)
    rng = np.random.default_rng(5)
    X1 = rng.random((40, 10))
    X2 = rng.random((40, 10))

    original = OptimizedDistanceMatrix._compute_uncached
    calls = {"count": 0}

    def tracker(self, X1_, X2_, metric, batch_size="auto", **kwargs):
        calls["count"] += 1
        return original(self, X1_, X2_, metric, batch_size=batch_size, **kwargs)

    monkeypatch.setattr(OptimizedDistanceMatrix, "_compute_uncached", tracker)
    optimizer.compute_distance_matrix(X1, X2, metric="euclidean")
    optimizer.compute_distance_matrix(X1, X2, metric="euclidean")

    assert calls["count"] == 1
