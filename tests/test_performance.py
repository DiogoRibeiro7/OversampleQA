import time

import numpy as np

from oversampleqa.distance import _METRICS
from oversampleqa.optimized_distance import OptimizedDistanceMatrix


def _naive_distance_matrix(X1: np.ndarray, X2: np.ndarray, metric: str) -> np.ndarray:
    func = _METRICS[metric]
    result = np.empty((len(X1), len(X2)), dtype=float)
    for i, u in enumerate(X1):
        for j, v in enumerate(X2):
            result[i, j] = func(u, v)
    return result


def _timeit(callable_obj, loops: int = 3) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        callable_obj()
    return time.perf_counter() - start


def test_vectorized_euclidean_speedup():
    rng = np.random.default_rng(42)
    X1 = rng.random((250, 25))
    X2 = rng.random((250, 25))
    optimizer = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)

    baseline = _timeit(lambda: _naive_distance_matrix(X1, X2, "euclidean"), loops=2)
    optimized = _timeit(
        lambda: optimizer.compute_distance_matrix(X1, X2, metric="euclidean"), loops=2
    )

    assert baseline > optimized * 2.0
