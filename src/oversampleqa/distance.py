"""Distance metrics used within oversampleqa."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .optimized_distance import OptimizedDistanceMatrix
from .caching import ValidationCache
from .extended_distances import (
    minkowski_distance,
    chebyshev_distance,
    mahalanobis_distance,
    canberra_distance,
    hamming_distance,
    jaccard_distance,
    braycurtis_distance,
    correlation_distance,
    energy_distance,
    wasserstein_1d_distance,
    hellinger_distance,
    jensen_shannon_distance,
)

__all__ = [
    "hassanat_distance",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "distance_matrix",
    "minkowski_distance",
    "chebyshev_distance",
    "mahalanobis_distance",
    "canberra_distance",
    "hamming_distance",
    "jaccard_distance",
    "braycurtis_distance",
    "correlation_distance",
    "energy_distance",
    "wasserstein_1d_distance",
    "hellinger_distance",
    "jensen_shannon_distance",
]


def hassanat_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute the Hassanat distance between two vectors."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    max_vals = np.maximum(np.abs(x1), np.abs(x2))
    min_vals = np.minimum(np.abs(x1), np.abs(x2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(max_vals == 0, 0.0, 1.0 - min_vals / max_vals)
    return float(np.sum(d))


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    return float(np.linalg.norm(x1 - x2))


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Manhattan distance between two vectors."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    return float(np.sum(np.abs(x1 - x2)))


def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cosine distance between two vectors."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    dot = np.dot(x1, x2)
    denom = np.linalg.norm(x1) * np.linalg.norm(x2)
    if denom == 0:
        return 0.0
    return float(1.0 - dot / denom)


MetricFunc = Callable[[NDArray[np.floating], NDArray[np.floating]], float]

_METRICS: dict[str, MetricFunc] = {
    "hassanat": hassanat_distance,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "cosine": cosine_distance,
    "minkowski": minkowski_distance,
    "chebyshev": chebyshev_distance,
    "mahalanobis": mahalanobis_distance,
    "canberra": canberra_distance,
    "hamming": hamming_distance,
    "jaccard": jaccard_distance,
    "braycurtis": braycurtis_distance,
    "correlation": correlation_distance,
    "energy": energy_distance,
    "wasserstein": wasserstein_1d_distance,
    "hellinger": hellinger_distance,
    "jensen_shannon": jensen_shannon_distance,
}

_CACHE = ValidationCache()
_OPTIMIZER = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=_CACHE)


def distance_matrix(
    X1: NDArray[np.floating],
    X2: NDArray[np.floating],
    metric: str = "hassanat",
    *,
    batch_size: int | str = "auto",
    **metric_kwargs: Any,
) -> NDArray[np.floating]:
    """Compute pairwise distance matrix using the given metric.

    Parameters
    ----------
    X1, X2 : ndarray
        Input matrices containing observations.
    metric : str, default="hassanat"
        Identifier of the distance metric to use.
    batch_size : int or {"auto", "stream"}, default="auto"
        Controls batching strategy. ``"auto"`` selects a batch size that fits
        ``memory_limit_gb`` of :class:`OptimizedDistanceMatrix`. ``"stream"``
        forces row-wise streaming when memory is constrained.
    **metric_kwargs :
        Additional keyword arguments are forwarded to the metric function. This
        enables configuration of metrics that require extra parameters, such as
        the inverse covariance matrix for Mahalanobis distance.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unsupported metric '{metric}'")
    metric_kwargs = metric_kwargs or {}
    return _OPTIMIZER.compute_distance_matrix(
        X1,
        X2,
        metric=metric,
        batch_size=batch_size,
        **metric_kwargs,
    )
