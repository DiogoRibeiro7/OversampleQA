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


def hassanat_distance(x1: NDArray[np.floating], x2: NDArray[np.floating]) -> float:
    r"""Compute the Hassanat distance between two vectors.

    For each dimension :math:`i`, with :math:`m = \min(a_i, b_i)` and
    :math:`M = \max(a_i, b_i)`:

    .. math::

       D(a_i, b_i) = \begin{cases}
         1 - \dfrac{1 + m}{1 + M} & m \ge 0 \\[2ex]
         1 - \dfrac{1 + m + |m|}{1 + M + |m|} & m < 0
       \end{cases}

    and :math:`HD(a, b) = \sum_i D(a_i, b_i)`.

    Every per-dimension term lies in :math:`[0, 1)`, which is what makes the
    metric invariant to feature scale and robust to outliers: no single
    dimension can contribute more than 1 regardless of its magnitude.

    Parameters
    ----------
    x1, x2 : NDArray[np.floating]
        Input vectors of identical shape.

    Returns
    -------
    float
        Hassanat distance, in ``[0, n_features)``.

    Raises
    ------
    ValueError
        If the two vectors do not have the same shape.

    References
    ----------
    Hassanat, A. B. (2014). Dimensionality invariant similarity measure.
    *Journal of American Science*, 10(8).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    mn = np.minimum(x1, x2)
    mx = np.maximum(x1, x2)
    # Adding |min| on the negative branch shifts both terms up so the ratio
    # stays in (0, 1]. The denominator is 1 + mx + shift; since mx >= mn and
    # shift = max(-mn, 0), we have mx + shift >= mn + shift >= 0, so the
    # denominator is >= 1 and can never vanish. No division guard is needed.
    shift = np.where(mn < 0.0, -mn, 0.0)
    return float(np.sum(1.0 - (1.0 + mn + shift) / (1.0 + mx + shift)))


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        x1: First vector.
        x2: Second vector.

    Returns:
        Euclidean distance.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    return float(np.linalg.norm(x1 - x2))


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Manhattan distance between two vectors.

    Args:
        x1: First vector.
        x2: Second vector.

    Returns:
        Manhattan distance.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    return float(np.sum(np.abs(x1 - x2)))


def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cosine distance between two vectors.

    Args:
        x1: First vector.
        x2: Second vector.

    Returns:
        Cosine distance.
    """
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
