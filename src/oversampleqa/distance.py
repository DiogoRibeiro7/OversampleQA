"""Distance metrics used within oversampleqa."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .caching import ValidationCache
from .extended_distances import (
    braycurtis_distance,
    canberra_distance,
    chebyshev_distance,
    correlation_distance,
    energy_distance,
    hamming_distance,
    hellinger_distance,
    jaccard_distance,
    jensen_shannon_distance,
    mahalanobis_distance,
    minkowski_distance,
    wasserstein_1d_distance,
)
from .optimized_distance import OptimizedDistanceMatrix

__all__ = [
    "braycurtis_distance",
    "canberra_distance",
    "chebyshev_distance",
    "correlation_distance",
    "cosine_distance",
    "distance_matrix",
    "energy_distance",
    "euclidean_distance",
    "hamming_distance",
    "hassanat_distance",
    "hellinger_distance",
    "jaccard_distance",
    "jensen_shannon_distance",
    "mahalanobis_distance",
    "manhattan_distance",
    "minkowski_distance",
    "wasserstein_1d_distance",
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

# Constructed without a cache. Importing this package must not touch the
# filesystem, and an always-on cache is the wrong default anyway: content
# hashing reads every input byte, which costs more than recomputing a
# BLAS-backed metric. Pass an explicit ValidationCache when it pays.
_OPTIMIZER = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)



def resolve_metric(name: str) -> MetricFunc | None:
    """Return a plugin metric callable, or ``None`` for a built-in.

    Registering a metric plugin used to accomplish nothing beyond making it
    retrievable from the registry: ``distance_matrix`` and every validator that
    funnels through it consulted only the built-in table, so a plugin metric was
    rejected as unsupported by the exact functions it exists to be used by.

    Resolution happens per call rather than at import, because plugins register
    at runtime -- often from ``discover_entry_points`` -- and the built-in table
    is bound when this module is imported.

    Args:
        name: Metric identifier.

    Returns:
        A callable for a registered plugin metric, or ``None`` when the name is
        a built-in and the default registry already covers it.

    Raises:
        ValueError: If the name is neither a built-in nor a registered plugin.
    """
    if name in _METRICS:
        return None

    # Local import: plugin_system reads _METRICS from this module, so importing
    # it at module scope would be a cycle.
    from .plugin_system import plugin_manager

    try:
        registered = plugin_manager.get_metric(name)
    except KeyError:
        known = ", ".join(sorted(_METRICS))
        raise ValueError(
            f"Unsupported metric {name!r}. Built-in metrics: {known}. "
            "If this is a plugin, register it first -- "
            "plugin_manager.discover_entry_points() for an installed package, "
            "or plugin_manager.register_metric(...) directly."
        ) from None
    return registered() if isinstance(registered, type) else registered


def distance_matrix(
    X1: NDArray[np.floating],
    X2: NDArray[np.floating],
    metric: str = "hassanat",
    *,
    batch_size: int | str = "auto",
    cache: ValidationCache | None = None,
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
    cache : ValidationCache, optional
        Opt-in cache. Caching is off by default: nothing is written to disk and
        no directory is created unless you supply one. Worth it for expensive
        metrics such as ``hassanat``; a net loss for ``euclidean``, where
        hashing the inputs costs more than recomputing the result.
    **metric_kwargs :
        Additional keyword arguments are forwarded to the metric function. This
        enables configuration of metrics that require extra parameters, such as
        the inverse covariance matrix for Mahalanobis distance.

    Returns
    -------
    ndarray
        Distance matrix. When ``cache`` is supplied the array is **read-only**;
        call ``.copy()`` before modifying it.
    """
    plugin = resolve_metric(metric)
    registry = _METRICS if plugin is None else {**_METRICS, metric: plugin}
    metric_kwargs = metric_kwargs or {}
    optimizer = (
        _OPTIMIZER
        if cache is None and plugin is None
        else OptimizedDistanceMatrix(metric_registry=registry, cache=cache)
    )
    return optimizer.compute_distance_matrix(
        X1,
        X2,
        metric=metric,
        batch_size=batch_size,
        **metric_kwargs,
    )
