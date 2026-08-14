"""
Extended distance metrics for oversampleqa package.

This module adds comprehensive distance metrics beyond the basic ones,
with proper validation and testing strategies.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable
import warnings


def minkowski_distance(x1: np.ndarray, x2: np.ndarray, p: float = 3.0) -> float:
    """Compute Minkowski distance between two vectors.
    
    Parameters
    ----------
    x1, x2 : np.ndarray
        Input vectors of same shape
    p : float, default=3.0
        Order of the norm (p >= 1)
        
    Returns
    -------
    float
        Minkowski distance
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    if p < 1:
        raise ValueError("p must be >= 1")
    
    diff = np.abs(x1 - x2)
    return float(np.sum(diff ** p) ** (1/p))


def chebyshev_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Chebyshev (L-infinity) distance between two vectors.
    
    This is the maximum absolute difference across all dimensions.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    return float(np.max(np.abs(x1 - x2)))


def mahalanobis_distance(
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    cov_inv: NDArray[np.floating] | None = None,
) -> float:
    """Compute Mahalanobis distance between two vectors.
    
    Parameters
    ----------
    x1, x2 : np.ndarray
        Input vectors
    cov_inv : np.ndarray, optional
        Inverse covariance matrix. If None, uses identity (equivalent to Euclidean)
        
    Returns
    -------
    float
        Mahalanobis distance
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    diff = x1 - x2
    
    if cov_inv is None:
        # Fall back to Euclidean if no covariance provided
        return float(np.sqrt(np.dot(diff, diff)))
    
    return float(np.sqrt(np.dot(diff, np.dot(cov_inv, diff))))


def canberra_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Canberra distance between two vectors.
    
    Canberra distance is a weighted version of Manhattan distance,
    useful when dealing with features of different scales.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    numerator = np.abs(x1 - x2)
    denominator = np.abs(x1) + np.abs(x2)
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(denominator == 0, 0.0, numerator / denominator)
    
    return float(np.sum(ratio))


def hamming_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Hamming distance between two vectors.
    
    Counts the number of positions where elements differ.
    Useful for categorical or binary features.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    return float(np.sum(x1 != x2))


def jaccard_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Jaccard distance between two binary vectors.
    
    Jaccard distance = 1 - Jaccard similarity
    where Jaccard similarity = :math:`|intersection| / |union|`
    """
    x1 = np.asarray(x1, dtype=bool)
    x2 = np.asarray(x2, dtype=bool)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    intersection = np.sum(x1 & x2)
    union = np.sum(x1 | x2)
    
    if union == 0:
        return 0.0  # Both vectors are all zeros
    
    return 1.0 - (intersection / union)


def braycurtis_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Bray-Curtis distance between two vectors.
    
    Often used in ecology and environmental science.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    numerator = np.sum(np.abs(x1 - x2))
    denominator = np.sum(np.abs(x1 + x2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def correlation_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute correlation distance between two vectors.
    
    Correlation distance = 1 - Pearson correlation coefficient
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    if len(x1) < 2:
        return 0.0
    
    corr_coef = np.corrcoef(x1, x2)[0, 1]
    
    # Handle NaN correlation (constant vectors)
    if np.isnan(corr_coef):
        return 0.0

    return 1.0 - corr_coef


def hellinger_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute the Hellinger distance between two probability vectors.

    The input vectors are normalized to sum to ``1`` and must contain
    non-negative values. The distance is bounded between ``0`` and ``1``.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    if np.any(x1 < 0) or np.any(x2 < 0):
        raise ValueError("Hellinger distance requires non-negative inputs")

    p = x1 / x1.sum() if x1.sum() != 0 else np.zeros_like(x1)
    q = x2 / x2.sum() if x2.sum() != 0 else np.zeros_like(x2)

    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))


def jensen_shannon_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute the Jensen-Shannon distance between two probability vectors.

    The Jensen-Shannon distance is the square root of the
    Jensen-Shannon divergence and is symmetric and bounded between ``0`` and
    ``sqrt(log(2))`` when using natural logarithms.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    if np.any(x1 < 0) or np.any(x2 < 0):
        raise ValueError("Jensen-Shannon distance requires non-negative inputs")

    p = x1 / x1.sum() if x1.sum() != 0 else np.zeros_like(x1)
    q = x2 / x2.sum() if x2.sum() != 0 else np.zeros_like(x2)
    m = 0.5 * (p + q)

    def _kl_div(a: np.ndarray, b: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a == 0, 1.0, a / b)
            log_term = np.log(ratio)
        return float(np.sum(np.where(a == 0, 0.0, a * log_term)))

    js_div = 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)
    return float(np.sqrt(js_div))


def energy_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute energy distance between two 1D or 2D vectors.

    The implementation follows the definition from energy statistics.

    .. warning::

       This is a **sample-based** metric, not a point metric. A 1-D input is
       reshaped to ``(len(x), 1)`` and treated as a *set of scalar
       observations*, not as one point in ``len(x)``-dimensional feature
       space. It therefore does not measure the same kind of quantity as
       ``euclidean`` or ``hassanat``, even though it is reachable through the
       same registry. Use it to compare two samples, not two points.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1.reshape(len(x1), -1)
    x2 = x2.reshape(len(x2), -1)

    diff_cross = np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=-1)
    term_a = diff_cross.mean()

    if len(x1) > 1:
        diff_x1 = np.linalg.norm(x1[:, None, :] - x1[None, :, :], axis=-1)
        term_b = diff_x1[np.triu_indices(len(x1), 1)].mean()
    else:
        term_b = 0.0

    if len(x2) > 1:
        diff_x2 = np.linalg.norm(x2[:, None, :] - x2[None, :, :], axis=-1)
        term_c = diff_x2[np.triu_indices(len(x2), 1)].mean()
    else:
        term_c = 0.0

    return float(2.0 * term_a - term_b - term_c)


def wasserstein_1d_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute the 1D Wasserstein distance between two empirical distributions.

    .. warning::

       This is a **sample-based** metric, not a point metric. The input vector
       is flattened and treated as a *set of scalar observations* drawn from a
       distribution, not as one point in feature space. It therefore does not
       measure the same kind of quantity as ``euclidean`` or ``hassanat``,
       even though it is reachable through the same registry. Use it to
       compare two samples, not two points.

    Args:
        x1: Samples from distribution 1.
        x2: Samples from distribution 2.

    Returns:
        Wasserstein distance.
    """
    x1 = np.sort(np.asarray(x1, dtype=float).ravel())
    x2 = np.sort(np.asarray(x2, dtype=float).ravel())

    n = len(x1)
    m = len(x2)
    i = j = 0
    cdf1 = cdf2 = 0.0
    last_x = min(x1[0] if n else 0.0, x2[0] if m else 0.0)
    dist = 0.0

    while i < n and j < m:
        if x1[i] <= x2[j]:
            x = x1[i]
            cdf1 = (i + 1) / n
            i += 1
        else:
            x = x2[j]
            cdf2 = (j + 1) / m
            j += 1
        dist += abs(cdf1 - cdf2) * (x - last_x)
        last_x = x

    while i < n:
        x = x1[i]
        cdf1 = (i + 1) / n
        dist += abs(cdf1 - cdf2) * (x - last_x)
        last_x = x
        i += 1

    while j < m:
        x = x2[j]
        cdf2 = (j + 1) / m
        dist += abs(cdf1 - cdf2) * (x - last_x)
        last_x = x
        j += 1

    return float(dist)
