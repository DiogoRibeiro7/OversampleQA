"""Distance metrics used within oversampleqa."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

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


def distance_matrix(
    X1: NDArray[np.floating],
    X2: NDArray[np.floating],
    metric: str = "hassanat",
    **metric_kwargs: Any,
) -> NDArray[np.floating]:
    """Compute pairwise distance matrix using the given metric.

    Additional keyword arguments are forwarded to the metric function. This
    enables configuration of metrics that require extra parameters, such as the
    inverse covariance matrix for Mahalanobis distance.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unsupported metric '{metric}'")
    func = _METRICS[metric]
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if metric == "euclidean":
        diff = X1[:, None, :] - X2[None, :, :]
        return np.linalg.norm(diff, axis=2)

    if metric == "manhattan":
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        return diff.sum(axis=2)

    if metric == "cosine":
        dot = X1 @ X2.T
        norm1 = np.linalg.norm(X1, axis=1)
        norm2 = np.linalg.norm(X2, axis=1)
        denom = norm1[:, None] * norm2[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            res = 1.0 - np.where(denom == 0, 0.0, dot / denom)
        return np.nan_to_num(res)

    if metric == "canberra":
        numerator = np.abs(X1[:, None, :] - X2[None, :, :])
        denominator = np.abs(X1[:, None, :]) + np.abs(X2[None, :, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denominator == 0, 0.0, numerator / denominator)
        return ratio.sum(axis=2)

    if metric == "braycurtis":
        num = np.abs(X1[:, None, :] - X2[None, :, :]).sum(axis=2)
        denom = np.abs(X1[:, None, :] + X2[None, :, :]).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(denom == 0, 0.0, num / denom)
        return res

    if metric == "correlation":
        X1_c = X1 - X1.mean(axis=1, keepdims=True)
        X2_c = X2 - X2.mean(axis=1, keepdims=True)
        dot = X1_c @ X2_c.T
        norm1 = np.linalg.norm(X1_c, axis=1)
        norm2 = np.linalg.norm(X2_c, axis=1)
        denom = norm1[:, None] * norm2[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom == 0, 0.0, dot / denom)
        corr = np.nan_to_num(corr)
        return 1.0 - corr

    if metric == "chebyshev":
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        return diff.max(axis=2)

    if metric == "minkowski":
        p = metric_kwargs.get("p", 3.0)
        diff = np.abs(X1[:, None, :] - X2[None, :, :]) ** p
        return np.sum(diff, axis=2) ** (1.0 / p)

    if metric == "mahalanobis":
        diff = X1[:, None, :] - X2[None, :, :]
        cov_inv = metric_kwargs.get("cov_inv")
        if cov_inv is None:
            return np.linalg.norm(diff, axis=2)
        res = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        return np.sqrt(res)

    if metric == "jaccard":
        X1_b = X1.astype(bool)
        X2_b = X2.astype(bool)
        intersection = (X1_b[:, None, :] & X2_b[None, :, :]).sum(axis=2)
        union = (X1_b[:, None, :] | X2_b[None, :, :]).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union == 0, 1.0, intersection / union)
        return 1.0 - sim

    if metric == "hamming":
        diff = X1[:, None, :] != X2[None, :, :]
        return diff.sum(axis=2).astype(float)

    if metric == "energy":
        X1_r = X1.reshape(len(X1), -1)
        X2_r = X2.reshape(len(X2), -1)
        d = X1_r.shape[1]
        diff_cross = np.abs(
            X1_r[:, None, :, None] - X2_r[None, :, None, :]
        )
        term_a = diff_cross.mean(axis=(2, 3))
        if d > 1:
            mask = np.triu(np.ones((d, d), dtype=bool), 1)
            diff_x1 = np.abs(X1_r[:, :, None] - X1_r[:, None, :])
            term_b = diff_x1[:, mask].mean(axis=1)
            diff_x2 = np.abs(X2_r[:, :, None] - X2_r[:, None, :])
            term_c = diff_x2[:, mask].mean(axis=1)
        else:
            term_b = np.zeros(len(X1_r))
            term_c = np.zeros(len(X2_r))
        return 2.0 * term_a - term_b[:, None] - term_c[None, :]

    if metric == "wasserstein":
        X1_s = np.sort(X1, axis=1)
        X2_s = np.sort(X2, axis=1)
        diff = np.abs(X1_s[:, None, :] - X2_s[None, :, :])
        return diff.mean(axis=2)

    if metric == "jensen_shannon":
        if np.any(X1 < 0) or np.any(X2 < 0):
            raise ValueError("Jensen-Shannon distance requires non-negative inputs")
        p_sum = X1.sum(axis=1, keepdims=True)
        q_sum = X2.sum(axis=1, keepdims=True)
        p = np.where(p_sum == 0, 0.0, X1 / p_sum)
        q = np.where(q_sum == 0, 0.0, X2 / q_sum)
        m = 0.5 * (p[:, None, :] + q[None, :, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.where(p[:, None, :] == 0, 0.0, p[:, None, :] * np.log(p[:, None, :] / m))
            kl_qm = np.where(q[None, :, :] == 0, 0.0, q[None, :, :] * np.log(q[None, :, :] / m))
        js = 0.5 * kl_pm.sum(axis=2) + 0.5 * kl_qm.sum(axis=2)
        return np.sqrt(js)

    if metric == "hellinger":
        p_sum = X1.sum(axis=1, keepdims=True)
        q_sum = X2.sum(axis=1, keepdims=True)
        p = np.where(p_sum == 0, 0, X1 / p_sum)
        q = np.where(q_sum == 0, 0, X2 / q_sum)
        diff = np.sqrt(p)[:, None, :] - np.sqrt(q)[None, :, :]
        return np.linalg.norm(diff, axis=2) / np.sqrt(2.0)

    # Default: Hassanat or any custom metric callable
    dm = np.empty((len(X1), len(X2)), dtype=float)
    for i, u in enumerate(X1):
        for j, v in enumerate(X2):
            dm[i, j] = func(u, v, **metric_kwargs)
    return dm
