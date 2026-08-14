"""Oversampling validation utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

from .distance import distance_matrix
from .metrics import calculate_error_rate


def _validate_hidden_ratio(hidden_ratio: float) -> None:
    """Validate that ``hidden_ratio`` is a fraction strictly between 0 and 1.

    Args:
        hidden_ratio: Fraction of samples to hide during validation.

    Raises:
        ValueError: If ``hidden_ratio`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < hidden_ratio < 1.0:
        raise ValueError(
            f"hidden_ratio must be in the open interval (0, 1); got {hidden_ratio!r}"
        )


def _split_classes(
    X: NDArray[np.floating], y: NDArray[np.integer], minority_label: int
):
    """Split features into minority and majority subsets.

    Args:
        X: Feature matrix aligned with ``y``.
        y: Target labels.
        minority_label: Label value identifying the minority class.

    Returns:
        Tuple of ``(minority, majority)`` feature arrays.
    """
    minority_mask = y == minority_label
    minority = X[minority_mask]
    majority = X[~minority_mask]
    return minority, majority


def extract_synthetic_samples(
    X_original: NDArray[np.floating],
    X_resampled: NDArray[np.floating],
    y_resampled: NDArray[np.integer],
    minority_label: int,
) -> NDArray[np.floating]:
    """Return synthetic minority samples from a resampled dataset.

    Parameters
    ----------
    X_original : ndarray
        Original feature matrix used for fitting the oversampler.
    X_resampled : ndarray
        Feature matrix returned by ``oversampler.fit_resample``.
    y_resampled : ndarray
        Corresponding labels for ``X_resampled``.
    minority_label : int
        Label of the minority class that was oversampled.

    Returns
    -------
    ndarray
        Array containing only the synthetic minority samples.
    """

    start = len(X_original)
    if start >= len(X_resampled):
        return np.empty((0, X_original.shape[1]))
    return X_resampled[start:][y_resampled[start:] == minority_label]


def validate_oversampling(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    oversampler: BaseOverSampler,
    hidden_ratio: float = 0.1,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    return_details: bool = False,
) -> float | tuple[float, int, NDArray[np.floating], NDArray[np.floating]]:
    """Validate oversampling using the hidden majority approach.

    Parameters
    ----------
    X, y : ndarray
        Input data and labels.
    minority_label : int
        Label of the minority class.
    oversampler : BaseOverSampler
        Instance of an `imbalanced-learn` oversampler.
    hidden_ratio : float, default=0.1
        Fraction of majority samples to hide during validation.
    metric : str, default="hassanat"
        Distance metric to use.
    metric_kwargs : dict, optional
        Additional keyword arguments passed to :func:`distance_matrix`.
    return_details : bool, default=False
        If ``True`` return the number of errors and distance matrices
        in addition to the error rate.

    Returns
    -------
    float or tuple
        Error rate by default. If ``return_details`` is ``True`` a tuple
        ``(error_rate, n_errors, dist_hidden, dist_min)`` is returned.
    """
    _validate_hidden_ratio(hidden_ratio)
    labels = np.unique(y)
    if minority_label not in labels:
        raise ValueError(f"minority_label {minority_label} not found in y")
    if len(labels) != 2:
        raise ValueError(
            "validate_oversampling expects binary labels; use validate_multiclass_oversampling for multi-class data"
        )
    majority_label = int(labels[labels != minority_label][0])

    minority, majority = _split_classes(X, y, minority_label)

    vis_majority, hid_majority = train_test_split(
        majority, test_size=hidden_ratio, random_state=42
    )

    X_train = np.vstack([vis_majority, minority])
    y_train = np.hstack(
        [
            np.full(len(vis_majority), majority_label, dtype=y.dtype),
            np.full(len(minority), minority_label, dtype=y.dtype),
        ]
    )

    try:
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Oversampler failed during fit_resample")
        raise

    synthetic = extract_synthetic_samples(X_train, X_res, y_res, minority_label)

    if len(synthetic) == 0:
        if return_details:
            empty = np.empty((0, 0))
            return 0.0, 0, empty, empty
        return 0.0

    kwargs = metric_kwargs or {}
    dist_hidden = distance_matrix(synthetic, hid_majority, metric, **kwargs)
    dist_min = distance_matrix(synthetic, minority, metric, **kwargs)

    nearest_hidden = dist_hidden.min(axis=1)
    nearest_min = dist_min.min(axis=1)

    errors = int(np.sum(nearest_hidden <= nearest_min))
    rate = calculate_error_rate(errors, len(synthetic))

    if return_details:
        return rate, errors, dist_hidden, dist_min

    return rate


def validate_multiclass_oversampling(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    oversampler: BaseOverSampler,
    hidden_ratio: float = 0.1,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    return_matrix: bool = False,
) -> dict[int, float] | tuple[dict[int, float], NDArray[np.floating]]:
    """Validate oversampling for multi-class datasets.

    For each class label, a portion of samples is hidden and the remainder
    is used for training the oversampler along with all other visible
    classes. The nearest hidden class to each synthetic sample determines
    the error attribution. The function returns the per-class error rates
    and optionally the full error matrix where ``matrix[i, j]`` counts how
    many synthetic samples generated for class ``i`` are closest to hidden
    samples from class ``j``.

    Parameters
    ----------
    X, y : ndarray
        Input data and labels.
    oversampler : BaseOverSampler
        Instance of an `imbalanced-learn` oversampler supporting
        multi-class data.
    hidden_ratio : float, default=0.1
        Fraction of each class to hide during validation.
    metric : str, default="hassanat"
        Distance metric to use.
    metric_kwargs : dict, optional
        Additional keyword arguments passed to :func:`distance_matrix`.
    return_matrix : bool, default=False
        If ``True`` also return the error matrix.

    Returns
    -------
    dict or tuple
        Mapping of ``class_label -> error_rate``. If ``return_matrix`` is
        ``True`` the second element is the error matrix.
    """

    _validate_hidden_ratio(hidden_ratio)
    labels = np.unique(y)
    rng = np.random.default_rng(42)

    visible = {}
    hidden = {}
    for label in labels:
        cls_samples = X[y == label]
        n_hidden = int(len(cls_samples) * hidden_ratio)
        if n_hidden == 0:
            visible[label] = cls_samples
            hidden[label] = np.empty((0, X.shape[1]))
            continue
        idx = rng.permutation(len(cls_samples))
        hidden[label] = cls_samples[idx[:n_hidden]]
        visible[label] = cls_samples[idx[n_hidden:]]

    X_train = np.vstack([visible[l] for l in labels])
    y_train = np.hstack([[l] * len(visible[l]) for l in labels])

    try:
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Oversampler failed during fit_resample")
        raise

    start = len(X_train)
    X_syn = X_res[start:]
    y_syn = y_res[start:]

    metric_kwargs = metric_kwargs or {}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Precompute distance matrices to hidden samples of each class
    hidden_dists: dict[int, NDArray[np.floating] | None] = {
        lbl: (
            distance_matrix(X_syn, hidden[lbl], metric, **metric_kwargs)
            if len(hidden[lbl]) > 0
            else None
        )
        for lbl in labels
    }

    for i, lbl in enumerate(labels):
        syn_i = X_syn[y_syn == lbl]
        if len(syn_i) == 0:
            continue
        nearest_dist = np.full(len(syn_i), np.inf)
        nearest_lbl_idx = np.full(len(syn_i), i)
        for j, lbl_hid in enumerate(labels):
            arr = hidden_dists[lbl_hid]
            if arr is None:
                continue
            d = arr[y_syn == lbl]
            if d.size == 0:
                continue
            n = d.min(axis=1)
            mask = n < nearest_dist
            nearest_dist[mask] = n[mask]
            nearest_lbl_idx[mask] = j
        for j in range(len(labels)):
            matrix[i, j] = np.sum(nearest_lbl_idx == j)

    error_rates = {}
    for i, lbl in enumerate(labels):
        n_syn = matrix[i].sum()
        if n_syn == 0:
            error_rates[int(lbl)] = 0.0
            continue
        errors = n_syn - matrix[i, i]
        error_rates[int(lbl)] = calculate_error_rate(errors, n_syn)

    if return_matrix:
        return error_rates, matrix

    return error_rates
