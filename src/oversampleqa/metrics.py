"""Validation metrics for oversampleqa."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def calculate_error_rate(errors: int, total: int) -> float:
    """Return error rate given the number of errors and total samples.

    Args:
        errors: Number of error samples.
        total: Total number of samples.

    Returns:
        Error rate in the range [0, 1], or ``nan`` when ``total`` is zero.

    Notes:
        A zero denominator means nothing was measured. Returning ``0.0`` in
        that case would be indistinguishable from a perfect score, so ``nan``
        is returned instead. Callers that aggregate error rates must use
        ``nan``-aware reductions (``np.nanmean``) deliberately.
    """
    if total == 0:
        return float("nan")
    return errors / total


def duplication_rate(
    synthetic: NDArray[np.floating],
    reference: NDArray[np.floating],
    *,
    atol: float = 0.0,
) -> float:
    """Fraction of synthetic points that coincide with a reference point.

    Parameters
    ----------
    synthetic : ndarray
        Synthetic samples of shape ``(n_synthetic, n_features)``.
    reference : ndarray
        Real samples the synthetic points may have been copied from.
    atol : float, default=0.0
        Absolute tolerance for treating a synthetic point as a duplicate.
        The default of ``0.0`` requires exact equality.

    Returns
    -------
    float
        Value in ``[0, 1]``; ``nan`` when there are no synthetic samples.

    Notes
    -----
    An oversampler that duplicates rather than synthesises -- such as
    ``RandomOverSampler`` -- scores ``1.0``. Its validation error rate is then
    uninformative about synthesis quality, because every "synthetic" point sits
    exactly on top of a real one.
    """
    if len(synthetic) == 0:
        return float("nan")
    if len(reference) == 0:
        return 0.0

    matches = 0
    for point in synthetic:
        deltas = np.abs(reference - point).max(axis=1)
        if bool(np.any(deltas <= atol)):
            matches += 1
    return matches / len(synthetic)


def confidence_ratio(dist_min: float, dist_maj: float) -> float:
    """Return ratio between distances to minority and majority classes.

    Args:
        dist_min: Distance to minority class.
        dist_maj: Distance to majority class.

    Returns:
        Ratio ``dist_min / dist_maj`` (inf if ``dist_maj`` is zero).
    """
    if dist_maj == 0:
        return float("inf")
    return dist_min / dist_maj


def local_density_divergence(
    synthetic_samples: np.ndarray, reference_samples: np.ndarray, k: int = 5
) -> float:
    """Compute divergence of local densities between synthetic and reference data.

    This metric compares the average distance to the ``k`` nearest neighbours
    for synthetic samples against the same statistic computed on the reference
    samples themselves. A higher value indicates that synthetic samples reside
    in sparser regions of the space compared to the reference distribution.

    Parameters
    ----------
    synthetic_samples, reference_samples : ndarray
        Arrays of shape ``(n_samples, n_features)`` representing synthetic and
        reference data respectively.
    k : int, default=5
        Number of nearest neighbours to consider when estimating local density.

    Returns
    -------
    float
        Relative difference in mean neighbourhood radii. ``0.0`` indicates that
        both sets have similar local density.
    """

    if np.array_equal(synthetic_samples, reference_samples):
        return 0.0

    if len(reference_samples) < 2 or len(synthetic_samples) == 0:
        return 0.0

    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(reference_samples) - 1)

    nbrs_ref = NearestNeighbors(n_neighbors=k + 1).fit(reference_samples)
    ref_dists, _ = nbrs_ref.kneighbors(reference_samples)
    mean_ref = ref_dists[:, 1:].mean()

    nbrs_syn = NearestNeighbors(n_neighbors=k).fit(reference_samples)
    syn_dists, _ = nbrs_syn.kneighbors(synthetic_samples)
    mean_syn = syn_dists.mean()

    if mean_ref == 0:
        return 0.0
    divergence: float = (mean_syn - mean_ref) / mean_ref
    return divergence


def minority_recall_loss(
    y_true: np.ndarray, y_pred: np.ndarray, minority_label: int
) -> float:
    """Return recall loss for the minority class.

    Parameters
    ----------
    y_true, y_pred : ndarray
        True and predicted class labels.
    minority_label : int
        Label of the minority class.

    Returns
    -------
    float
        ``1 - recall`` for the minority class.
    """

    from sklearn.metrics import recall_score

    recall = recall_score(y_true == minority_label, y_pred == minority_label)
    achieved: float = recall
    return 1.0 - achieved


def umap_manifold_distance(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_neighbors: int = 15,
    random_state: int | None = None,
) -> float:
    """Return Wasserstein distance between real and synthetic data in UMAP space.

    Args:
        real: Real samples.
        synthetic: Synthetic samples.
        n_neighbors: UMAP neighborhood size.
        random_state: Optional random seed.

    Returns:
        Mean Wasserstein distance across UMAP dimensions.
    """

    from umap import UMAP

    from .extended_distances import wasserstein_1d_distance

    if len(synthetic) == 0 or len(real) == 0:
        return 0.0

    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        random_state=random_state,
        n_jobs=1,
    )
    X = np.vstack([real, synthetic])
    embed = reducer.fit_transform(X)
    real_emb = embed[: len(real)]
    synth_emb = embed[len(real) :]
    d1 = wasserstein_1d_distance(real_emb[:, 0], synth_emb[:, 0])
    d2 = wasserstein_1d_distance(real_emb[:, 1], synth_emb[:, 1])
    return float((d1 + d2) / 2)


def check_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    minority_label: int,
) -> float:
    """Return absolute difference in minority recall across protected groups.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        protected_attr: Protected group labels.
        minority_label: Minority class label.

    Returns:
        Absolute recall gap between the two groups.
    """

    from sklearn.metrics import recall_score

    groups = np.unique(protected_attr)
    if len(groups) != 2:
        raise ValueError("protected_attr must have exactly two groups")

    recalls = []
    for g in groups:
        mask = protected_attr == g
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(
                recall_score(
                    y_true[mask] == minority_label, y_pred[mask] == minority_label
                )
            )

    return abs(recalls[0] - recalls[1])


def flip_labels(
    y: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of ``y`` with ``indices`` relabelled to a *different* class.

    Drawing replacements from all classes, as this used to, lets a selected
    point keep its own label. The realised noise was then
    ``requested * (k - 1) / k`` -- on binary data, half of what was asked for.

    The offset is taken within the sorted label list, which guarantees a change
    and is uniform over the ``k - 1`` alternatives.

    Args:
        y: Label array.
        indices: Positions to relabel.
        labels: Sorted unique labels.
        rng: Source of randomness.

    Returns:
        A new array; ``y`` is not modified.
    """
    flipped = y.copy()
    if len(indices) == 0:
        return flipped
    original = np.searchsorted(labels, y[indices])
    offset = rng.integers(1, len(labels), size=len(indices))
    flipped[indices] = labels[(original + offset) % len(labels)]
    return flipped


def noise_sensitivity_diagnostic(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int,
    oversampler: Any,
    noise_levels: list[float] | None = None,
    hidden_ratio: float = 0.1,
    metric: str = "hassanat",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate error rate under different label noise levels.

    Args:
        X: Feature matrix.
        y: Target labels.
        minority_label: Minority class label.
        oversampler: Oversampler instance.
        noise_levels: Noise levels to evaluate.
        hidden_ratio: Fraction of majority to hide.
        metric: Distance metric name.
        random_state: Optional random seed.

    Returns:
        DataFrame with ``noise``, ``error_rate`` and ``n_flipped`` -- the number
        of labels actually changed, so the applied noise can be checked against
        the requested level rather than assumed.

    Raises:
        ValueError: If ``y`` contains fewer than two classes, leaving no label
            to flip to.

    Notes:
        Replacement labels are drawn from the *other* classes. Drawing from all
        classes, as this used to, lets a selected point keep its own label, so
        the realised noise was ``noise * (k - 1) / k``: on binary data -- this
        package's main case -- **half** the requested level. A run labelled
        ``noise=0.3`` applied about 0.15, and the x-axis of every
        noise-sensitivity plot was overstated by that factor.
    """

    from .validator import validate_oversampling

    noise_levels = noise_levels or [0.0, 0.1, 0.2, 0.3]
    rng = np.random.default_rng(random_state)
    results = []
    labels = np.unique(y)
    if len(labels) < 2:
        raise ValueError(
            "noise_sensitivity_diagnostic needs at least two classes: with one "
            "class there is no other label to flip to, so no noise level is "
            "distinguishable from zero."
        )

    for noise in noise_levels:
        y_noisy = y.copy()
        n_flipped = 0
        if noise > 0:
            n_flip = int(len(y) * noise)
            idx = rng.choice(len(y), n_flip, replace=False)
            y_noisy = flip_labels(y, idx, labels, rng)
            n_flipped = int(np.sum(y_noisy != y))

        err = validate_oversampling(
            X,
            y_noisy,
            minority_label=minority_label,
            oversampler=oversampler,
            hidden_ratio=hidden_ratio,
            metric=metric,
        )
        results.append(
            {"noise": noise, "error_rate": err, "n_flipped": n_flipped}
        )

    return pd.DataFrame(results)
