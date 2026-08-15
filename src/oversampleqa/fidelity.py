"""Fidelity and diversity metrics for synthetic samples.

The hidden-majority error rate is a single scalar, and it conflates two failures
that call for opposite remedies:

**Low fidelity** -- synthetic points land in implausible regions: between
clusters, inside majority territory, off the data manifold. The fix is a more
conservative generator.

**Low diversity** -- synthetic points are perfectly realistic but merely copy the
training minority, adding no information. The fix is a *less* conservative one.

``RandomOverSampler`` is the clean demonstration: it duplicates real points, so
it is maximally realistic and completely uninformative, and a single scalar
scores it perfectly. Separating the two axes is what this module does.

References
----------
Sajjadi, M. S. M. et al. (2018). Assessing generative models via precision and
recall. *NeurIPS*.

Kynkaanniemi, T. et al. (2019). Improved precision and recall metric for
assessing generative models. *NeurIPS*.

Naeem, M. F. et al. (2020). Reliable fidelity and diversity metrics for
generative models. *ICML*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .distance import distance_matrix
from .exceptions import ValidationError
from .types import ValidationDetails

__all__ = [
    "BoundaryReport",
    "FidelityReport",
    "ManifoldMetrics",
    "MemorisationReport",
    "UtilityReport",
    "boundary_violation_rate",
    "downstream_utility",
    "fidelity_report",
    "memorisation_report",
    "precision_recall_density_coverage",
    "sweep_k",
]


@dataclass(frozen=True)
class ManifoldMetrics:
    """k-NN manifold estimates of fidelity and diversity.

    Attributes
    ----------
    precision:
        Fraction of synthetic points inside the real manifold. **Fidelity**:
        are the generated points plausible?
    recall:
        Fraction of real points inside the synthetic manifold. **Diversity**:
        does the generator cover the real distribution?
    density:
        Like precision, but counts *how many* real k-NN spheres contain each
        synthetic point. Not saturated by a single real outlier whose sphere is
        enormous, which is precision's main failure mode.
    coverage:
        Fraction of real points with at least one synthetic point inside their
        own k-NN sphere. More robust than recall for the same reason.

    Notes
    -----
    **Density and coverage are the more reliable pair** (Naeem et al. 2020) and
    are what the report surfaces first. Precision and recall are reported too,
    because their disagreement with density/coverage is itself informative: it
    usually means an outlier is inflating one manifold.
    """

    precision: float
    recall: float
    density: float
    coverage: float
    k: int
    metric: str
    n_synthetic: int
    n_real: int

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the reporting layer."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "density": self.density,
            "coverage": self.coverage,
            "k": self.k,
            "metric": self.metric,
            "n_synthetic": self.n_synthetic,
            "n_real": self.n_real,
        }


@dataclass(frozen=True)
class MemorisationReport:
    """How much of the "synthetic" output is really copied training data.

    Attributes
    ----------
    distance_ratio:
        **The headline number.** Median nearest-neighbour distance from
        synthetic points to their training set, divided by the median
        nearest-neighbour distance *within* the real minority. Below 1 means
        the generator sits closer to its training points than real points sit
        to each other -- it is copying. Near 0 means outright duplication.
    exact_duplicate_rate:
        Fraction of synthetic points exactly coinciding with a training point.
    near_duplicate_rates:
        Fraction within a threshold taken from the real minority's own
        nearest-neighbour distance distribution, keyed by quantile. Deriving
        the threshold from the data makes it scale-free: an absolute tolerance
        means something different on every dataset.
    """

    distance_ratio: float
    exact_duplicate_rate: float
    near_duplicate_rates: dict[float, float]
    median_distance_to_train: float
    median_real_nn_distance: float
    metric: str
    n_synthetic: int

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the reporting layer."""
        payload: dict[str, Any] = {
            "memorisation_distance_ratio": self.distance_ratio,
            "exact_duplicate_rate": self.exact_duplicate_rate,
            "median_distance_to_train": self.median_distance_to_train,
            "median_real_nn_distance": self.median_real_nn_distance,
            "metric": self.metric,
            "n_synthetic": self.n_synthetic,
        }
        for quantile, rate in self.near_duplicate_rates.items():
            payload[f"near_duplicate_rate_q{quantile:g}"] = rate
        return payload

    def interpret(self) -> str:
        """One-line reading of the headline ratio."""
        if np.isnan(self.distance_ratio):
            return "Not enough data to assess memorisation."
        if self.distance_ratio < 0.1:
            return (
                f"ratio {self.distance_ratio:.3f}: synthetic points sit essentially "
                "on top of training points -- this generator is copying."
            )
        if self.distance_ratio < 0.5:
            return (
                f"ratio {self.distance_ratio:.3f}: synthetic points are much closer "
                "to training data than real points are to each other."
            )
        return (
            f"ratio {self.distance_ratio:.3f}: synthetic points are about as far "
            "from training data as real points are from each other."
        )


@dataclass(frozen=True)
class BoundaryReport:
    """How often synthetic points land in majority territory."""

    strict_rate: float
    graded_rate: float
    k: int
    metric: str
    n_synthetic: int

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the reporting layer."""
        return {
            "boundary_violation_strict": self.strict_rate,
            "boundary_violation_graded": self.graded_rate,
            "boundary_k": self.k,
            "metric": self.metric,
            "n_synthetic": self.n_synthetic,
        }


def _check_sizes(
    synthetic: NDArray[np.floating], real: NDArray[np.floating], k: int
) -> None:
    """Reject inputs too small for a k-NN manifold estimate."""
    if len(synthetic) == 0:
        raise ValidationError("synthetic is empty; nothing to measure")
    if len(real) < k + 1:
        raise ValidationError(
            f"a k={k} manifold estimate needs at least {k + 1} real points, got "
            f"{len(real)}. Lower k, or supply more real minority data -- a "
            "sphere built from fewer neighbours than k is not a manifold "
            "estimate, it is noise."
        )


def _warn_if_high_dimensional(real: NDArray[np.floating]) -> None:
    """Warn when the sample is too small for the dimension.

    Every k-NN manifold metric degrades as dimension grows: distances
    concentrate, so the spheres stop distinguishing anything.
    """
    n_real, n_features = real.shape[0], (real.shape[1] if real.ndim > 1 else 1)
    if n_features > n_real / 10:
        warnings.warn(
            f"{n_features} features against {n_real} real points: k-NN manifold "
            "metrics concentrate in high dimension and become unreliable well "
            "before they become obviously wrong. Reduce dimension first (see "
            "umap_manifold_distance) or read these numbers as indicative only.",
            UserWarning,
            stacklevel=3,
        )


def _knn_radii(
    points: NDArray[np.floating],
    k: int,
    metric: str,
    metric_kwargs: dict[str, Any] | None,
) -> NDArray[np.floating]:
    """Distance from each point to its k-th nearest neighbour within the set."""
    within = distance_matrix(points, points, metric, **(metric_kwargs or {}))
    within = np.array(within, copy=True)
    np.fill_diagonal(within, np.inf)  # a point is not its own neighbour
    ordered = np.sort(within, axis=1)
    return ordered[:, k - 1]


def precision_recall_density_coverage(
    synthetic: NDArray[np.floating],
    real: NDArray[np.floating],
    *,
    k: int = 5,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
) -> ManifoldMetrics:
    """Estimate fidelity and diversity from k-NN manifolds.

    The real manifold is the union of hyperspheres centred on each real point
    with radius its k-th nearest neighbour distance; the synthetic manifold is
    the same construction on synthetic points.

    Parameters
    ----------
    synthetic, real : ndarray
        Synthetic points and real held-out minority points.
    k : int, default=5
        Neighbours defining each sphere. **These metrics are sensitive to k**;
        use :func:`sweep_k` rather than trusting one value.
    metric : str, default="hassanat"
        Any metric from the package registry.

    Returns
    -------
    ManifoldMetrics

    Raises
    ------
    ValidationError
        If ``synthetic`` is empty or ``real`` has fewer than ``k + 1`` points.
    """
    synthetic = np.asarray(synthetic, dtype=float)
    real = np.asarray(real, dtype=float)
    _check_sizes(synthetic, real, k)
    _warn_if_high_dimensional(real)

    kwargs = metric_kwargs or {}
    real_radii = _knn_radii(real, k, metric, kwargs)
    cross = distance_matrix(synthetic, real, metric, **kwargs)

    # Precision: a synthetic point is inside the real manifold if it falls in
    # any real point's sphere.
    inside_real = cross <= real_radii[None, :]
    precision = float(inside_real.any(axis=1).mean())

    # Density: how many real spheres contain it, normalised by k. Counting
    # rather than thresholding is what stops one outsized outlier sphere from
    # certifying every synthetic point at once.
    density = float(inside_real.sum(axis=1).mean() / k)

    # Coverage: fraction of real points with a synthetic point in their sphere.
    coverage = float(inside_real.any(axis=0).mean())

    # Recall needs the synthetic manifold, so it needs enough synthetic points.
    if len(synthetic) >= k + 1:
        synthetic_radii = _knn_radii(synthetic, k, metric, kwargs)
        inside_synthetic = synthetic_radii[None, :] >= cross.T
        recall = float(inside_synthetic.any(axis=1).mean())
    else:
        recall = float("nan")

    return ManifoldMetrics(
        precision=precision,
        recall=recall,
        density=density,
        coverage=coverage,
        k=k,
        metric=metric,
        n_synthetic=len(synthetic),
        n_real=len(real),
    )


def sweep_k(
    synthetic: NDArray[np.floating],
    real: NDArray[np.floating],
    *,
    ks: tuple[int, ...] = (3, 5, 10, 20),
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Recompute the manifold metrics across several ``k``.

    These metrics are known to be k-sensitive. Reporting one value and hiding
    that sensitivity would repeat the error rate's original sin -- a single
    number with no indication of what it depends on. A metric that moves sharply
    with k is telling you the manifold estimate is unstable, not that the
    generator changed.

    Returns
    -------
    pandas.DataFrame
        One row per ``k``; ``k`` values too large for the sample are skipped.
    """
    rows = []
    for k in ks:
        if len(real) < k + 1:
            continue
        rows.append(
            precision_recall_density_coverage(
                synthetic, real, k=k, metric=metric, metric_kwargs=metric_kwargs
            ).to_dict()
        )
    if not rows:
        raise ValidationError(
            f"none of k={ks} is usable with {len(real)} real points; the largest "
            f"usable k is {max(1, len(real) - 1)}"
        )
    return pd.DataFrame(rows)


def memorisation_report(
    synthetic: NDArray[np.floating],
    train_minority: NDArray[np.floating],
    *,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    quantiles: tuple[float, ...] = (0.01, 0.05),
) -> MemorisationReport:
    """Assess how much of the output is copied from the training minority.

    The headline is ``distance_ratio``: the median distance from a synthetic
    point to its nearest training point, over the median nearest-neighbour
    distance *within* the real minority. That denominator is what makes the
    number legible -- it is the natural spacing of real data, so a ratio well
    below 1 says the generator sits closer to its training points than real
    points sit to each other.

    Near-duplicate thresholds come from the same distribution rather than an
    absolute tolerance, so they mean the same thing on any dataset.

    Parameters
    ----------
    synthetic, train_minority : ndarray
        Synthetic points and the minority data the sampler was fitted on.
    quantiles : tuple of float, default=(0.01, 0.05)
        Quantiles of the real nearest-neighbour distance distribution to use as
        near-duplicate thresholds.

    Returns
    -------
    MemorisationReport
    """
    synthetic = np.asarray(synthetic, dtype=float)
    train_minority = np.asarray(train_minority, dtype=float)
    if len(synthetic) == 0:
        raise ValidationError("synthetic is empty; nothing to measure")
    if len(train_minority) < 2:
        raise ValidationError(
            "memorisation needs at least 2 training minority points to establish "
            f"the real spacing; got {len(train_minority)}"
        )

    kwargs = metric_kwargs or {}
    to_train = distance_matrix(synthetic, train_minority, metric, **kwargs).min(axis=1)

    within = distance_matrix(train_minority, train_minority, metric, **kwargs)
    within = np.array(within, copy=True)
    np.fill_diagonal(within, np.inf)
    real_nn = within.min(axis=1)

    median_to_train = float(np.median(to_train))
    median_real_nn = float(np.median(real_nn))
    ratio = median_to_train / median_real_nn if median_real_nn > 0 else float("nan")

    near_rates = {
        q: float((to_train <= np.quantile(real_nn, q)).mean()) for q in quantiles
    }

    # Not `== 0.0`. Metrics computed through the BLAS gram trick -- euclidean
    # among them -- lose the last bits to cancellation for identical points and
    # return ~1e-8 rather than exactly zero, while direct formulas such as
    # hassanat return exact zeros. An exact test would therefore report a
    # different duplicate rate for the same data depending on the metric. The
    # tolerance is scaled by the real spacing so it stays scale-free.
    duplicate_tolerance = max(median_real_nn * 1e-6, np.finfo(float).eps * 100)

    return MemorisationReport(
        distance_ratio=ratio,
        exact_duplicate_rate=float((to_train <= duplicate_tolerance).mean()),
        near_duplicate_rates=near_rates,
        median_distance_to_train=median_to_train,
        median_real_nn_distance=median_real_nn,
        metric=metric,
        n_synthetic=len(synthetic),
    )


def boundary_violation_rate(
    synthetic: NDArray[np.floating],
    X_real: NDArray[np.floating],
    y_real: NDArray[np.integer],
    minority_label: int,
    *,
    k: int = 5,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
) -> BoundaryReport:
    """Fraction of synthetic points sitting in majority territory.

    Measures the failure this package exists to detect, per point and without a
    hold-out -- so it can still be reported when the minority is too small for
    :func:`~oversampleqa.validate_oversampling`'s hold-out guard.

    Two versions are returned because they answer different questions:

    ``strict_rate``
        Fraction whose **all** ``k`` nearest real neighbours are majority.
        Unambiguous violations.
    ``graded_rate``
        Mean majority fraction among the ``k`` neighbours. Sensitive to points
        drifting toward the boundary before they cross it.

    This is unrelated to
    :func:`~oversampleqa.noise_sensitivity_diagnostic`, which measures how the
    error rate responds to injected *label noise* -- a different question, so
    the two do not overlap.

    Returns
    -------
    BoundaryReport
    """
    synthetic = np.asarray(synthetic, dtype=float)
    X_real = np.asarray(X_real, dtype=float)
    y_real = np.asarray(y_real)
    if len(synthetic) == 0:
        raise ValidationError("synthetic is empty; nothing to measure")
    if len(X_real) < k:
        raise ValidationError(
            f"need at least k={k} real points to inspect neighbours, got {len(X_real)}"
        )

    distances = distance_matrix(synthetic, X_real, metric, **(metric_kwargs or {}))
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :k]
    is_majority = (y_real != minority_label)[neighbours]

    return BoundaryReport(
        strict_rate=float(is_majority.all(axis=1).mean()),
        graded_rate=float(is_majority.mean()),
        k=k,
        metric=metric,
        n_synthetic=len(synthetic),
    )


@dataclass(frozen=True)
class UtilityReport:
    """Whether oversampling actually helps a downstream classifier."""

    score_with: float
    score_without: float
    difference: float
    ci_lower: float
    ci_upper: float
    scoring: str
    n_folds: int
    fold_differences: tuple[float, ...] = ()

    @property
    def helps(self) -> bool:
        """Whether the improvement interval excludes zero."""
        return self.ci_lower > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the reporting layer."""
        return {
            "utility_with": self.score_with,
            "utility_without": self.score_without,
            "utility_difference": self.difference,
            "utility_ci_lower": self.ci_lower,
            "utility_ci_upper": self.ci_upper,
            "utility_scoring": self.scoring,
            "utility_helps": self.helps,
        }


def downstream_utility(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    oversampler: Any,
    estimator: Any = None,
    *,
    cv: int = 5,
    scoring: str = "average_precision",
    n_resamples: int = 2000,
    random_state: int | None = 42,
) -> UtilityReport:
    """Compare a classifier trained with and without oversampling.

    The question practitioners actually have. Geometry can look fine while the
    classifier gains nothing at all.

    .. danger::

       **Oversampling must happen inside each training fold.** Resampling before
       the split leaks synthetic points derived from validation-fold minority
       samples into training, and the score is inflated -- sometimes
       dramatically, because a SMOTE point interpolated from a validation point
       is nearly that point.

       This uses :class:`imblearn.pipeline.Pipeline`, which resamples within
       each fold. ``sklearn.pipeline.Pipeline`` does not handle samplers this
       way. ``tests/test_fidelity.py`` builds the leaky version deliberately and
       asserts it scores higher, so the correct construction is pinned by
       evidence rather than by comment.

    Parameters
    ----------
    X, y : ndarray
        Full dataset.
    oversampler : object
        An ``imbalanced-learn`` sampler.
    estimator : object, optional
        Classifier. Defaults to a small random forest.
    cv : int, default=5
        Stratified folds.
    scoring : str, default="average_precision"
        **Not accuracy.** On imbalanced data accuracy is dominated by the
        majority class -- predicting the majority for everything scores well
        while being useless. Average precision (PR-AUC) reflects performance on
        the minority, which is the class of interest.
    n_resamples : int, default=2000
        Bootstrap resamples for the paired-difference interval.
    random_state : int, optional
        Seeds the folds, the sampler and the estimator.

    Returns
    -------
    UtilityReport
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=1
        )

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # A pipeline, not a pre-resampled X: this is what keeps the sampler inside
    # the fold.
    with_sampling = ImbPipeline(
        [("sampler", clone(oversampler)), ("model", clone(estimator))]
    )
    scores_with = cross_val_score(
        with_sampling, X, y, cv=splitter, scoring=scoring, n_jobs=1
    )
    scores_without = cross_val_score(
        clone(estimator), X, y, cv=splitter, scoring=scoring, n_jobs=1
    )

    differences = np.asarray(scores_with) - np.asarray(scores_without)
    rng = np.random.default_rng(random_state)
    draws = rng.choice(differences, size=(n_resamples, len(differences)), replace=True)
    boot = draws.mean(axis=1)

    return UtilityReport(
        score_with=float(np.mean(scores_with)),
        score_without=float(np.mean(scores_without)),
        difference=float(np.mean(differences)),
        ci_lower=float(np.percentile(boot, 2.5)),
        ci_upper=float(np.percentile(boot, 97.5)),
        scoring=scoring,
        n_folds=cv,
        fold_differences=tuple(float(d) for d in differences),
    )


@dataclass(frozen=True)
class FidelityReport:
    """Every fidelity signal for one oversampler on one dataset."""

    error_rate: float
    manifold: ManifoldMetrics
    memorisation: MemorisationReport
    boundary: BoundaryReport
    utility: UtilityReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping across every component."""
        payload: dict[str, Any] = {"error_rate": self.error_rate}
        payload.update(self.manifold.to_dict())
        payload.update(self.memorisation.to_dict())
        payload.update(self.boundary.to_dict())
        if self.utility is not None:
            payload.update(self.utility.to_dict())
        return payload

    def to_frame(self) -> pd.DataFrame:
        """Single-row frame, for concatenating across samplers."""
        return pd.DataFrame([self.to_dict()])

    def interpret(self) -> list[str]:
        """Readings of the patterns that matter, in plain language."""
        notes: list[str] = []
        if self.memorisation.distance_ratio < 0.1:
            notes.append(
                "Memorisation: synthetic points sit on top of training points. "
                "The error rate cannot say anything about synthesis quality here."
            )
        if self.manifold.coverage < 0.5:
            notes.append(
                f"Low coverage ({self.manifold.coverage:.2f}): the generator misses "
                "much of the real minority distribution."
            )
        if self.manifold.precision < 0.5:
            notes.append(
                f"Low precision ({self.manifold.precision:.2f}): many synthetic "
                "points fall outside the real manifold."
            )
        if self.boundary.strict_rate > 0.1:
            notes.append(
                f"Boundary violations ({self.boundary.strict_rate:.2f}): synthetic "
                "points are landing in majority territory."
            )
        if self.utility is not None and not self.utility.helps:
            notes.append("No downstream gain: the improvement interval includes zero.")
        if not notes:
            notes.append("No fidelity concerns detected.")
        return notes


def fidelity_report(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    oversampler: Any,
    *,
    metric: str = "hassanat",
    k: int = 5,
    hidden_ratio: float = 0.1,
    random_state: int | None = 42,
    include_utility: bool = False,
) -> FidelityReport:
    """Run the full fidelity suite for one oversampler.

    Parameters
    ----------
    X, y : ndarray
        Full dataset.
    minority_label : int
        Minority class label.
    oversampler : object
        An ``imbalanced-learn`` sampler.
    metric : str, default="hassanat"
        Distance metric for every geometric measure.
    k : int, default=5
        Neighbours for the manifold and boundary estimates.
    hidden_ratio : float, default=0.1
        Fraction held out, matching ``validate_oversampling``.
    include_utility : bool, default=False
        Fit models to measure downstream gain. Off by default because it is far
        slower than the geometric measures.

    Returns
    -------
    FidelityReport
    """
    from .validator import (
        extract_synthetic_samples,
        prepare_validation_split,
        validate_oversampling,
    )

    labels = np.unique(y)
    if len(labels) != 2:
        raise ValidationError("fidelity_report expects binary labels")
    majority_label = int(labels[labels != minority_label][0])

    split = prepare_validation_split(
        X, y, minority_label, majority_label, hidden_ratio, random_state=random_state
    )
    X_res, y_res = oversampler.fit_resample(split.X_train, split.y_train)
    synthetic = extract_synthetic_samples(split.X_train, X_res, y_res, minority_label)
    if len(synthetic) == 0:
        raise ValidationError(
            f"{type(oversampler).__name__} produced no synthetic samples"
        )

    error_rate = validate_oversampling(
        X,
        y,
        minority_label,
        oversampler,
        hidden_ratio=hidden_ratio,
        metric=metric,
        random_state=random_state,
    )

    # return_details=False always yields a float; narrow it at the boundary
    # rather than suppressing the union.
    if isinstance(error_rate, ValidationDetails):  # pragma: no cover
        raise ValidationError(
            "validate_oversampling(return_details=False) must return a float"
        )

    utility = None
    if include_utility:
        utility = downstream_utility(X, y, oversampler, random_state=random_state)

    return FidelityReport(
        error_rate=float(error_rate),
        manifold=precision_recall_density_coverage(
            synthetic, split.reference_minority, k=k, metric=metric
        ),
        memorisation=memorisation_report(synthetic, split.fit_minority, metric=metric),
        boundary=boundary_violation_rate(
            synthetic, X, y, minority_label, k=k, metric=metric
        ),
        utility=utility,
    )
