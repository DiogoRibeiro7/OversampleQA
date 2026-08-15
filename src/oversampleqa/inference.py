"""Inference for the hidden-majority error rate.

The error rate on its own is uninterpretable: it depends on dimensionality,
minority density, ``hidden_ratio`` and the metric, so 0.13 means nothing without
a reference. This module supplies the two things that make it a diagnostic
rather than a number:

**A null distribution.** Score *real held-out minority points* through the exact
same pipeline. That is the rate an ideal generator would achieve -- one drawing
from the true minority distribution -- so the observed rate can be reported as a
position within it. A ceiling reference, from deliberately bad points, bounds the
other end.

**Two-sample tests.** Counting how often a point's nearest neighbour comes from
the other sample *is* the nearest-neighbour two-sample statistic of Schilling
(1986) and Henze (1988). Naming it that brings a null distribution, a
permutation test, and a literature along with it.

References
----------
Schilling, M. F. (1986). Multivariate two-sample tests based on nearest
neighbors. *JASA* 81(395).

Henze, N. (1988). A multivariate two-sample test based on the number of nearest
neighbor type coincidences. *Annals of Statistics* 16(2).

Friedman, J. H. & Rafsky, L. C. (1979). Multivariate generalizations of the
Wald-Wolfowitz and Smirnov two-sample tests. *Annals of Statistics* 7(4).

Rosenbaum, P. R. (2005). An exact distribution-free test comparing two
multivariate distributions based on adjacency. *JRSS-B* 67(4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from ._rng import RandomStateLike, as_generator, spawn_generators
from .distance import distance_matrix
from .exceptions import ValidationError
from .metrics import calculate_error_rate
from .validator import prepare_validation_split, score_nearest_distances

__all__ = [
    "NullCalibration",
    "TwoSampleTestResult",
    "cross_match_test",
    "mst_two_sample_test",
    "nn_two_sample_test",
    "null_error_rate",
]


@dataclass(frozen=True)
class NullCalibration:
    """Where an observed error rate sits against known reference points.

    Attributes
    ----------
    observed:
        The error rate being interpreted.
    null_rates:
        Error rates from scoring *real* held-out minority points -- what an
        ideal generator, drawing from the true minority distribution, achieves.
    ceiling_rates:
        Error rates from deliberately bad points drawn from the majority
        region. The other end of the scale.
    z_score:
        ``(observed - null_mean) / null_sd``. Positive means worse than ideal.
        ``nan`` when the null has no spread.
    percentile:
        Empirical percentile of ``observed`` within ``null_rates``.
    scaled:
        Position on a 0-1 scale where 0 is the null mean and 1 the ceiling
        mean. Above 1 is worse than a deliberately bad generator.
    """

    observed: float
    null_rates: tuple[float, ...]
    ceiling_rates: tuple[float, ...]
    z_score: float
    percentile: float
    scaled: float
    metric: str
    n_draws: int

    @property
    def null_mean(self) -> float:
        """Mean of the null distribution."""
        return float(np.mean(self.null_rates)) if self.null_rates else float("nan")

    @property
    def null_sd(self) -> float:
        """Standard deviation of the null distribution."""
        if len(self.null_rates) < 2:
            return float("nan")
        return float(np.std(self.null_rates, ddof=1))

    @property
    def ceiling_mean(self) -> float:
        """Mean of the ceiling distribution."""
        return (
            float(np.mean(self.ceiling_rates)) if self.ceiling_rates else float("nan")
        )

    def null_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Percentile interval of the null distribution."""
        if len(self.null_rates) < 2:
            return (float("nan"), float("nan"))
        alpha = 1.0 - confidence
        arr = np.asarray(self.null_rates)
        return (
            float(np.percentile(arr, 100 * alpha / 2)),
            float(np.percentile(arr, 100 * (1 - alpha / 2))),
        )

    def interpret(self) -> str:
        """One-line reading of where the observed rate falls."""
        low, high = self.null_interval()
        if np.isnan(low):
            return "Not enough draws to calibrate."
        if self.observed <= high:
            return (
                f"{self.observed:.3f} is within the null interval "
                f"[{low:.3f}, {high:.3f}] -- indistinguishable from an ideal "
                "generator on this data."
            )
        return (
            f"{self.observed:.3f} is above the null interval "
            f"[{low:.3f}, {high:.3f}] (z={self.z_score:.2f}) -- worse than an "
            "ideal generator would achieve here."
        )

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the reporting layer."""
        low, high = self.null_interval()
        return {
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "null_ci_lower": low,
            "null_ci_upper": high,
            "ceiling_mean": self.ceiling_mean,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "scaled": self.scaled,
            "metric": self.metric,
            "n_draws": self.n_draws,
        }


@dataclass(frozen=True)
class TwoSampleTestResult:
    """Outcome of a two-sample test between synthetic and real points.

    A **high** p-value is weak evidence that the two samples are
    distributionally indistinguishable -- which is what good synthesis looks
    like. See the warning in :func:`nn_two_sample_test` about what failing to
    reject does *not* mean.
    """

    name: str
    statistic: float
    p_value: float
    n_synthetic: int
    n_real: int
    n_permutations: int
    asymptotic_p_value: float | None = None
    null_statistics: tuple[float, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the reporting layer."""
        return {
            "test": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "asymptotic_p_value": self.asymptotic_p_value,
            "n_synthetic": self.n_synthetic,
            "n_real": self.n_real,
            "n_permutations": self.n_permutations,
        }


def _score_against(
    candidates: NDArray[np.floating],
    hidden_majority: NDArray[np.floating],
    reference_minority: NDArray[np.floating],
    metric: str,
    metric_kwargs: dict[str, Any] | None = None,
) -> float:
    """Run the standard scoring pipeline over arbitrary candidate points.

    This is the same comparison ``validate_oversampling`` performs; isolating it
    is what lets the null be computed with *real* points substituted for
    synthetic ones, so the two rates are directly comparable.
    """
    if len(candidates) == 0:
        return float("nan")
    kwargs = metric_kwargs or {}
    dist_hidden = distance_matrix(candidates, hidden_majority, metric, **kwargs)
    dist_min = distance_matrix(candidates, reference_minority, metric, **kwargs)
    errors, _ties = score_nearest_distances(
        dist_hidden.min(axis=1), dist_min.min(axis=1)
    )
    return calculate_error_rate(errors, len(candidates))


def null_error_rate(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    observed: float,
    *,
    hidden_ratio: float = 0.1,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    n_draws: int = 200,
    min_hidden: int = 5,
    random_state: RandomStateLike = 42,
) -> NullCalibration:
    """Calibrate an observed error rate against ideal and worst-case references.

    The null is built by scoring **real held-out minority points** through the
    identical pipeline. Those points are, by construction, drawn from the true
    minority distribution, so their error rate is what a perfect generator would
    score. Anything an actual oversampler achieves can then be read as a
    position relative to that.

    The ceiling uses points drawn from the majority region -- what a
    deliberately bad generator produces -- bounding the other end of the scale.

    Parameters
    ----------
    X, y : ndarray
        Input data and labels.
    minority_label : int
        Label of the minority class.
    observed : float
        The error rate to interpret, e.g. from
        :func:`~oversampleqa.validate_oversampling`.
    hidden_ratio : float, default=0.1
        Fraction held out. Must match the run that produced ``observed``, or
        the comparison is meaningless.
    metric : str, default="hassanat"
        Distance metric. Must also match.
    n_draws : int, default=200
        Independent splits behind the null distribution.
    min_hidden : int, default=5
        Minimum held-out minority points per draw.
    random_state : int, Generator, SeedSequence or None, default=42
        Seeds the draws.

    Returns
    -------
    NullCalibration

    Raises
    ------
    ValidationError
        If the labels are not binary or the minority is too small.

    Notes
    -----
    ``hidden_ratio`` and ``metric`` must match the run that produced
    ``observed``. The error rate's scale depends on both, so calibrating
    against a null computed with different settings compares two different
    quantities.
    """
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValidationError(
            f"null_error_rate expects binary labels; got {len(labels)} distinct values"
        )
    if minority_label not in labels:
        raise ValidationError(f"minority_label {minority_label} not found in y")
    majority_label = int(labels[labels != minority_label][0])

    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1; got {n_draws}")

    generators = spawn_generators(random_state, n_draws)
    majority = X[y != minority_label]

    null_rates: list[float] = []
    ceiling_rates: list[float] = []

    for gen in generators:
        split = prepare_validation_split(
            X,
            y,
            minority_label,
            majority_label,
            hidden_ratio,
            reference="hidden_minority",
            min_hidden=min_hidden,
            random_state=gen,
        )
        # The null: score the held-out *real* minority points. They cannot be
        # scored against themselves, so the fitted minority is the reference.
        null_rates.append(
            _score_against(
                split.reference_minority,
                split.hid_majority,
                split.fit_minority,
                metric,
                metric_kwargs,
            )
        )
        # The ceiling: majority points standing in for synthetic ones, i.e. a
        # generator that has learned the wrong distribution entirely.
        n_bad = len(split.reference_minority)
        bad_idx = gen.choice(
            len(majority), size=min(n_bad, len(majority)), replace=False
        )
        ceiling_rates.append(
            _score_against(
                majority[bad_idx],
                split.hid_majority,
                split.fit_minority,
                metric,
                metric_kwargs,
            )
        )

    null_arr = np.asarray(null_rates, dtype=float)
    finite_null = null_arr[np.isfinite(null_arr)]
    ceiling_arr = np.asarray(ceiling_rates, dtype=float)
    finite_ceiling = ceiling_arr[np.isfinite(ceiling_arr)]

    null_mean = float(np.mean(finite_null)) if finite_null.size else float("nan")
    null_sd = (
        float(np.std(finite_null, ddof=1)) if finite_null.size > 1 else float("nan")
    )
    ceiling_mean = (
        float(np.mean(finite_ceiling)) if finite_ceiling.size else float("nan")
    )

    z = (observed - null_mean) / null_sd if null_sd and null_sd > 0 else float("nan")
    percentile = (
        float((finite_null <= observed).mean() * 100.0)
        if finite_null.size
        else float("nan")
    )
    span = ceiling_mean - null_mean
    scaled = (
        (observed - null_mean) / span if span and abs(span) > 1e-12 else float("nan")
    )

    return NullCalibration(
        observed=observed,
        null_rates=tuple(null_rates),
        ceiling_rates=tuple(ceiling_rates),
        z_score=z,
        percentile=percentile,
        scaled=scaled,
        metric=metric,
        n_draws=n_draws,
    )


def _pooled_distances(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    metric: str,
    metric_kwargs: dict[str, Any] | None = None,
) -> NDArray[np.floating]:
    """Pooled pairwise distance matrix for the two samples stacked A-then-B.

    Computed once and reused across every permutation: permutations only
    relabel points, they do not move them, so recomputing distances per
    permutation would repeat identical work ``n_permutations`` times. This is
    the core optimisation of the permutation tests here.
    """
    pooled = np.vstack([A, B])
    return distance_matrix(pooled, pooled, metric, **(metric_kwargs or {}))


def _nn_coincidences(
    distances: NDArray[np.floating], labels: NDArray[np.integer], k: int
) -> int:
    """Count k-nearest-neighbour pairs sharing a sample label."""
    # A point is its own nearest neighbour at distance 0; exclude it.
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    neighbours = np.argsort(masked, axis=1, kind="stable")[:, :k]
    return int((labels[neighbours] == labels[:, None]).sum())


def nn_two_sample_test(
    synthetic: NDArray[np.floating],
    real: NDArray[np.floating],
    *,
    k: int = 3,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    n_permutations: int = 999,
    random_state: RandomStateLike = 42,
) -> TwoSampleTestResult:
    """Schilling-Henze nearest-neighbour two-sample test.

    Of the ``k`` nearest neighbours of each point in the pooled sample, count
    how many share its sample label. If the two samples come from the same
    distribution, neighbours are labelled roughly at the base rate; if they are
    separated, points cluster with their own kind and the count rises.

    Applied to synthetic points against held-out real minority points, this
    tests the question a user actually has: *are these synthetic points
    distributionally indistinguishable from real ones?* A **high p-value is
    evidence of good synthesis**.

    .. warning::

       **Failing to reject is not proof of equality.** The power of every
       nearest-neighbour test collapses as dimension grows, so on
       high-dimensional data a large p-value may reflect a lack of power rather
       than genuine similarity. Always read it next to ``n_synthetic`` and
       ``n_real``, which are returned for exactly this reason.

    Parameters
    ----------
    synthetic, real : ndarray
        The two samples.
    k : int, default=3
        Neighbours considered per point.
    metric : str, default="hassanat"
        Any metric from the package registry, so ``hassanat`` composes with the
        inferential layer.
    n_permutations : int, default=999
        Permutations behind the p-value. The pooled distance matrix is computed
        once and reused; permutations only relabel.
    random_state : int, Generator, SeedSequence or None, default=42
        Seeds the permutations.

    Returns
    -------
    TwoSampleTestResult
        Carries both the permutation p-value and the asymptotic normal
        approximation, so the user can see where they disagree.
    """
    n1, n2 = len(synthetic), len(real)
    if n1 == 0 or n2 == 0:
        raise ValidationError("both samples must be non-empty")
    n = n1 + n2
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the pooled size {n}")

    distances = _pooled_distances(synthetic, real, metric, metric_kwargs)
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    observed = _nn_coincidences(distances, labels, k)

    rng = as_generator(random_state)
    null: list[int] = []
    for _ in range(n_permutations):
        null.append(_nn_coincidences(distances, rng.permutation(labels), k))

    null_arr = np.asarray(null)
    # +1 in both terms: the observed value is itself one draw from the null,
    # which keeps the p-value valid (never exactly zero).
    p_perm = float((np.sum(null_arr >= observed) + 1) / (n_permutations + 1))

    # Asymptotic normal approximation (Schilling 1986).
    lam1, lam2 = n1 / n, n2 / n
    mean = n * k * (lam1**2 + lam2**2)
    var = n * k * (lam1 * lam2 + 4 * lam1**2 * lam2**2)
    p_asym = (
        float(1.0 - stats.norm.cdf((observed - mean) / np.sqrt(var)))
        if var > 0
        else float("nan")
    )

    return TwoSampleTestResult(
        name="schilling_henze_nn",
        statistic=float(observed),
        p_value=p_perm,
        asymptotic_p_value=p_asym,
        n_synthetic=n1,
        n_real=n2,
        n_permutations=n_permutations,
        null_statistics=tuple(float(v) for v in null_arr),
    )


def _mst_cross_edges(
    distances: NDArray[np.floating], labels: NDArray[np.integer]
) -> int:
    """Count minimum-spanning-tree edges joining the two samples."""
    tree = minimum_spanning_tree(csr_matrix(distances))
    rows, cols = tree.nonzero()
    return int((labels[rows] != labels[cols]).sum())


def mst_two_sample_test(
    synthetic: NDArray[np.floating],
    real: NDArray[np.floating],
    *,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    n_permutations: int = 999,
    random_state: RandomStateLike = 42,
) -> TwoSampleTestResult:
    """Friedman-Rafsky minimum-spanning-tree two-sample test.

    Build the MST on the pooled sample and count edges joining the two samples.
    Well-mixed samples produce many cross edges; separated ones produce few, so
    **small** counts are evidence against equality and the p-value is
    left-tailed.

    The same power caveat as :func:`nn_two_sample_test` applies.

    Returns
    -------
    TwoSampleTestResult
    """
    n1, n2 = len(synthetic), len(real)
    if n1 == 0 or n2 == 0:
        raise ValidationError("both samples must be non-empty")

    distances = _pooled_distances(synthetic, real, metric, metric_kwargs)
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    observed = _mst_cross_edges(distances, labels)

    rng = as_generator(random_state)
    null = [
        _mst_cross_edges(distances, rng.permutation(labels))
        for _ in range(n_permutations)
    ]
    null_arr = np.asarray(null)
    # Left-tailed: few cross edges means the samples separate.
    p_perm = float((np.sum(null_arr <= observed) + 1) / (n_permutations + 1))

    return TwoSampleTestResult(
        name="friedman_rafsky_mst",
        statistic=float(observed),
        p_value=p_perm,
        n_synthetic=n1,
        n_real=n2,
        n_permutations=n_permutations,
        null_statistics=tuple(float(v) for v in null_arr),
    )


def _greedy_cross_matches(
    distances: NDArray[np.floating], labels: NDArray[np.integer]
) -> int:
    """Count cross-sample pairs in a greedy non-bipartite matching.

    Optimal non-bipartite matching is the exact Rosenbaum construction; this is
    the documented greedy approximation -- repeatedly take the globally closest
    unmatched pair. It is deterministic given the distances, which is what the
    permutation test needs, but it does not minimise total matched distance, so
    the statistic is not identical to Rosenbaum's.
    """
    n = len(labels)
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    unmatched = np.ones(n, dtype=bool)
    cross = 0
    for _ in range(n // 2):
        sub = np.where(unmatched[:, None] & unmatched[None, :], masked, np.inf)
        flat = int(np.argmin(sub))
        i, j = divmod(flat, n)
        if not np.isfinite(sub[i, j]):
            break
        unmatched[i] = unmatched[j] = False
        if labels[i] != labels[j]:
            cross += 1
    return cross


def cross_match_test(
    synthetic: NDArray[np.floating],
    real: NDArray[np.floating],
    *,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    n_permutations: int = 999,
    random_state: RandomStateLike = 42,
) -> TwoSampleTestResult:
    """Rosenbaum cross-match test, with a greedy matching.

    Pair up the pooled sample and count how many pairs join the two samples.
    Well-mixed samples yield many cross pairs, so the p-value is left-tailed.

    .. note::

       Rosenbaum's test uses **optimal** non-bipartite matching, which
       minimises total matched distance and admits an exact null distribution.
       This implementation uses a greedy nearest-available matching instead, so
       the exact distribution does not apply and the p-value comes from
       permutation. The greedy statistic is generally close but not identical;
       treat it as an approximation to the published test rather than the test
       itself.

    Returns
    -------
    TwoSampleTestResult
    """
    n1, n2 = len(synthetic), len(real)
    if n1 == 0 or n2 == 0:
        raise ValidationError("both samples must be non-empty")

    distances = _pooled_distances(synthetic, real, metric, metric_kwargs)
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    observed = _greedy_cross_matches(distances, labels)

    rng = as_generator(random_state)
    null = [
        _greedy_cross_matches(distances, rng.permutation(labels))
        for _ in range(n_permutations)
    ]
    null_arr = np.asarray(null)
    p_perm = float((np.sum(null_arr <= observed) + 1) / (n_permutations + 1))

    return TwoSampleTestResult(
        name="rosenbaum_cross_match_greedy",
        statistic=float(observed),
        p_value=p_perm,
        n_synthetic=n1,
        n_real=n2,
        n_permutations=n_permutations,
        null_statistics=tuple(float(v) for v in null_arr),
    )
