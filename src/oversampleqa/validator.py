"""Oversampling validation utilities."""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from numpy.typing import NDArray
from sklearn.base import clone

from ._rng import RandomStateLike, as_generator, integer_seed, spawn_generators
from .distance import distance_matrix
from .metrics import calculate_error_rate, duplication_rate
from .types import ReferenceSet, ValidationDetails

logger = logging.getLogger(__name__)


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


class ValidationSplit(NamedTuple):
    """Training data and reference sets for one validation run.

    Shared by :func:`validate_oversampling`, ``MemoryEfficientValidator`` and
    ``TypedValidator`` so the three cannot drift apart on what the error rate
    measures.
    """

    X_train: NDArray[np.floating]
    y_train: NDArray[np.integer]
    hid_majority: NDArray[np.floating]
    fit_minority: NDArray[np.floating]
    reference_minority: NDArray[np.floating]
    hidden_majority_index: NDArray[np.integer]


def _holdout_indices(
    n: int,
    ratio: float,
    rng: np.random.Generator,
    strata: NDArray[Any] | None = None,
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Split ``n`` positions into ``(visible, hidden)`` index arrays.

    Uses ``rng.permutation`` directly rather than ``train_test_split``: it keeps
    the binary and multiclass code paths structurally identical, and it accepts
    a ``Generator``, which scikit-learn's splitter does not.

    When ``strata`` is given, the fraction is taken within each stratum, so a
    hold-out cannot miss a group entirely.
    """
    if strata is None:
        order = rng.permutation(n)
        n_hidden = int(n * ratio)
        return order[n_hidden:], order[:n_hidden]

    strata = np.asarray(strata)
    if len(strata) != n:
        raise ValueError(
            f"stratify_by has length {len(strata)} but the class being split has "
            f"{n} samples; it must be aligned with that class."
        )
    visible_parts: list[NDArray[np.integer]] = []
    hidden_parts: list[NDArray[np.integer]] = []
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        order = rng.permutation(len(members))
        n_hidden = int(len(members) * ratio)
        hidden_parts.append(members[order[:n_hidden]])
        visible_parts.append(members[order[n_hidden:]])
    return np.concatenate(visible_parts), np.concatenate(hidden_parts)


def prepare_validation_split(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    majority_label: int,
    hidden_ratio: float,
    *,
    reference: ReferenceSet = "hidden_minority",
    minority_hidden_ratio: float | None = None,
    min_hidden: int = 5,
    random_state: RandomStateLike = 42,
    stratify_by: NDArray[Any] | None = None,
) -> ValidationSplit:
    """Build the train/hidden split defining the validation estimand.

    See :func:`validate_oversampling` for the meaning of ``reference``,
    ``random_state`` and ``stratify_by``.

    Raises
    ------
    ValueError
        If the held-out minority would contain fewer than ``min_hidden``
        points, or if ``stratify_by`` is not aligned with the majority class.
    """
    minority, majority = _split_classes(X, y, minority_label)
    rng = as_generator(random_state)

    majority_strata = None
    if stratify_by is not None:
        stratify_by = np.asarray(stratify_by)
        if len(stratify_by) != len(y):
            raise ValueError(
                f"stratify_by has length {len(stratify_by)} but y has {len(y)}; "
                "it must be aligned with the full dataset."
            )
        majority_strata = stratify_by[y != minority_label]

    vis_idx, hid_idx = _holdout_indices(
        len(majority), hidden_ratio, rng, majority_strata
    )
    vis_majority = majority[vis_idx]
    hid_majority = majority[hid_idx]

    if reference == "hidden_minority":
        ratio = hidden_ratio if minority_hidden_ratio is None else minority_hidden_ratio
        _validate_hidden_ratio(ratio)
        n_hidden_minority = int(len(minority) * ratio)
        if n_hidden_minority < min_hidden:
            raise ValueError(
                f"Holding out {ratio:.3g} of a minority class of "
                f"{len(minority)} leaves {n_hidden_minority} held-out points, "
                f"below min_hidden={min_hidden}. A nearest-neighbour comparison "
                "against so few points is not meaningful. Either supply more "
                "minority data, raise minority_hidden_ratio, or pass "
                "reference='train_minority' -- noting that it compares against "
                "the oversampler's own training data and is biased toward zero."
            )
        fit_idx, ref_idx = _holdout_indices(len(minority), ratio, rng)
        fit_minority = minority[fit_idx]
        reference_minority = minority[ref_idx]
    else:
        fit_minority = minority
        reference_minority = minority

    X_train = np.vstack([vis_majority, fit_minority])
    y_train = np.hstack(
        [
            np.full(len(vis_majority), majority_label, dtype=y.dtype),
            np.full(len(fit_minority), minority_label, dtype=y.dtype),
        ]
    )
    return ValidationSplit(
        X_train, y_train, hid_majority, fit_minority, reference_minority, hid_idx
    )


def score_nearest_distances(
    nearest_hidden: NDArray[np.floating],
    nearest_min: NDArray[np.floating],
) -> tuple[int, int]:
    """Count errors and ties from nearest-neighbour distances.

    Returns ``(n_errors, n_ties)``. The comparison is strict: a point exactly
    equidistant from both reference sets is not evidence of a majority-like
    artefact, and counting ties as errors is a one-directional bias on
    discrete or quantised features.
    """
    errors = int(np.sum(nearest_hidden < nearest_min))
    ties = int(np.sum(nearest_hidden == nearest_min))
    return errors, ties


def warn_reference_bias(reference: ReferenceSet, stacklevel: int = 3) -> None:
    """Emit the ``train_minority`` bias warning."""
    if reference == "train_minority":
        warnings.warn(
            "reference='train_minority' compares synthetic points against the "
            "minority data the oversampler interpolated from. Held-out data on "
            "one side and training data on the other biases the error rate "
            "toward zero by an amount that depends on minority density, not on "
            "oversampler quality. Use reference='hidden_minority' for a "
            "comparison where both sides are unseen.",
            FutureWarning,
            stacklevel=stacklevel,
        )


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

    Raises
    ------
    ValueError
        If the oversampler did not preserve the original samples as a prefix
        of its output, so synthetic rows cannot be identified positionally.

    Notes
    -----
    Synthetic samples are identified positionally: everything after the
    original rows is assumed to be new. That holds for the SMOTE family and
    ``RandomOverSampler``, which append. It does **not** hold for combined
    over/under-samplers such as ``SMOTEENN`` and ``SMOTETomek``, which delete
    original rows. A length check alone does not catch this -- ``SMOTEENN``
    can still return more rows than it was given while having removed some of
    the originals -- so the prefix is compared element-wise.
    """
    n = len(X_original)
    if len(X_resampled) < n or not np.array_equal(X_resampled[:n], X_original):
        raise ValueError(
            "The oversampler did not preserve the original samples as a prefix of "
            "its output, so synthetic samples cannot be identified positionally. "
            "This is expected for combined over/under-samplers such as SMOTEENN "
            "and SMOTETomek, which are not supported by validate_oversampling."
        )
    return X_resampled[n:][y_resampled[n:] == minority_label]


def validate_oversampling(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    oversampler: BaseOverSampler,
    hidden_ratio: float = 0.1,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    return_details: bool = False,
    *,
    reference: ReferenceSet = "hidden_minority",
    minority_hidden_ratio: float | None = None,
    min_hidden: int = 5,
    duplication_warn_threshold: float = 0.5,
    random_state: RandomStateLike = 42,
    stratify_by: NDArray[Any] | None = None,
    n_repeats: int = 1,
    reseed_oversampler: bool = False,
) -> float | ValidationDetails:
    """Validate oversampling using the hidden majority approach.

    A fraction of the majority class is held out. The oversampler is fitted on
    what remains, and each synthetic minority point is scored by comparing its
    nearest-neighbour distance to the hidden majority against its
    nearest-neighbour distance to a minority reference set. A synthetic point
    strictly closer to the hidden majority counts as an error.

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
        If ``True`` return a :class:`~oversampleqa.types.ValidationDetails`
        instead of the bare error rate.
    reference : {"hidden_minority", "train_minority"}, default="hidden_minority"
        Which minority set the synthetic points are compared against.

        ``"hidden_minority"`` also holds out part of the minority class and
        compares against those held-out points. Both sides of the comparison
        are then unseen, which is the same estimand
        :func:`validate_multiclass_oversampling` uses.

        ``"train_minority"`` compares against the full minority class -- the
        very data the oversampler interpolated from. This is the historical
        behaviour, retained so old numbers can be reproduced. It biases the
        result toward "no error" by an amount that depends on minority
        density rather than on oversampler quality, and emits a
        ``FutureWarning``.
    minority_hidden_ratio : float, optional
        Fraction of the minority class to hide when
        ``reference="hidden_minority"``. Defaults to ``hidden_ratio``.
    min_hidden : int, default=5
        Minimum number of held-out minority points required. A
        nearest-neighbour comparison against fewer than a handful of points is
        not meaningful, so this raises rather than warns.
    duplication_warn_threshold : float, default=0.5
        Emit a ``UserWarning`` when this fraction of synthetic points coincide
        with a real point.
    random_state : int, Generator, SeedSequence or None, default=42
        Seeds the hold-out split. **Which** points get hidden is the single
        largest driver of the error rate, so varying this varies the result.
        The default of 42 reproduces previously documented numbers; ``None``
        draws fresh entropy and is not reproducible.
    stratify_by : ndarray, optional
        Group labels aligned with ``y``. When given, the majority hold-out
        takes ``hidden_ratio`` within each group, so a hold-out cannot miss a
        cluster entirely. Strata are never inferred automatically.
    n_repeats : int, default=1
        Number of independent hold-out splits. Above 1, the returned details
        carry the per-repeat vector and its dispersion. Repeat streams are
        spawned from a ``SeedSequence`` rather than derived as ``seed + i``,
        which would correlate them.
    reseed_oversampler : bool, default=False
        Give the oversampler a fresh seed on each repeat. This changes what the
        dispersion covers -- see Notes.

    Returns
    -------
    float or ValidationDetails
        Error rate by default, ``nan`` if no synthetic samples were produced.
        With ``n_repeats > 1`` the bare return is the mean across repeats.

    Raises
    ------
    ValueError
        If the labels are not binary, if ``minority_label`` is absent, if the
        held-out minority would be smaller than ``min_hidden``, or if the
        oversampler does not preserve the original samples as a prefix.

    Notes
    -----
    The error rate is a **relative** quantity. Its scale depends on
    ``hidden_ratio``, on the density of the data, and on dimensionality, so
    values are not comparable across datasets. See :doc:`/concepts`.

    **What the repeat interval covers.** With ``n_repeats > 1`` the reported
    interval is a percentile bootstrap over the per-repeat error rates. It
    describes the variability of the **hold-out split**, conditional on this
    dataset and on the oversampler's own seed. It is *not* a confidence
    interval for a population quantity, and with ``reseed_oversampler=False``
    it does not include the oversampler's own randomness at all. Setting
    ``reseed_oversampler=True`` clones the sampler with a fresh seed per
    repeat, so the dispersion then covers both sources together -- a different,
    wider decomposition.

    Synthetic points generated from shared parent points are **not
    independent**, so a binomial interval on the error rate would be too
    narrow. Nothing here claims more than the repeat-level bootstrap supports.
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

    if reference not in ("hidden_minority", "train_minority"):
        raise ValueError(
            f"reference must be 'hidden_minority' or 'train_minority'; got {reference!r}"
        )
    warn_reference_bias(reference, stacklevel=3)

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1; got {n_repeats}")

    if n_repeats > 1:
        return _validate_repeated(
            X,
            y,
            minority_label,
            oversampler,
            hidden_ratio=hidden_ratio,
            metric=metric,
            metric_kwargs=metric_kwargs,
            return_details=return_details,
            reference=reference,
            minority_hidden_ratio=minority_hidden_ratio,
            min_hidden=min_hidden,
            duplication_warn_threshold=duplication_warn_threshold,
            random_state=random_state,
            stratify_by=stratify_by,
            n_repeats=n_repeats,
            reseed_oversampler=reseed_oversampler,
        )

    minority, _ = _split_classes(X, y, minority_label)
    split = prepare_validation_split(
        X,
        y,
        minority_label,
        majority_label,
        hidden_ratio,
        reference=reference,
        minority_hidden_ratio=minority_hidden_ratio,
        min_hidden=min_hidden,
        random_state=random_state,
        stratify_by=stratify_by,
    )
    X_train = split.X_train
    y_train = split.y_train
    hid_majority = split.hid_majority
    fit_minority = split.fit_minority
    reference_minority = split.reference_minority

    try:
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
    except Exception as exc:
        logger.exception("Oversampler failed during fit_resample")
        raise ValueError(
            f"{type(oversampler).__name__} failed to fit on the reduced training "
            f"set ({len(fit_minority)} minority points after holding out "
            f"{len(minority) - len(fit_minority)}). Neighbour-based samplers "
            "such as SMOTE require more minority points than their k_neighbors "
            "setting. Lower k_neighbors, lower minority_hidden_ratio, or supply "
            f"more minority data. Original error: {exc}"
        ) from exc

    synthetic = extract_synthetic_samples(X_train, X_res, y_res, minority_label)

    kwargs = metric_kwargs or {}
    empty = np.empty((0, 0))

    if len(synthetic) == 0:
        warnings.warn(
            f"{type(oversampler).__name__} produced no synthetic minority "
            "samples, so there is nothing to validate. Returning nan rather "
            "than 0.0, which would be indistinguishable from a perfect score.",
            UserWarning,
            stacklevel=2,
        )
        rate = float("nan")
        if return_details:
            return ValidationDetails(
                error_rate=rate,
                n_errors=0,
                n_synthetic=0,
                n_ties=0,
                duplication_rate=float("nan"),
                reference=reference,
                dist_hidden=empty,
                dist_min=empty,
            )
        return rate

    dup_rate = duplication_rate(synthetic, fit_minority)
    if dup_rate >= duplication_warn_threshold:
        warnings.warn(
            f"{dup_rate:.0%} of the synthetic samples produced by "
            f"{type(oversampler).__name__} are exact copies of real minority "
            "points. The validation error rate is not informative for a sampler "
            "that mostly duplicates: copied points sit at distance zero from the "
            "minority set and so can never be scored as errors.",
            UserWarning,
            stacklevel=2,
        )

    dist_hidden = distance_matrix(synthetic, hid_majority, metric, **kwargs)
    dist_min = distance_matrix(synthetic, reference_minority, metric, **kwargs)

    nearest_hidden = dist_hidden.min(axis=1)
    nearest_min = dist_min.min(axis=1)

    errors, n_ties = score_nearest_distances(nearest_hidden, nearest_min)
    rate = calculate_error_rate(errors, len(synthetic))

    if n_ties > 0.01 * len(synthetic):
        warnings.warn(
            f"{n_ties} of {len(synthetic)} synthetic points are exactly "
            "equidistant from the hidden majority and the minority reference "
            "set. Ties are excluded from the error count, but this many "
            "suggests duplicated or heavily quantised features.",
            UserWarning,
            stacklevel=2,
        )

    if return_details:
        return ValidationDetails(
            error_rate=rate,
            n_errors=errors,
            n_synthetic=len(synthetic),
            n_ties=n_ties,
            duplication_rate=dup_rate,
            reference=reference,
            dist_hidden=dist_hidden,
            dist_min=dist_min,
        )

    return rate


def _reseeded(
    oversampler: BaseOverSampler, rng: np.random.Generator
) -> BaseOverSampler:
    """Clone ``oversampler`` with a fresh seed, if it takes one.

    Samplers without a ``random_state`` parameter are cloned unchanged rather
    than being rejected -- a sampler can legitimately be deterministic.
    """
    clone_ = clone(oversampler)
    if "random_state" in clone_.get_params(deep=False):
        clone_.set_params(random_state=integer_seed(rng))
    return clone_


def _bootstrap_interval(
    values: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    lower = float(np.percentile(draws, 100 * alpha / 2))
    upper = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return lower, upper


def _validate_repeated(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    oversampler: BaseOverSampler,
    *,
    hidden_ratio: float,
    metric: str,
    metric_kwargs: dict[str, Any] | None,
    return_details: bool,
    reference: ReferenceSet,
    minority_hidden_ratio: float | None,
    min_hidden: int,
    duplication_warn_threshold: float,
    random_state: RandomStateLike,
    stratify_by: NDArray[Any] | None,
    n_repeats: int,
    reseed_oversampler: bool,
) -> float | ValidationDetails:
    """Run ``n_repeats`` independent hold-out splits and summarise dispersion.

    Child generators are spawned from a single ``SeedSequence`` so the repeat
    streams are independent; ``seed + i`` would correlate them.
    """
    children = spawn_generators(random_state, n_repeats + 1)
    split_rngs, boot_rng = children[:-1], children[-1]

    rates: list[float] = []
    last: ValidationDetails | None = None

    for split_rng in split_rngs:
        sampler = (
            _reseeded(oversampler, split_rng) if reseed_oversampler else oversampler
        )
        details = validate_oversampling(
            X,
            y,
            minority_label,
            sampler,
            hidden_ratio=hidden_ratio,
            metric=metric,
            metric_kwargs=metric_kwargs,
            return_details=True,
            reference=reference,
            minority_hidden_ratio=minority_hidden_ratio,
            min_hidden=min_hidden,
            duplication_warn_threshold=duplication_warn_threshold,
            random_state=split_rng,
            stratify_by=stratify_by,
            n_repeats=1,
        )
        assert isinstance(details, ValidationDetails)
        rates.append(details.error_rate)
        last = details

    arr = np.asarray(rates, dtype=float)
    finite = arr[np.isfinite(arr)]
    mean = float(np.nanmean(arr)) if finite.size else float("nan")

    if not return_details:
        return mean

    assert last is not None
    return replace(
        last,
        error_rate=mean,
        n_repeats=n_repeats,
        rates=tuple(rates),
        mean=mean,
        std=float(np.nanstd(arr, ddof=1)) if finite.size > 1 else float("nan"),
        interval=_bootstrap_interval(finite, boot_rng) if finite.size > 1 else None,
    )


def validate_multiclass_oversampling(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    oversampler: BaseOverSampler,
    hidden_ratio: float = 0.1,
    metric: str = "hassanat",
    metric_kwargs: dict[str, Any] | None = None,
    return_matrix: bool = False,
    *,
    random_state: RandomStateLike = 42,
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
    random_state : int, Generator, SeedSequence or None, default=42
        Seeds the per-class hold-out. Defaults to 42, which reproduces
        previously documented numbers.

    Returns
    -------
    dict or tuple
        Mapping of ``class_label -> error_rate``. If ``return_matrix`` is
        ``True`` the second element is the error matrix.
    """

    _validate_hidden_ratio(hidden_ratio)
    labels = np.unique(y)
    rng = as_generator(random_state)

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
