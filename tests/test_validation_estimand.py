"""Tests for the validation estimand and its degenerate cases.

The binary validator used to compare held-out majority data against the *full*
minority set -- the very data the oversampler interpolated from. That biased the
error rate toward zero and gave ``RandomOverSampler``, which only duplicates, a
perfect score. These tests pin the corrected estimand and the guards around it.
"""

from __future__ import annotations

import numpy as np
import pytest
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification

from oversampleqa.memory_efficient_validator import MemoryEfficientValidator
from oversampleqa.metrics import calculate_error_rate, duplication_rate
from oversampleqa.types import ValidationConfig, ValidationDetails
from oversampleqa.typed_validator import TypedValidator
from oversampleqa.validator import extract_synthetic_samples, validate_oversampling


@pytest.fixture
def imbalanced_data():
    return make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        random_state=0,
    )


def test_default_reference_is_hidden_minority(imbalanced_data):
    X, y = imbalanced_data
    details = validate_oversampling(X, y, 1, SMOTE(random_state=0), return_details=True)
    assert isinstance(details, ValidationDetails)
    assert details.reference == "hidden_minority"


def test_train_minority_is_available_and_warns(imbalanced_data):
    """The legacy estimand stays available, but says so.

    The pinned value changed when the hold-out split moved from
    ``train_test_split`` to a ``Generator`` permutation: the same seed selects
    different points under a different splitter, so this no longer reproduces
    the pre-0.2 number. The estimand itself is unchanged -- it still compares
    against the full minority class -- which is what this pins.
    """
    X, y = imbalanced_data
    with pytest.warns(FutureWarning, match="biases the error rate"):
        rate = validate_oversampling(
            X, y, 1, SMOTE(random_state=0), reference="train_minority"
        )
    assert rate == pytest.approx(0.007125890736342043)


def test_same_seed_is_bit_identical(imbalanced_data):
    X, y = imbalanced_data
    a = validate_oversampling(X, y, 1, SMOTE(random_state=0), random_state=123)
    b = validate_oversampling(X, y, 1, SMOTE(random_state=0), random_state=123)
    assert a == b


def test_different_seed_changes_the_result(imbalanced_data):
    """Which points are hidden is the dominant driver of the error rate."""
    X, y = imbalanced_data
    a = validate_oversampling(X, y, 1, SMOTE(random_state=0), random_state=42)
    b = validate_oversampling(X, y, 1, SMOTE(random_state=0), random_state=7)
    assert a != b


def test_repeats_give_distinct_splits(imbalanced_data):
    """Assert on the split indices, not just the rates."""
    from oversampleqa._rng import spawn_generators
    from oversampleqa.validator import prepare_validation_split

    X, y = imbalanced_data
    seen = set()
    for gen in spawn_generators(42, 20):
        split = prepare_validation_split(X, y, 1, 0, 0.1, random_state=gen)
        seen.add(tuple(sorted(split.hidden_majority_index.tolist())))
    assert len(seen) == 20


def test_repeats_report_dispersion(imbalanced_data):
    X, y = imbalanced_data
    details = validate_oversampling(
        X, y, 1, SMOTE(random_state=0), n_repeats=10, return_details=True
    )
    assert details.n_repeats == 10
    assert len(details.rates) == 10
    assert details.std > 0
    assert details.interval is not None
    assert details.interval[0] < details.mean < details.interval[1]


def test_stratify_by_reaches_the_split(imbalanced_data):
    from oversampleqa.validator import prepare_validation_split

    X, y = imbalanced_data
    strata = np.arange(len(y)) % 3
    split = prepare_validation_split(
        X, y, 1, 0, 0.3, random_state=1, stratify_by=strata
    )
    majority_strata = strata[y != 1]
    held = majority_strata[split.hidden_majority_index]
    counts = np.bincount(held, minlength=3)
    # Each stratum contributes its own share, so none can be missed entirely.
    assert counts.min() > 0
    assert counts.max() - counts.min() <= 2


def test_reseed_oversampler_widens_the_dispersion(imbalanced_data):
    """Reseeding the sampler per repeat adds a second variance source."""
    X, y = imbalanced_data
    split_only = validate_oversampling(
        X, y, 1, SMOTE(random_state=0), n_repeats=8, return_details=True
    )
    both = validate_oversampling(
        X,
        y,
        1,
        SMOTE(random_state=0),
        n_repeats=8,
        reseed_oversampler=True,
        return_details=True,
    )
    assert both.rates != split_only.rates
    assert both.n_repeats == 8


def test_reseed_tolerates_samplers_without_random_state(imbalanced_data):
    """A deterministic sampler is cloned unchanged rather than rejected."""
    from oversampleqa.validator import _reseeded

    class Deterministic(SMOTE):
        def get_params(self, deep=True):
            params = super().get_params(deep=deep)
            params.pop("random_state", None)
            return params

    sampler = Deterministic(random_state=0)
    clone_ = _reseeded(sampler, np.random.default_rng(0))
    assert isinstance(clone_, SMOTE)


def test_n_repeats_must_be_positive(imbalanced_data):
    X, y = imbalanced_data
    with pytest.raises(ValueError, match="n_repeats must be at least 1"):
        validate_oversampling(X, y, 1, SMOTE(random_state=0), n_repeats=0)


def test_multiclass_accepts_random_state():
    from oversampleqa.validator import validate_multiclass_oversampling

    # Overlapping classes, so which points are hidden actually changes the
    # attribution; well-separated clusters score 0.0 under every seed.
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(0.0, 1.0, (200, 4)),
            rng.normal(0.7, 1.0, (120, 4)),
            rng.normal(1.4, 1.0, (90, 4)),
        ]
    )
    y = np.hstack([np.zeros(200), np.ones(120), np.full(90, 2)]).astype(int)
    a = validate_multiclass_oversampling(X, y, SMOTE(random_state=0), random_state=1)
    b = validate_multiclass_oversampling(X, y, SMOTE(random_state=0), random_state=1)
    c = validate_multiclass_oversampling(X, y, SMOTE(random_state=0), random_state=99)
    assert a == b
    assert a != c


def test_stratify_by_rejects_misaligned_input(imbalanced_data):
    from oversampleqa.validator import prepare_validation_split

    X, y = imbalanced_data
    with pytest.raises(ValueError, match="aligned with the full dataset"):
        prepare_validation_split(
            X, y, 1, 0, 0.1, random_state=1, stratify_by=np.arange(5)
        )


def test_hidden_minority_differs_from_train_minority(imbalanced_data):
    """The two estimands are genuinely different quantities."""
    X, y = imbalanced_data
    hidden = validate_oversampling(X, y, 1, SMOTE(random_state=0))
    with pytest.warns(FutureWarning):
        train = validate_oversampling(
            X, y, 1, SMOTE(random_state=0), reference="train_minority"
        )
    assert hidden > train


def test_random_oversampler_is_flagged_as_duplication(imbalanced_data):
    """A sampler that only copies must not score a perfect zero silently."""
    X, y = imbalanced_data
    with pytest.warns(UserWarning, match="exact copies"):
        details = validate_oversampling(
            X, y, 1, RandomOverSampler(random_state=0), return_details=True
        )
    assert details.duplication_rate == pytest.approx(1.0)


def test_smoteenn_raises_instead_of_returning_a_number(imbalanced_data):
    """Combined samplers break positional extraction; that must be loud.

    SMOTEENN removes original rows, so the prefix assumption fails. It still
    returns more rows than it was given, so a length check alone does not catch
    it -- the old code silently scored a mix of surviving originals and
    synthetics and produced a plausible-looking number.
    """
    X, y = imbalanced_data
    with pytest.raises(ValueError, match="did not preserve the original samples"):
        validate_oversampling(X, y, 1, SMOTEENN(random_state=0))


def test_extract_synthetic_samples_rejects_broken_prefix():
    original = np.array([[0.0, 0.0], [1.0, 1.0]])
    shuffled = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="did not preserve the original samples"):
        extract_synthetic_samples(original, shuffled, labels, 1)


def test_extract_synthetic_samples_accepts_appending_sampler():
    original = np.array([[0.0, 0.0], [1.0, 1.0]])
    resampled = np.vstack([original, [[2.0, 2.0]]])
    labels = np.array([0, 1, 1])
    out = extract_synthetic_samples(original, resampled, labels, 1)
    assert out.shape == (1, 2)
    assert np.array_equal(out[0], [2.0, 2.0])


def test_small_minority_raises_rather_than_producing_noise():
    """Hiding 10% of a tiny minority leaves too few points to compare against."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 4)), rng.normal(4, 1, (20, 4))])
    y = np.hstack([np.zeros(200, dtype=int), np.ones(20, dtype=int)])
    with pytest.raises(ValueError, match="below min_hidden"):
        validate_oversampling(X, y, 1, SMOTE(random_state=0), hidden_ratio=0.1)


def test_min_hidden_is_configurable():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 4)), rng.normal(4, 1, (20, 4))])
    y = np.hstack([np.zeros(200, dtype=int), np.ones(20, dtype=int)])
    rate = validate_oversampling(
        X, y, 1, SMOTE(random_state=0), hidden_ratio=0.1, min_hidden=2
    )
    assert 0.0 <= rate <= 1.0


def test_ties_are_not_counted_as_errors():
    """Points equidistant from both sets are reported, not scored as errors."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 1, (300, 3)), rng.normal(0.2, 1, (60, 3))])
    y = np.hstack([np.zeros(300, dtype=int), np.ones(60, dtype=int)])
    details = validate_oversampling(X, y, 1, SMOTE(random_state=1), return_details=True)
    assert details.n_errors + details.n_ties <= details.n_synthetic
    assert details.n_ties >= 0


def test_calculate_error_rate_returns_nan_for_empty():
    """0.0 would be indistinguishable from a perfect score."""
    assert np.isnan(calculate_error_rate(0, 0))
    assert calculate_error_rate(1, 4) == 0.25


def test_duplication_rate_extremes():
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert duplication_rate(reference.copy(), reference) == pytest.approx(1.0)
    assert duplication_rate(np.array([[5.0, 5.0]]), reference) == pytest.approx(0.0)
    assert np.isnan(duplication_rate(np.empty((0, 2)), reference))


def test_all_three_validators_agree(imbalanced_data):
    """The three validators must measure the same thing.

    They each carried their own copy of the binary logic, which is how they
    drifted; they now share ``prepare_validation_split`` and
    ``score_nearest_distances``.
    """
    X, y = imbalanced_data
    X = X.astype(float)

    plain = validate_oversampling(X, y, 1, SMOTE(random_state=0))

    mem = MemoryEfficientValidator(cache=None).validate_oversampling(
        X, y, 1, SMOTE(random_state=0)
    )

    typed = TypedValidator().validate(
        X, y, 1, SMOTE(random_state=0), ValidationConfig()
    )

    assert plain == pytest.approx(mem)
    assert plain == pytest.approx(typed["error_rate"])
