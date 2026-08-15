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


def test_train_minority_reproduces_old_numbers_and_warns(imbalanced_data):
    """The legacy estimand stays available, but says so."""
    X, y = imbalanced_data
    with pytest.warns(FutureWarning, match="biases the error rate"):
        rate = validate_oversampling(
            X, y, 1, SMOTE(random_state=0), reference="train_minority"
        )
    # Value recorded from the implementation before the estimand changed.
    assert rate == pytest.approx(0.009523809523809525)


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
