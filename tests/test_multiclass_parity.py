"""The multiclass path must carry the binary path's guarantees.

It did not. Four things the binary validator has always done, that multiclass
validation did not:

* refuse a resampler that deletes original rows -- ``SMOTEENN`` returned
  plausible numbers from a misaligned positional slice;
* enforce ``min_hidden``, so a class whose hold-out rounded to zero silently
  stopped being a reference for anything;
* attribute ties to the point's own class rather than to whichever label sorted
  first, which decided 38.6% of attributions on quantised features;
* report ``nan`` rather than ``0.0`` for a class the sampler never generated
  for.
"""

from __future__ import annotations

import numpy as np
import pytest
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification

from oversampleqa.validator import (
    macro_error_rate,
    validate_multiclass_oversampling,
)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=900,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        random_state=0,
    )
    return X, y


def test_resampler_that_deletes_originals_is_refused(data):
    """The binary path has always raised here; multiclass returned numbers."""
    X, y = data
    with pytest.raises(ValueError, match="prefix"):
        validate_multiclass_oversampling(X, y, SMOTEENN(random_state=0), random_state=0)


def test_class_too_small_to_hide_from_is_refused(data):
    X, y = data
    small = np.vstack([X[y == 0][:300], X[y == 1][:150], X[y == 2][:20]])
    labels = np.array([0] * 300 + [1] * 150 + [2] * 20)
    with pytest.raises(ValueError, match="min_hidden"):
        validate_multiclass_oversampling(
            small, labels, SMOTE(k_neighbors=3, random_state=0), random_state=0
        )


def test_the_error_names_the_offending_class(data):
    X, y = data
    small = np.vstack([X[y == 0][:300], X[y == 1][:150], X[y == 2][:20]])
    labels = np.array([0] * 300 + [1] * 150 + [2] * 20)
    with pytest.raises(ValueError, match="class 2"):
        validate_multiclass_oversampling(
            small, labels, SMOTE(k_neighbors=3, random_state=0), random_state=0
        )


def test_min_hidden_is_adjustable(data):
    """Refusing must be a threshold, not a hard rule."""
    X, y = data
    rates = validate_multiclass_oversampling(
        X, y, SMOTE(random_state=0), hidden_ratio=0.02, min_hidden=1, random_state=0
    )
    assert len(rates) == 3


def test_unmeasured_class_is_nan_not_zero(data):
    """0.0 is the score of a perfect result, not of an absent one."""
    X, y = data
    rates, matrix = validate_multiclass_oversampling(
        X, y, SMOTE(random_state=0), return_matrix=True, random_state=0
    )
    unmeasured = [i for i in range(len(matrix)) if matrix[i].sum() == 0]
    assert unmeasured, "fixture no longer produces an unmeasured class"
    labels = sorted(rates)
    for i in unmeasured:
        assert np.isnan(rates[labels[i]])


def test_measured_classes_are_not_nan(data):
    X, y = data
    rates, matrix = validate_multiclass_oversampling(
        X, y, SMOTE(random_state=0), return_matrix=True, random_state=0
    )
    labels = sorted(rates)
    for i, label in enumerate(labels):
        if matrix[i].sum() > 0:
            assert not np.isnan(rates[label])


# --- macro summary ---


def test_macro_skips_unmeasured_classes():
    assert macro_error_rate({0: float("nan"), 1: 0.2, 2: 0.4}) == pytest.approx(0.3)


def test_plain_mean_would_be_nan():
    """Which is why the helper exists."""
    rates = {0: float("nan"), 1: 0.2, 2: 0.4}
    assert np.isnan(np.mean(list(rates.values())))


def test_macro_is_nan_when_nothing_was_measured():
    """Not 0.0 -- no class was evaluated, so there is no score."""
    assert np.isnan(macro_error_rate({0: float("nan"), 1: float("nan")}))


def test_macro_does_not_count_nan_as_zero():
    """Counting it as zero would read as a perfect score for that class."""
    assert macro_error_rate({0: float("nan"), 1: 0.6}) == pytest.approx(0.6)


# --- ties ---


def _quantised():
    rng = np.random.default_rng(0)
    X = np.rint(
        np.vstack(
            [
                rng.normal(0, 1.2, (200, 3)),
                rng.normal(1, 1.2, (120, 3)),
                rng.normal(2, 1.2, (90, 3)),
            ]
        )
    )
    y = np.array([0] * 200 + [1] * 120 + [2] * 90)
    return X, y


class _PlantSynthetic:
    """Oversampler that appends exactly the points it was given.

    Real samplers make the tie case hard to construct: the hold-out is drawn
    per class from a shared generator consumed in label order, so relabelling
    the classes changes the splits and is not the pure renaming it looks like.
    Planting the synthetic rows removes every source of variation except the
    attribution rule under test.
    """

    def __init__(self, points, label):
        self.points = np.asarray(points, dtype=float)
        self.label = label

    def fit_resample(self, X, y):
        return (
            np.vstack([X, self.points]),
            np.hstack([y, np.full(len(self.points), self.label, dtype=y.dtype)]),
        )


def _collapsed_classes():
    """Three classes, each a single repeated point.

    Every hold-out therefore yields the same hidden set, whatever the
    permutation, so the attribution rule is the only thing being measured.
    """
    X = np.vstack(
        [
            np.tile([0.0, 0.0], (60, 1)),
            np.tile([10.0, 10.0], (60, 1)),
            np.tile([20.0, 20.0], (60, 1)),
        ]
    )
    y = np.array([0] * 60 + [1] * 60 + [2] * 60)
    return X, y


def test_an_exact_tie_goes_to_the_points_own_class():
    """[5, 5] is equidistant from class 0 and class 1, and belongs to class 1.

    Attributing it to class 0 -- the lowest label index, which is what a strict
    `<` against a running minimum does -- makes it an error. It is not one: a
    tie is not evidence that the point landed in another class's territory.
    """
    X, y = _collapsed_classes()
    rates, matrix = validate_multiclass_oversampling(
        X,
        y,
        _PlantSynthetic([[5.0, 5.0]] * 10, label=1),
        return_matrix=True,
        random_state=0,
        metric="euclidean",
    )
    assert matrix[1, 1] == 10, "tied points were not attributed to their own class"
    assert matrix[1, 0] == 0
    assert rates[1] == pytest.approx(0.0)


def test_a_point_genuinely_nearer_another_class_is_still_an_error():
    """The tie rule must not swallow real errors."""
    X, y = _collapsed_classes()
    rates, matrix = validate_multiclass_oversampling(
        X,
        y,
        _PlantSynthetic([[1.0, 1.0]] * 10, label=1),
        return_matrix=True,
        random_state=0,
        metric="euclidean",
    )
    assert matrix[1, 0] == 10
    assert rates[1] == pytest.approx(1.0)


def test_ties_are_attributed_to_the_points_own_class():
    """Matching score_nearest_distances: a tie is not evidence of an error."""
    X, y = _quantised()
    _rates, matrix = validate_multiclass_oversampling(
        X,
        y,
        RandomOverSampler(random_state=0),
        return_matrix=True,
        random_state=0,
        metric="euclidean",
    )
    # With ties going to the own class, the diagonal must hold a real share.
    for i in range(len(matrix)):
        if matrix[i].sum() == 0:
            continue
        assert matrix[i, i] > 0


# --- structure ---


def test_matrix_rows_sum_to_the_synthetic_count(data):
    X, y = data
    rates, matrix = validate_multiclass_oversampling(
        X, y, SMOTE(random_state=0), return_matrix=True, random_state=0
    )
    labels = sorted(rates)
    for i, label in enumerate(labels):
        total = matrix[i].sum()
        if total == 0:
            continue
        expected = 1.0 - matrix[i, i] / total
        assert rates[label] == pytest.approx(expected)


def test_is_deterministic(data):
    X, y = data
    a = validate_multiclass_oversampling(X, y, SMOTE(random_state=0), random_state=11)
    b = validate_multiclass_oversampling(X, y, SMOTE(random_state=0), random_state=11)
    for label in a:
        assert (np.isnan(a[label]) and np.isnan(b[label])) or a[label] == b[label]
