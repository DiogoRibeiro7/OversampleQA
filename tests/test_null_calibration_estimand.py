"""The null must calibrate the same experiment the validator measured.

Two defects motivated these tests.

The null scored held-out minority points against ``fit_minority`` -- roughly 90%
of the minority -- while ``validate_oversampling`` scores synthetic points
against ``reference_minority``, the held-out 10%. A denser reference means
closer nearest neighbours and fewer errors, so the null sat at 0.0325 where the
same experiment gives 0.1325: a bar four times too low, against which an
ordinary sampler looks significantly worse than ideal.

The ceiling drew its deliberately-bad points from the *full* majority, which
contains the hidden majority they are scored against. A candidate that is
itself in the reference sits at distance zero from it and is an error by
construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from oversampleqa import inference
from oversampleqa.exceptions import ValidationError
from oversampleqa.inference import NullCalibration, null_error_rate


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        random_state=0,
    )
    return X, y


def _rows(array):
    return {tuple(row) for row in np.asarray(array)}


@pytest.fixture
def captured(monkeypatch, data):
    """Record every (candidates, hidden, reference) triple scored."""
    calls = []
    original = inference._score_against

    def spy(candidates, hidden_majority, reference_minority, metric, metric_kwargs=None):
        calls.append((candidates, hidden_majority, reference_minority))
        return original(candidates, hidden_majority, reference_minority, metric, metric_kwargs)

    monkeypatch.setattr(inference, "_score_against", spy)
    X, y = data
    null_error_rate(X, y, 1, 0.1, hidden_ratio=0.1, n_draws=4, random_state=7)
    return calls


def test_null_candidates_are_disjoint_from_their_reference(captured):
    """Scoring a point against a set it belongs to measures nothing."""
    for candidates, _hidden, reference in captured:
        assert not (_rows(candidates) & _rows(reference))


def test_null_and_ceiling_share_one_reference(captured):
    """Two scales built on different references cannot be compared."""
    # Calls alternate null, ceiling within each draw.
    for i in range(0, len(captured), 2):
        _, _, null_ref = captured[i]
        _, _, ceiling_ref = captured[i + 1]
        assert _rows(null_ref) == _rows(ceiling_ref)


def test_ceiling_candidates_are_never_hidden_majority_points(captured):
    """They would sit at distance zero from the reference: errors by fiat."""
    for i in range(1, len(captured), 2):
        candidates, hidden, _ = captured[i]
        assert not (_rows(candidates) & _rows(hidden))


def test_reference_is_the_held_out_minority_not_the_fitted_set(captured, data):
    """The reference must be the small held-out set the validator uses.

    A reference of roughly 90% of the minority is a different quantity, and was
    what made the null four times too low.
    """
    _X, y = data
    n_minority = int(np.sum(y == 1))
    for _candidates, _hidden, reference in captured:
        assert len(reference) < n_minority * 0.5


def test_calibration_is_deterministic(data):
    X, y = data
    a = null_error_rate(X, y, 1, 0.1, n_draws=10, random_state=3)
    b = null_error_rate(X, y, 1, 0.1, n_draws=10, random_state=3)
    assert a.null_rates == b.null_rates
    assert a.ceiling_rates == b.ceiling_rates


def test_ceiling_sits_above_the_null(data):
    """Majority-drawn points must score worse than real minority ones."""
    X, y = data
    cal = null_error_rate(X, y, 1, 0.1, n_draws=30, random_state=5)
    assert cal.ceiling_mean > cal.null_mean


def test_minority_too_small_for_three_way_split_raises(data):
    X, y = data
    rng = np.random.default_rng(0)
    X_small = np.vstack([X[y == 0][:200], rng.normal(3, 1, (12, X.shape[1]))])
    y_small = np.array([0] * 200 + [1] * 12)
    with pytest.raises(ValidationError, match="three disjoint minority sets"):
        null_error_rate(X_small, y_small, 1, 0.1, n_draws=5, random_state=0)


# --- interpret() ---


def _calibration(observed: float, rates: tuple[float, ...]) -> NullCalibration:
    return NullCalibration(
        observed=observed,
        null_rates=rates,
        ceiling_rates=(0.9,),
        z_score=0.0,
        percentile=50.0,
        scaled=0.0,
        metric="hassanat",
        n_draws=len(rates),
    )


NULL = tuple(np.linspace(0.10, 0.20, 50))


def test_interpret_reports_above_the_interval():
    assert "above the null interval" in _calibration(0.9, NULL).interpret()


def test_interpret_reports_within_the_interval():
    assert "within the null interval" in _calibration(0.15, NULL).interpret()


def test_interpret_reports_below_the_interval():
    """A value under `low` was previously reported as 'within'."""
    text = _calibration(0.01, NULL).interpret()
    assert "below the null interval" in text
    assert "within" not in text


def test_interpret_below_warns_about_memorisation():
    """Beating real held-out minority points is a red flag, not a win."""
    assert "memorisation" in _calibration(0.01, NULL).interpret().lower()


def test_interpret_boundaries_are_consistent_with_the_interval():
    cal = _calibration(0.0, NULL)
    low, high = cal.null_interval()
    for observed, expected in [
        (low - 0.01, "below"),
        ((low + high) / 2, "within"),
        (high + 0.01, "above"),
    ]:
        assert expected in _calibration(observed, NULL).interpret()


def test_interpret_without_enough_draws():
    assert "Not enough draws" in _calibration(0.1, (0.1,)).interpret()
