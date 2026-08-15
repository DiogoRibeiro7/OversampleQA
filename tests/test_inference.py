"""Tests for null calibration and the nearest-neighbour two-sample tests.

The error rate on its own is uninterpretable. These tests pin the two things
that make it readable: a null built from real held-out minority points, and
two-sample tests that answer whether synthetic points are distributionally
indistinguishable from real ones.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from oversampleqa.exceptions import ValidationError
from oversampleqa.inference import (
    TwoSampleTestResult,
    _score_against,
    cross_match_test,
    mst_two_sample_test,
    nn_two_sample_test,
    null_error_rate,
)
from oversampleqa.validator import prepare_validation_split

ALL_TESTS = (nn_two_sample_test, mst_two_sample_test, cross_match_test)


@pytest.fixture
def separable():
    """Majority and minority that are clearly distinguishable."""
    rng = np.random.default_rng(0)
    majority = rng.normal(0.0, 1.0, (400, 4))
    minority = rng.normal(2.5, 1.0, (150, 4))
    X = np.vstack([majority, minority])
    y = np.hstack([np.zeros(400), np.ones(150)]).astype(int)
    return X, y


@pytest.fixture
def split(separable):
    X, y = separable
    return prepare_validation_split(X, y, 1, 0, 0.1, random_state=7)


def _ideal_points(n=60, seed=101):
    """Points drawn from the true minority distribution."""
    return np.random.default_rng(seed).normal(2.5, 1.0, (n, 4))


def _bad_points(n=60, seed=202):
    """Points drawn from the majority distribution -- a wrong generator."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, 4))


def test_ideal_points_are_indistinguishable_from_the_null(separable, split):
    """A generator drawing from the true distribution must look like the null.

    This is the property that makes the calibration meaningful: if points drawn
    from the real minority distribution scored far from the null, the null would
    not be a reference for anything.
    """
    X, y = separable
    observed = _score_against(
        _ideal_points(), split.hid_majority, split.fit_minority, "hassanat"
    )
    calibration = null_error_rate(X, y, 1, observed, n_draws=60, random_state=1)

    low, high = calibration.null_interval()
    assert low <= calibration.observed <= high
    assert abs(calibration.z_score) < 3.0
    assert "indistinguishable" in calibration.interpret()


def test_majority_drawn_points_are_flagged(separable, split):
    """The other end of the scale: a generator that learned the wrong thing."""
    X, y = separable
    observed = _score_against(
        _bad_points(), split.hid_majority, split.fit_minority, "hassanat"
    )
    calibration = null_error_rate(X, y, 1, observed, n_draws=60, random_state=1)

    _low, high = calibration.null_interval()
    assert calibration.observed > high
    assert calibration.z_score > 5.0
    assert "worse than an ideal generator" in calibration.interpret()


def test_ceiling_is_above_the_null(separable):
    """The scale needs both ends, or 'worse' has no magnitude."""
    X, y = separable
    calibration = null_error_rate(X, y, 1, 0.5, n_draws=40, random_state=2)
    assert calibration.ceiling_mean > calibration.null_mean


def test_scaled_position_places_observed_between_the_references(separable, split):
    X, y = separable
    bad = _score_against(
        _bad_points(), split.hid_majority, split.fit_minority, "hassanat"
    )
    calibration = null_error_rate(X, y, 1, bad, n_draws=40, random_state=3)
    # 0 is the null mean, 1 the ceiling mean; a majority-drawn generator is at
    # the ceiling, so it should land near 1.
    assert calibration.scaled > 0.5


def test_calibration_is_deterministic_given_a_seed(separable):
    X, y = separable
    a = null_error_rate(X, y, 1, 0.2, n_draws=20, random_state=5)
    b = null_error_rate(X, y, 1, 0.2, n_draws=20, random_state=5)
    assert a.null_rates == b.null_rates
    assert a.ceiling_rates == b.ceiling_rates


def test_calibration_to_dict_is_flat(separable):
    X, y = separable
    payload = null_error_rate(X, y, 1, 0.2, n_draws=10, random_state=6).to_dict()
    assert isinstance(payload, dict)
    assert {"observed", "null_mean", "ceiling_mean", "z_score"} <= set(payload)
    assert all(not isinstance(v, (list, tuple, dict)) for v in payload.values())


def test_calibration_rejects_multiclass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 3))
    y = np.repeat([0, 1, 2], 30)
    with pytest.raises(ValidationError, match="binary labels"):
        null_error_rate(X, y, 1, 0.1, n_draws=5)


@pytest.mark.parametrize("test_fn", ALL_TESTS)
def test_ideal_points_are_not_rejected(test_fn, split):
    """Points from the true distribution should not be flagged as different."""
    result = test_fn(
        _ideal_points(), split.reference_minority, n_permutations=199, random_state=3
    )
    assert isinstance(result, TwoSampleTestResult)
    assert result.p_value > 0.05


@pytest.mark.parametrize("test_fn", ALL_TESTS)
def test_majority_drawn_points_are_rejected(test_fn, split):
    """Every test must have power against an obviously wrong generator."""
    result = test_fn(
        _bad_points(), split.reference_minority, n_permutations=199, random_state=3
    )
    assert result.p_value < 0.05


@pytest.mark.parametrize("test_fn", ALL_TESTS)
def test_results_report_sample_sizes(test_fn, split):
    """Sizes travel with the p-value because power depends on them."""
    result = test_fn(
        _ideal_points(), split.reference_minority, n_permutations=99, random_state=3
    )
    assert result.n_synthetic == 60
    assert result.n_real == len(split.reference_minority)
    assert result.n_permutations == 99


@pytest.mark.parametrize("test_fn", ALL_TESTS)
def test_tests_are_deterministic_given_a_seed(test_fn, split):
    a = test_fn(
        _ideal_points(), split.reference_minority, n_permutations=99, random_state=4
    )
    b = test_fn(
        _ideal_points(), split.reference_minority, n_permutations=99, random_state=4
    )
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value


@pytest.mark.parametrize("test_fn", ALL_TESTS)
def test_p_values_are_never_zero(test_fn, split):
    """The observed value is itself a draw from the null, hence the +1 terms."""
    result = test_fn(
        _bad_points(), split.reference_minority, n_permutations=99, random_state=3
    )
    assert result.p_value >= 1.0 / (99 + 1)


@pytest.mark.parametrize("test_fn", ALL_TESTS)
def test_empty_sample_raises(test_fn, split):
    with pytest.raises(ValidationError, match="non-empty"):
        test_fn(np.empty((0, 4)), split.reference_minority, n_permutations=9)


def test_nn_statistic_matches_a_hand_computed_reference():
    """Validate against a case worked out by hand, not against itself.

    Two points per sample, arranged so each point's single nearest neighbour is
    its own partner: every one of the 4 nearest-neighbour pairs is a same-sample
    coincidence, so the statistic is 4.
    """
    a = np.array([[0.0], [0.1]])
    b = np.array([[10.0], [10.1]])
    result = nn_two_sample_test(a, b, k=1, n_permutations=49, metric="euclidean")
    assert result.statistic == 4.0


def test_mst_statistic_matches_a_hand_computed_reference():
    """Two well-separated clusters are joined by exactly one MST edge."""
    a = np.array([[0.0], [0.1], [0.2]])
    b = np.array([[10.0], [10.1], [10.2]])
    result = mst_two_sample_test(a, b, n_permutations=49, metric="euclidean")
    assert result.statistic == 1.0


def test_nn_asymptotic_and_permutation_agree_on_a_clear_case(split):
    """Both p-values are reported so disagreement is visible; here they agree."""
    result = nn_two_sample_test(
        _bad_points(), split.reference_minority, n_permutations=199, random_state=3
    )
    assert result.asymptotic_p_value is not None
    assert result.p_value < 0.05
    assert result.asymptotic_p_value < 0.05


def test_tests_accept_any_registered_metric(split):
    """hassanat must compose with the inferential layer, as must the others."""
    for metric in ("hassanat", "euclidean", "manhattan"):
        result = nn_two_sample_test(
            _ideal_points(),
            split.reference_minority,
            metric=metric,
            n_permutations=49,
            random_state=3,
        )
        assert 0.0 < result.p_value <= 1.0


@pytest.mark.slow
def test_permutation_p_values_are_uniform_under_the_null():
    """Under H0 a valid permutation p-value is uniform on [0, 1].

    Both samples are drawn from the same distribution, so any departure from
    uniformity means the test is miscalibrated -- rejecting too often or too
    seldom regardless of the data.
    """
    rng = np.random.default_rng(0)
    p_values = []
    for _ in range(150):
        pooled = rng.normal(size=(40, 3))
        result = nn_two_sample_test(
            pooled[:20],
            pooled[20:],
            n_permutations=199,
            random_state=int(rng.integers(1e6)),
        )
        p_values.append(result.p_value)

    # A discrete permutation p-value is only approximately uniform, so this
    # checks for gross miscalibration rather than exact uniformity.
    ks = stats.kstest(p_values, "uniform")
    assert ks.pvalue > 0.001, f"p-values look miscalibrated (KS p={ks.pvalue:.4g})"
    assert 0.01 < np.mean(np.array(p_values) < 0.05) < 0.15
