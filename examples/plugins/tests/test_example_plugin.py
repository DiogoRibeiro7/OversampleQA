"""Tests for the reference plugin.

These live with the plugin rather than in the host suite, because they are the
tests a plugin author should copy: they check the plugin against the contract
the host enforces, not against oversampleqa's internals.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa_example_plugin.metric import LorentzianDistance
from oversampleqa_example_plugin.validator import MedianRatioValidator


@pytest.fixture
def metric():
    return LorentzianDistance()


def test_identity_of_indiscernibles(metric):
    """The check the built-in hassanat metric would once have failed."""
    x = np.array([1.0, -2.0, 3.5])
    assert metric(x, x) == pytest.approx(0.0)


def test_distinct_points_are_at_positive_distance(metric):
    assert metric(np.array([-5.0]), np.array([5.0])) > 0


def test_symmetry(metric):
    a, b = np.array([1.0, 4.0]), np.array([2.0, -1.0])
    assert metric(a, b) == pytest.approx(metric(b, a))


def test_non_negativity(metric):
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.normal(size=4), rng.normal(size=4)
        assert metric(a, b) >= 0


def test_finite_on_random_input(metric):
    rng = np.random.default_rng(1)
    for _ in range(50):
        assert np.isfinite(metric(rng.normal(size=5), rng.normal(size=5)))


def test_triangle_inequality(metric):
    """The axiom smoke check does not test this one, so the plugin must.

    Passing registration is necessary, not sufficient. A function can satisfy
    identity, symmetry and non-negativity and still not be a metric.
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        a, b, c = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        assert metric(a, c) <= metric(a, b) + metric(b, c) + 1e-12


def test_large_differences_are_damped(metric):
    """The reason to choose this over Manhattan: outliers do not dominate."""
    near = metric(np.array([0.0]), np.array([1.0]))
    far = metric(np.array([0.0]), np.array([1000.0]))
    assert far > near
    # Manhattan would make this ratio 1000.
    assert far / near < 20


def test_passes_the_host_axiom_check():
    """The gate registration actually applies."""
    from oversampleqa.plugin_contract import check_metric_axioms

    report = check_metric_axioms(LorentzianDistance(), "lorentzian", domain="real")
    assert report.ok, report.failures


def test_registers_through_the_host():
    from oversampleqa.plugin_system import PluginManager

    manager = PluginManager()
    manager.register_metric("lorentzian", LorentzianDistance)
    assert manager.get_metric("lorentzian") is LorentzianDistance


class _Duplicating:
    """Oversampler that copies existing minority points."""

    def fit_resample(self, X, y):
        minority = X[y == 1]
        extra = minority[: len(minority)]
        return np.vstack([X, extra]), np.concatenate([y, np.ones(len(extra), int)])


class _Empty:
    def fit_resample(self, X, y):
        return X, y


def _dataset(seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (40, 2)), rng.normal(3, 1, (10, 2))])
    y = np.array([0] * 40 + [1] * 10)
    return X, y


def test_validator_scores_a_duplicating_sampler_near_zero():
    X, y = _dataset()
    score = MedianRatioValidator().validate(X, y, 1, _Duplicating())
    assert score == pytest.approx(0.0)


def test_validator_returns_nan_when_nothing_was_generated():
    """nan, not 0.0 -- a 0.0 would look like a perfect copy of the input."""
    X, y = _dataset()
    assert np.isnan(MedianRatioValidator().validate(X, y, 1, _Empty()))


def test_validator_returns_nan_for_a_minority_of_one():
    X, y = _dataset()
    y[y == 1] = 0
    y[0] = 1
    assert np.isnan(MedianRatioValidator().validate(X, y, 1, _Empty()))


def test_validator_registers_through_the_host():
    from oversampleqa.plugin_system import PluginManager

    manager = PluginManager()
    manager.register_validator("median_ratio", MedianRatioValidator)
    assert manager.get_validator("median_ratio") is MedianRatioValidator
