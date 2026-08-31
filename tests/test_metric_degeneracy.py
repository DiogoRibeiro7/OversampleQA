"""Deterministic tests for degenerate metric inputs.

The axiom smoke check draws random vectors, so it will essentially never draw
an exact zero vector or an exact constant vector -- which is precisely where
these metrics break. Four defects survived that way, all of the same shape as
the original hassanat bug: a distance of zero between points that are not
identical.

    cosine(0, x)             = 0.0        zero vector "identical" to everything
    cosine(const, const)     = -2.22e-16  a negative distance
    correlation(const, x)    = 0.0        constant vector "perfectly correlated"
    braycurtis([-1,0],[1,0]) = 0.0        negatives cancel the denominator

`METRIC_DOMAINS` had documented the correlation case as undefined all along.
The code disagreed with the comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa.distance import _METRICS, cosine_distance, distance_matrix
from oversampleqa.extended_distances import braycurtis_distance, correlation_distance
from oversampleqa.plugin_contract import METRIC_DOMAINS

ZERO = np.zeros(3)
CONST = np.full(3, 2.0)
X = np.array([1.0, 2.0, 3.0])


# --- cosine ---


def test_cosine_rejects_a_zero_vector():
    with pytest.raises(ValueError, match="no direction"):
        cosine_distance(ZERO, X)


def test_cosine_rejects_two_zero_vectors():
    with pytest.raises(ValueError, match="undefined"):
        cosine_distance(ZERO, ZERO.copy())


def test_cosine_of_a_constant_vector_with_itself_is_exactly_zero():
    """It was -2.22e-16: a negative distance feeding a `<` comparison."""
    assert cosine_distance(CONST, CONST.copy()) == 0.0


def test_cosine_is_never_negative():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.normal(size=4), rng.normal(size=4)
        assert cosine_distance(a, b) >= 0.0


def test_cosine_still_measures_direction():
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        1.0
    )
    assert cosine_distance(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(
        0.0
    )


# --- correlation ---


def test_correlation_rejects_a_constant_vector():
    with pytest.raises(ValueError, match="constant"):
        correlation_distance(CONST, X)


def test_correlation_rejects_a_zero_vector():
    """Zero is constant, so it has no variance either."""
    with pytest.raises(ValueError, match="constant"):
        correlation_distance(ZERO, X)


def test_correlation_rejects_too_short_a_vector():
    with pytest.raises(ValueError, match="length"):
        correlation_distance(np.array([1.0]), np.array([2.0]))


def test_correlation_matches_the_documented_domain_note():
    """METRIC_DOMAINS says this case is undefined. The code now agrees."""
    assert METRIC_DOMAINS["correlation"] == "non_negative"
    with pytest.raises(ValueError):
        correlation_distance(CONST, CONST.copy())


def test_correlation_still_measures_association():
    same = correlation_distance(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    opposed = correlation_distance(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    assert same == pytest.approx(0.0, abs=1e-12)
    assert opposed > 1.0


# --- properties that must hold for every metric ---


def _in_domain(name: str) -> np.ndarray | None:
    """A pair of distinct vectors inside the metric's declared domain."""
    domain = METRIC_DOMAINS.get(name, "real")
    if domain == "boolean":
        return np.array([1.0, 0.0, 1.0])
    if domain in ("non_negative", "sample"):
        return np.array([1.0, 2.0, 3.0])
    return np.array([1.0, -2.0, 3.0])


@pytest.mark.parametrize("name", sorted(n for n in _METRICS if n != "mahalanobis"))
def test_identical_points_are_at_distance_zero(name):
    """d(x, x) == 0 exactly, with no floating-point residue."""
    if METRIC_DOMAINS.get(name) == "sample":
        pytest.skip("sample metrics are rejected for pointwise use")
    x = _in_domain(name)
    assert _METRICS[name](x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_energy_violates_the_identity_axiom_pointwise():
    """Recorded, not hidden: this is why sample metrics are excluded.

    Applied to a single pair, energy scores a point against *itself* at
    -0.889 -- both non-zero and negative. It is a valid statistic between two
    samples and not a distance between two points, which is what
    require_pointwise_metric enforces.
    """
    x = np.array([1.0, -2.0, 3.0])
    assert _METRICS["energy"](x, x.copy()) < 0.0


@pytest.mark.parametrize("name", sorted(n for n in _METRICS if n != "mahalanobis"))
def test_no_metric_returns_a_negative_distance_on_its_domain(name):
    """energy returns negatives, which is why it is declared sample-level."""
    if METRIC_DOMAINS.get(name) == "sample":
        pytest.skip("sample metrics are rejected for pointwise use")
    x = _in_domain(name)
    y = x + 1.0
    assert _METRICS[name](x, y) >= 0.0


@pytest.mark.parametrize("name", sorted(n for n in _METRICS if n != "mahalanobis"))
def test_metrics_are_symmetric_on_their_domain(name):
    if METRIC_DOMAINS.get(name) == "sample":
        pytest.skip("sample metrics are rejected for pointwise use")
    x = _in_domain(name)
    y = x + 1.0
    assert _METRICS[name](x, y) == pytest.approx(_METRICS[name](y, x))


# --- bray-curtis and its declared domain ---


def test_braycurtis_rejects_negative_input():
    """It is declared `non_negative`; its siblings enforce that, it did not.

    The zero-denominator guard returns 0.0, which is right when the inputs are
    non-negative -- the absolute sums vanish only when both vectors are all
    zero. With a negative present the terms cancel instead, and two distinct
    points come back at distance zero.
    """
    with pytest.raises(ValueError, match="non-negative"):
        braycurtis_distance(np.array([-1.0, 0.0]), np.array([1.0, 0.0]))


def test_braycurtis_rejected_the_identity_violation_it_used_to_return():
    """d([-1, 0], [1, 0]) was 0.0: distinct points, distance zero.

    The same failure as the original hassanat implementation, which is what
    `check_metric_axioms` was written for.
    """
    x1, x2 = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    assert not np.array_equal(x1, x2)

    with pytest.raises(ValueError):
        braycurtis_distance(x1, x2)


def test_braycurtis_matrix_path_rejects_negatives_too():
    """Two implementations, so two guards.

    The vectorised kernel is a separate code path; a check in the scalar
    function is not a check in this one.
    """
    with pytest.raises(ValueError, match="non-negative"):
        distance_matrix(
            np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]), "braycurtis"
        )


def test_braycurtis_still_accepts_its_own_domain():
    """Enforcement must not reject the input the metric is for."""
    d = braycurtis_distance(np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0]))
    assert 0.0 <= d <= 1.0

    both_zero = braycurtis_distance(np.zeros(3), np.zeros(3))
    assert both_zero == 0.0
