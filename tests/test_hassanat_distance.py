"""Property and regression tests for the Hassanat (2014) distance.

The implementation this replaces was ``1 - min(|a|,|b|)/max(|a|,|b|)``, which is
not the Hassanat distance: it is discontinuous at the origin and violates
identity of indiscernibles. The tests here pin the properties that the real
metric has and the broken one did not.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa.distance import distance_matrix, hassanat_distance


def reference_hassanat(a: np.ndarray, b: np.ndarray) -> float:
    """Plain-loop Hassanat reference with an explicit sign branch."""
    total = 0.0
    for ai, bi in zip(a, b):
        mn = min(ai, bi)
        mx = max(ai, bi)
        if mn >= 0:
            total += 1.0 - (1.0 + mn) / (1.0 + mx)
        else:
            total += 1.0 - (1.0 + mn + abs(mn)) / (1.0 + mx + abs(mn))
    return total


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([-5.0], [5.0], 0.909091),
        ([0.0], [1e-9], 0.0),
        ([0.1], [0.2], 0.083333),
        ([1.0, 2.0, 3.0], [4.0, 0.5, -1.0], 1.900000),
    ],
)
def test_reference_values(a, b, expected):
    got = hassanat_distance(np.array(a), np.array(b))
    assert np.isclose(got, expected, atol=1e-6)


def test_identity_of_indiscernibles():
    """d(a, b) == 0 if and only if a == b."""
    rng = np.random.default_rng(20260815)
    for _ in range(200):
        a = rng.normal(0, 10, size=4)
        assert hassanat_distance(a, a) == pytest.approx(0.0)
        b = a + rng.normal(0, 1, size=4)
        if not np.allclose(a, b):
            assert hassanat_distance(a, b) > 0.0


def test_sign_symmetric_points_are_not_identical():
    """Regression guard: the old implementation scored [-5] vs [5] as zero."""
    d = hassanat_distance(np.array([-5.0]), np.array([5.0]))
    assert d > 0.0
    assert np.isclose(d, 0.909091, atol=1e-6)


def test_continuity_near_zero():
    """The old implementation jumped to 1.0 for any epsilon > 0."""
    previous = 1.0
    for exponent in range(1, 12):
        eps = 10.0**-exponent
        d = hassanat_distance(np.array([0.0]), np.array([eps]))
        assert d < previous
        previous = d
    assert previous < 1e-10


def test_per_dimension_bound_over_wide_dynamic_range():
    """Every one-dimensional distance lies in [0, 1), at any feature scale."""
    rng = np.random.default_rng(7)
    for scale in (50.0, 1e6):
        pairs = rng.normal(0.0, scale, size=(5000, 2))
        for x, y in pairs:
            d = hassanat_distance(np.array([x]), np.array([y]))
            assert 0.0 <= d < 1.0


def test_symmetry():
    rng = np.random.default_rng(11)
    for _ in range(500):
        a = rng.normal(0, 100, size=3)
        b = rng.normal(0, 100, size=3)
        assert hassanat_distance(a, b) == pytest.approx(hassanat_distance(b, a))


def test_triangle_inequality():
    """Hassanat is a proven metric; check on 10 000 random triples."""
    rng = np.random.default_rng(1234)
    triples = rng.normal(0.0, 25.0, size=(10_000, 3, 2))
    for a, b, c in triples:
        ab = hassanat_distance(a, b)
        bc = hassanat_distance(b, c)
        ac = hassanat_distance(a, c)
        assert ac <= ab + bc + 1e-12


def test_matches_independent_reference():
    rng = np.random.default_rng(99)
    for _ in range(300):
        a = rng.normal(0, 30, size=6)
        b = rng.normal(0, 30, size=6)
        assert hassanat_distance(a, b) == pytest.approx(reference_hassanat(a, b))


def test_vectorized_matches_scalar_dense():
    rng = np.random.default_rng(2024)
    X1 = rng.normal(0, 40, size=(17, 5))
    X2 = rng.normal(0, 40, size=(13, 5))
    matrix = distance_matrix(X1, X2, "hassanat")
    expected = np.array([[hassanat_distance(a, b) for b in X2] for a in X1])
    assert matrix.shape == (17, 13)
    assert np.allclose(matrix, expected)


def test_vectorized_matches_scalar_single_row():
    rng = np.random.default_rng(2025)
    X1 = rng.normal(0, 40, size=(1, 4))
    X2 = rng.normal(0, 40, size=(6, 4))
    matrix = distance_matrix(X1, X2, "hassanat")
    expected = np.array([[hassanat_distance(a, b) for b in X2] for a in X1])
    assert matrix.shape == (1, 6)
    assert np.allclose(matrix, expected)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        hassanat_distance(np.array([1.0, 2.0]), np.array([1.0]))
