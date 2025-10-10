import numpy as np
import pytest

from oversampleqa.distance import (
    hellinger_distance,
    jensen_shannon_distance,
    distance_matrix,
)


def test_hellinger_negative_values_raise() -> None:
    """Ensure negative inputs are rejected."""
    p = np.array([-0.1, 1.1])
    q = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        hellinger_distance(p, q)


def test_jensen_shannon_negative_values_raise() -> None:
    """Ensure negative inputs are rejected."""
    p = np.array([0.5, -0.5])
    q = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        jensen_shannon_distance(p, q)


def test_jensen_shannon_zero_distance_identical() -> None:
    """Identical distributions should yield zero distance."""
    p = np.array([0.3, 0.7])
    assert jensen_shannon_distance(p, p) == pytest.approx(0.0)


def test_jensen_shannon_max_distance() -> None:
    """Opposite distributions reach the theoretical maximum."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    expected = np.sqrt(np.log(2.0))
    assert jensen_shannon_distance(p, q) == pytest.approx(expected)


def test_probability_distance_matrix_normalizes() -> None:
    """distance_matrix should normalize probability vectors before computing distances."""
    X1 = np.array([[2.0, 1.0]])
    X2 = np.array([[1.0, 1.0], [3.0, 0.0]])
    dm = distance_matrix(X1, X2, metric="jensen_shannon")

    p = X1[0] / X1[0].sum()
    q0 = X2[0] / X2[0].sum()
    q1 = X2[1] / X2[1].sum()
    expected = np.array([
        jensen_shannon_distance(p, q0),
        jensen_shannon_distance(p, q1),
    ])
    assert dm.shape == (1, 2)
    assert np.allclose(dm[0], expected)
