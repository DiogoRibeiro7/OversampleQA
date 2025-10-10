import numpy as np
from oversampleqa.distance import (
    minkowski_distance,
    manhattan_distance,
    euclidean_distance,
    chebyshev_distance,
    jaccard_distance,
    hellinger_distance,
    jensen_shannon_distance,
    distance_matrix,
)


def test_minkowski_equivalence():
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    assert np.isclose(minkowski_distance(x1, x2, p=1), manhattan_distance(x1, x2))
    assert np.isclose(minkowski_distance(x1, x2, p=2), euclidean_distance(x1, x2))


def test_chebyshev_distance():
    x1 = np.array([1.0, 4.0, 2.0])
    x2 = np.array([2.0, 1.0, 3.0])
    expected = np.max(np.abs(x1 - x2))
    assert np.isclose(chebyshev_distance(x1, x2), expected)


def test_jaccard_distance():
    x = np.array([1, 1, 0, 0], dtype=bool)
    y = np.array([1, 0, 1, 0], dtype=bool)
    expected = 1.0 - 1 / 3
    assert np.isclose(jaccard_distance(x, y), expected)


def test_hellinger_distance():
    p = np.array([0.6, 0.4])
    q = np.array([0.5, 0.5])
    expected = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    assert np.isclose(hellinger_distance(p, q), expected)


def test_jensen_shannon_distance():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        term_p = np.where(p == 0, 0.0, p * np.log(p / m))
        term_q = np.where(q == 0, 0.0, q * np.log(q / m))
    expected = np.sqrt(0.5 * (np.sum(term_p) + np.sum(term_q)))
    assert np.isclose(jensen_shannon_distance(p, q), expected)


def test_distance_matrix_with_new_metric():
    X1 = np.array([[0, 1], [1, 0]])
    X2 = np.array([[1, 1], [0, 0]])
    dm = distance_matrix(X1, X2, metric="chebyshev")
    assert dm.shape == (2, 2)


def test_hellinger_distance_matrix():
    X1 = np.array([[0.6, 0.4]])
    X2 = np.array([[0.5, 0.5], [0.2, 0.8]])
    dm = distance_matrix(X1, X2, metric="hellinger")
    assert dm.shape == (1, 2)
    assert np.isclose(dm[0, 0], hellinger_distance(X1[0], X2[0]))


def test_jensen_shannon_distance_matrix():
    X1 = np.array([[0.5, 0.5], [1.0, 0.0]])
    X2 = np.array([[0.5, 0.5]])
    dm = distance_matrix(X1, X2, metric="jensen_shannon")
    assert dm.shape == (2, 1)
    assert np.isclose(dm[0, 0], jensen_shannon_distance(X1[0], X2[0]))


def test_mahalanobis_distance_matrix_with_cov():
    X1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    X2 = np.array([[1.0, 0.0], [1.0, 2.0]])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    cov_inv = np.linalg.inv(cov)

    dm = distance_matrix(X1, X2, metric="mahalanobis", cov_inv=cov_inv)

    diff = X1[0] - X2[0]
    expected = np.sqrt(np.dot(diff, cov_inv @ diff))
    assert np.isclose(dm[0, 0], expected)

    dm_default = distance_matrix(X1, X2, metric="mahalanobis")
    dm_euclidean = distance_matrix(X1, X2, metric="euclidean")
    assert np.allclose(dm_default, dm_euclidean)
