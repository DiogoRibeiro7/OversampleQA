import numpy as np
from oversampleqa.distance import hassanat_distance, distance_matrix, cosine_distance


def test_hassanat_distance_zero():
    x = np.array([1, 2, 3])
    assert hassanat_distance(x, x) == 0.0


def test_distance_matrix_shape():
    X1 = np.array([[0, 1], [1, 0]])
    X2 = np.array([[1, 1], [0, 0], [1, 0]])
    dm = distance_matrix(X1, X2)
    assert dm.shape == (2, 3)


def test_cosine_distance_matrix_values():
    X1 = np.array([[1, 0], [1, 1]])
    X2 = np.array([[1, 0], [0, 1]])
    dm = distance_matrix(X1, X2, metric="cosine")
    assert np.isclose(dm[0, 0], 0.0)
    assert np.isclose(dm[0, 1], 1.0)
    assert np.isclose(cosine_distance([1, 0], [0, 1]), 1.0)
