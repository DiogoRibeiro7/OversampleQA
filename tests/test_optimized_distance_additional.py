import numpy as np
import pytest

from oversampleqa.distance import distance_matrix


def test_distance_matrix_streaming():
    X1 = np.array([[0.0, 1.0], [2.0, 3.0]])
    X2 = np.array([[1.0, 0.0]])
    dm = distance_matrix(X1, X2, metric="euclidean", batch_size="stream")
    assert dm.shape == (2, 1)
    assert np.all(dm >= 0)


def test_distance_matrix_invalid_batch_size():
    X1 = np.array([[0.0, 1.0]])
    X2 = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        distance_matrix(X1, X2, metric="euclidean", batch_size=0)
