import numpy as np
import pytest

from oversampleqa.distance import (
    hassanat_distance,
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)


@pytest.mark.parametrize(
    "metric", [hassanat_distance, euclidean_distance, manhattan_distance, cosine_distance]
)
def test_metric_basic_properties(metric):
    x = np.array([1.0, -2.0, 3.0])
    y = np.array([-1.0, 2.0, -3.0])
    z = np.array([0.5, 0.5, 0.5])

    d_xy = metric(x, y)
    d_yx = metric(y, x)
    d_xx = metric(x, x)

    assert d_xy >= 0
    assert d_xx == 0
    assert pytest.approx(d_xy) == d_yx

    d_xz = metric(x, z)
    d_zy = metric(z, y)
    assert d_xz <= d_xy + d_zy + 1e-8


def test_metrics_with_various_dtypes():
    xi = np.array([1, 2, 3], dtype=int)
    yi = np.array([4, 5, 6], dtype=int)
    xb = np.array([1, 0, 1], dtype=bool)
    yb = np.array([0, 1, 1], dtype=bool)

    for metric in (
        hassanat_distance,
        euclidean_distance,
        manhattan_distance,
        cosine_distance,
    ):
        assert metric(xi, yi) >= 0
        assert metric(xb, yb) >= 0
