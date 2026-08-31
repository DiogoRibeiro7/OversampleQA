import numpy as np
import pytest

from oversampleqa import check_metric_axioms
from oversampleqa.distance import (
    chebyshev_distance,
    distance_matrix,
    euclidean_distance,
    hellinger_distance,
    jaccard_distance,
    jensen_shannon_distance,
    manhattan_distance,
    minkowski_distance,
)
from oversampleqa.extended_distances import mahalanobis_distance


def test_minkowski_equivalence():
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    assert np.isclose(minkowski_distance(x1, x2, p=1), manhattan_distance(x1, x2))
    assert np.isclose(minkowski_distance(x1, x2, p=2), euclidean_distance(x1, x2))


def test_minkowski_at_infinite_p_is_the_chebyshev_distance():
    """`p=inf` returned 1.0 for every input, whatever the data.

    The limit as p grows is the Chebyshev distance, but the general formula
    cannot produce it: every |d| > 1 raised to inf is inf, so the sum is inf,
    and inf ** (1 / inf) is inf ** 0 -- which is 1.0.
    """
    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.0, 2.0, 3.0])

    assert minkowski_distance(x1, x2, p=np.inf) == chebyshev_distance(x1, x2) == 3.0


@pytest.mark.parametrize("seed", range(5))
def test_minkowski_approaches_chebyshev_as_p_grows(seed):
    """The fix must agree with the limit it is meant to be, not just at inf."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=5), rng.normal(size=5)

    assert np.isclose(
        minkowski_distance(x1, x2, p=200.0),
        chebyshev_distance(x1, x2),
        rtol=1e-6,
    )


def test_minkowski_does_not_overflow_at_large_p():
    """`diff ** p` overflowed to inf at p=1000, as it does in scipy.

    The answer there is simply the largest term, so scaling before
    exponentiating gives it rather than an infinity.
    """
    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.0, 2.0, 3.0])

    assert minkowski_distance(x1, x2, p=1000.0) == 3.0


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0])
def test_minkowski_is_unchanged_where_it_was_already_correct(p):
    """Scaling must not move results that never overflowed."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        x1, x2 = rng.normal(size=4) * 10, rng.normal(size=4) * 10
        direct = float(np.sum(np.abs(x1 - x2) ** p) ** (1 / p))
        assert np.isclose(minkowski_distance(x1, x2, p=p), direct, rtol=1e-12)


def test_minkowski_still_rejects_p_below_one():
    with pytest.raises(ValueError, match="p must be >= 1"):
        minkowski_distance(np.array([0.0]), np.array([1.0]), p=0.5)


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

    # Omitting cov_inv used to fall back to Euclidean, and this test asserted
    # it did. That is not a weaker Mahalanobis, it is a different metric under
    # the wrong name -- and it silently produced duplicate rows in the advanced
    # benchmark, where mahalanobis was a default.
    with pytest.raises(ValueError, match="requires cov_inv"):
        distance_matrix(X1, X2, metric="mahalanobis")


# --- mahalanobis and a covariance that is not positive semi-definite ---


def test_mahalanobis_rejects_a_non_psd_cov_inv():
    """A negative squared distance used to become nan.

    `np.sqrt` returned nan with nothing but "invalid value encountered in
    sqrt" -- a warning users routinely filter, naming a line inside this
    library rather than the matrix they passed.
    """
    not_psd = np.array([[1.0, 0.0], [0.0, -2.0]])

    with pytest.raises(ValueError, match="positive semi-definite"):
        mahalanobis_distance(
            np.array([0.0, 0.0]), np.array([1.0, 1.0]), cov_inv=not_psd
        )


def test_mahalanobis_tolerates_rounding_noise_from_a_near_singular_inverse():
    """A tiny negative is floating point, not a broken matrix.

    Raising on it would reject the ill-conditioned covariances that real,
    correlated data produces.
    """
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(200, 4))
    sample[:, 1] = sample[:, 0] + 1e-9 * rng.normal(size=200)
    cov_inv = np.linalg.pinv(np.cov(sample, rowvar=False))

    distances = [
        mahalanobis_distance(sample[i], sample[j], cov_inv=cov_inv)
        for i in range(40)
        for j in range(i + 1, 40)
    ]

    assert np.all(np.isfinite(distances))
    assert mahalanobis_distance(sample[0], sample[0], cov_inv=cov_inv) == 0.0


def test_mahalanobis_satisfies_the_axioms_with_a_real_covariance():
    """Not with an identity inverse, under which it is merely Euclidean."""
    sample = np.random.default_rng(0).normal(size=(200, 4))
    cov_inv = np.linalg.pinv(np.cov(sample, rowvar=False))

    report = check_metric_axioms(
        mahalanobis_distance, "mahalanobis", domain="real", cov_inv=cov_inv
    )

    assert report.ok, report.failures
