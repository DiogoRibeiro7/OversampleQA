import numpy as np
import pytest
from scipy import stats as scipy_stats
from scipy.spatial import distance as scipy_distance

from oversampleqa.distance import (
    hassanat_distance,
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    chebyshev_distance,
    minkowski_distance,
    canberra_distance,
    braycurtis_distance,
    correlation_distance,
    energy_distance as pkg_energy_distance,
    jaccard_distance,
    hamming_distance,
    wasserstein_1d_distance,
    mahalanobis_distance,
    hellinger_distance,
    jensen_shannon_distance,
)


def reference_hassanat(a: np.ndarray, b: np.ndarray) -> float:
    """Independent Hassanat (2014) reference.

    Deliberately a plain Python loop with an explicit branch, so this is a real
    cross-check rather than a restatement of the vectorised implementation.
    """
    total = 0.0
    for ai, bi in zip(a, b):
        mn = min(ai, bi)
        mx = max(ai, bi)
        if mn >= 0:
            total += 1.0 - (1.0 + mn) / (1.0 + mx)
        else:
            total += 1.0 - (1.0 + mn + abs(mn)) / (1.0 + mx + abs(mn))
    return total


def test_distance_formula_consistency():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 0.5, -1.0])

    assert np.isclose(hassanat_distance(x, y), reference_hassanat(x, y))

    cov = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.0], [0.1, 0.0, 0.5]])
    cov_inv = np.linalg.inv(cov)

    assert np.isclose(euclidean_distance(x, y), np.linalg.norm(x - y))
    assert np.isclose(manhattan_distance(x, y), np.sum(np.abs(x - y)))
    cos_exp = 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    assert np.isclose(cosine_distance(x, y), cos_exp)
    assert np.isclose(chebyshev_distance(x, y), np.max(np.abs(x - y)))
    mink_exp = np.sum(np.abs(x - y) ** 3) ** (1.0 / 3)
    assert np.isclose(minkowski_distance(x, y, p=3), mink_exp)
    can_exp = np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))
    assert np.isclose(canberra_distance(x, y), can_exp)
    bc_num = np.sum(np.abs(x - y))
    bc_den = np.sum(np.abs(x + y))
    assert np.isclose(braycurtis_distance(x, y), bc_num / bc_den)
    corr_exp = 1.0 - np.corrcoef(x, y)[0, 1]
    assert np.isclose(correlation_distance(x, y), corr_exp)
    wass_exp = np.mean(np.abs(np.sort(x) - np.sort(y)))
    assert np.isclose(wasserstein_1d_distance(x, y), wass_exp)
    # Energy distance reference using same formula as implementation
    x_set = x.reshape(len(x), -1)
    y_set = y.reshape(len(y), -1)
    diff_cross = np.linalg.norm(x_set[:, None, :] - y_set[None, :, :], axis=-1)
    term_a = diff_cross.mean()
    diff_x = np.linalg.norm(x_set[:, None, :] - x_set[None, :, :], axis=-1)
    term_b = diff_x[np.triu_indices(len(x_set), 1)].mean()
    diff_y = np.linalg.norm(y_set[:, None, :] - y_set[None, :, :], axis=-1)
    term_c = diff_y[np.triu_indices(len(y_set), 1)].mean()
    energy_exp = 2.0 * term_a - term_b - term_c
    assert np.isclose(pkg_energy_distance(x, y), energy_exp)
    maha_exp = np.sqrt(np.dot(x - y, cov_inv @ (x - y)))
    assert np.isclose(mahalanobis_distance(x, y, cov_inv=cov_inv), maha_exp)

    p = np.abs(x)
    q = np.abs(y)
    p /= p.sum()
    q /= q.sum()
    hell_exp = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    assert np.isclose(hellinger_distance(p, q), hell_exp)

    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        term_p = np.where(p == 0, 0.0, p * np.log(p / m))
        term_q = np.where(q == 0, 0.0, q * np.log(q / m))
    js_exp = np.sqrt(0.5 * (np.sum(term_p) + np.sum(term_q)))
    assert np.isclose(jensen_shannon_distance(p, q), js_exp)

    xb = np.array([1, 0, 1, 1], dtype=bool)
    yb = np.array([1, 1, 0, 1], dtype=bool)
    inter = np.sum(xb & yb)
    union = np.sum(xb | yb)
    jac_exp = 1.0 - inter / union
    assert np.isclose(jaccard_distance(xb, yb), jac_exp)
    ham_exp = np.sum(xb != yb)
    assert np.isclose(hamming_distance(xb, yb), ham_exp)


def test_metrics_agree_with_scipy():
    """Each metric must be the thing it is named after, not just self-consistent.

    SciPy is the independent authority wherever it implements the metric.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 0.5, -1.0])

    cov = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.0], [0.1, 0.0, 0.5]])
    cov_inv = np.linalg.inv(cov)

    assert np.isclose(euclidean_distance(x, y), scipy_distance.euclidean(x, y))
    assert np.isclose(manhattan_distance(x, y), scipy_distance.cityblock(x, y))
    assert np.isclose(cosine_distance(x, y), scipy_distance.cosine(x, y))
    assert np.isclose(chebyshev_distance(x, y), scipy_distance.chebyshev(x, y))
    assert np.isclose(canberra_distance(x, y), scipy_distance.canberra(x, y))
    assert np.isclose(
        minkowski_distance(x, y, p=3), scipy_distance.minkowski(x, y, p=3)
    )
    assert np.isclose(
        correlation_distance(x, y), scipy_distance.correlation(x, y)
    )
    assert np.isclose(
        mahalanobis_distance(x, y, cov_inv=cov_inv),
        scipy_distance.mahalanobis(x, y, cov_inv),
    )

    # Bray-Curtis: SciPy divides by sum|x_i + y_i|, matching this package.
    assert np.isclose(
        braycurtis_distance(x, y), scipy_distance.braycurtis(x, y)
    )

    # Hamming: SciPy returns the *fraction* of differing components; this
    # package returns the raw count. Scale to compare.
    xb = np.array([1, 0, 1, 1], dtype=bool)
    yb = np.array([1, 1, 0, 1], dtype=bool)
    assert np.isclose(
        hamming_distance(xb, yb), scipy_distance.hamming(xb, yb) * xb.size
    )
    assert np.isclose(jaccard_distance(xb, yb), scipy_distance.jaccard(xb, yb))

    # Jensen-Shannon: SciPy's jensenshannon is the *distance* (sqrt of the
    # divergence) with natural log when base is unset, matching this package.
    p = np.abs(x) / np.abs(x).sum()
    q = np.abs(y) / np.abs(y).sum()
    assert np.isclose(
        jensen_shannon_distance(p, q), scipy_distance.jensenshannon(p, q)
    )


def test_energy_and_wasserstein_are_sample_based():
    """Pin the sample-based semantics of the two distribution metrics.

    Both treat their input as a *set of observations*, not as a single point in
    feature space. Permuting a vector leaves the sample unchanged, so both
    metrics are permutation-invariant -- which is exactly what distinguishes
    them from the point metrics in the same registry.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=8)
    y = rng.normal(size=8)
    x_shuffled = rng.permutation(x)

    assert np.isclose(pkg_energy_distance(x, y), pkg_energy_distance(x_shuffled, y))
    assert np.isclose(
        wasserstein_1d_distance(x, y), wasserstein_1d_distance(x_shuffled, y)
    )

    # A point metric is *not* permutation invariant -- the contrast is the point.
    assert not np.isclose(euclidean_distance(x, y), euclidean_distance(x_shuffled, y))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "wasserstein_1d_distance does not integrate |F1 - F2| correctly: its "
        "loop stops when either sample is exhausted, dropping the tail, and it "
        "advances the CDF before adding the interval contribution. Out of scope "
        "for the Hassanat correction; fixing it changes numeric output and "
        "needs its own changelog incomparability note."
    ),
)
def test_wasserstein_agrees_with_scipy():
    """Known-failing: pins the discrepancy so it cannot be forgotten.

    ``[0, 1]`` vs ``[0, 3]`` has true W1 = 1.0 (SciPy and the equal-size closed
    form ``mean|sort(x) - sort(y)|`` both agree); this implementation returns
    0.5. When the implementation is fixed, this test flips to passing and the
    strict xfail turns that into a failure, prompting removal of the marker.
    """
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 3.0])
    assert np.isclose(
        wasserstein_1d_distance(a, b), scipy_stats.wasserstein_distance(a, b)
    )


def test_probability_metrics_reject_negative_inputs():
    """Hellinger and Jensen-Shannon require normalisable inputs.

    They must raise rather than silently taking absolute values, which would
    turn an invalid call into a plausible-looking number.
    """
    x = np.array([1.0, -2.0, 3.0])
    y = np.array([0.5, 2.0, -1.0])

    with pytest.raises(ValueError, match="non-negative"):
        hellinger_distance(x, y)
    with pytest.raises(ValueError, match="non-negative"):
        jensen_shannon_distance(x, y)

