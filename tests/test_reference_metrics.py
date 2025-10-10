import numpy as np

from oversampleqa.distance import (
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


def test_distance_formula_consistency():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 0.5, -1.0])

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

