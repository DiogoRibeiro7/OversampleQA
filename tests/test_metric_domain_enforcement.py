"""Nearest-neighbour validation needs a metric between two points.

`energy` and `wasserstein` compare two *samples*. Applied to a single pair they
treat each point's own coordinates as the sample, so feature identity vanishes:

    [0, 5] vs [5, 0]   euclidean 7.0711   hassanat 1.6667
                       wasserstein 0.0    energy -5.0

Neither raised. Both produced plausible error rates -- 0.53 and 0.78 against
hassanat's 0.50 on the same run -- and `energy` fed a *negative* distance into a
`nearest_hidden < nearest_minority` comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa import validate_oversampling
from oversampleqa.distance import distance_matrix
from oversampleqa.exceptions import MetricError
from oversampleqa.inference import null_error_rate
from oversampleqa.plugin_contract import METRIC_DOMAINS, require_pointwise_metric
from oversampleqa.validator import validate_multiclass_oversampling

SAMPLE_METRICS = [n for n, d in METRIC_DOMAINS.items() if d == "sample"]
POINT_METRICS = ["hassanat", "euclidean", "manhattan", "chebyshev"]


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=600,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        random_state=0,
    )
    return X, y


def test_the_registry_declares_some_sample_metrics():
    """Guards the premise: no sample metrics would make this file vacuous."""
    assert set(SAMPLE_METRICS) >= {"energy", "wasserstein"}


@pytest.mark.parametrize("metric", SAMPLE_METRICS)
def test_require_pointwise_metric_rejects_sample_metrics(metric):
    with pytest.raises(MetricError, match="sample-level"):
        require_pointwise_metric(metric)


@pytest.mark.parametrize("metric", POINT_METRICS)
def test_require_pointwise_metric_allows_point_metrics(metric):
    require_pointwise_metric(metric)


def test_the_error_says_what_to_use_instead():
    with pytest.raises(MetricError) as excinfo:
        require_pointwise_metric("wasserstein")
    message = str(excinfo.value)
    assert "hassanat" in message
    assert "inference" in message, "two-sample tests are the right tool; say so"


@pytest.mark.parametrize("metric", SAMPLE_METRICS)
def test_validate_oversampling_rejects_sample_metrics(binary_data, metric):
    X, y = binary_data
    with pytest.raises(MetricError, match="sample-level"):
        validate_oversampling(X, y, 1, SMOTE(random_state=0), metric=metric)


@pytest.mark.parametrize("metric", SAMPLE_METRICS)
def test_null_error_rate_rejects_sample_metrics(binary_data, metric):
    X, y = binary_data
    with pytest.raises(MetricError, match="sample-level"):
        null_error_rate(X, y, 1, 0.3, metric=metric, n_draws=3)


def test_multiclass_rejects_sample_metrics():
    X, y = make_classification(
        n_samples=900,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        random_state=0,
    )
    with pytest.raises(MetricError, match="sample-level"):
        validate_multiclass_oversampling(X, y, SMOTE(random_state=0), metric="energy")


def test_point_metrics_still_validate(binary_data):
    """The guard must not block the metrics validation is built on."""
    X, y = binary_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate = validate_oversampling(
            X, y, 1, SMOTE(random_state=0), metric="hassanat", random_state=0
        )
    assert 0.0 <= rate <= 1.0


# --- the behaviour that motivated the guard ---


def test_sample_metrics_discard_feature_identity():
    """Two clearly different points, scored as identical."""
    p = np.array([[0.0, 5.0]])
    q = np.array([[5.0, 0.0]])
    assert float(distance_matrix(p, q, "euclidean")[0, 0]) > 7.0
    assert float(distance_matrix(p, q, "wasserstein")[0, 0]) == pytest.approx(0.0)


def test_energy_can_return_a_negative_distance():
    """Which then feeds a `nearest_hidden < nearest_minority` comparison."""
    p = np.array([[0.0, 5.0]])
    q = np.array([[5.0, 0.0]])
    assert float(distance_matrix(p, q, "energy")[0, 0]) < 0.0


def test_distance_matrix_still_computes_them():
    """The guard belongs in validation, not in the distance layer.

    These metrics are legitimate between two *samples*, which is how
    `oversampleqa.inference` uses them. Blocking them outright would remove a
    correct use to prevent an incorrect one.
    """
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [5.0, 5.0]])
    assert distance_matrix(a, b, "wasserstein").shape == (2, 2)


# --- mahalanobis without a covariance inverse ---


def test_mahalanobis_without_cov_inv_raises():
    """It silently returned Euclidean, which is a different metric.

    Mahalanobis with an identity covariance *is* Euclidean, so the fallback did
    not degrade the metric -- it reported one under another's name.
    """
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    with pytest.raises(ValueError, match="requires cov_inv"):
        distance_matrix(a, b, "mahalanobis")


def test_mahalanobis_with_cov_inv_is_not_euclidean():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(60, 3)) @ np.array(
        [[2.0, 0.4, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.5]]
    )
    cov_inv = np.linalg.pinv(np.cov(data, rowvar=False))
    a, b = data[:5], data[5:10]
    mahal = distance_matrix(a, b, "mahalanobis", cov_inv=cov_inv)
    euclid = distance_matrix(a, b, "euclidean")
    assert not np.allclose(mahal, euclid)


def test_the_error_says_how_to_estimate_the_inverse():
    a, b = np.array([[0.0]]), np.array([[1.0]])
    with pytest.raises(ValueError) as excinfo:
        distance_matrix(a, b, "mahalanobis")
    assert "np.cov" in str(excinfo.value)


def test_mahalanobis_is_not_a_default_benchmark_metric():
    """Every mahalanobis row duplicated the euclidean row exactly.

    That double-weighted euclidean in the rankings, since ranking happens per
    metric, and split one pairwise comparison into two for the correction.
    """
    import inspect

    from oversampleqa.advanced_benchmark import StatisticalBenchmark

    source = inspect.getsource(StatisticalBenchmark.run_comprehensive_benchmark)
    assert 'metrics or ("hassanat", "euclidean")' in source


def test_no_config_template_requests_mahalanobis():
    """A template asking for it would fail at run time now that it raises."""
    from oversampleqa.config_templates import CONFIG_TEMPLATES

    for name, template in CONFIG_TEMPLATES.items():
        metrics = template.get("params", {}).get("metrics", [])
        assert "mahalanobis" not in metrics, name
