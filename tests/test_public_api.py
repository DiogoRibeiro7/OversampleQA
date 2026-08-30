"""Tests for the public surface: estimator, report object, plugin contract.

The axiom check here is the institutionalised form of this codebase's most
serious defect. The built-in ``hassanat`` shipped for the project's entire
history scoring ``[-5]`` and ``[5]`` as distance zero -- it was not a metric,
and nothing checked. ``test_builtin_registry_satisfies_the_axioms`` is that
check, running in CI so the package cannot repeat it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.datasets import make_classification

from oversampleqa.distance import _METRICS
from oversampleqa.estimator import OversamplingValidator, validation_scorer
from oversampleqa.exceptions import PluginError, ValidationError
from oversampleqa.plugin_contract import (
    METRIC_DOMAINS,
    check_metric_axioms,
    validate_metric_signature,
)
from oversampleqa.plugin_system import PluginManager
from oversampleqa.reports import SCHEMA_VERSION, RunMetadata, ValidationReport


@pytest.fixture
def data():
    return make_classification(
        n_samples=700,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        random_state=0,
    )


# --- the axiom check, and the bug it exists to prevent --------------------


def _original_broken_hassanat(x1, x2, **_):
    """The implementation that shipped before the Hassanat correction.

    Reproduced verbatim so the check is tested against the real defect rather
    than a stand-in.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    maxima = np.maximum(np.abs(x1), np.abs(x2))
    minima = np.minimum(np.abs(x1), np.abs(x2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(np.where(maxima == 0, 0.0, 1.0 - minima / maxima)))


def test_the_axiom_check_rejects_the_original_hassanat():
    """The whole point of the check.

    It compared absolute values, so d([-5], [5]) == 0: two distinct points at
    distance zero. Everything else about it looked fine, which is why it
    survived.
    """
    report = check_metric_axioms(_original_broken_hassanat, "broken")
    assert not report.ok
    assert not report.identity_of_indiscernibles
    assert any("distance zero" in f for f in report.failures)


def test_the_axiom_check_accepts_the_corrected_hassanat():
    report = check_metric_axioms(_METRICS["hassanat"], "hassanat")
    assert report.ok


@pytest.mark.parametrize("name", sorted(_METRICS))
def test_builtin_registry_satisfies_the_axioms(name):
    """Run the plugin check against the package's own metrics, in CI.

    A check that only applies to third-party plugins would not have caught the
    defect it was written for.
    """
    kwargs = {"cov_inv": np.eye(4)} if name == "mahalanobis" else {}
    report = check_metric_axioms(
        _METRICS[name], name, domain=METRIC_DOMAINS[name], **kwargs
    )
    assert report.ok, f"{name} violates: {report.failures}"


def test_every_builtin_metric_has_a_declared_domain():
    """A new metric must declare its domain, or it is checked on the wrong input."""
    assert set(_METRICS) <= set(METRIC_DOMAINS)


def test_sample_metrics_are_exempt_from_point_axioms():
    """energy and wasserstein compare distributions, not points."""
    report = check_metric_axioms(_METRICS["energy"], "energy", domain="sample")
    assert report.ok


def test_signature_check_rejects_a_one_argument_callable():
    with pytest.raises(PluginError, match="two positional arguments"):
        validate_metric_signature(lambda x: 0.0, "one_arg")


def test_signature_check_rejects_a_non_callable():
    with pytest.raises(PluginError, match="not callable"):
        validate_metric_signature(42, "not_callable")


# --- plugin registration --------------------------------------------------


def _euclidean(x1, x2, **_):
    return float(np.linalg.norm(np.asarray(x1) - np.asarray(x2)))


def test_registration_accepts_a_well_behaved_metric():
    manager = PluginManager()
    manager.register_metric("custom", _euclidean)
    assert manager.get_metric("custom") is _euclidean


def test_registration_rejects_a_metric_failing_the_axioms():
    manager = PluginManager()
    with pytest.raises(PluginError, match="does not satisfy the distance axioms"):
        manager.register_metric("bad", _original_broken_hassanat)


def test_registration_refuses_to_shadow_a_builtin():
    """Silently overriding a built-in would change results for every caller."""
    manager = PluginManager()
    with pytest.raises(PluginError, match="already a built-in"):
        manager.register_metric("hassanat", _euclidean)


def test_registration_refuses_a_duplicate_plugin_name():
    manager = PluginManager()
    manager.register_metric("custom", _euclidean)
    with pytest.raises(PluginError, match="already registered"):
        manager.register_metric("custom", _euclidean)


def test_unregister_allows_deliberate_replacement():
    manager = PluginManager()
    manager.register_metric("custom", _euclidean)
    manager.unregister_metric("custom")
    manager.register_metric("custom", _euclidean)


def test_unregister_rejects_an_unknown_name():
    with pytest.raises(PluginError, match="no metric plugin"):
        PluginManager().unregister_metric("never_registered")


def test_axiom_check_can_be_skipped_deliberately():
    """Opt-out exists, but it must be explicit."""
    manager = PluginManager()
    manager.register_metric("odd", _original_broken_hassanat, check_axioms=False)
    assert manager.get_metric("odd") is _original_broken_hassanat


# --- the estimator --------------------------------------------------------


def test_estimator_follows_the_constructor_contract():
    """__init__ stores parameters unchanged: no validation, no computation."""
    sampler = SMOTE(random_state=0)
    validator = OversamplingValidator(sampler, hidden_ratio=0.2, metric="euclidean")
    assert validator.oversampler is sampler
    assert validator.hidden_ratio == 0.2
    assert validator.metric == "euclidean"
    # Nothing fitted yet.
    assert not hasattr(validator, "error_rate_")


def test_get_params_round_trips():
    validator = OversamplingValidator(SMOTE(random_state=0), hidden_ratio=0.2)
    params = validator.get_params(deep=False)
    rebuilt = OversamplingValidator(**params)
    assert rebuilt.get_params()["hidden_ratio"] == 0.2


def test_estimator_is_cloneable():
    """clone() relies on the constructor being a pure assignment."""
    validator = OversamplingValidator(SMOTE(random_state=0), metric="euclidean")
    copy = clone(validator)
    assert copy.metric == "euclidean"
    assert copy is not validator


def test_fit_sets_underscored_attributes(data):
    X, y = data
    validator = OversamplingValidator(SMOTE(random_state=0)).fit(X, y)
    assert hasattr(validator, "error_rate_")
    assert hasattr(validator, "report_")
    assert 0.0 <= validator.error_rate_ <= 1.0


def test_fit_returns_self(data):
    X, y = data
    validator = OversamplingValidator(SMOTE(random_state=0))
    assert validator.fit(X, y) is validator


def test_minority_label_is_inferred(data):
    X, y = data
    validator = OversamplingValidator(SMOTE(random_state=0)).fit(X, y)
    assert validator.minority_label_ == 1


def test_score_is_negated_so_greater_is_better(data):
    """A higher error rate is worse, but sklearn maximises the score.

    Returning the raw rate would make GridSearchCV select the worst sampler.
    """
    X, y = data
    validator = OversamplingValidator(SMOTE(random_state=0)).fit(X, y)
    assert validator.score() == pytest.approx(-validator.error_rate_)


def test_score_before_fit_raises():
    with pytest.raises(ValidationError, match="call fit"):
        OversamplingValidator(SMOTE(random_state=0)).score()


def test_fit_rejects_mismatched_lengths(data):
    X, y = data
    with pytest.raises(ValidationError, match="rows"):
        OversamplingValidator(SMOTE(random_state=0)).fit(X[:10], y)


def test_scorer_works_with_the_sklearn_signature(data):
    X, y = data
    validator = OversamplingValidator(SMOTE(random_state=0))
    assert validation_scorer(validator, X, y) <= 0.0


# --- the report object ----------------------------------------------------


def test_metadata_captures_provenance(data):
    X, y = data
    meta = RunMetadata.capture(X, y, SMOTE(random_state=0), minority_label=1)
    assert meta.oversampler == "SMOTE"
    assert meta.n_samples == 700
    assert meta.n_features == 6
    assert meta.dataset_hash
    assert meta.numpy_version
    assert meta.timestamp.endswith("+00:00")


def test_dataset_hash_changes_with_the_data(data):
    X, y = data
    first = RunMetadata.capture(X, y, SMOTE()).dataset_hash
    same = RunMetadata.capture(X.copy(), y.copy(), SMOTE()).dataset_hash
    altered = X.copy()
    altered[0, 0] += 1.0
    different = RunMetadata.capture(altered, y, SMOTE()).dataset_hash
    assert first == same
    assert first != different


def test_report_round_trips_through_dict(data):
    X, y = data
    report = OversamplingValidator(SMOTE(random_state=0)).fit(X, y).report_
    restored = ValidationReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()


def test_report_json_is_strictly_valid(data):
    """JSON has no NaN; a bare NaN token is invalid and strict parsers reject it."""
    X, y = data
    report = OversamplingValidator(SMOTE(random_state=0)).fit(X, y).report_
    text = report.to_json()
    assert json.loads(text)["schema_version"] == SCHEMA_VERSION


def test_non_finite_values_export_as_null():
    report = ValidationReport(error_rate=float("nan"), metadata=RunMetadata())
    payload = report.to_dict()
    assert payload["error_rate"] is None
    json.dumps(payload, allow_nan=False)  # must not raise


def test_report_rejects_an_incompatible_schema_version():
    """A major-version change means a field was removed or changed meaning."""
    payload = ValidationReport(error_rate=0.1, metadata=RunMetadata()).to_dict()
    payload["schema_version"] = "99.0"
    with pytest.raises(ValueError, match="not compatible"):
        ValidationReport.from_dict(payload)


def test_report_to_frame_is_one_tidy_row(data):
    X, y = data
    report = OversamplingValidator(SMOTE(random_state=0)).fit(X, y).report_
    frame = report.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert "error_rate" in frame.columns
    assert {
        "oversampler",
        "metric",
        "hidden_ratio",
        "reference",
        "random_state",
        "n_repeats",
        "minority_label",
        "oversampleqa_version",
    }.issubset(frame.columns)
    assert any(c.startswith("meta_") for c in frame.columns)


def test_report_renders_for_the_cli(data):
    X, y = data
    report = OversamplingValidator(SMOTE(random_state=0)).fit(X, y).report_
    rendered = report.__rich__()
    assert "error rate" in rendered
    assert "SMOTE" in rendered


def test_report_composes_optional_components(data):
    """calibration, inference and fidelity are optional because each costs time."""
    X, y = data
    report = OversamplingValidator(SMOTE(random_state=0)).fit(X, y).report_
    assert report.calibration is None
    enriched = report.with_components(calibration={"null_mean": 0.1})
    assert enriched.calibration == {"null_mean": 0.1}
    assert report.calibration is None  # frozen: the original is untouched
