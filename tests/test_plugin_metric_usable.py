"""A registered metric plugin must be usable, not merely retrievable.

Registering one used to accomplish nothing beyond storing it: `distance_matrix`
and every validator funnelling through it consulted only the built-in table, so
a plugin metric was rejected as unsupported by the exact functions it exists to
be used by. `PluginManager.get_metric` returned it happily the whole time.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa import validate_oversampling
from oversampleqa.distance import _METRICS, distance_matrix, resolve_metric
from oversampleqa.plugin_system import PluginManager, plugin_manager


class Lorentzian:
    """log1p-damped L1. A genuine metric, not in the built-in table."""

    def __call__(self, x1, x2, **kwargs):
        a = np.asarray(x1, dtype=float)
        b = np.asarray(x2, dtype=float)
        return float(np.log1p(np.abs(a - b)).sum())


@pytest.fixture
def registered():
    """Register into the global manager, then remove it again."""
    name = "test_lorentzian"
    plugin_manager.register_metric(name, Lorentzian)
    try:
        yield name
    finally:
        plugin_manager.unregister_metric(name)


def test_distance_matrix_accepts_a_plugin_metric(registered):
    d = distance_matrix(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), registered)
    assert float(d[0, 0]) == pytest.approx(2 * np.log1p(1.0))


def test_validate_oversampling_accepts_a_plugin_metric(registered):
    from imblearn.over_sampling import SMOTE
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        random_state=0,
    )
    rate = validate_oversampling(
        X, y, 1, SMOTE(random_state=0), metric=registered, random_state=0
    )
    assert 0.0 <= rate <= 1.0


def test_typed_config_accepts_a_plugin_metric(registered):
    """The config rejected it before any validation could run."""
    from oversampleqa.typed_validator import ValidationConfig

    assert ValidationConfig(metric=registered).metric == registered


def test_plugin_metric_is_removed_again_after_unregistering():
    name = "temporary_metric"
    plugin_manager.register_metric(name, Lorentzian)
    plugin_manager.unregister_metric(name)
    with pytest.raises(ValueError, match="Unsupported metric"):
        distance_matrix(np.array([[0.0]]), np.array([[1.0]]), name)


# --- resolve_metric ---


def test_builtin_resolves_to_none():
    """None means 'the default registry already covers it'."""
    assert resolve_metric("euclidean") is None


def test_plugin_resolves_to_a_callable(registered):
    resolved = resolve_metric(registered)
    assert callable(resolved)
    assert resolved(np.array([0.0]), np.array([1.0])) == pytest.approx(np.log1p(1.0))


def test_a_class_is_instantiated(registered):
    """Registration stores the class; callers need an instance."""
    assert not isinstance(resolve_metric(registered), type)


def test_unknown_metric_names_the_builtins():
    with pytest.raises(ValueError) as excinfo:
        resolve_metric("definitely_not_a_metric")
    message = str(excinfo.value)
    assert "hassanat" in message
    assert "discover_entry_points" in message, "the message should say what to do"


def test_registering_does_not_touch_the_builtin_table(registered):
    """Plugins must not mutate the shared built-in registry."""
    assert registered not in _METRICS


def test_a_separate_manager_does_not_leak_into_resolution():
    """resolve_metric reads the global registry, not any manager instance."""
    isolated = PluginManager()
    isolated.register_metric("isolated_metric", Lorentzian)
    with pytest.raises(ValueError, match="Unsupported metric"):
        resolve_metric("isolated_metric")
