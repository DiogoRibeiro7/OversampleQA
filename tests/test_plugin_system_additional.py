import types

import pytest

from oversampleqa.plugin_system import PluginManager


class DummyMetric:
    def __call__(self, x1, x2, **_):
        return float((x1 - x2).sum())


class DummyValidator:
    def validate(self, X, y, minority_label, oversampler, **_):
        return 0.0


def test_register_and_get_metric_validator():
    manager = PluginManager()
    manager.register_metric("dummy_metric", DummyMetric)
    manager.register_validator("dummy_validator", DummyValidator)
    assert manager.get_metric("dummy_metric") is DummyMetric
    assert manager.get_validator("dummy_validator") is DummyValidator

    with pytest.raises(KeyError):
        manager.get_metric("missing")
    with pytest.raises(KeyError):
        manager.get_validator("missing")


def test_register_module_discovers_classes():
    module = types.ModuleType("dummy_module")

    class MetricClass:
        def __call__(self, x1, x2, **_):
            return 0.0

    class ValidatorClass:
        def validate(self, X, y, minority_label, oversampler, **_):
            return 0.0

    module.MetricClass = MetricClass
    module.ValidatorClass = ValidatorClass

    manager = PluginManager()
    manager._register_module(module)
    assert manager.get_metric("metricclass") is MetricClass
    # Classes are callable, so current registration treats them as metrics.
    assert manager.get_metric("validatorclass") is ValidatorClass
    with pytest.raises(KeyError):
        manager.get_validator("validatorclass")
