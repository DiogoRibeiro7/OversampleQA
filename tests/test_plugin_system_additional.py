import types

import pytest

from oversampleqa.plugin_system import PluginManager


class DummyMetric:
    """A real metric (Manhattan).

    This used to return ``(x1 - x2).sum()``, which is not a distance: it goes
    negative, is not symmetric, and is zero for distinct points. Registration
    now runs an axiom check, so a stand-in has to actually be a metric.
    """

    def __call__(self, x1, x2, **_):
        import numpy as np

        return float(np.abs(np.asarray(x1) - np.asarray(x2)).sum())


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
            import numpy as np

            return float(np.abs(np.asarray(x1) - np.asarray(x2)).sum())

    class ValidatorClass:
        def validate(self, X, y, minority_label, oversampler, **_):
            return 0.0

    module.MetricClass = MetricClass
    module.ValidatorClass = ValidatorClass

    manager = PluginManager()
    manager._register_module(module)
    assert manager.get_metric("metricclass") is MetricClass

    # A validator must be discovered as a validator.
    #
    # This test previously asserted the opposite, with the comment "Classes are
    # callable, so current registration treats them as metrics" -- documenting
    # a bug rather than flagging it. `_implements_metric` was
    # `callable(getattr(cls, "__call__", None))`, which is True for every class
    # because `SomeClass.__call__` resolves to the metaclass hook. Every class
    # was therefore classified as a metric and the validator branch was
    # unreachable.
    assert manager.get_validator("validatorclass") is ValidatorClass
    with pytest.raises(KeyError):
        manager.get_metric("validatorclass")
