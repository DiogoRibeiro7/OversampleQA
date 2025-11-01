"""Simple plugin management for metrics and validators."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any, Dict, Type

import numpy as np

from .types import DistanceMetricProtocol, ValidatorProtocol, FloatArray


class PluginManager:
    """Runtime registry for pluggable components."""

    def __init__(self) -> None:
        self._metric_plugins: Dict[str, Type[DistanceMetricProtocol]] = {}
        self._validator_plugins: Dict[str, Type[ValidatorProtocol]] = {}

    def register_metric(self, name: str, metric_cls: Type[DistanceMetricProtocol]) -> None:
        self._metric_plugins[name] = metric_cls

    def register_validator(self, name: str, validator_cls: Type[ValidatorProtocol]) -> None:
        self._validator_plugins[name] = validator_cls

    def get_metric(self, name: str) -> Type[DistanceMetricProtocol]:
        if name not in self._metric_plugins:
            raise KeyError(f"Metric '{name}' is not registered")
        return self._metric_plugins[name]

    def get_validator(self, name: str) -> Type[ValidatorProtocol]:
        if name not in self._validator_plugins:
            raise KeyError(f"Validator '{name}' is not registered")
        return self._validator_plugins[name]

    def discover_plugins(self, package: str = "oversampleqa_plugins") -> None:
        """Discover plugins within a namespace package."""

        try:
            module = importlib.import_module(package)
        except ImportError:
            return
        if not hasattr(module, "__path__"):
            return
        for _, name, _ in pkgutil.iter_modules(module.__path__):
            discovered = importlib.import_module(f"{package}.{name}")
            self._register_module(discovered)

    def _register_module(self, module: Any) -> None:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if self._implements_metric(cls):
                self.register_metric(cls.__name__.lower(), cls)
            elif self._implements_validator(cls):
                self.register_validator(cls.__name__.lower(), cls)

    @staticmethod
    def _implements_metric(cls: Type) -> bool:
        return callable(getattr(cls, "__call__", None))

    @staticmethod
    def _implements_validator(cls: Type) -> bool:
        return callable(getattr(cls, "validate", None))


plugin_manager = PluginManager()


def register_metric(name: str):
    def decorator(cls: Type[DistanceMetricProtocol]) -> Type[DistanceMetricProtocol]:
        plugin_manager.register_metric(name, cls)
        return cls

    return decorator


def register_validator(name: str):
    def decorator(cls: Type[ValidatorProtocol]) -> Type[ValidatorProtocol]:
        plugin_manager.register_validator(name, cls)
        return cls

    return decorator


@register_metric("custom_euclidean")
class CustomEuclideanMetric:
    """Example metric plugin."""

    def __call__(self, x1: FloatArray, x2: FloatArray, **_: Any) -> float:
        return float(np.linalg.norm(x1 - x2))
