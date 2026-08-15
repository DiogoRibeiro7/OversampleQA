"""Simple plugin management for metrics and validators."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any

import numpy as np

from .plugin_contract import MetricDomain
from .types import DistanceMetricProtocol, FloatArray, ValidatorProtocol


class PluginManager:
    """Runtime registry for pluggable components."""

    def __init__(self) -> None:
        self._metric_plugins: dict[str, type[DistanceMetricProtocol]] = {}
        self._validator_plugins: dict[str, type[ValidatorProtocol]] = {}

    def register_metric(
        self,
        name: str,
        metric_cls: type[DistanceMetricProtocol],
        *,
        domain: MetricDomain = "real",
        check_axioms: bool = True,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Register a metric class by name, after checking it behaves like one.

        Registration used to be a bare dictionary assignment: any object could
        be registered under any name, silently replacing a built-in, and a
        metric that violated the axioms would be discovered only by whoever
        eventually distrusted its numbers.

        Three checks now run, each raising :class:`~oversampleqa.PluginError`:

        1. **Name collision.** A name already taken by a built-in or another
           plugin is refused rather than overwritten.
        2. **Signature.** The callable must accept two positional arguments.
        3. **Axioms.** ``d(x, x) == 0``, ``d(x, y) > 0`` for distinct points,
           symmetry, non-negativity and finiteness on random input.

        The third is not hypothetical. The built-in ``hassanat`` shipped for
        this project's entire history scoring ``[-5]`` and ``[5]`` as distance
        zero, because it compared absolute values -- it was not a metric, and
        nothing checked. This is that check.

        Args:
            name: Metric identifier.
            metric_cls: Metric callable or class implementing ``__call__``.
            domain: Input the metric is defined on. See
                :data:`~oversampleqa.plugin_contract.METRIC_DOMAINS`.
            check_axioms: Run the axiom smoke check. Disable only for a metric
                you have verified another way, and say why.
            metric_kwargs: Extra arguments the metric needs, such as
                ``cov_inv`` for Mahalanobis.

        Raises:
            PluginError: On collision, bad signature, or axiom failure.
        """
        from .distance import _METRICS
        from .exceptions import PluginError
        from .plugin_contract import check_metric_axioms, validate_metric_signature

        if name in _METRICS:
            raise PluginError(
                f"metric {name!r} is already a built-in. Registering over it "
                "would silently change results for every caller using that "
                "name; choose a different one."
            )
        if name in self._metric_plugins:
            raise PluginError(
                f"metric {name!r} is already registered by another plugin. "
                "Unregister it first if replacement is intended."
            )

        candidate = metric_cls() if isinstance(metric_cls, type) else metric_cls
        validate_metric_signature(candidate, name)

        if check_axioms:
            report = check_metric_axioms(
                candidate, name, domain=domain, **(metric_kwargs or {})
            )
            if not report.ok:
                detail = "\n  ".join(report.failures)
                raise PluginError(
                    f"metric {name!r} does not satisfy the distance axioms:\n"
                    f"  {detail}\n"
                    "A metric that violates these produces numbers that look "
                    "plausible and mean nothing."
                )

        self._metric_plugins[name] = metric_cls

    def unregister_metric(self, name: str) -> None:
        """Remove a registered metric plugin.

        Args:
            name: Metric identifier.

        Raises:
            PluginError: If no plugin is registered under that name.
        """
        from .exceptions import PluginError

        if name not in self._metric_plugins:
            raise PluginError(f"no metric plugin registered as {name!r}")
        del self._metric_plugins[name]

    def register_validator(
        self, name: str, validator_cls: type[ValidatorProtocol]
    ) -> None:
        """Register a validator class by name.

        Args:
            name: Validator identifier.
            validator_cls: Validator class implementing ``validate``.
        """
        self._validator_plugins[name] = validator_cls

    def get_metric(self, name: str) -> type[DistanceMetricProtocol]:
        """Retrieve a registered metric class by name.

        Args:
            name: Metric identifier.

        Returns:
            Registered metric class.
        """
        if name not in self._metric_plugins:
            raise KeyError(f"Metric '{name}' is not registered")
        return self._metric_plugins[name]

    def get_validator(self, name: str) -> type[ValidatorProtocol]:
        """Retrieve a registered validator class by name.

        Args:
            name: Validator identifier.

        Returns:
            Registered validator class.
        """
        if name not in self._validator_plugins:
            raise KeyError(f"Validator '{name}' is not registered")
        return self._validator_plugins[name]

    def discover_plugins(self, package: str = "oversampleqa_plugins") -> None:
        """Discover plugins within a namespace package.

        Args:
            package: Namespace package to scan.
        """

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
        """Register all compatible classes in a module.

        Args:
            module: Imported module object.
        """
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if self._implements_metric(cls):
                self.register_metric(cls.__name__.lower(), cls)
            elif self._implements_validator(cls):
                self.register_validator(cls.__name__.lower(), cls)

    @staticmethod
    def _implements_metric(cls: type) -> bool:
        """Return True if instances of ``cls`` are callable.

        Args:
            cls: Class object.

        Returns:
            ``True`` if the class, or one of its bases, defines ``__call__``.

        Notes:
            This used to be ``callable(getattr(cls, "__call__", None))``, which
            is ``True`` for **every** class: ``SomeClass.__call__`` resolves to
            ``type.__call__``, the metaclass hook used to instantiate it, and
            that is callable. Module discovery therefore classified every class
            as a metric, and the validator branch below was unreachable -- no
            validator has ever been discovered from a module.

            Walking ``__mro__`` and looking in each ``__dict__`` asks the right
            question: does the class itself define ``__call__``? ``object`` does
            not, and the metaclass is not in the MRO.
        """
        return any("__call__" in klass.__dict__ for klass in cls.__mro__)

    @staticmethod
    def _implements_validator(cls: type) -> bool:
        """Return True if class looks like a validator.

        Args:
            cls: Class object.

        Returns:
            ``True`` if the class defines ``validate``.
        """
        return callable(getattr(cls, "validate", None))


plugin_manager = PluginManager()


def register_metric(name: str) -> Callable[[Any], Any]:
    """Decorator to register a metric plugin by name.

    Args:
        name: Metric identifier.
    """

    def decorator(cls: type[DistanceMetricProtocol]) -> type[DistanceMetricProtocol]:
        plugin_manager.register_metric(name, cls)
        return cls

    return decorator


def register_validator(name: str) -> Callable[[Any], Any]:
    """Decorator to register a validator plugin by name.

    Args:
        name: Validator identifier.
    """

    def decorator(cls: type[ValidatorProtocol]) -> type[ValidatorProtocol]:
        plugin_manager.register_validator(name, cls)
        return cls

    return decorator


@register_metric("custom_euclidean")
class CustomEuclideanMetric:
    """Example metric plugin."""

    def __call__(self, x1: FloatArray, x2: FloatArray, **_: Any) -> float:
        return float(np.linalg.norm(x1 - x2))
