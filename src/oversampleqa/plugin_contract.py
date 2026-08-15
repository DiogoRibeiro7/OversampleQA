"""Registration checks for metric plugins.

Task 01 of this codebase's remediation found that the built-in ``hassanat``
implementation was not a metric at all: it scored two distinct points ``[-5]``
and ``[5]`` as distance zero, violating identity of indiscernibles, and it was
discontinuous at the origin. It sat in the registry as the package default for
the project's entire history because nothing ever checked the axioms.

:func:`check_metric_axioms` is that check, applied at registration time so a
plugin cannot repeat the mistake -- and applied to the **built-in registry in
CI**, so the package cannot either.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .exceptions import PluginError

__all__ = [
    "METRIC_DOMAINS",
    "AxiomReport",
    "MetricDomain",
    "MetricPlugin",
    "check_metric_axioms",
    "validate_metric_signature",
]

MetricDomain = Literal["real", "non_negative", "boolean", "sample"]
"""Input a metric is defined on.

Checking every metric on arbitrary real vectors would report failures that are
really out-of-domain calls. ``hellinger`` correctly *raises* on negative input;
``jaccard`` treats -5 and 5 as identical because it is a set metric on booleans,
which is right rather than wrong.
"""

METRIC_DOMAINS: dict[str, MetricDomain] = {
    "hassanat": "real",
    "euclidean": "real",
    "manhattan": "real",
    "chebyshev": "real",
    "minkowski": "real",
    "mahalanobis": "real",
    "cosine": "real",
    "canberra": "real",
    # Compositional: sum|x+y| vanishes when x = -y, so the ratio is undefined
    # off the non-negative orthant.
    "braycurtis": "non_negative",
    # 1 - correlation is undefined for a constant vector (zero variance).
    "correlation": "non_negative",
    # Require inputs normalisable to a probability vector, and raise otherwise.
    "hellinger": "non_negative",
    "jensen_shannon": "non_negative",
    # Set metrics on booleans: sign and magnitude are deliberately discarded.
    "jaccard": "boolean",
    "hamming": "boolean",
    # Sample-based, not point metrics: they compare distributions, so the
    # point-metric axioms do not apply.
    "energy": "sample",
    "wasserstein": "sample",
}


@runtime_checkable
class MetricPlugin(Protocol):
    """A distance metric: two vectors in, one float out."""

    def __call__(
        self, x1: NDArray[np.floating], x2: NDArray[np.floating], **kwargs: Any
    ) -> float:
        """Return the distance between ``x1`` and ``x2``."""
        ...


@dataclass(frozen=True)
class AxiomReport:
    """Which metric axioms a callable satisfied, and how it failed."""

    identity: bool
    identity_of_indiscernibles: bool
    symmetry: bool
    non_negativity: bool
    finiteness: bool
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every checked axiom held."""
        return not self.failures

    def __bool__(self) -> bool:
        """Truthy when every axiom held."""
        return self.ok


def validate_metric_signature(func: Any, name: str) -> None:
    """Reject a callable that cannot be used as a metric.

    Raises:
        PluginError: If it is not callable or cannot take two positional
            arguments.
    """
    if not callable(func):
        raise PluginError(
            f"metric {name!r} is not callable (got {type(func).__name__})"
        )
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return

    positional = [
        p
        for p in signature.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        for p in signature.parameters.values()
    )
    if len(positional) < 2 and not has_varargs:
        raise PluginError(
            f"metric {name!r} must accept two positional arguments (x1, x2); "
            f"its signature is {signature}"
        )


def check_metric_axioms(
    func: Any,
    name: str = "metric",
    *,
    domain: MetricDomain = "real",
    n_trials: int = 50,
    n_features: int = 4,
    tolerance: float = 1e-9,
    random_state: int = 0,
    **metric_kwargs: Any,
) -> AxiomReport:
    """Check that a callable behaves like a distance metric.

    Checks, on random vectors:

    ``identity``
        ``d(x, x) == 0``.
    ``identity_of_indiscernibles``
        ``d(x, y) > 0`` whenever ``x != y``. **This is the check the built-in
        Hassanat implementation failed** -- it scored ``[-5]`` against ``[5]``
        as zero, because it compared absolute values.
    ``symmetry``
        ``d(x, y) == d(y, x)``.
    ``non_negativity``
        ``d(x, y) >= 0``.
    ``finiteness``
        No ``nan`` or ``inf`` on ordinary input.

    The triangle inequality is deliberately **not** checked: several useful
    registry entries are genuine semi-metrics, so requiring it would reject
    metrics the package intends to support. Identity of indiscernibles is the
    one whose violation makes a metric silently meaningless.

    Args:
        func: Candidate metric.
        name: Name used in failure messages.
        domain: Input the metric is defined on. ``"sample"`` metrics compare
            distributions rather than points, so the point-metric axioms are
            skipped for them. See :data:`METRIC_DOMAINS`.
        n_trials: Random vector pairs to test.
        n_features: Dimension of the test vectors.
        tolerance: Numerical slack.
        random_state: Seed, so failures reproduce.
        **metric_kwargs: Extra arguments forwarded to the metric.

    Returns:
        AxiomReport, falsy when any axiom failed.
    """
    if domain == "sample":
        # Sample-based metrics answer a different question -- they compare two
        # sets of observations, not two points -- so identity of indiscernibles
        # is not even meaningful for them.
        return AxiomReport(True, True, True, True, True, ())

    rng = np.random.default_rng(random_state)
    failures: list[str] = []

    def draw() -> NDArray[np.floating]:
        if domain == "non_negative":
            return rng.random(n_features) + 0.1
        if domain == "boolean":
            return (rng.random(n_features) < 0.5).astype(float)
        return rng.normal(0, 5, size=n_features)

    identity = True
    indiscernibles = True
    symmetry = True
    non_negative = True
    finite = True

    for _ in range(n_trials):
        x = draw()
        y = draw()
        if domain == "boolean" and np.array_equal(x, y):
            continue  # boolean draws collide; that is not a violation

        try:
            d_xy = float(func(x, y, **metric_kwargs))
            d_yx = float(func(y, x, **metric_kwargs))
            d_xx = float(func(x, x, **metric_kwargs))
        except Exception as exc:
            failures.append(f"raised {type(exc).__name__}: {exc}")
            return AxiomReport(False, False, False, False, False, tuple(failures))

        if not (np.isfinite(d_xy) and np.isfinite(d_xx)):
            finite = False
        if abs(d_xx) > tolerance:
            identity = False
        if d_xy < -tolerance:
            non_negative = False
        if abs(d_xy - d_yx) > tolerance:
            symmetry = False
        if d_xy <= tolerance:
            # Random continuous vectors are distinct with probability 1.
            indiscernibles = False

    # The specific case the broken Hassanat passed everything else on. Only
    # meaningful where sign carries information: a boolean set metric is
    # *supposed* to map -5 and 5 to the same element, and a non-negative domain
    # has no mirrored pair.
    mirrored = np.full(n_features, 5.0)
    if domain != "real":
        return AxiomReport(
            identity=identity,
            identity_of_indiscernibles=indiscernibles,
            symmetry=symmetry,
            non_negativity=non_negative,
            finiteness=finite,
            failures=tuple(
                f"{name}: {f}"
                for f in _collect(
                    identity, indiscernibles, symmetry, non_negative, finite, failures
                )
            ),
        )
    try:
        d_mirror = float(func(-mirrored, mirrored, **metric_kwargs))
        if abs(d_mirror) <= tolerance:
            indiscernibles = False
            failures.append(
                "d(-x, x) == 0 for x = 5: distinct points at distance zero. "
                "This usually means the metric compares magnitudes and discards "
                "sign -- the exact defect found in the original hassanat "
                "implementation."
            )
    except Exception:
        pass

    failures = _collect(
        identity, indiscernibles, symmetry, non_negative, finite, failures
    )

    return AxiomReport(
        identity=identity,
        identity_of_indiscernibles=indiscernibles,
        symmetry=symmetry,
        non_negativity=non_negative,
        finiteness=finite,
        failures=tuple(f"{name}: {f}" for f in failures),
    )


def _collect(
    identity: bool,
    indiscernibles: bool,
    symmetry: bool,
    non_negative: bool,
    finite: bool,
    failures: list[str],
) -> list[str]:
    """Turn the boolean outcomes into messages."""
    if not identity:
        failures.append("d(x, x) != 0")
    if not indiscernibles and not any("distance zero" in f for f in failures):
        failures.append("d(x, y) == 0 for distinct x and y")
    if not symmetry:
        failures.append("d(x, y) != d(y, x)")
    if not non_negative:
        failures.append("d(x, y) < 0")
    if not finite:
        failures.append("returned nan or inf on ordinary input")
    return failures
