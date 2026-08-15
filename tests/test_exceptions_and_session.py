"""Tests for the exception hierarchy and the error paths that reach it.

Every defect these cover was latent because nothing exercised the failure
path: ``ConfigurationError`` was raised without being defined, a ``return``
inside ``finally`` swallowed exceptions, and the async mode called
``run_until_complete`` in contexts where it always throws.
"""

from __future__ import annotations

import asyncio
import typing

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

from oversampleqa.exceptions import (
    ConfigurationError,
    MetricError,
    OversampleQAError,
    PluginError,
    UnsupportedSamplerError,
    ValidationError,
)
from oversampleqa.typed_validator import (
    ServiceRegistry,
    TypedValidator,
    validation_session,
)
from oversampleqa.types import ValidationConfig, ValidationMode


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (240, 4)), rng.normal(3.0, 1.0, (80, 4))])
    y = np.hstack([np.zeros(240), np.ones(80)]).astype(int)
    return X, y


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError,
        ValidationError,
        UnsupportedSamplerError,
        MetricError,
        PluginError,
    ],
)
def test_every_error_derives_from_the_base(error):
    """Callers must be able to catch the whole family with one except clause."""
    assert issubclass(error, OversampleQAError)
    with pytest.raises(OversampleQAError):
        raise error("boom")


def test_unsupported_sampler_is_a_validation_error():
    """It is a validation failure, so catching ValidationError must catch it."""
    assert issubclass(UnsupportedSamplerError, ValidationError)


def test_types_module_still_re_exports_the_errors():
    """Existing imports from oversampleqa.types must keep working."""
    from oversampleqa import types

    assert types.ValidationError is ValidationError
    assert types.ConfigurationError is ConfigurationError


def test_service_registry_hit():
    registry = ServiceRegistry()
    sentinel = object()
    registry.register(str, sentinel)
    assert registry.get(str) is sentinel


def test_service_registry_miss_raises_a_defined_error():
    """This path used to raise NameError from an undefined ConfigurationError.

    The error handler failing is worse than the original error: it replaces a
    clear "not registered" message with a confusing NameError.
    """
    registry = ServiceRegistry()
    with pytest.raises(ConfigurationError, match="not registered"):
        registry.get(int)


def test_service_registry_collision_takes_the_last_registration():
    registry = ServiceRegistry()
    registry.register(str, "first")
    registry.register(str, "second")
    assert registry.get(str) == "second"


def test_wilson_interval_type_hints_resolve():
    """A bare `Tuple` annotation was a latent NameError under get_type_hints."""
    hints = typing.get_type_hints(TypedValidator._wilson_confidence_interval)
    assert hints["return"] == tuple[float, float]


def test_wilson_interval_stays_within_bounds_near_zero():
    """Wald degenerates to zero width at rate=0; Wilson does not.

    Error rates near zero are the common case for this package, so the
    behaviour at the boundary is the behaviour that matters.
    """
    lower, upper = TypedValidator._wilson_confidence_interval(0.0, 100)
    assert lower == 0.0
    assert upper > 0.0, "a zero-width interval at rate=0 claims impossible certainty"

    # At p = 1 the algebra gives exactly (1 + z^2/n) / (1 + z^2/n) = 1, so the
    # residue here is floating-point rounding rather than a real gap.
    lower, upper = TypedValidator._wilson_confidence_interval(1.0, 100)
    assert upper == pytest.approx(1.0)
    assert lower < 1.0


def test_wilson_interval_never_leaves_the_unit_range():
    for rate in (0.0, 0.01, 0.5, 0.99, 1.0):
        lower, upper = TypedValidator._wilson_confidence_interval(rate, 30)
        assert 0.0 <= lower <= upper <= 1.0


def test_wilson_interval_handles_empty_sample():
    assert TypedValidator._wilson_confidence_interval(0.0, 0) == (0.0, 1.0)


def test_validation_session_yields_a_validator():
    async def run():
        async with validation_session(ValidationConfig()) as validator:
            assert isinstance(validator, TypedValidator)

    asyncio.run(run())


def test_validation_session_propagates_exceptions():
    """`return` inside `finally` swallowed whatever was in flight.

    A failure inside the session body vanished and the caller saw a clean exit.
    """

    async def run():
        async with validation_session(ValidationConfig()):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())


def test_async_mode_raises_instead_of_driving_a_loop(data):
    """It called run_until_complete, which throws if a loop is already running.

    Rather than papering over that, the mode now explains why it cannot work:
    the validation is CPU-bound NumPy, so asyncio offers it no concurrency.
    """
    X, y = data
    validator = TypedValidator(mode=ValidationMode.ASYNC)
    with pytest.raises(ConfigurationError, match="cannot be driven"):
        validator.validate(X, y, 1, SMOTE(random_state=0), ValidationConfig())


def test_async_mode_error_survives_inside_a_running_loop(data):
    """The original failure mode: a loop already running."""
    X, y = data

    async def run():
        validator = TypedValidator(mode=ValidationMode.ASYNC)
        with pytest.raises(ConfigurationError):
            validator.validate(X, y, 1, SMOTE(random_state=0), ValidationConfig())

    asyncio.run(run())


def test_validate_async_works_from_async_code(data):
    """The supported route for async callers."""
    X, y = data

    async def run():
        validator = TypedValidator()
        return await validator.validate_async(
            X, y, 1, SMOTE(random_state=0), ValidationConfig()
        )

    result = asyncio.run(run())
    assert 0.0 <= result["error_rate"] <= 1.0


def test_unsupported_metric_raises_metric_error(data):
    X, y = data
    validator = TypedValidator()
    with pytest.raises(MetricError, match="Unsupported metric"):
        validator.validate(
            X,
            y,
            1,
            SMOTE(random_state=0),
            ValidationConfig(metric="not_a_metric"),
        )


def test_bad_input_raises_validation_error(data):
    X, y = data
    validator = TypedValidator()
    with pytest.raises(ValidationError, match="same number of rows"):
        validator.validate(X[:10], y, 1, SMOTE(random_state=0), ValidationConfig())
