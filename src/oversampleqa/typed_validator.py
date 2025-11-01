"""Typed validator with runtime validation and async support."""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Union, cast, overload

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .types import (
    FloatArray,
    IntArray,
    MetricError,
    MetricName,
    OversamplerProtocol,
    ValidationConfig,
    ValidationError,
    ValidationMode,
    ValidationResult,
    BaseValidator,
)
from .distance import _METRICS


class PydanticValidationConfig(BaseModel):
    """Runtime validation for configuration parameters."""

    hidden_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    metric: str = Field(default="hassanat")
    return_details: bool = Field(default=False)
    random_state: Optional[int] = Field(default=None)

    @field_validator("metric")
    def validate_metric(cls, value: str) -> str:
        allowed = set(_METRICS.keys())
        if value not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}")
        return value

    @field_validator("random_state")
    def validate_random_state(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (0 <= value < 2**31):
            raise ValueError("random_state must be between 0 and 2**31 - 1")
        return value


class TypedValidator(BaseValidator[ValidationResult]):
    """Type-safe validator wrapper with runtime validation."""

    def __init__(self, mode: ValidationMode = ValidationMode.STANDARD) -> None:
        self.mode = mode

    @overload
    def validate(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: ValidationConfig,
    ) -> ValidationResult:
        ...

    @overload
    def validate(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        *,
        hidden_ratio: float = 0.1,
        metric: str = "hassanat",
        return_details: bool = False,
        random_state: Optional[int] = None,
    ) -> ValidationResult:
        ...

    def validate(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: Optional[ValidationConfig] = None,
        **kwargs: Any,
    ) -> ValidationResult:
        if config is None:
            parsed = PydanticValidationConfig(**kwargs)
            metric_name = cast(MetricName, parsed.metric)
            config = ValidationConfig(
                hidden_ratio=parsed.hidden_ratio,
                metric=metric_name,
                return_details=parsed.return_details,
                random_state=parsed.random_state,
            )

        self._validate_inputs(X, y, minority_label, oversampler, config)

        if self.mode == ValidationMode.MEMORY_EFFICIENT:
            return self._validate_standard(X, y, minority_label, oversampler, config)
        if self.mode == ValidationMode.PARALLEL:
            # Future: add parallel implementation
            return self._validate_standard(X, y, minority_label, oversampler, config)
        if self.mode == ValidationMode.ASYNC:
            return asyncio.get_event_loop().run_until_complete(
                self.validate_async(X, y, minority_label, oversampler, config)
            )
        return self._validate_standard(X, y, minority_label, oversampler, config)

    async def validate_async(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: ValidationConfig,
    ) -> ValidationResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.validate, X, y, minority_label, oversampler, config
        )

    def _validate_inputs(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: ValidationConfig,
    ) -> None:
        if not isinstance(X, np.ndarray) or not np.issubdtype(X.dtype, np.floating):
            raise ValidationError("X must be a floating-point numpy array")
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.integer):
            raise ValidationError("y must be an integer numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValidationError("X and y must have the same number of rows")
        if minority_label not in y:
            raise ValidationError(f"minority_label {minority_label} not present in y")
        if not hasattr(oversampler, "fit_resample"):
            raise ValidationError("oversampler must implement fit_resample")
        if config.metric not in _METRICS:
            raise MetricError(f"Unsupported metric '{config.metric}'")

    def _validate_standard(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: ValidationConfig,
    ) -> ValidationResult:
        from .validator import validate_oversampling

        try:
            if config.return_details:
                error_rate, n_errors, dist_hidden, dist_min = validate_oversampling(
                    X,
                    y,
                    minority_label,
                    oversampler,
                    hidden_ratio=config.hidden_ratio,
                    metric=config.metric,
                    return_details=True,
                )
                n_synthetic = dist_hidden.shape[0]
                ci = self._wald_confidence_interval(error_rate, max(n_synthetic, 1))
                return ValidationResult(
                    error_rate=error_rate,
                    n_errors=n_errors,
                    n_synthetic=n_synthetic,
                    confidence_interval=ci,
                    metadata={
                        "distance_matrices": {"hidden": dist_hidden, "minority": dist_min}
                    },
                )
            error_rate = validate_oversampling(
                X,
                y,
                minority_label,
                oversampler,
                hidden_ratio=config.hidden_ratio,
                metric=config.metric,
                return_details=False,
            )
            ci = self._wald_confidence_interval(error_rate, len(y))
            return ValidationResult(
                error_rate=error_rate,
                n_errors=0,
                n_synthetic=0,
                confidence_interval=ci,
                metadata={},
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationError(f"Validation failed: {exc}") from exc

    @staticmethod
    def _wald_confidence_interval(rate: float, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n <= 0:
            return (0.0, 1.0)
        se = math.sqrt(rate * (1 - rate) / n)
        lower = max(0.0, rate - z * se)
        upper = min(1.0, rate + z * se)
        return (lower, upper)


@asynccontextmanager
async def validation_session(config: ValidationConfig) -> AsyncIterator[TypedValidator]:
    validator = TypedValidator()
    try:
        yield validator
    finally:
        return


class ServiceRegistry:
    """Minimal dependency injection container."""

    def __init__(self) -> None:
        self._services: Dict[type, Any] = {}

    def register(self, service_type: type, implementation: Any) -> None:
        self._services[service_type] = implementation

    def get(self, service_type: type) -> Any:
        if service_type not in self._services:
            raise ConfigurationError(f"Service {service_type} not registered")
        return self._services[service_type]


registry = ServiceRegistry()
