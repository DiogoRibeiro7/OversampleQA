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
    ValidationDetails,
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
        """Validate that the metric is supported.

        Args:
            value: Metric name.

        Returns:
            The validated metric name.
        """
        allowed = set(_METRICS.keys())
        if value not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}")
        return value

    @field_validator("random_state")
    def validate_random_state(cls, value: Optional[int]) -> Optional[int]:
        """Validate random_state bounds when provided.

        Args:
            value: Optional random state.

        Returns:
            The validated random state.
        """
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
        """Validate using a prebuilt ValidationConfig.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig.

        Returns:
            ValidationResult.
        """
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
        """Validate using keyword configuration parameters.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            hidden_ratio: Fraction of majority to hide.
            metric: Distance metric name.
            return_details: Whether to include distance matrices.
            random_state: Optional random seed.

        Returns:
            ValidationResult.
        """
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
        """Validate oversampling with typed configuration.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig, or None to build from kwargs.
            **kwargs: ValidationConfig fields when ``config`` is None.

        Returns:
            ValidationResult with error rate and optional details.
        """
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
        """Async wrapper around validate using an executor.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig.

        Returns:
            ValidationResult.
        """
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
        """Validate input arrays and configuration.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig.
        """
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
        """Execute the standard validation pipeline.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig.

        Returns:
            ValidationResult.
        """
        from .validator import validate_oversampling

        try:
            if config.return_details:
                details = validate_oversampling(
                    X,
                    y,
                    minority_label,
                    oversampler,
                    hidden_ratio=config.hidden_ratio,
                    metric=config.metric,
                    return_details=True,
                    reference=config.reference,
                    random_state=config.random_state,
                    n_repeats=config.n_repeats,
                )
                # return_details=True always yields ValidationDetails; the
                # runtime check narrows the union without suppressing the type.
                if not isinstance(details, ValidationDetails):
                    raise TypeError(
                        "validate_oversampling(return_details=True) must return "
                        f"ValidationDetails, got {type(details).__name__}"
                    )
                n_synthetic = details.n_synthetic
                ci = self._wald_confidence_interval(
                    details.error_rate, max(n_synthetic, 1)
                )
                return ValidationResult(
                    error_rate=details.error_rate,
                    n_errors=details.n_errors,
                    n_synthetic=n_synthetic,
                    confidence_interval=ci,
                    metadata={
                        "distance_matrices": {
                            "hidden": details.dist_hidden,
                            "minority": details.dist_min,
                        },
                        "n_ties": details.n_ties,
                        "duplication_rate": details.duplication_rate,
                        "reference": details.reference,
                        "random_state": config.random_state,
                        "n_repeats": details.n_repeats,
                        "rates": details.rates,
                        "std": details.std,
                        "repeat_interval": details.interval,
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
                reference=config.reference,
                random_state=config.random_state,
                n_repeats=config.n_repeats,
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
        """Compute a Wald confidence interval for a binomial proportion.

        Args:
            rate: Estimated proportion.
            n: Sample size.
            z: Z-score for the confidence level.

        Returns:
            Lower and upper confidence bounds.
        """
        if n <= 0:
            return (0.0, 1.0)
        se = math.sqrt(rate * (1 - rate) / n)
        lower = max(0.0, rate - z * se)
        upper = min(1.0, rate + z * se)
        return (lower, upper)


@asynccontextmanager
async def validation_session(config: ValidationConfig) -> AsyncIterator[TypedValidator]:
    """Async context manager that yields a TypedValidator.

    Args:
        config: ValidationConfig (reserved for future use).

    Yields:
        TypedValidator instance.
    """
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
        """Register a service implementation by type.

        Args:
            service_type: Key type.
            implementation: Service implementation instance.
        """
        self._services[service_type] = implementation

    def get(self, service_type: type) -> Any:
        """Retrieve a registered service implementation.

        Args:
            service_type: Key type.

        Returns:
            Registered service implementation.
        """
        if service_type not in self._services:
            raise ConfigurationError(f"Service {service_type} not registered")
        return self._services[service_type]


registry = ServiceRegistry()
