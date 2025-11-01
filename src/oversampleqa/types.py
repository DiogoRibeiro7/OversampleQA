"""Core protocol and type definitions for oversampleqa."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    TypedDict,
    Union,
)

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


MetricName = Literal[
    "hassanat",
    "euclidean",
    "manhattan",
    "cosine",
    "minkowski",
    "chebyshev",
    "mahalanobis",
    "canberra",
    "hamming",
    "jaccard",
    "braycurtis",
    "correlation",
    "energy",
    "wasserstein",
    "hellinger",
    "jensen_shannon",
]


class ValidationMode(Enum):
    """Validation execution modes."""

    STANDARD = "standard"
    MEMORY_EFFICIENT = "memory_efficient"
    PARALLEL = "parallel"
    ASYNC = "async"


class DistanceMetricProtocol(Protocol):
    """Protocol for distance metric callables."""

    def __call__(self, x1: FloatArray, x2: FloatArray, **kwargs: Any) -> float:
        ...


class OversamplerProtocol(Protocol):
    """Protocol for oversampler-like objects."""

    def fit_resample(self, X: FloatArray, y: IntArray) -> Tuple[FloatArray, IntArray]:
        ...

    @property
    def random_state(self) -> Optional[int]:
        ...

    @random_state.setter
    def random_state(self, value: Optional[int]) -> None:
        ...


class ValidatorProtocol(Protocol):
    """Protocol for validator implementations."""

    def validate(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        **kwargs: Any,
    ) -> float:
        ...


@dataclass(frozen=True)
class ValidationConfig:
    """Immutable validation configuration."""

    hidden_ratio: float = 0.1
    metric: str = "hassanat"
    return_details: bool = False
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 < self.hidden_ratio < 1:
            raise ValueError("hidden_ratio must be between 0 and 1")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    n_runs: int = 10
    hidden_ratios: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    metrics: List[str] = field(default_factory=lambda: ["hassanat", "euclidean"])
    validation_mode: ValidationMode = ValidationMode.STANDARD
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if any(ratio <= 0 or ratio >= 1 for ratio in self.hidden_ratios):
            raise ValueError("All hidden ratios must be in (0, 1)")


class ValidationResult(TypedDict, total=False):
    """Typed structure for validation result."""

    error_rate: float
    n_errors: int
    n_synthetic: int
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any]


T = TypeVar("T")
DatasetType = TypeVar("DatasetType", bound="Dataset")
ValidatorType = TypeVar("ValidatorType", bound="BaseValidator[Any]")


class BaseValidator(ABC, Generic[T]):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: ValidationConfig,
    ) -> T:
        ...

    @abstractmethod
    async def validate_async(
        self,
        X: FloatArray,
        y: IntArray,
        minority_label: int,
        oversampler: OversamplerProtocol,
        config: ValidationConfig,
    ) -> T:
        ...


class Dataset(ABC):
    """Abstract dataset definition."""

    @property
    @abstractmethod
    def X(self) -> FloatArray:
        ...

    @property
    @abstractmethod
    def y(self) -> IntArray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def minority_label(self) -> int:
        ...


class OversampleQAError(Exception):
    """Base exception for oversampleqa."""


class ValidationError(OversampleQAError):
    """Validation-specific error."""


class ConfigurationError(OversampleQAError):
    """Configuration-related errors."""


class MetricError(OversampleQAError):
    """Distance metric computation errors."""


class ValidatorFactory(Protocol):
    """Factory for validators."""

    def create_validator(self, mode: ValidationMode, **kwargs: Any) -> BaseValidator[Any]:
        ...


class MetricFactory(Protocol):
    """Factory for distance metrics."""

    def create_metric(self, name: str, **kwargs: Any) -> DistanceMetricProtocol:
        ...
