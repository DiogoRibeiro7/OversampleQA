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


ReferenceSet = Literal["hidden_minority", "train_minority"]
"""Which minority set validation compares synthetic points against."""


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
        """Fit and resample the dataset, returning resampled arrays.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Tuple of resampled ``(X, y)`` arrays.
        """
        ...

    @property
    def random_state(self) -> Optional[int]:
        """Return the random state, if supported.

        Returns:
            Random state value or ``None``.
        """
        ...

    @random_state.setter
    def random_state(self, value: Optional[int]) -> None:
        """Set the random state, if supported.

        Args:
            value: Random state to set.
        """
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
        """Validate an oversampler and return an error rate.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            **kwargs: Implementation-specific options.

        Returns:
            Error rate.
        """
        ...


@dataclass(frozen=True)
class ValidationConfig:
    """Immutable validation configuration."""

    hidden_ratio: float = 0.1
    metric: str = "hassanat"
    return_details: bool = False
    random_state: Optional[int] = 42
    reference: ReferenceSet = "hidden_minority"
    n_repeats: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.hidden_ratio < 1:
            raise ValueError("hidden_ratio must be between 0 and 1")


@dataclass(frozen=True)
class ValidationDetails:
    """Detailed outcome of a single validation run.

    Replaces the former ``(error_rate, n_errors, dist_hidden, dist_min)``
    4-tuple returned by ``return_details=True``.

    Attributes
    ----------
    error_rate:
        Fraction of synthetic points strictly closer to the hidden majority
        than to the minority reference set. ``nan`` when no synthetic samples
        were produced -- that is an absent measurement, not a perfect score.
    n_errors:
        Count behind ``error_rate``.
    n_synthetic:
        Number of synthetic points scored.
    n_ties:
        Points exactly equidistant from both reference sets. Counted
        separately rather than scored as errors; a large value indicates
        duplicated or heavily quantised features.
    duplication_rate:
        Fraction of synthetic points coinciding with a reference point. A
        sampler that only duplicates scores 1.0, and its error rate carries
        no information about synthesis quality.
    reference:
        Which minority set the comparison used.
    dist_hidden, dist_min:
        Distance matrices from synthetic points to the hidden majority and to
        the minority reference set.
    """

    error_rate: float
    n_errors: int
    n_synthetic: int
    n_ties: int
    duplication_rate: float
    reference: ReferenceSet
    dist_hidden: FloatArray
    dist_min: FloatArray
    n_repeats: int = 1
    rates: tuple[float, ...] = ()
    mean: float = float("nan")
    std: float = float("nan")
    interval: tuple[float, float] | None = None

    @property
    def has_dispersion(self) -> bool:
        """Whether more than one hold-out split was drawn."""
        return self.n_repeats > 1


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
        """Run validation and return a result.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig.

        Returns:
            Validation result.
        """
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
        """Run validation asynchronously and return a result.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            config: ValidationConfig.

        Returns:
            Validation result.
        """
        ...


class Dataset(ABC):
    """Abstract dataset definition."""

    @property
    @abstractmethod
    def X(self) -> FloatArray:
        """Return feature matrix.

        Returns:
            Feature matrix.
        """
        ...

    @property
    @abstractmethod
    def y(self) -> IntArray:
        """Return target labels.

        Returns:
            Target labels.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return dataset name.

        Returns:
            Dataset name.
        """
        ...

    @property
    @abstractmethod
    def minority_label(self) -> int:
        """Return the minority class label.

        Returns:
            Minority class label.
        """
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
        """Create a validator instance for the given mode.

        Args:
            mode: Validation execution mode.
            **kwargs: Implementation-specific options.

        Returns:
            Validator instance.
        """
        ...


class MetricFactory(Protocol):
    """Factory for distance metrics."""

    def create_metric(self, name: str, **kwargs: Any) -> DistanceMetricProtocol:
        """Create a distance metric by name.

        Args:
            name: Metric identifier.
            **kwargs: Metric-specific parameters.

        Returns:
            Distance metric callable.
        """
        ...
