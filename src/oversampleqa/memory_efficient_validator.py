"""Memory optimised validator utilities."""

from __future__ import annotations

import hashlib
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .distance import _METRICS, hassanat_distance  # noqa: F401 - ensures registry populated
from .metrics import calculate_error_rate, duplication_rate
from .types import ReferenceSet, ValidationDetails
from .optimized_distance import OptimizedDistanceMatrix, get_available_memory_gb
from .validator import (
    extract_synthetic_samples,
    prepare_validation_split,
    score_nearest_distances,
    warn_reference_bias,
)
from .caching import ValidationCache


class MemoryEfficientValidator:
    """Drop-in replacement for :func:`validate_oversampling` with memory safeguards."""

    def __init__(
        self,
        memory_limit_gb: float = 4.0,
        batch_size: int | str = "auto",
        show_progress: bool = False,
        temp_dir: str | None = None,
        cache: ValidationCache | None = None,
    ) -> None:
        self.memory_limit_gb = memory_limit_gb
        self.batch_size = batch_size
        self.temp_root = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "oversampleqa"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.cache = cache or ValidationCache()
        self.distance_computer = OptimizedDistanceMatrix(
            memory_limit_gb=memory_limit_gb,
            metric_registry=_METRICS,
            show_progress=show_progress,
            cache=self.cache,
        )
        self._stream_dirs: list[Path] = []

    def validate_oversampling(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        minority_label: int,
        oversampler: BaseOverSampler,
        hidden_ratio: float = 0.1,
        metric: str = "hassanat",
        metric_kwargs: dict[str, Any] | None = None,
        return_details: bool = False,
        *,
        reference: ReferenceSet = "hidden_minority",
        minority_hidden_ratio: float | None = None,
        min_hidden: int = 5,
    ) -> float | ValidationDetails:
        """Validate oversampling with streaming-aware distance calculations.

        Uses the same estimand as :func:`oversampleqa.validate_oversampling`
        via the shared :func:`~oversampleqa.validator.prepare_validation_split`
        helper, so the two cannot drift apart.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance.
            hidden_ratio: Fraction of majority to hide.
            metric: Distance metric name.
            metric_kwargs: Metric keyword arguments.
            return_details: Whether to return a ``ValidationDetails``.
            reference: Which minority set to compare against. See
                :func:`oversampleqa.validate_oversampling`.
            minority_hidden_ratio: Fraction of the minority to hide.
            min_hidden: Minimum held-out minority points.

        Returns:
            Error rate, or ``ValidationDetails`` when ``return_details`` is True.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        metric_kwargs = metric_kwargs or {}
        warn_reference_bias(reference, stacklevel=3)

        params_hash = None
        if self.cache is not None and not return_details:
            payload = {
                "data_hash": self.cache.get_data_hash(X, y),
                "minority_label": minority_label,
                "oversampler": oversampler.__class__.__qualname__,
                "oversampler_params": oversampler.get_params(deep=True),
                "hidden_ratio": hidden_ratio,
                "metric": metric,
                "metric_kwargs": metric_kwargs,
                "reference": reference,
                "minority_hidden_ratio": minority_hidden_ratio,
            }
            params_hash = hashlib.sha256(pickle.dumps(payload)).hexdigest()
            cached = self.cache.load_validation_result(params_hash)
            if cached is not None:
                return cached

        labels = np.unique(y)
        majority_labels = labels[labels != minority_label]
        if len(majority_labels) == 0:
            raise ValueError(f"minority_label {minority_label} is the only label in y")
        majority_label = int(majority_labels[0])

        split = prepare_validation_split(
            X,
            y,
            minority_label,
            majority_label,
            hidden_ratio,
            reference=reference,
            minority_hidden_ratio=minority_hidden_ratio,
            min_hidden=min_hidden,
        )
        X_train = split.X_train
        y_train = split.y_train
        hid_majority = split.hid_majority
        fit_minority = split.fit_minority
        minority = split.reference_minority

        X_res, y_res = oversampler.fit_resample(X_train, y_train)
        synthetic = extract_synthetic_samples(X_train, X_res, y_res, minority_label)

        empty = np.empty((0, 0))
        if len(synthetic) == 0:
            warnings.warn(
                f"{type(oversampler).__name__} produced no synthetic minority "
                "samples, so there is nothing to validate. Returning nan rather "
                "than 0.0, which would be indistinguishable from a perfect score.",
                UserWarning,
                stacklevel=2,
            )
            if return_details:
                return ValidationDetails(
                    error_rate=float("nan"),
                    n_errors=0,
                    n_synthetic=0,
                    n_ties=0,
                    duplication_rate=float("nan"),
                    reference=reference,
                    dist_hidden=empty,
                    dist_min=empty,
                )
            return float("nan")

        dtype = np.result_type(synthetic.dtype, minority.dtype, np.float64)
        est_hidden = self.distance_computer.estimate_memory_gb(
            len(synthetic), len(hid_majority), dtype=dtype
        )
        est_minority = self.distance_computer.estimate_memory_gb(
            len(synthetic), len(minority), dtype=dtype
        )
        available = get_available_memory_gb()
        requires_stream = max(len(synthetic), len(minority), len(hid_majority)) > 10_000
        if est_hidden > self.memory_limit_gb or est_minority > self.memory_limit_gb:
            warnings.warn(
                "Distance matrices exceed configured memory limit; activating streaming mode.",
                ResourceWarning,
            )
            requires_stream = True
        elif est_hidden > available or est_minority > available:
            warnings.warn(
                "Estimated distance matrices exceed available system memory; switching to streaming mode.",
                ResourceWarning,
            )
            requires_stream = True

        if requires_stream:
            return self._streaming_validation(
                synthetic,
                hid_majority,
                minority,
                metric=metric,
                metric_kwargs=metric_kwargs,
                return_details=return_details,
                reference=reference,
                fit_minority=fit_minority,
            )

        dist_hidden = self.distance_computer.compute_distance_matrix(
            synthetic,
            hid_majority,
            metric=metric,
            batch_size=self.batch_size,
            **metric_kwargs,
        )
        dist_min = self.distance_computer.compute_distance_matrix(
            synthetic,
            minority,
            metric=metric,
            batch_size=self.batch_size,
            **metric_kwargs,
        )

        nearest_hidden = (
            dist_hidden.min(axis=1) if dist_hidden.size else np.full(len(synthetic), np.inf)
        )
        nearest_min = (
            dist_min.min(axis=1) if dist_min.size else np.full(len(synthetic), np.inf)
        )
        errors, n_ties = score_nearest_distances(nearest_hidden, nearest_min)
        rate = calculate_error_rate(errors, len(synthetic))

        if return_details:
            return ValidationDetails(
                error_rate=rate,
                n_errors=errors,
                n_synthetic=len(synthetic),
                n_ties=n_ties,
                duplication_rate=duplication_rate(synthetic, fit_minority),
                reference=reference,
                dist_hidden=dist_hidden,
                dist_min=dist_min,
            )

        if params_hash is not None:
            self.cache.cache_validation_result(params_hash, rate)
        return rate

    def _streaming_validation(
        self,
        synthetic: NDArray[np.floating],
        hidden_majority: NDArray[np.floating],
        minority: NDArray[np.floating],
        metric: str,
        metric_kwargs: dict[str, Any],
        return_details: bool,
        reference: ReferenceSet = "hidden_minority",
        fit_minority: NDArray[np.floating] | None = None,
    ) -> float | ValidationDetails:
        """Compute validation statistics using chunked distance matrices.

        Args:
            synthetic: Synthetic samples.
            hidden_majority: Hidden majority samples.
            minority: Minority reference samples.
            metric: Distance metric name.
            metric_kwargs: Metric keyword arguments.
            return_details: Whether to return distance matrices.
            reference: Which minority set was used.
            fit_minority: Minority points the oversampler trained on, for the
                duplication diagnostic.

        Returns:
            Error rate, or ``ValidationDetails`` when ``return_details`` is True.
        """
        dtype = np.result_type(synthetic.dtype, minority.dtype, np.float64)
        n_syn = len(synthetic)
        n_hidden = len(hidden_majority)
        n_minority = len(minority)

        chunk_cols = max(1, n_hidden, n_minority)
        chunk_size = min(n_syn, self._stream_chunk_size(chunk_cols, dtype=dtype))
        errors = 0
        n_ties = 0

        hidden_store: NDArray[np.floating] | None
        min_store: NDArray[np.floating] | None
        hidden_store = None
        min_store = None

        if return_details:
            temp_dir = Path(tempfile.mkdtemp(prefix="oversampleqa_stream_", dir=self.temp_root))
            self._stream_dirs.append(temp_dir)
            if n_hidden > 0:
                hidden_store = np.memmap(
                    temp_dir / "hidden.dat",
                    dtype=dtype,
                    mode="w+",
                    shape=(n_syn, n_hidden),
                )
            else:
                hidden_store = np.empty((n_syn, 0), dtype=dtype)
            if n_minority > 0:
                min_store = np.memmap(
                    temp_dir / "minority.dat",
                    dtype=dtype,
                    mode="w+",
                    shape=(n_syn, n_minority),
                )
            else:
                min_store = np.empty((n_syn, 0), dtype=dtype)

        for start in range(0, n_syn, chunk_size):
            end = min(start + chunk_size, n_syn)
            chunk = synthetic[start:end]

            if n_hidden > 0:
                dist_hidden = self.distance_computer.compute_distance_matrix(
                    chunk,
                    hidden_majority,
                    metric=metric,
                    batch_size=self.batch_size,
                    **metric_kwargs,
                )
                if return_details and hidden_store is not None:
                    hidden_store[start:end] = dist_hidden
                nearest_hidden = dist_hidden.min(axis=1)
            else:
                nearest_hidden = np.full(len(chunk), np.inf)

            if n_minority > 0:
                dist_min = self.distance_computer.compute_distance_matrix(
                    chunk,
                    minority,
                    metric=metric,
                    batch_size=self.batch_size,
                    **metric_kwargs,
                )
                if return_details and min_store is not None:
                    min_store[start:end] = dist_min
                nearest_min = dist_min.min(axis=1)
            else:
                nearest_min = np.full(len(chunk), np.inf)

            chunk_errors, chunk_ties = score_nearest_distances(
                nearest_hidden, nearest_min
            )
            errors += chunk_errors
            n_ties += chunk_ties

        rate = calculate_error_rate(errors, n_syn)

        if return_details:
            if isinstance(hidden_store, np.memmap):
                hidden_store.flush()
            if isinstance(min_store, np.memmap):
                min_store.flush()
            hidden_return = hidden_store if hidden_store is not None else np.empty((n_syn, 0), dtype=dtype)
            min_return = min_store if min_store is not None else np.empty((n_syn, 0), dtype=dtype)
            dup = (
                duplication_rate(synthetic, fit_minority)
                if fit_minority is not None
                else float("nan")
            )
            return ValidationDetails(
                error_rate=rate,
                n_errors=errors,
                n_synthetic=n_syn,
                n_ties=n_ties,
                duplication_rate=dup,
                reference=reference,
                dist_hidden=hidden_return,
                dist_min=min_return,
            )

        return rate

    def _stream_chunk_size(self, n_cols: int, dtype: np.dtype) -> int:
        """Return streaming chunk size based on memory limit.

        Args:
            n_cols: Number of columns in distance matrix.
            dtype: Data type of the distance matrix.

        Returns:
            Maximum number of rows to process per chunk.
        """
        limit_bytes = int(self.memory_limit_gb * (1024**3))
        per_row = max(1, n_cols * np.dtype(dtype).itemsize)
        return max(1, limit_bytes // per_row)

    def cleanup(self) -> None:
        """Remove temporary files created during streaming computations.

        This cleans any memmap-backed temporary directories created during
        streaming validation.
        """
        for path in self._stream_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self._stream_dirs.clear()
