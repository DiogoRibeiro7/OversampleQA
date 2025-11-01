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
from .metrics import calculate_error_rate
from .optimized_distance import OptimizedDistanceMatrix, get_available_memory_gb
from .validator import extract_synthetic_samples
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
    ) -> float | tuple[float, int, NDArray[np.floating], NDArray[np.floating]]:
        """Validate oversampling with streaming-aware distance calculations."""
        X = np.asarray(X)
        y = np.asarray(y)
        metric_kwargs = metric_kwargs or {}

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
            }
            params_hash = hashlib.sha256(pickle.dumps(payload)).hexdigest()
            cached = self.cache.load_validation_result(params_hash)
            if cached is not None:
                return cached

        minority_mask = y == minority_label
        minority = X[minority_mask]
        majority = X[~minority_mask]

        if len(majority) == 0:
            if return_details:
                empty = np.empty((0, 0))
                return 0.0, 0, empty, empty
            return 0.0

        vis_majority, hid_majority = train_test_split(
            majority, test_size=hidden_ratio, random_state=42
        )

        X_train = np.vstack([vis_majority, minority])
        y_train = np.hstack(
            [
                np.zeros(len(vis_majority), dtype=int),
                np.full(len(minority), minority_label, dtype=int),
            ]
        )

        X_res, y_res = oversampler.fit_resample(X_train, y_train)
        synthetic = extract_synthetic_samples(X_train, X_res, y_res, minority_label)

        if len(synthetic) == 0:
            if return_details:
                empty = np.empty((0, 0))
                return 0.0, 0, empty, empty
            return 0.0

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
        errors = int(np.sum(nearest_hidden <= nearest_min))
        rate = calculate_error_rate(errors, len(synthetic))

        if return_details:
            return rate, errors, dist_hidden, dist_min

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
    ) -> float | tuple[float, int, NDArray[np.floating], NDArray[np.floating]]:
        dtype = np.result_type(synthetic.dtype, minority.dtype, np.float64)
        n_syn = len(synthetic)
        n_hidden = len(hidden_majority)
        n_minority = len(minority)

        chunk_cols = max(1, n_hidden, n_minority)
        chunk_size = min(n_syn, self._stream_chunk_size(chunk_cols, dtype=dtype))
        errors = 0

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

            errors += int(np.sum(nearest_hidden <= nearest_min))

        rate = calculate_error_rate(errors, n_syn)

        if return_details:
            if isinstance(hidden_store, np.memmap):
                hidden_store.flush()
            if isinstance(min_store, np.memmap):
                min_store.flush()
            hidden_return = hidden_store if hidden_store is not None else np.empty((n_syn, 0), dtype=dtype)
            min_return = min_store if min_store is not None else np.empty((n_syn, 0), dtype=dtype)
            return rate, errors, hidden_return, min_return

        return rate

    def _stream_chunk_size(self, n_cols: int, dtype: np.dtype) -> int:
        limit_bytes = int(self.memory_limit_gb * (1024**3))
        per_row = max(1, n_cols * np.dtype(dtype).itemsize)
        return max(1, limit_bytes // per_row)

    def cleanup(self) -> None:
        """Remove temporary files created during streaming computations."""
        for path in self._stream_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self._stream_dirs.clear()
