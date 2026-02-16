"""Optimised distance matrix computation utilities."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Union, TYPE_CHECKING

import numpy as np
import psutil
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .caching import ValidationCache

try:  # pragma: no cover - tqdm is optional
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm is unavailable
    tqdm = None

DistanceCallable = Callable[[NDArray[np.floating], NDArray[np.floating]], float]


def get_available_memory_gb() -> float:
    """Return currently available system memory in gigabytes.

    Returns:
        Available memory in GB.
    """
    return psutil.virtual_memory().available / (1024**3)


class OptimizedDistanceMatrix:
    """Memory-aware distance matrix computation with vectorisation and batching."""

    def __init__(
        self,
        cache_size: int = 128,
        memory_limit_gb: float = 4.0,
        metric_registry: Optional[Dict[str, DistanceCallable]] = None,
        show_progress: bool = False,
        progress_threshold: int = 10_000,
        cache: "ValidationCache | None" = None,
    ) -> None:
        self.cache_size = cache_size
        self.memory_limit_gb = memory_limit_gb
        self.metric_registry = metric_registry or {}
        self.show_progress = show_progress
        self.progress_threshold = progress_threshold
        self.cache = cache

        self._vectorized_dispatch: Dict[str, Callable[..., NDArray[np.floating]]] = {
            "euclidean": self._vectorized_euclidean,
            "manhattan": self._vectorized_manhattan,
            "cosine": self._vectorized_cosine,
            "chebyshev": self._vectorized_chebyshev,
            "canberra": self._vectorized_canberra,
            "braycurtis": self._vectorized_braycurtis,
            "correlation": self._vectorized_correlation,
            "minkowski": self._vectorized_minkowski,
            "mahalanobis": self._vectorized_mahalanobis,
        }

    def compute_distance_matrix(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric: str = "hassanat",
        batch_size: Union[int, str] = "auto",
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Compute pairwise distances with automatic optimisation.

        Parameters
        ----------
        X1, X2:
            Input matrices of shape ``(n_samples, n_features)``.
        metric:
            Name of the distance metric registered in ``metric_registry``.
        batch_size:
            ``"auto"`` selects the largest batch size fitting within
            ``memory_limit_gb``. An integer enforces a specific chunk length.
            ``"stream"`` yields rows sequentially without storing the full
            matrix in memory.
        kwargs:
            Extra keyword arguments forwarded to the underlying metric.
        """
        if metric not in self.metric_registry:
            raise ValueError(f"Unsupported metric '{metric}'")

        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        n1, n2 = X1.shape[0], X2.shape[0]
        dtype = np.result_type(X1.dtype, X2.dtype, np.float64)
        X1 = X1.astype(dtype, copy=False)
        X2 = X2.astype(dtype, copy=False)

        if n1 == 0 or n2 == 0:
            return np.empty((n1, n2), dtype=dtype)

        if isinstance(batch_size, str):
            batch_key = batch_size.lower()
        else:
            batch_key = ""

        if self.cache is not None and batch_key != "stream":
            return self.cache.cached_distance_matrix(
                optimizer=self,
                X1=X1,
                X2=X2,
                metric=metric,
                batch_size=batch_size,
                **kwargs,
            )

        return self._compute_uncached(
            X1,
            X2,
            metric=metric,
            **kwargs,
        )

    def _compute_uncached(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric: str,
        batch_size: Union[int, str] = "auto",
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Compute distances without using the cache.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            metric: Distance metric name.
            batch_size: Batch size or mode.
            **kwargs: Metric keyword arguments.

        Returns:
            Distance matrix.
        """
        vectorized = self._vectorized_dispatch.get(metric)
        n1, n2 = len(X1), len(X2)
        dtype = X1.dtype
        memory_required = self._estimate_memory_usage(n1, n2, dtype)
        available_memory = min(self.memory_limit_gb, get_available_memory_gb())

        if isinstance(batch_size, str):
            batch_key = batch_size.lower()
        else:
            batch_key = ""

        if batch_key == "stream":
            return self._streaming_computation(X1, X2, metric, **kwargs)

        if batch_size == "auto":
            if memory_required <= available_memory:
                if vectorized is not None:
                    return vectorized(X1, X2, **kwargs)
                batch_size = len(X1)
            else:
                batch_size = self._auto_batch_size(n2, dtype=dtype)
        elif not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be 'auto', 'stream', or a positive integer")

        batch_size = min(int(batch_size), max(1, n1))

        if vectorized is not None and batch_size >= n1:
            return vectorized(X1, X2, **kwargs)

        return self._batched_computation(
            X1,
            X2,
            metric=metric,
            batch_size=int(batch_size),
            vectorized=vectorized,
            **kwargs,
        )

    def _vectorized_euclidean(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Euclidean distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        x1_norm = np.einsum("ij,ij->i", X1, X1)
        x2_norm = np.einsum("ij,ij->i", X2, X2)
        distances = x1_norm[:, None] + x2_norm[None, :] - 2.0 * (X1 @ X2.T)
        np.maximum(distances, 0.0, out=distances)
        return np.sqrt(distances, out=distances)

    def _vectorized_manhattan(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Manhattan distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        return diff.sum(axis=2)

    def _vectorized_cosine(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized cosine distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        dot = X1 @ X2.T
        norm1 = np.linalg.norm(X1, axis=1)
        norm2 = np.linalg.norm(X2, axis=1)
        denom = norm1[:, None] * norm2[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            res = 1.0 - np.where(denom == 0, 0.0, dot / denom)
        return np.nan_to_num(res)

    def _vectorized_chebyshev(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Chebyshev distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        return diff.max(axis=2)

    def _vectorized_canberra(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Canberra distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        numerator = np.abs(X1[:, None, :] - X2[None, :, :])
        denominator = np.abs(X1[:, None, :]) + np.abs(X2[None, :, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denominator == 0, 0.0, numerator / denominator)
        return ratio.sum(axis=2)

    def _vectorized_braycurtis(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Bray-Curtis distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        num = np.abs(X1[:, None, :] - X2[None, :, :]).sum(axis=2)
        denom = np.abs(X1[:, None, :] + X2[None, :, :]).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(denom == 0, 0.0, num / denom)
        return res

    def _vectorized_correlation(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized correlation distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        X1_c = X1 - X1.mean(axis=1, keepdims=True)
        X2_c = X2 - X2.mean(axis=1, keepdims=True)
        dot = X1_c @ X2_c.T
        norm1 = np.linalg.norm(X1_c, axis=1)
        norm2 = np.linalg.norm(X2_c, axis=1)
        denom = norm1[:, None] * norm2[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom == 0, 0.0, dot / denom)
        corr = np.nan_to_num(corr)
        return 1.0 - corr

    def _vectorized_minkowski(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Minkowski distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            **kwargs: Metric keyword arguments (e.g., ``p``).

        Returns:
            Distance matrix.
        """
        p = kwargs.get("p", 3.0)
        diff = np.abs(X1[:, None, :] - X2[None, :, :]) ** p
        return np.sum(diff, axis=2) ** (1.0 / p)

    def _vectorized_mahalanobis(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Mahalanobis distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            **kwargs: Metric keyword arguments (e.g., ``cov_inv``).

        Returns:
            Distance matrix.
        """
        cov_inv = kwargs.get("cov_inv")
        if cov_inv is None:
            return self._vectorized_euclidean(X1, X2)
        diff = X1[:, None, :] - X2[None, :, :]
        res = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        np.maximum(res, 0.0, out=res)
        return np.sqrt(res, out=res)

    def _batched_computation(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric: str,
        batch_size: int,
        vectorized: Optional[Callable[..., NDArray[np.floating]]] = None,
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Compute distances in batches to limit memory usage.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            metric: Distance metric name.
            batch_size: Rows per batch.
            vectorized: Optional vectorized kernel.
            **kwargs: Metric keyword arguments.

        Returns:
            Distance matrix.
        """
        result = np.empty((len(X1), len(X2)), dtype=X1.dtype)
        iterator = range(0, len(X1), batch_size)
        iterator = self._progress(iterator, total=len(X1))  # type: ignore[assignment]
        metric_func = self.metric_registry[metric]

        for start in iterator:
            end = min(start + batch_size, len(X1))
            chunk = X1[start:end]
            if vectorized is not None:
                result[start:end] = vectorized(chunk, X2, **kwargs)
            else:
                result[start:end] = self._pairwise(chunk, X2, metric_func, **kwargs)
        return result

    def _streaming_computation(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric: str,
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Compute distances row-by-row to minimize memory usage.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            metric: Distance metric name.
            **kwargs: Metric keyword arguments.

        Returns:
            Distance matrix.
        """
        metric_func = self.metric_registry[metric]
        result = np.empty((len(X1), len(X2)), dtype=X1.dtype)
        iterator = self._progress(range(len(X1)), total=len(X1))
        for idx in iterator:
            row = self._pairwise(X1[idx : idx + 1], X2, metric_func, **kwargs)
            result[idx] = row[0]
        return result

    def _pairwise(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric_func: DistanceCallable,
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Compute pairwise distances using a Python loop.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            metric_func: Metric callable.
            **kwargs: Metric keyword arguments.

        Returns:
            Distance matrix.
        """
        dm = np.empty((len(X1), len(X2)), dtype=X1.dtype)
        for i, u in enumerate(X1):
            for j, v in enumerate(X2):
                dm[i, j] = metric_func(u, v, **kwargs)
        return dm

    def _progress(self, iterable: Iterable[int], total: int) -> Iterable[int]:
        """Wrap an iterable with a progress bar if enabled.

        Args:
            iterable: Base iterator.
            total: Total size for progress display.

        Returns:
            Iterator wrapped with tqdm when enabled.
        """
        if (
            not self.show_progress
            or tqdm is None
            or total < self.progress_threshold
        ):
            return iterable
        return tqdm(iterable, total=math.ceil(total))  # pragma: no cover - requires tqdm

    def _auto_batch_size(self, n_cols: int, dtype: np.dtype) -> int:
        """Estimate a safe batch size under the memory limit.

        Args:
            n_cols: Number of columns in distance matrix.
            dtype: Data type of the distance matrix.

        Returns:
            Batch size in rows.
        """
        limit_bytes = int(self.memory_limit_gb * (1024**3))
        row_bytes = max(1, n_cols * np.dtype(dtype).itemsize)
        batch = max(1, limit_bytes // max(row_bytes, 1))
        return batch

    def _estimate_memory_usage(
        self,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype,
    ) -> float:
        """Estimate memory usage (GB) for a dense distance matrix.

        Args:
            n_rows: Number of rows.
            n_cols: Number of columns.
            dtype: Data type of the distance matrix.

        Returns:
            Estimated memory usage in gigabytes.
        """
        itemsize = np.dtype(dtype).itemsize
        result_bytes = n_rows * n_cols * itemsize
        overhead_bytes = (n_rows + n_cols) * itemsize
        return (result_bytes + overhead_bytes) / (1024**3)

    def estimate_memory_gb(
        self,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype | None = None,
    ) -> float:
        """Public helper returning estimated memory footprint of a distance matrix.

        Args:
            n_rows: Number of rows.
            n_cols: Number of columns.
            dtype: Data type of the distance matrix.

        Returns:
            Estimated memory usage in gigabytes.
        """
        dtype = dtype or np.dtype(np.float64)
        return self._estimate_memory_usage(n_rows, n_cols, dtype)
