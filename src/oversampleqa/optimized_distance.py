"""Optimised distance matrix computation utilities."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .caching import ValidationCache

try:  # pragma: no cover - psutil is optional
    import psutil
except ImportError:  # pragma: no cover - fallback if psutil is unavailable
    psutil = None

try:  # pragma: no cover - tqdm is optional
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm is unavailable
    tqdm = None

# Conservative assumption used when psutil is not installed, so memory-aware
# batching still picks a safe (smaller) batch size rather than failing.
_DEFAULT_AVAILABLE_MEMORY_GB = 1.0

# Peak allocation of each kernel, as multiples of the (n1, n2) output array:
#
#     peak ~= output * (flat + per_feature * d)
#
# Two terms are needed because the kernels split into two families. Some hold
# only (n1, n2)-shaped temporaries and do not scale with the feature dimension
# at all -- ``euclidean`` goes through a BLAS gram trick and peaks at ~3x the
# output regardless of d. Others broadcast to (n1, n2, d) and scale linearly:
# ``hassanat`` peaks at ~96x the output at d=16. A single multiplier cannot
# express both, and assuming the wrong family is how the batching logic came to
# be bypassed.
#
# These are fitted to measured peaks (tracemalloc, n1=120 n2=150 d=16), not
# counted by eye -- an earlier hand-count was wrong for six of the fourteen
# kernels. Two entries are re-verified on every run in
# tests/test_kernels.py::test_multiplier_table_matches_measured_peak.
#
# Metrics routed through _pairwise build no intermediate beyond the output row.
_PEAK_MODEL: dict[str, tuple[float, float]] = {
    # metric            flat   per_feature       measured peak/output at d=16
    "euclidean": (3.0, 0.0),  # 3.02x
    "cosine": (5.0, 0.0),  # 4.65x
    "correlation": (5.0, 0.0),  # 4.89x
    "hamming": (1.0, 0.25),  # 4.01x  (bool intermediate, 1 byte/elem)
    "jaccard": (1.0, 0.25),  # 4.50x  (two bool intermediates)
    "mahalanobis": (1.0, 1.2),  # 17.92x
    "manhattan": (1.0, 2.0),  # 32.00x
    "chebyshev": (1.0, 2.0),  # 32.00x
    "minkowski": (1.0, 2.0),  # 32.01x
    "braycurtis": (1.0, 2.0),  # 33.00x
    "hellinger": (1.0, 2.2),  # 33.25x
    "jensen_shannon": (1.0, 4.1),  # 64.67x
    "canberra": (1.0, 4.2),  # 66.02x
    "hassanat": (1.0, 6.0),  # 96.00x
    # _pairwise: one row at a time, so only the output array persists.
    "energy": (1.0, 0.0),
    "wasserstein": (1.0, 0.0),
}

# Used for an unregistered (plugin) metric. Assumes a broadcasting kernel with
# several intermediates -- overestimating costs a smaller batch, while
# underestimating risks an allocation the caller did not budget for.
_DEFAULT_PEAK_MODEL = (1.0, 4.0)


def peak_multiple(metric: str, n_features: int) -> float:
    """Return peak allocation as a multiple of the output array.

    Args:
        metric: Metric name.
        n_features: Feature dimension ``d``.

    Returns:
        Multiplier to apply to the ``(n1, n2)`` output size.
    """
    flat, per_feature = _PEAK_MODEL.get(metric, _DEFAULT_PEAK_MODEL)
    return flat + per_feature * max(1, n_features)


DistanceCallable = Callable[[NDArray[np.floating], NDArray[np.floating]], float]

_psutil_warned = False


def get_available_memory_gb() -> float:
    """Return currently available system memory in gigabytes.

    Falls back to a conservative constant when ``psutil`` is not installed.
    That fallback changes batching behaviour, so it is logged once at INFO --
    otherwise performance differs silently depending on whether an optional
    dependency happens to be present.

    Returns:
        Available memory in GB.
    """
    global _psutil_warned
    if psutil is None:
        if not _psutil_warned:
            _psutil_warned = True
            logger.info(
                "psutil is not installed, so available memory cannot be measured. "
                "Assuming %.1f GB, which makes batching more conservative than it "
                "needs to be on a larger machine. Install the 'performance' extra "
                "(pip install 'oversampleqa[performance]') to use the real value.",
                _DEFAULT_AVAILABLE_MEMORY_GB,
            )
        return _DEFAULT_AVAILABLE_MEMORY_GB
    available: float = psutil.virtual_memory().available / (1024**3)
    return available


class OptimizedDistanceMatrix:
    """Memory-aware distance matrix computation with vectorisation and batching.

    .. note::

       The effective memory limit is ``min(memory_limit_gb, available)``, where
       ``available`` comes from ``psutil``. **Without ``psutil`` installed it is
       assumed to be 1 GB**, regardless of the machine, so batching is more
       conservative and throughput differs from an otherwise identical
       environment that has it. The fallback is logged once at INFO. Install the
       ``performance`` extra to get the real figure.

    Parameters
    ----------
    cache_size : int, default=128
        Retained for API compatibility.
    memory_limit_gb : float, default=4.0
        Upper bound on the memory one computation may use.
    metric_registry : dict, optional
        Name-to-callable mapping of metrics.
    show_progress : bool, default=False
        Display a progress bar for large computations.
    progress_threshold : int, default=10000
        Row count above which progress is shown.
    cache : ValidationCache, optional
        Opt-in cache. ``None`` means nothing is written to disk.
    safety_factor : float, default=0.8
        Fraction of the limit a batched computation is allowed to plan against.
        The remainder is headroom for allocator overhead and transient copies,
        which the analytic estimate does not model.
    """

    def __init__(
        self,
        cache_size: int = 128,
        memory_limit_gb: float = 4.0,
        metric_registry: dict[str, DistanceCallable] | None = None,
        show_progress: bool = False,
        progress_threshold: int = 10_000,
        cache: ValidationCache | None = None,
        safety_factor: float = 0.8,
    ) -> None:
        if not 0.0 < safety_factor <= 1.0:
            raise ValueError(f"safety_factor must be in (0, 1]; got {safety_factor!r}")
        self.cache_size = cache_size
        self.memory_limit_gb = memory_limit_gb
        self.metric_registry = metric_registry or {}
        self.show_progress = show_progress
        self.progress_threshold = progress_threshold
        self.cache = cache
        self.safety_factor = safety_factor

        self._vectorized_dispatch: dict[str, Callable[..., NDArray[np.floating]]] = {
            "hassanat": self._vectorized_hassanat,
            "hamming": self._vectorized_hamming,
            "jaccard": self._vectorized_jaccard,
            "hellinger": self._vectorized_hellinger,
            "jensen_shannon": self._vectorized_jensen_shannon,
            # "energy" is deliberately absent. It is a sample-based metric whose
            # scalar form computes three pairwise-norm terms per (i, j) -- the
            # cross term plus a within-term for each input row. Broadcasting that
            # needs an (n1, n2, d, d) intermediate, which is larger than the work
            # it saves at any realistic size, so it stays on _pairwise.
            #
            # "wasserstein" is also absent, for a different reason. A vectorised
            # kernel exists below and is correct -- it uses the equal-length
            # closed form mean|sort(x) - sort(y)|, which agrees with SciPy. The
            # scalar wasserstein_1d_distance does not: its CDF walk drops the
            # tail once either sample is exhausted and advances the CDF before
            # adding each interval, so [0,1] vs [0,3] returns 0.5 where the true
            # W1 is 1.0 (see the strict xfail in tests/test_reference_metrics).
            # Registering the kernel would make the two paths disagree; matching
            # them would mean reproducing the bug. It stays on _pairwise until
            # the scalar is fixed, which is task 07's scope.
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
        batch_size: int | str = "auto",
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
            batch_size=batch_size,
            **kwargs,
        )

    def _compute_uncached(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric: str,
        batch_size: int | str = "auto",
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
        n_features = X1.shape[1] if X1.ndim > 1 else 1
        # Estimate the *peak*, including the (n1, n2, d) intermediates the
        # kernel allocates -- not just the output array. Underestimating here
        # is what let the whole-input path run when it should have batched.
        memory_required = self._estimate_memory_usage(n1, n2, dtype, n_features, metric)
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
                batch_size = self._auto_batch_size(
                    n2,
                    dtype=dtype,
                    n_features=n_features,
                    metric=metric,
                    n_rows=n1,
                )
        elif not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size must be 'auto', 'stream', or a positive integer"
            )

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
        result: NDArray[np.floating] = np.sqrt(distances, out=distances)
        return result

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
        result: NDArray[np.floating] = diff.sum(axis=2)
        return result

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

    def _vectorized_hassanat(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Hassanat distance matrix.

        Mirrors :func:`oversampleqa.distance.hassanat_distance`. The
        denominator is ``1 + mx + shift``, which is always ``>= 1``, so no
        division guard is needed.

        Note: this allocates an ``(n1, n2, d)`` intermediate. Memory
        accounting for the batched paths is handled by the caller.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        mn = np.minimum(X1[:, None, :], X2[None, :, :])
        mx = np.maximum(X1[:, None, :], X2[None, :, :])
        shift = np.where(mn < 0.0, -mn, 0.0)
        ratio = (1.0 + mn + shift) / (1.0 + mx + shift)
        result: NDArray[np.floating] = np.sum(1.0 - ratio, axis=-1)
        return result

    def _vectorized_hamming(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Hamming distance matrix.

        Matches the scalar form, which returns the raw **count** of differing
        components rather than SciPy's fraction.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        differing = X1[:, None, :] != X2[None, :, :]
        result: NDArray[np.floating] = differing.sum(axis=-1).astype(float)
        return result

    def _vectorized_jaccard(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Jaccard distance matrix.

        The scalar form casts to ``bool`` and computes set Jaccard, not the
        weighted Ruzicka variant, so this does the same. A pair whose union is
        empty is defined as distance 0.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        b1 = X1.astype(bool)[:, None, :]
        b2 = X2.astype(bool)[None, :, :]
        intersection = np.logical_and(b1, b2).sum(axis=-1)
        union = np.logical_or(b1, b2).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(union == 0, 1.0, intersection / union)
        result: NDArray[np.floating] = 1.0 - similarity
        return result

    @staticmethod
    def _normalise_rows(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Scale each row to sum to 1, leaving all-zero rows as zeros.

        Mirrors the scalar probability metrics, which divide by the sum unless
        it is zero.
        """
        totals = X.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(totals == 0, 0.0, X / totals)

    def _vectorized_hellinger(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Hellinger distance matrix.

        Rows are normalised once each rather than per pair, which is where the
        saving comes from.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.

        Raises:
            ValueError: If either input contains negative values.
        """
        if np.any(X1 < 0) or np.any(X2 < 0):
            raise ValueError("Hellinger distance requires non-negative inputs")
        root_p = np.sqrt(self._normalise_rows(X1))
        root_q = np.sqrt(self._normalise_rows(X2))
        diff = root_p[:, None, :] - root_q[None, :, :]
        result: NDArray[np.floating] = np.sqrt((diff**2).sum(axis=-1)) / np.sqrt(2.0)
        return result

    def _vectorized_jensen_shannon(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized Jensen-Shannon distance matrix.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.

        Raises:
            ValueError: If either input contains negative values.
        """
        if np.any(X1 < 0) or np.any(X2 < 0):
            raise ValueError("Jensen-Shannon distance requires non-negative inputs")
        p = self._normalise_rows(X1)[:, None, :]
        q = self._normalise_rows(X2)[None, :, :]
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            term_p = np.where(p == 0, 0.0, p * np.log(p / m))
            term_q = np.where(q == 0, 0.0, q * np.log(q / m))
        divergence = 0.5 * (term_p.sum(axis=-1) + term_q.sum(axis=-1))
        result: NDArray[np.floating] = np.sqrt(np.clip(divergence, 0.0, None))
        return result

    def _vectorized_wasserstein(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        **_: Any,
    ) -> NDArray[np.floating]:
        """Vectorized 1-D Wasserstein distance matrix.

        Sample-based, like ``energy``: each row is a set of observations. The
        sort each pair needs is hoisted out of the pair loop -- both inputs are
        sorted once, then broadcast -- which is where the win comes from.

        Only valid when both inputs have the same number of columns, which the
        equal-length closed form ``mean|sort(x) - sort(y)|`` requires. The
        caller guarantees this: distance matrices are computed between matrices
        with matching feature counts.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Distance matrix.
        """
        sorted_1 = np.sort(X1, axis=1)
        sorted_2 = np.sort(X2, axis=1)
        diff = np.abs(sorted_1[:, None, :] - sorted_2[None, :, :])
        result: NDArray[np.floating] = diff.mean(axis=-1)
        return result

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
        result: NDArray[np.floating] = diff.max(axis=2)
        return result

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
        result: NDArray[np.floating] = ratio.sum(axis=2)
        return result

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
        result: NDArray[np.floating] = 1.0 - corr
        return result

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
        result: NDArray[np.floating] = np.sum(diff, axis=2) ** (1.0 / p)
        return result

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
        result: NDArray[np.floating] = np.sqrt(res, out=res)
        return result

    def _batched_computation(
        self,
        X1: NDArray[np.floating],
        X2: NDArray[np.floating],
        metric: str,
        batch_size: int,
        vectorized: Callable[..., NDArray[np.floating]] | None = None,
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
        if not self.show_progress or tqdm is None or total < self.progress_threshold:
            return iterable
        wrapped: Iterable[int] = tqdm(  # pragma: no cover - requires tqdm
            iterable, total=math.ceil(total)
        )
        return wrapped

    def _auto_batch_size(
        self,
        n_cols: int,
        dtype: np.dtype,
        n_features: int = 1,
        metric: str = "",
        n_rows: int = 0,
    ) -> int:
        """Estimate a safe batch size under the memory limit.

        Reserves the accumulating result array before dividing what remains
        into batches, and scales a batch's cost by the metric's intermediate
        multiplier. The previous version allowed every batch to consume the
        entire limit, leaving no headroom for the ``(n1, n2)`` result that lives
        for the whole computation, nor for the ``(batch, n2, d)`` intermediate a
        broadcasting kernel allocates.

        Args:
            n_cols: Number of columns in the distance matrix.
            dtype: Data type of the distance matrix.
            n_features: Feature dimension ``d``.
            metric: Metric name, used to look up the intermediate multiplier.
            n_rows: Total rows, used to reserve the result array.

        Returns:
            Batch size in rows.
        """
        itemsize = np.dtype(dtype).itemsize
        limit_bytes = int(self.memory_limit_gb * (1024**3) * self.safety_factor)

        # The full result array outlives every batch, so subtract it first.
        result_bytes = n_rows * n_cols * itemsize if n_rows else 0
        usable = max(itemsize, limit_bytes - result_bytes)

        # A batch row costs its slice of the output times the kernel's peak
        # multiple, which already includes the output itself.
        row_bytes = max(1, int(n_cols * itemsize * peak_multiple(metric, n_features)))
        return max(1, usable // row_bytes)

    def _estimate_memory_usage(
        self,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype,
        n_features: int = 1,
        metric: str = "",
    ) -> float:
        """Estimate peak memory usage (GB) for a distance computation.

        The output array is ``(n_rows, n_cols)``, but a broadcasting kernel
        also allocates one or more ``(n_rows, n_cols, n_features)``
        intermediates -- so peak use is roughly ``n_features`` times the output,
        multiplied again by how many intermediates the kernel holds at once.
        Ignoring that was how the batching logic got bypassed: the check passed,
        then the kernel allocated far more than the check had permitted.

        Args:
            n_rows: Number of rows.
            n_cols: Number of columns.
            dtype: Data type of the distance matrix.
            n_features: Feature dimension ``d``.
            metric: Metric name; selects the multiplier.

        Returns:
            Estimated peak memory usage in gigabytes.
        """
        itemsize = np.dtype(dtype).itemsize
        result_bytes = n_rows * n_cols * itemsize
        overhead_bytes = (n_rows + n_cols) * itemsize
        peak_bytes = result_bytes * peak_multiple(metric, n_features)
        return (peak_bytes + overhead_bytes) / (1024**3)

    def estimate_memory_gb(
        self,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype | None = None,
        n_features: int = 1,
        metric: str = "",
    ) -> float:
        """Public helper returning estimated peak footprint of a distance matrix.

        Args:
            n_rows: Number of rows.
            n_cols: Number of columns.
            dtype: Data type of the distance matrix.
            n_features: Feature dimension.
            metric: Metric name; selects the intermediate multiplier.

        Returns:
            Estimated memory usage in gigabytes.
        """
        dtype = dtype or np.dtype(np.float64)
        return self._estimate_memory_usage(n_rows, n_cols, dtype, n_features, metric)
