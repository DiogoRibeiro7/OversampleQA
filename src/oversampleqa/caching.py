"""Caching utilities for oversampleqa."""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import joblib
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .optimized_distance import OptimizedDistanceMatrix

logger = logging.getLogger(__name__)


def default_cache_dir() -> Path:
    """Return the per-user cache directory for OversampleQA.

    Uses ``platformdirs`` when available so the cache lands in the platform's
    conventional location rather than the current working directory. Falls back
    to ``~/.cache/oversampleqa``.

    The directory is **not** created here; see :class:`ValidationCache`.
    """
    try:
        from platformdirs import user_cache_dir
    except ImportError:  # pragma: no cover - optional dependency
        return Path.home() / ".cache" / "oversampleqa"
    return Path(user_cache_dir("oversampleqa"))


class ValidationCache:
    """Caching layer for validation results and distance computations.

    Caching is **opt-in**. Constructing this class is the caller's decision;
    nothing in the package builds one at import time, and no directory is
    created until the first write.

    .. warning::

       **Not thread-safe across instances, and not process-safe.** A single
       instance guards its own in-memory bookkeeping with a lock, so concurrent
       reads and writes through one instance will not corrupt its accounting.
       ``joblib`` on-disk writes are *not* atomic, so two processes (or two
       instances pointed at the same directory) writing the same key can
       interleave and leave a truncated file. Give each process its own
       ``cache_dir``.

    .. note::

       Whether caching pays depends entirely on how expensive the metric is
       relative to hashing its inputs. Content hashing must read every input
       byte, so for a BLAS-backed metric such as ``euclidean`` the cache is a
       net loss; for ``hassanat`` it is worth tens of times the compute. See
       :doc:`/reproducibility`.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Where to store cached artefacts. Defaults to the per-user cache
        directory, never the working directory.
    max_entries : int, default=128
        Upper bound on in-memory distance matrices. Least-recently-used entries
        are evicted first.
    memory_mb : int, default=1000
        Upper bound on the in-memory tier, in megabytes. Enforced: entries are
        evicted oldest-first until the total fits.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        memory_mb: int = 1000,
        max_entries: int = 128,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.bytes_limit = memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self._memory: joblib.Memory | None = None
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._nbytes = 0

    def _ensure_dir(self) -> None:
        """Create the cache directory. Called on first write, never on import."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def memory(self) -> joblib.Memory:
        """Lazily-created joblib store; creates the directory on first use."""
        if self._memory is None:
            self._ensure_dir()
            self._memory = joblib.Memory(self.cache_dir, verbose=0)
        return self._memory

    @property
    def size_bytes(self) -> int:
        """Bytes currently held by the in-memory tier."""
        with self._lock:
            return self._nbytes

    def clear(self) -> None:
        """Drop everything held in memory. Does not touch the disk store."""
        with self._lock:
            self._store.clear()
            self._nbytes = 0

    def get_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Return stable SHA256 hash for dataset.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            SHA256 hex digest.
        """
        hasher = hashlib.sha256()
        self._update_hasher(hasher, X)
        self._update_hasher(hasher, y)
        return hasher.hexdigest()

    def cache_validation_result(self, params_hash: str, result: float) -> None:
        """Persist validation result using joblib.

        Args:
            params_hash: Cache key for the run parameters.
            result: Error rate to persist.
        """
        self._ensure_dir()
        path = self.cache_dir / f"validation_{params_hash}.pkl"
        joblib.dump(result, path)

    def load_validation_result(self, params_hash: str) -> float | None:
        """Retrieve cached validation result if present.

        Args:
            params_hash: Cache key for the run parameters.

        Returns:
            Cached error rate if available.
        """
        path = self.cache_dir / f"validation_{params_hash}.pkl"
        if path.exists():
            return joblib.load(path)
        return None

    def cached_distance_matrix(
        self,
        optimizer: "OptimizedDistanceMatrix",
        X1: np.ndarray,
        X2: np.ndarray,
        metric: str,
        batch_size: int | str = "auto",
        **kwargs: Any,
    ) -> np.ndarray:
        """Return cached distance matrix or compute and cache it.

        The returned array is **read-only**. Cache hits hand back the stored
        array rather than a copy, so an in-place operation downstream would
        otherwise corrupt every later hit silently; the write flag turns that
        into a loud ``ValueError`` instead. Call ``.copy()`` if you need to
        modify it.

        ``batch_size`` is deliberately **not** part of the key: batching splits
        the same computation into chunks and concatenates them, so it cannot
        change the result. ``test_caching.py`` pins that invariant for every
        registered metric.

        Args:
            optimizer: OptimizedDistanceMatrix instance. Used only to compute a
                miss -- it is never part of the cache key.
            X1: First feature matrix.
            X2: Second feature matrix.
            metric: Distance metric name.
            batch_size: Batch size or mode.
            **kwargs: Metric keyword arguments.

        Returns:
            Read-only distance matrix.
        """
        key = self._distance_key(X1, X2, metric, kwargs)

        with self._lock:
            hit = self._store.get(key)
            if hit is not None:
                self._store.move_to_end(key)
                return hit

        result = optimizer._compute_uncached(  # type: ignore[attr-defined]
            X1, X2, metric=metric, batch_size=batch_size, **kwargs
        )
        result.setflags(write=False)
        self._remember(key, result)
        return result

    def _remember(self, key: str, arr: np.ndarray) -> None:
        """Store ``arr`` under ``key``, evicting until the limits are met."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return
            self._store[key] = arr
            self._nbytes += arr.nbytes
            while self._store and (
                self._nbytes > self.bytes_limit or len(self._store) > self.max_entries
            ):
                _, evicted = self._store.popitem(last=False)
                self._nbytes -= evicted.nbytes
                logger.debug(
                    "Evicted a %d-byte distance matrix; %d bytes still cached",
                    evicted.nbytes,
                    self._nbytes,
                )

    def _distance_key(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        metric: str,
        kwargs: Dict[str, Any],
    ) -> str:
        """Return a stable key for distance matrix caching.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.
            metric: Distance metric name.
            kwargs: Metric keyword arguments.

        Returns:
            Cache key as a hex digest.
        """
        hasher = hashlib.sha256()
        self._update_hasher(hasher, X1)
        self._update_hasher(hasher, X2)
        hasher.update(metric.encode("utf-8"))
        if kwargs:
            serialized = pickle.dumps(sorted(kwargs.items(), key=lambda item: item[0]))
            hasher.update(serialized)
        return hasher.hexdigest()

    @staticmethod
    def _update_hasher(hasher: "hashlib._Hash", arr: np.ndarray) -> None:
        """Update the hasher with array shape, dtype, and data bytes.

        Args:
            hasher: Hash object to update.
            arr: Array to serialize into the hash.
        """
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(arr.tobytes(order="C"))
