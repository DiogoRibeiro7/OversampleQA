"""Caching utilities for oversampleqa."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple
from functools import lru_cache

import joblib
import numpy as np

from .optimized_distance import OptimizedDistanceMatrix


def _distance_disk(
    X1: np.ndarray,
    X2: np.ndarray,
    metric: str,
    optimizer: OptimizedDistanceMatrix,
    batch_size: int | str,
    kwargs_tuple: Tuple[Tuple[str, Any], ...],
) -> np.ndarray:
    kwargs = dict(kwargs_tuple)
    return optimizer._compute_uncached(  # type: ignore[attr-defined]
        X1,
        X2,
        metric=metric,
        batch_size=batch_size,
        **kwargs,
    )


class ValidationCache:
    """Caching layer for validation results and distance computations."""

    def __init__(self, cache_dir: str = ".oversampleqa_cache", memory_mb: int = 1000) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = joblib.Memory(self.cache_dir, verbose=0)
        self.bytes_limit = memory_mb * 1024 * 1024
        self._distance_args: Dict[
            str,
            Tuple[
                np.ndarray,
                np.ndarray,
                str,
                OptimizedDistanceMatrix,
                int | str,
                Tuple[Tuple[str, Any], ...],
            ],
        ] = {}
        self._disk_distance = self.memory.cache(_distance_disk)

    def get_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Return stable SHA256 hash for dataset."""
        hasher = hashlib.sha256()
        self._update_hasher(hasher, X)
        self._update_hasher(hasher, y)
        return hasher.hexdigest()

    def cache_validation_result(self, params_hash: str, result: float) -> None:
        """Persist validation result using joblib."""
        path = self.cache_dir / f"validation_{params_hash}.pkl"
        joblib.dump(result, path)

    def load_validation_result(self, params_hash: str) -> float | None:
        """Retrieve cached validation result if present."""
        path = self.cache_dir / f"validation_{params_hash}.pkl"
        if path.exists():
            return joblib.load(path)
        return None

    def cached_distance_matrix(
        self,
        optimizer: OptimizedDistanceMatrix,
        X1: np.ndarray,
        X2: np.ndarray,
        metric: str,
        batch_size: int | str = "auto",
        **kwargs: Any,
    ) -> np.ndarray:
        """Return cached distance matrix or compute and cache it."""
        key = self._distance_key(X1, X2, metric, kwargs)
        kwargs_tuple = tuple(sorted(kwargs.items(), key=lambda item: item[0]))
        self._distance_args[key] = (X1, X2, metric, optimizer, batch_size, kwargs_tuple)
        result = self._lru_distance(key)
        self._distance_args.pop(key, None)
        return result

    @lru_cache(maxsize=128)
    def _lru_distance(self, key: str) -> np.ndarray:
        args = self._distance_args.get(key)
        if args is None:
            raise ValueError(f"No arguments registered for key {key}")

        X1, X2, metric, optimizer, batch_size, kwargs_tuple = args
        try:
            return self._disk_distance(X1, X2, metric, optimizer, batch_size, kwargs_tuple)
        finally:
            self._distance_args.pop(key, None)

    def _distance_key(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        metric: str,
        kwargs: Dict[str, Any],
    ) -> str:
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
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(arr.tobytes(order="C"))
