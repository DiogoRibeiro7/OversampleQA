"""Tests for the caching layer.

The cache used to be constructed at import time, keyed on a live optimizer
object, and fronted by an ``lru_cache`` on an instance method fed through a
mutable side channel. These tests pin the replacements.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from oversampleqa.caching import ValidationCache, default_cache_dir
from oversampleqa.distance import _METRICS, distance_matrix
from oversampleqa.optimized_distance import OptimizedDistanceMatrix

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_import_creates_nothing_in_cwd(tmp_path):
    """Regression guard: importing a library must not touch the filesystem.

    ``ValidationCache.__init__`` used to ``mkdir`` its cache directory, and
    ``distance.py`` constructed one at module scope, so merely importing the
    package created ``.oversampleqa_cache`` in whatever directory the process
    happened to be in -- a home directory, a repo root, a container's ``/``.
    """
    env = {"PYTHONPATH": REPO_SRC, "PATH": ""}
    result = subprocess.run(
        [sys.executable, "-c", "import oversampleqa"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), **env},
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cache_dir_is_not_created_until_first_write(tmp_path):
    target = tmp_path / "cache_here"
    cache = ValidationCache(cache_dir=target)
    assert not target.exists()
    _ = cache.memory  # first use of the disk store
    assert target.exists()


def test_default_cache_dir_is_not_the_working_directory():
    assert default_cache_dir().is_absolute()
    assert default_cache_dir() != Path.cwd()


def test_no_lru_cache_on_instance_methods():
    """``lru_cache`` on a method pins ``self`` alive forever and is shared."""
    cache = ValidationCache(cache_dir="unused")
    for name in dir(cache):
        attr = getattr(type(cache), name, None)
        assert not hasattr(attr, "cache_info"), f"{name} is lru_cache-wrapped"
    assert not hasattr(cache, "_distance_args")


def test_cached_result_is_read_only(tmp_path):
    """A mutation must raise rather than silently corrupt later hits."""
    cache = ValidationCache(cache_dir=tmp_path / "c")
    rng = np.random.default_rng(0)
    X1 = rng.random((20, 4))
    X2 = rng.random((15, 4))

    first = distance_matrix(X1, X2, "euclidean", cache=cache)
    with pytest.raises(ValueError):
        first[0, 0] = 999.0

    second = distance_matrix(X1, X2, "euclidean", cache=cache)
    assert np.array_equal(first, second)


def test_cache_hit_returns_same_values(tmp_path):
    cache = ValidationCache(cache_dir=tmp_path / "c")
    rng = np.random.default_rng(1)
    X1 = rng.random((25, 5))
    X2 = rng.random((18, 5))
    a = distance_matrix(X1, X2, "hassanat", cache=cache)
    b = distance_matrix(X1, X2, "hassanat", cache=cache)
    assert np.array_equal(a, b)


def test_bytes_limit_is_enforced(tmp_path):
    """``memory_mb`` used to be stored and never read."""
    cache = ValidationCache(cache_dir=tmp_path / "c", memory_mb=0)
    rng = np.random.default_rng(2)
    for i in range(4):
        X1 = rng.random((30 + i, 4))
        X2 = rng.random((20, 4))
        distance_matrix(X1, X2, "euclidean", cache=cache)
    # A zero-byte budget evicts everything as soon as it is stored.
    assert cache.size_bytes == 0


def test_max_entries_evicts_oldest(tmp_path):
    cache = ValidationCache(cache_dir=tmp_path / "c", max_entries=2)
    rng = np.random.default_rng(3)
    for i in range(5):
        X1 = rng.random((10 + i, 3))
        X2 = rng.random((8, 3))
        distance_matrix(X1, X2, "euclidean", cache=cache)
    assert len(cache._store) <= 2


def test_clear_empties_the_memory_tier(tmp_path):
    cache = ValidationCache(cache_dir=tmp_path / "c")
    rng = np.random.default_rng(4)
    distance_matrix(rng.random((12, 3)), rng.random((9, 3)), "euclidean", cache=cache)
    assert cache.size_bytes > 0
    cache.clear()
    assert cache.size_bytes == 0


@pytest.mark.parametrize("metric", sorted(_METRICS))
def test_batched_and_unbatched_agree(metric):
    """Batching must not change the result -- the invariant that lets us keep
    ``batch_size`` out of the cache key."""
    rng = np.random.default_rng(5)
    X1 = np.abs(rng.random((24, 4))) + 0.1
    X2 = np.abs(rng.random((17, 4))) + 0.1

    kwargs = {}
    if metric == "mahalanobis":
        kwargs["cov_inv"] = np.linalg.pinv(np.cov(np.vstack([X1, X2]).T))

    opt = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)
    whole = opt.compute_distance_matrix(
        X1, X2, metric=metric, batch_size=1000, **kwargs
    )
    batched = opt.compute_distance_matrix(X1, X2, metric=metric, batch_size=3, **kwargs)
    assert np.allclose(whole, batched, equal_nan=True)


def test_batch_size_reaches_the_uncached_path(monkeypatch):
    """``batch_size`` was dropped on the no-cache path and silently became "auto"."""
    opt = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)
    seen = {}
    original = opt._compute_uncached

    def spy(X1, X2, *, metric, batch_size="auto", **kwargs):
        seen["batch_size"] = batch_size
        return original(X1, X2, metric=metric, batch_size=batch_size, **kwargs)

    monkeypatch.setattr(opt, "_compute_uncached", spy)
    rng = np.random.default_rng(6)
    opt.compute_distance_matrix(
        rng.random((10, 3)), rng.random((8, 3)), metric="euclidean", batch_size=7
    )
    assert seen["batch_size"] == 7


def test_concurrent_use_of_one_instance_is_consistent(tmp_path):
    """One instance guards its own bookkeeping; hammer it from many threads."""
    cache = ValidationCache(cache_dir=tmp_path / "c", max_entries=8)
    rng = np.random.default_rng(7)
    X1 = rng.random((30, 4))
    X2 = rng.random((22, 4))
    expected = distance_matrix(X1, X2, "euclidean", cache=cache)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                got = distance_matrix(X1, X2, "euclidean", cache=cache)
                assert np.array_equal(got, expected)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert cache.size_bytes >= 0
