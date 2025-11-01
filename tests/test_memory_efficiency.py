from __future__ import annotations

import numpy as np
import pytest

from imblearn.over_sampling import SMOTE

from oversampleqa.distance import _METRICS, distance_matrix
from oversampleqa.memory_efficient_validator import MemoryEfficientValidator
from oversampleqa.optimized_distance import OptimizedDistanceMatrix
from oversampleqa.validator import validate_oversampling
from oversampleqa.caching import ValidationCache


def test_batched_computation_limits_memory(monkeypatch):
    optimizer = OptimizedDistanceMatrix(
        memory_limit_gb=1e-4,
        metric_registry=_METRICS,
        cache=None,
    )
    rng = np.random.default_rng(0)
    X1 = rng.random((180, 40))
    X2 = rng.random((180, 40))

    captured: dict[str, int] = {}
    original = OptimizedDistanceMatrix._batched_computation

    def spy(self, X1_, X2_, metric, batch_size, vectorized=None, **kwargs):
        captured["batch_size"] = batch_size
        return original(self, X1_, X2_, metric, batch_size, vectorized=vectorized, **kwargs)

    monkeypatch.setattr(OptimizedDistanceMatrix, "_batched_computation", spy)
    result = optimizer.compute_distance_matrix(X1, X2, metric="euclidean")

    assert "batch_size" in captured
    full_mem = optimizer.estimate_memory_gb(*result.shape, dtype=result.dtype)
    chunk_mem = optimizer.estimate_memory_gb(captured["batch_size"], result.shape[1], dtype=result.dtype)
    assert chunk_mem <= full_mem * 0.5


def test_memory_validator_streams_when_memory_capped(monkeypatch):
    validator = MemoryEfficientValidator(memory_limit_gb=1e-6, batch_size="auto")
    rng = np.random.default_rng(42)
    X_majority = rng.normal(size=(11_500, 4))
    X_minority = rng.normal(loc=1.0, size=(600, 4))
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([np.zeros(len(X_majority), dtype=int), np.ones(len(X_minority), dtype=int)])

    invoked = {}

    def fake_streaming(self, synthetic, hidden_majority, minority, metric, metric_kwargs, return_details):
        invoked["called"] = True
        return 0.0

    monkeypatch.setattr(MemoryEfficientValidator, "_streaming_validation", fake_streaming)
    rate = validator.validate_oversampling(X, y, minority_label=1, oversampler=SMOTE())
    assert rate == 0.0
    assert invoked.get("called", False)


class DummyOversampler:
    def __init__(self):
        self.calls = 0

    def fit_resample(self, X, y):
        self.calls += 1
        minority = X[y == 1]
        if len(minority) == 0:
            return X, y
        synthetic = minority[: min(5, len(minority))]
        X_aug = np.vstack([X, synthetic])
        y_aug = np.hstack([y, np.ones(len(synthetic), dtype=int)])
        return X_aug, y_aug

    def get_params(self, deep=True):
        return {"dummy": True}


def _build_dataset():
    rng = np.random.default_rng(7)
    X_majority = rng.normal(size=(300, 3))
    X_minority = rng.normal(loc=1.0, size=(60, 3))
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([np.zeros(len(X_majority), dtype=int), np.ones(len(X_minority), dtype=int)])
    return X, y


def test_validator_caching_reuses_results(tmp_path):
    cache = ValidationCache(cache_dir=tmp_path / "cache")
    validator = MemoryEfficientValidator(cache=cache)
    oversampler = DummyOversampler()
    X, y = _build_dataset()

    result_first = validator.validate_oversampling(X, y, minority_label=1, oversampler=oversampler)
    result_second = validator.validate_oversampling(X, y, minority_label=1, oversampler=oversampler)

    assert result_first == result_second
    assert oversampler.calls == 1


def test_validator_cache_invalidated_on_data_change(tmp_path):
    cache = ValidationCache(cache_dir=tmp_path / "cache")
    validator = MemoryEfficientValidator(cache=cache)
    oversampler = DummyOversampler()
    X, y = _build_dataset()

    validator.validate_oversampling(X, y, minority_label=1, oversampler=oversampler)
    X_changed = X.copy()
    X_changed[0, 0] += 5.0
    validator.validate_oversampling(X_changed, y, minority_label=1, oversampler=oversampler)

    assert oversampler.calls == 2


def test_memory_validator_matches_standard():
    X, y = _build_dataset()
    oversampler = SMOTE(random_state=0)
    standard = validate_oversampling(X, y, minority_label=1, oversampler=oversampler, metric="euclidean")
    memory_validator = MemoryEfficientValidator()
    optimized = memory_validator.validate_oversampling(X, y, minority_label=1, oversampler=oversampler, metric="euclidean")
    assert pytest.approx(standard, rel=1e-9) == optimized
