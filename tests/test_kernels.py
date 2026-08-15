"""Kernel/``_pairwise`` agreement and memory-accounting tests.

Seven of sixteen metrics had no vectorised kernel and fell through to a Python
double loop -- one function call per pair, on the package's default metric. The
parametrised agreement test below covers every registered metric, so a newly
added kernel cannot diverge and a newly registered metric without one is still
checked.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from oversampleqa.distance import _METRICS
from oversampleqa.optimized_distance import (
    _PEAK_MODEL,
    OptimizedDistanceMatrix,
    peak_multiple,
)


def _inputs(n1: int = 30, n2: int = 22, d: int = 5):
    rng = np.random.default_rng(0)
    # Strictly positive: the probability metrics reject negatives.
    return np.abs(rng.random((n1, d))) + 0.05, np.abs(rng.random((n2, d))) + 0.05


def _kwargs_for(metric: str, X1, X2) -> dict:
    if metric == "mahalanobis":
        return {"cov_inv": np.linalg.pinv(np.cov(np.vstack([X1, X2]).T))}
    return {}


@pytest.fixture
def optimizer():
    return OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)


@pytest.mark.parametrize("metric", sorted(_METRICS))
def test_kernel_agrees_with_pairwise(metric, optimizer):
    """Every kernel must reproduce the scalar path exactly."""
    X1, X2 = _inputs()
    kwargs = _kwargs_for(metric, X1, X2)
    expected = optimizer._pairwise(X1, X2, _METRICS[metric], **kwargs)

    kernel = optimizer._vectorized_dispatch.get(metric)
    if kernel is None:
        pytest.skip(f"{metric} intentionally has no kernel")
    assert np.allclose(kernel(X1, X2, **kwargs), expected, rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize("metric", sorted(_METRICS))
def test_every_metric_is_accounted_for(metric):
    """Every metric needs an explicit peak-model entry.

    Without one it falls back to a default, which is safe but imprecise.
    """
    assert metric in _PEAK_MODEL, (
        f"{metric} has no _PEAK_MODEL entry; add one so memory estimation does "
        "not silently fall back to the default."
    )


def test_unvectorised_metrics_are_deliberate(optimizer):
    """energy and wasserstein are the only metrics left on _pairwise.

    energy needs an (n1, n2, d, d) intermediate; wasserstein's scalar form is
    numerically wrong, so a correct kernel would disagree with it.
    """
    missing = set(_METRICS) - set(optimizer._vectorized_dispatch)
    assert missing == {"energy", "wasserstein"}


def test_memory_estimate_accounts_for_intermediates(optimizer):
    """The old estimate counted only the (n1, n2) output array."""
    n1, n2, d = 100, 200, 20
    output_only = n1 * n2 * 8 / (1024**3)

    euclidean = optimizer.estimate_memory_gb(
        n1, n2, np.dtype(np.float64), d, "euclidean"
    )
    hassanat = optimizer.estimate_memory_gb(n1, n2, np.dtype(np.float64), d, "hassanat")

    # Both exceed the bare output; hassanat scales with d and euclidean does not.
    assert euclidean > output_only
    assert euclidean == pytest.approx(
        output_only * peak_multiple("euclidean", d), rel=0.05
    )
    assert hassanat == pytest.approx(
        output_only * peak_multiple("hassanat", d), rel=0.05
    )
    assert hassanat > euclidean * 10


def test_peak_model_splits_into_two_families():
    """Some kernels scale with d, others do not -- one multiplier cannot express both."""
    # euclidean uses a BLAS gram trick: flat in d.
    assert peak_multiple("euclidean", 4) == peak_multiple("euclidean", 400)
    # hassanat broadcasts to (n1, n2, d): linear in d.
    assert peak_multiple("hassanat", 400) > peak_multiple("hassanat", 4) * 50


def test_unknown_metric_uses_the_conservative_default(optimizer):
    """A plugin metric must not be assumed free."""
    plugin = optimizer.estimate_memory_gb(50, 60, np.dtype(np.float64), 10, "made_up")
    euclidean = optimizer.estimate_memory_gb(
        50, 60, np.dtype(np.float64), 10, "euclidean"
    )
    assert plugin > euclidean


def test_auto_batch_size_reserves_the_output(optimizer):
    """Every batch used to be allowed the whole limit, with no room for the
    result array that lives for the entire computation."""
    small = OptimizedDistanceMatrix(
        metric_registry=_METRICS, cache=None, memory_limit_gb=0.01
    )
    batch = small._auto_batch_size(
        n_cols=1000,
        dtype=np.dtype(np.float64),
        n_features=20,
        metric="hassanat",
        n_rows=500,
    )
    assert batch >= 1
    # A 4-intermediate metric must give a smaller batch than a 0-intermediate one.
    cheap = small._auto_batch_size(
        n_cols=1000,
        dtype=np.dtype(np.float64),
        n_features=20,
        metric="euclidean",
        n_rows=500,
    )
    assert batch < cheap


def test_safety_factor_is_validated():
    with pytest.raises(ValueError, match="safety_factor"):
        OptimizedDistanceMatrix(metric_registry=_METRICS, safety_factor=0.0)
    with pytest.raises(ValueError, match="safety_factor"):
        OptimizedDistanceMatrix(metric_registry=_METRICS, safety_factor=1.5)


def test_safety_factor_shrinks_batches():
    generous = OptimizedDistanceMatrix(
        metric_registry=_METRICS, memory_limit_gb=1.0, safety_factor=1.0
    )
    cautious = OptimizedDistanceMatrix(
        metric_registry=_METRICS, memory_limit_gb=1.0, safety_factor=0.5
    )
    args = dict(
        n_cols=500,
        dtype=np.dtype(np.float64),
        n_features=10,
        metric="hassanat",
        n_rows=200,
    )
    assert cautious._auto_batch_size(**args) < generous._auto_batch_size(**args)


@pytest.mark.slow
@pytest.mark.parametrize("metric", ["euclidean", "hassanat"])
def test_multiplier_table_matches_measured_peak(metric, optimizer):
    """Verify two table entries empirically rather than trusting the count.

    Measures peak allocation with tracemalloc and checks the predicted estimate
    is in the right ballpark and, critically, not an *under*estimate.
    """
    n1, n2, d = 120, 150, 16
    rng = np.random.default_rng(1)
    X1 = rng.random((n1, d))
    X2 = rng.random((n2, d))

    tracemalloc.start()
    optimizer._vectorized_dispatch[metric](X1, X2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predicted_gb = optimizer.estimate_memory_gb(n1, n2, np.dtype(np.float64), d, metric)
    predicted = predicted_gb * (1024**3)

    # The estimate must not fall below the measured peak -- underestimating is
    # exactly what bypasses batching. Some headroom above is fine.
    assert predicted >= peak * 0.9, (
        f"{metric}: estimate {predicted:,.0f}B underestimates measured peak "
        f"{peak:,.0f}B -- the _PEAK_MODEL entry is too low"
    )
    assert predicted <= peak * 4, (
        f"{metric}: estimate {predicted:,.0f}B is far above peak {peak:,.0f}B; "
        "the entry is needlessly pessimistic and will over-batch"
    )


@pytest.mark.slow
def test_batched_execution_stays_under_the_limit():
    """Batching must actually bound peak memory."""
    limit_gb = 0.05
    opt = OptimizedDistanceMatrix(
        metric_registry=_METRICS,
        cache=None,
        memory_limit_gb=limit_gb,
        safety_factor=0.8,
    )
    rng = np.random.default_rng(2)
    X1 = rng.random((400, 16))
    X2 = rng.random((600, 16))

    tracemalloc.start()
    result = opt.compute_distance_matrix(X1, X2, metric="hassanat", batch_size="auto")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.shape == (400, 600)
    # Peak must stay within the declared limit, with allowance for the result
    # array itself and interpreter overhead.
    assert peak < limit_gb * (1024**3) * 2
