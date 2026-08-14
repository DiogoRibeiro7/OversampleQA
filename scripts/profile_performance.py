"""Performance profiling and regression checking for OversampleQA.

This is an *optional* tool, not part of the default test suite. It times the
hot paths most likely to regress -- distance-matrix computation and the
validator -- and can save a JSON baseline or compare the current timings against
one, failing if anything is slower than a tolerance factor.

Usage
-----
Print a timing table::

    python scripts/profile_performance.py

Save a baseline::

    python scripts/profile_performance.py --save perf_baseline.json

Check against a baseline (exit code 1 on regression)::

    python scripts/profile_performance.py --check perf_baseline.json --tolerance 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from oversampleqa.distance import _METRICS, distance_matrix
from oversampleqa.optimized_distance import OptimizedDistanceMatrix
from oversampleqa.validator import validate_oversampling


def _timeit(func: Callable[[], object], loops: int = 3) -> float:
    """Return the best wall-clock time over ``loops`` runs of ``func``.

    Args:
        func: Zero-argument callable to time.
        loops: Number of repetitions; the minimum is returned to reduce noise.

    Returns:
        The fastest observed run time in seconds.
    """
    best = float("inf")
    for _ in range(loops):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def run_profile(quick: bool = False) -> Dict[str, float]:
    """Run the performance profile and return per-benchmark best times.

    Args:
        quick: Use small inputs and fewer loops for a fast smoke run.

    Returns:
        Mapping of benchmark name to best wall-clock seconds.
    """
    from imblearn.over_sampling import SMOTE

    rng = np.random.default_rng(42)
    n = 120 if quick else 400
    dim = 10 if quick else 25
    loops = 1 if quick else 3

    X1 = rng.random((n, dim))
    X2 = rng.random((n, dim))
    optimizer = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)

    results: Dict[str, float] = {}
    for metric in ("euclidean", "hassanat", "manhattan"):
        results[f"optimized_distance_matrix[{metric}]"] = _timeit(
            lambda m=metric: optimizer.compute_distance_matrix(X1, X2, metric=m),
            loops=loops,
        )
        results[f"distance_matrix[{metric}]"] = _timeit(
            lambda m=metric: distance_matrix(X1, X2, m), loops=loops
        )

    # Validator hot path on an imbalanced dataset.
    Xv = rng.random((n, dim))
    yv = np.zeros(n, dtype=int)
    yv[: max(10, n // 10)] = 1
    results["validate_oversampling[euclidean]"] = _timeit(
        lambda: validate_oversampling(
            Xv, yv, 1, SMOTE(random_state=0), metric="euclidean"
        ),
        loops=loops,
    )
    return results


def compare(
    current: Dict[str, float], baseline: Dict[str, float], tolerance: float
) -> List[str]:
    """Return human-readable regression messages for slower benchmarks.

    A benchmark regresses when its current time exceeds the baseline time
    multiplied by ``tolerance``. Benchmarks missing from the baseline are
    ignored (they are new and have nothing to compare against).

    Args:
        current: Current benchmark timings.
        baseline: Baseline benchmark timings.
        tolerance: Allowed slowdown factor (e.g. ``1.5`` permits 50% slower).

    Returns:
        A list of regression messages; empty if nothing regressed.
    """
    regressions: List[str] = []
    for name, base_time in baseline.items():
        if name not in current:
            continue
        allowed = base_time * tolerance
        if current[name] > allowed:
            regressions.append(
                f"{name}: {current[name] * 1e3:.2f} ms vs baseline "
                f"{base_time * 1e3:.2f} ms (> {tolerance:.2f}x)"
            )
    return regressions


def _format_table(results: Dict[str, float]) -> str:
    """Render benchmark results as a simple aligned text table.

    Args:
        results: Mapping of benchmark name to seconds.

    Returns:
        A formatted multi-line string.
    """
    width = max((len(name) for name in results), default=0)
    lines = [f"{'benchmark'.ljust(width)}  time (ms)"]
    for name, seconds in results.items():
        lines.append(f"{name.ljust(width)}  {seconds * 1e3:10.2f}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    """Entry point for the profiling CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success, 1 on detected regression).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", type=Path, help="Write timings to a baseline JSON file."
    )
    parser.add_argument(
        "--check", type=Path, help="Compare timings against a baseline JSON file."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.5,
        help="Allowed slowdown factor when checking (default: 1.5).",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use small inputs for a fast run."
    )
    args = parser.parse_args(argv)

    results = run_profile(quick=args.quick)
    print(_format_table(results))

    if args.save:
        args.save.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nBaseline written to {args.save}")

    if args.check:
        baseline = json.loads(args.check.read_text(encoding="utf-8"))
        regressions = compare(results, baseline, args.tolerance)
        if regressions:
            print("\nPerformance regressions detected:")
            for message in regressions:
                print(f"  - {message}")
            return 1
        print("\nNo performance regressions detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
