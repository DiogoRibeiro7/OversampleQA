"""Benchmark core OversampleQA validation paths.

The output is a JSON artifact, not a hard merge gate. Wall-clock timings on
shared runners and developer laptops are too noisy for required CI, but a
structured artifact gives maintainers something concrete to compare when a
change is suspected of slowing the package down or increasing peak memory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oversampleqa import __version__
from oversampleqa.distance import _METRICS, distance_matrix
from oversampleqa.optimized_distance import OptimizedDistanceMatrix
from oversampleqa.validator import validate_oversampling


@dataclass(frozen=True)
class BenchmarkCase:
    """A named benchmark and the parameters needed to interpret it."""

    name: str
    group: str
    n_samples: int
    n_features: int
    metric: str


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured benchmark case."""

    name: str
    group: str
    n_samples: int
    n_features: int
    metric: str
    repeats: int
    median_seconds: float
    peak_memory_mb: float
    samples_seconds: tuple[float, ...]


def environment() -> dict[str, Any]:
    """Return metadata that makes benchmark artifacts interpretable."""
    return {
        "oversampleqa_version": __version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
    }


def core_cases(*, quick: bool) -> tuple[BenchmarkCase, ...]:
    """Return the default core-path benchmark cases."""
    n_samples = 100 if quick else 400
    n_features = 8 if quick else 25
    return (
        BenchmarkCase(
            "distance_matrix[euclidean]",
            "distance",
            n_samples,
            n_features,
            "euclidean",
        ),
        BenchmarkCase(
            "distance_matrix[hassanat]",
            "distance",
            n_samples,
            n_features,
            "hassanat",
        ),
        BenchmarkCase(
            "optimized_distance_matrix[euclidean]",
            "distance",
            n_samples,
            n_features,
            "euclidean",
        ),
        BenchmarkCase(
            "validate_oversampling[euclidean]",
            "validation",
            n_samples,
            n_features,
            "euclidean",
        ),
    )


def _distance_data(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    return (
        rng.random((case.n_samples, case.n_features)),
        rng.random((case.n_samples, case.n_features)),
    )


def _validation_data(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.random((case.n_samples, case.n_features))
    y = np.zeros(case.n_samples, dtype=int)
    minority_count = max(20, case.n_samples // 4)
    y[:minority_count] = 1
    return X, y


def build_case(case: BenchmarkCase) -> Callable[[], object]:
    """Build the callable measured for one benchmark case."""
    if case.name.startswith("optimized_distance_matrix"):
        X1, X2 = _distance_data(case)
        optimizer = OptimizedDistanceMatrix(metric_registry=_METRICS, cache=None)
        return lambda: optimizer.compute_distance_matrix(X1, X2, metric=case.metric)
    if case.name.startswith("distance_matrix"):
        X1, X2 = _distance_data(case)
        return lambda: distance_matrix(X1, X2, metric=case.metric)
    if case.name.startswith("validate_oversampling"):
        from imblearn.over_sampling import SMOTE

        X, y = _validation_data(case)
        return lambda: validate_oversampling(
            X,
            y,
            minority_label=1,
            oversampler=SMOTE(random_state=0),
            hidden_ratio=0.2,
            metric=case.metric,
            random_state=0,
        )
    raise ValueError(f"unknown benchmark case: {case.name}")


def measure_case(case: BenchmarkCase, *, repeats: int) -> BenchmarkResult:
    """Measure duration and peak traced memory for one benchmark case."""
    samples: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        benchmark = build_case(case)
        tracemalloc.start()
        start = time.perf_counter()
        benchmark()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(elapsed)
        peaks.append(peak)

    return BenchmarkResult(
        name=case.name,
        group=case.group,
        n_samples=case.n_samples,
        n_features=case.n_features,
        metric=case.metric,
        repeats=repeats,
        median_seconds=statistics.median(samples),
        peak_memory_mb=max(peaks) / 1024 / 1024,
        samples_seconds=tuple(samples),
    )


def run_benchmarks(*, quick: bool, repeats: int) -> dict[str, Any]:
    """Run the default benchmark suite and return a JSON-ready payload."""
    results = [measure_case(case, repeats=repeats) for case in core_cases(quick=quick)]
    return {
        "schema_version": "1.0",
        "suite": "core-paths",
        "mode": "quick" if quick else "standard",
        "environment": environment(),
        "results": [asdict(result) for result in results],
    }


def format_table(payload: dict[str, Any]) -> str:
    """Return a compact text table for humans reading the command output."""
    rows = payload["results"]
    width = max(len(row["name"]) for row in rows)
    lines = [f"{'benchmark'.ljust(width)}  median ms  peak MiB"]
    for row in rows:
        lines.append(
            f"{row['name'].ljust(width)}  "
            f"{row['median_seconds'] * 1e3:9.2f}  "
            f"{row['peak_memory_mb']:8.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the JSON artifact here.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller benchmark inputs for smoke checks and examples.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measurements per case.",
    )
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    payload = run_benchmarks(quick=args.quick, repeats=args.repeats)
    print(format_table(payload))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nBenchmark artifact written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
