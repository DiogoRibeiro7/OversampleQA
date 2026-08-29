"""Tests for the core-path benchmark scaffold."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_core_paths.py"
_spec = importlib.util.spec_from_file_location("benchmark_core_paths", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
benchmark_core_paths = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_core_paths"] = benchmark_core_paths
_spec.loader.exec_module(benchmark_core_paths)


def test_core_cases_include_distance_and_validation_paths():
    cases = benchmark_core_paths.core_cases(quick=True)
    names = {case.name for case in cases}

    assert "distance_matrix[euclidean]" in names
    assert "distance_matrix[hassanat]" in names
    assert "optimized_distance_matrix[euclidean]" in names
    assert "validate_oversampling[euclidean]" in names
    assert {case.group for case in cases} == {"distance", "validation"}


def test_measure_case_records_duration_and_peak_memory():
    case = benchmark_core_paths.BenchmarkCase(
        name="distance_matrix[euclidean]",
        group="distance",
        n_samples=8,
        n_features=3,
        metric="euclidean",
    )

    result = benchmark_core_paths.measure_case(case, repeats=1)

    assert result.name == case.name
    assert result.repeats == 1
    assert result.median_seconds >= 0.0
    assert result.peak_memory_mb >= 0.0
    assert len(result.samples_seconds) == 1


def test_main_writes_json_artifact(tmp_path, monkeypatch):
    output = tmp_path / "benchmarks" / "core_paths.json"
    payload = {
        "schema_version": "1.0",
        "suite": "core-paths",
        "mode": "quick",
        "environment": {"python_version": "3.test"},
        "results": [
            {
                "name": "distance_matrix[euclidean]",
                "group": "distance",
                "n_samples": 8,
                "n_features": 3,
                "metric": "euclidean",
                "repeats": 1,
                "median_seconds": 0.001,
                "peak_memory_mb": 0.01,
                "samples_seconds": [0.001],
            }
        ],
    }

    monkeypatch.setattr(benchmark_core_paths, "run_benchmarks", lambda **_: payload)

    assert (
        benchmark_core_paths.main(["--quick", "--repeats", "1", "--output", str(output)])
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
