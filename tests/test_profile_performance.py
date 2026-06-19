"""Tests for the optional performance profiling script."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "profile_performance.py"
_spec = importlib.util.spec_from_file_location("profile_performance", SCRIPT)
profile_performance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_performance)


def test_compare_flags_regressions():
    baseline = {"a": 1.0, "b": 2.0}
    current = {"a": 1.4, "b": 4.0}
    regressions = profile_performance.compare(current, baseline, tolerance=1.5)
    # 'a' is within 1.5x, 'b' is 2x -> only 'b' regresses.
    assert len(regressions) == 1
    assert regressions[0].startswith("b:")


def test_compare_ignores_missing_benchmarks():
    baseline = {"a": 1.0, "missing": 1.0}
    current = {"a": 1.0}
    assert profile_performance.compare(current, baseline, tolerance=1.0) == []


def test_run_profile_quick_returns_expected_keys():
    results = profile_performance.run_profile(quick=True)
    assert results
    assert "validate_oversampling[euclidean]" in results
    assert all(value >= 0.0 for value in results.values())
