"""Regression tests for copyable documentation examples."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_quickstart_teaches_repeated_seeded_validation():
    quickstart = read_text("docs/quickstart.md")

    assert "SMOTE(random_state=42)" in quickstart
    assert "random_state=42" in quickstart
    assert "n_repeats=10" in quickstart
    assert "return_details=True" in quickstart
    assert "details.mean" in quickstart
    assert "*.metadata.json" in quickstart


def test_advanced_tutorial_exposes_statistical_identifiers():
    tutorial = read_text("docs/tutorials/advanced_tutorial.md")

    assert "StatisticalBenchmark(n_folds=3, n_repeats=3, random_state=42)" in tutorial
    assert "bench.fold_results()" in tutorial
    for identifier in (
        "repeat",
        "fold",
        "split_seed",
        "metric",
        "hidden_ratio",
        "oversampleqa_version",
    ):
        assert identifier in tutorial


def test_benchmark_example_exports_auditable_report_metadata():
    example = read_text("examples/benchmark_example.py")

    assert "ADASYN(random_state=0)" in example
    assert "n_runs=3" in example
    assert "random_state=0" in example
    assert "include_plots=False" in example
    assert ".metadata.json" in example
    assert "metadata['environment']['oversampleqa_version']" in example
    assert "metadata['source']['row_count']" in example


def test_examples_index_mentions_fidelity_and_metadata():
    examples = read_text("docs/examples.md")
    gallery = read_text("docs/gallery/index.md")

    assert "repeated splits" in examples
    assert "metadata" in examples
    assert "fidelity examples separate realism from memorisation" in gallery
    assert "metadata sidecars" in gallery
