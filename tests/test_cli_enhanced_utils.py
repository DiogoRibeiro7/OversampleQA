import json
from pathlib import Path

import pandas as pd

from oversampleqa import cli_enhanced
from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa.cli_enhanced import (
    analyze_dataset,
    explain_ratio,
    generate_recommendations,
    interpret_error_rate,
    load_checkpoint,
    save_checkpoint,
)


def test_analyze_dataset(tmp_path: Path):
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 10.0],
            "target": [0, 0, 1, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    info = analyze_dataset(csv_path, target="target", minority_label=1)
    assert info["n_samples"] == 4
    assert info["minority"] == 2
    assert info["majority"] == 2


def test_checkpoint_roundtrip(tmp_path: Path):
    payload = {"status": "completed", "results": {"error_rate": 0.12}}
    save_checkpoint(tmp_path, payload)
    loaded = load_checkpoint(tmp_path)
    assert loaded == payload


def test_interpretation_helpers():
    assert "Excellent" in interpret_error_rate(0.05)
    assert "Acceptable" in interpret_error_rate(0.2)
    assert "Risky" in interpret_error_rate(0.4)

    assert "Highly imbalanced" in explain_ratio(0.05)
    assert "Moderately" in explain_ratio(0.2)
    assert "Near-balanced" in explain_ratio(0.6)

    recs = generate_recommendations(0.4, 0.05)
    assert any("advanced oversamplers" in r for r in recs)


def test_statistical_benchmark_outputs_get_metadata_sidecars(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "dataset_name": ["toy"],
            "oversampler_name": ["SMOTE"],
            "metric": ["hassanat"],
            "mean_error": [0.1],
            "std_error": [0.01],
            "ci_lower": [0.05],
            "ci_upper": [0.15],
            "n_observations": [5],
            "pairwise_p_values": [json.dumps({})],
            "pairwise_effect_sizes": [json.dumps({})],
        }
    )

    class DummyBenchmark:
        def __init__(self, n_folds, n_repeats):
            self.n_folds = n_folds
            self.n_repeats = n_repeats

        def run_comprehensive_benchmark(self, datasets, oversamplers):
            assert datasets == [{"name": "toy"}]
            assert len(oversamplers) == 2
            return frame

    monkeypatch.setattr(cli_enhanced, "StatisticalBenchmark", DummyBenchmark)

    cli_enhanced._run_statistical_benchmark(
        [{"name": "toy"}], tmp_path, folds=2, repeats=3
    )

    statistics = tmp_path / "benchmark_statistics.csv"
    summary = tmp_path / "benchmark_summary.md"
    report = tmp_path / "benchmark_report.html"

    for artifact in (statistics, summary, report):
        assert metadata_sidecar_path(artifact).exists()

    metadata = json.loads(metadata_sidecar_path(summary).read_text(encoding="utf-8"))
    assert metadata["benchmark_parameters"] == {"folds": 2, "repeats": 3}
