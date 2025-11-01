import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import RandomOverSampler, SMOTE

from oversampleqa.advanced_benchmark import (
    DatasetRepository,
    StatisticalBenchmark,
    create_benchmark_report,
)


def _toy_dataset():
    rng = np.random.default_rng(0)
    minority = rng.normal(loc=2.0, scale=0.3, size=(20, 4))
    majority = rng.normal(loc=0.0, scale=0.5, size=(60, 4))
    X = np.vstack([majority, minority])
    y = np.array([0] * len(majority) + [1] * len(minority))
    return {"name": "toy", "data": X, "target": y, "minority_label": 1}


def test_statistical_benchmark_generates_results():
    dataset = _toy_dataset()
    benchmark = StatisticalBenchmark(n_folds=2, n_repeats=1, random_state=1)
    oversamplers = [RandomOverSampler(), SMOTE()]
    df = benchmark.run_comprehensive_benchmark(
        [dataset], oversamplers, metrics=["euclidean"]
    )
    assert not df.empty
    assert {"pairwise_p_values", "pairwise_effect_sizes"}.issubset(df.columns)
    p_values = json.loads(df.iloc[0]["pairwise_p_values"])
    assert isinstance(p_values, dict)


def test_confidence_interval_and_power_estimate():
    dataset = _toy_dataset()
    benchmark = StatisticalBenchmark(n_folds=2, n_repeats=1, confidence_level=0.9)
    df = benchmark.run_comprehensive_benchmark(
        [dataset], [RandomOverSampler()], metrics=["hassanat"]
    )
    assert (df["ci_upper"] >= df["ci_lower"]).all()
    assert df["recommended_samples"].iloc[0] is None or df["recommended_samples"].iloc[0] > 0


def test_dataset_repository_synthetic_generation():
    repo = DatasetRepository()
    synthetic = repo.create_synthetic_benchmark_suite(["easy", "extreme"])
    assert synthetic
    difficulties = {ds["difficulty"] for ds in synthetic}
    assert {"easy", "extreme"}.issubset(difficulties)


def test_benchmark_report_creation(tmp_path: Path):
    df = pd.DataFrame(
        {
            "dataset_name": ["toy"],
            "oversampler_name": ["RandomOverSampler"],
            "metric": ["hassanat"],
            "mean_error": [0.1],
            "std_error": [0.01],
            "ci_lower": [0.05],
            "ci_upper": [0.15],
            "n_observations": [5],
            "error_rates": [[0.1, 0.09, 0.11, 0.08, 0.12]],
            "pairwise_p_values": [json.dumps({})],
            "pairwise_effect_sizes": [json.dumps({})],
            "recommended_samples": [42],
        }
    )
    report_path = create_benchmark_report(df, output_path=tmp_path / "report.html")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "OversampleQA Benchmark Report" in content
