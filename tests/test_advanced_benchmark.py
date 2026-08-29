import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa.advanced_benchmark import (
    DatasetRepository,
    StatisticalBenchmark,
    create_benchmark_report,
    format_statistical_summary,
)


def _toy_dataset():
    rng = np.random.default_rng(0)
    # Sized so each cross-validation fold still leaves >= min_hidden minority
    # points to hold out; a 20-point minority does not survive 5-fold CV.
    minority = rng.normal(loc=2.0, scale=0.3, size=(160, 4))
    majority = rng.normal(loc=0.0, scale=0.5, size=(480, 4))
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


def test_confidence_interval_is_ordered():
    dataset = _toy_dataset()
    benchmark = StatisticalBenchmark(n_folds=2, n_repeats=1, confidence_level=0.9)
    df = benchmark.run_comprehensive_benchmark(
        [dataset], [RandomOverSampler()], metrics=["hassanat"]
    )
    assert (df["ci_upper"] >= df["ci_lower"]).all()
    # recommended_samples was also asserted here, as "None or > 0" -- a
    # condition every integer satisfies. It reported 1 on every real run.
    assert "recommended_samples" not in df.columns


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
        }
    )
    report_path = create_benchmark_report(df, output_path=tmp_path / "report.html")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "OversampleQA Benchmark Report" in content
    metadata = json.loads(metadata_sidecar_path(report_path).read_text())
    assert metadata["export_kind"] == "statistical_benchmark_report"
    assert metadata["data"]["row_count"] == 1


def test_empty_benchmark_report_creation_writes_metadata(tmp_path: Path):
    report_path = create_benchmark_report(
        pd.DataFrame(), output_path=tmp_path / "empty.html"
    )

    assert report_path.exists()
    metadata = json.loads(metadata_sidecar_path(report_path).read_text())
    assert metadata["export_kind"] == "statistical_benchmark_report"


def test_format_statistical_summary_reports_significance():
    df = pd.DataFrame(
        {
            "dataset_name": ["toy", "toy"],
            "oversampler_name": ["SMOTE", "ADASYN"],
            "metric": ["hassanat", "hassanat"],
            "mean_error": [0.10, 0.25],
            "std_error": [0.01, 0.02],
            "ci_lower": [0.08, 0.21],
            "ci_upper": [0.12, 0.29],
            "n_observations": [10, 10],
            "pairwise_p_values": [json.dumps({"SMOTE_vs_ADASYN": 0.01}), None],
            "pairwise_effect_sizes": [json.dumps({"SMOTE_vs_ADASYN": 1.3}), None],
        }
    )
    summary = format_statistical_summary(df)
    assert "## Dataset: toy" in summary
    assert "| SMOTE | hassanat | 0.100 |" in summary
    assert "SMOTE_vs_ADASYN: p=0.0100, d=1.30" in summary


def test_format_statistical_summary_handles_empty():
    assert "No benchmark results" in format_statistical_summary(pd.DataFrame())


# --- silent empty results -------------------------------------------------


def test_empty_results_keep_their_columns():
    """A (0, 0) frame raises KeyError on any column access.

    A caller that correctly handles "no results" still breaks if the empty
    frame has no columns, so the shape must be stable whether or not any
    combination succeeded.
    """
    import warnings as _warnings

    from imblearn.over_sampling import SMOTE

    from oversampleqa import StatisticalBenchmark
    from oversampleqa.benchmark import load_standard_datasets

    # 50 minority split into 3 folds leaves 33 per training fold; a 10%
    # hold-out is then 3 points, below min_hidden, so every fold fails.
    datasets = [d for d in load_standard_datasets() if d["name"] in {"moons"}]
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        frame = StatisticalBenchmark(
            n_folds=3, n_repeats=1, random_state=42
        ).run_comprehensive_benchmark(
            datasets, [SMOTE(random_state=0)], metrics=["euclidean"]
        )

    assert frame.empty
    assert "mean_error" in frame.columns
    assert list(frame["mean_error"]) == []


def test_skipped_combinations_are_summarised_once():
    """Per-fold warnings number in the hundreds on a real sweep.

    Without a summary the caller sees an empty frame and has to reconstruct
    why from a flood of individual messages.
    """
    import warnings as _warnings

    from imblearn.over_sampling import SMOTE

    from oversampleqa import StatisticalBenchmark
    from oversampleqa.benchmark import load_standard_datasets

    datasets = [
        d for d in load_standard_datasets() if d["name"] in {"moons", "circles"}
    ]
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        StatisticalBenchmark(
            n_folds=3, n_repeats=1, random_state=42
        ).run_comprehensive_benchmark(
            datasets, [SMOTE(random_state=0)], metrics=["euclidean"]
        )

    summaries = [
        str(w.message)
        for w in caught
        if "combinations produced no usable folds" in str(w.message)
    ]
    assert len(summaries) == 1
    assert "moons" in summaries[0]
    assert "fewer" in summaries[0]


def test_reused_engine_does_not_accumulate_skips():
    """The skip list is per run, not per engine."""
    import warnings as _warnings

    from imblearn.over_sampling import SMOTE

    from oversampleqa import StatisticalBenchmark
    from oversampleqa.benchmark import load_standard_datasets

    datasets = [d for d in load_standard_datasets() if d["name"] in {"moons"}]
    engine = StatisticalBenchmark(n_folds=3, n_repeats=1, random_state=42)
    counts = []
    for _ in range(2):
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            engine.run_comprehensive_benchmark(
                datasets, [SMOTE(random_state=0)], metrics=["euclidean"]
            )
        summaries = [
            str(w.message)
            for w in caught
            if "combinations produced no usable folds" in str(w.message)
        ]
        counts.append(summaries[0].split()[0] if summaries else "0")
    assert counts[0] == counts[1], f"skip count grew across runs: {counts}"
