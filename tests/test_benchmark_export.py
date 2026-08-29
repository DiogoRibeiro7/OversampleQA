"""Tests for the long-format benchmark frame and its four exports.

The Markdown path here used `to_csv(sep="|")` — the same defect fixed in
`report.py`, which survived at this second site because the renderer was
duplicated rather than shared. Both now come from `oversampleqa._render`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa.benchmark import (
    _BENCHMARK_COLUMNS,
    compute_ranking,
    export_benchmark_results,
    run_benchmark,
)


@pytest.fixture(scope="module")
def dataset():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        random_state=0,
    )
    return [{"name": "d", "data": X, "target": y, "minority_label": 1}]


@pytest.fixture(scope="module")
def results(dataset):
    from imblearn.over_sampling import SMOTE

    return run_benchmark(
        dataset, [SMOTE(random_state=0)], hidden_ratios=[0.2], n_runs=2
    )


def test_frame_is_long_format(results):
    """One row per (dataset, oversampler, metric, hidden_ratio, run)."""
    key = ["dataset", "oversampler", "metric", "hidden_ratio", "run"]
    assert not results.duplicated(key).any()


def test_metric_is_a_column(results):
    assert "metric" in results.columns
    assert set(results["metric"]) == {"hassanat"}


def test_column_order_is_fixed(results):
    assert list(results.columns) == list(_BENCHMARK_COLUMNS)


def test_two_metrics_stay_distinguishable_when_concatenated(dataset):
    """Without the metric column these rows are indistinguishable.

    Error rates are not comparable across metrics, so averaging a concatenation
    of the two would produce a number that means nothing.
    """
    from imblearn.over_sampling import SMOTE

    frames = [
        run_benchmark(
            dataset,
            [SMOTE(random_state=0)],
            hidden_ratios=[0.2],
            n_runs=2,
            distance_metric=metric,
        )
        for metric in ("hassanat", "euclidean")
    ]
    combined = pd.concat(frames, ignore_index=True)
    assert set(combined["metric"]) == {"hassanat", "euclidean"}
    per_metric = combined.groupby("metric")["error_rate"].mean()
    assert len(per_metric) == 2


def test_empty_frame_keeps_its_columns():
    """A (0, 0) frame raises KeyError on any column access."""
    empty = run_benchmark([], [], hidden_ratios=[0.2], n_runs=1)
    assert empty.empty
    assert list(empty.columns) == list(_BENCHMARK_COLUMNS)
    assert empty["error_rate"].tolist() == []


@pytest.mark.parametrize("fmt", ["csv", "json", "markdown", "html"])
def test_every_format_writes_a_file(results, tmp_path, fmt):
    out = tmp_path / f"r.{fmt}"
    export_benchmark_results(results, str(out), fmt=fmt)
    assert out.exists() and out.stat().st_size > 0
    sidecar = metadata_sidecar_path(out)
    assert sidecar.exists() and sidecar.stat().st_size > 0


def test_benchmark_export_metadata_describes_source(results, tmp_path):
    out = tmp_path / "r.json"
    export_benchmark_results(results, str(out), fmt="json")

    metadata = json.loads(metadata_sidecar_path(out).read_text(encoding="utf-8"))

    assert metadata["export_kind"] == "benchmark_summary"
    assert metadata["artifact"] == out.name
    assert metadata["data"]["row_count"] == 1
    assert metadata["data"]["attrs"]["source"]["row_count"] == len(results)
    assert metadata["environment"]["oversampleqa_version"]


def test_markdown_export_is_actually_markdown(results, tmp_path):
    """The regression this file exists for."""
    out = tmp_path / "r.md"
    export_benchmark_results(results, str(out), fmt="markdown")
    lines = [
        line
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    assert len(lines) >= 3, "no table rows"
    separator = lines[1]
    assert set(separator.replace("|", "").replace(" ", "")) <= {"-", ":"}
    assert all(line.startswith("|") and line.endswith("|") for line in lines)


def test_html_export_is_a_table(results, tmp_path):
    out = tmp_path / "r.html"
    export_benchmark_results(results, str(out), fmt="html")
    assert "<table" in out.read_text(encoding="utf-8")


def test_json_export_writes_null_for_nan(tmp_path):
    """JSON has no NaN literal; strict parsers reject one."""
    frame = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "oversampler": ["A", "A"],
            "metric": ["hassanat"] * 2,
            "hidden_ratio": [0.2, 0.2],
            "run": [0, 1],
            "error_rate": [np.nan, np.nan],
        }
    )
    out = tmp_path / "r.json"
    export_benchmark_results(frame, str(out), fmt="json")
    payload = json.loads(out.read_text(encoding="utf-8"))  # would raise on NaN
    assert payload[0]["mean"] is None


def test_unknown_format_raises(results, tmp_path):
    with pytest.raises(ValueError, match="html"):
        export_benchmark_results(results, str(tmp_path / "x"), fmt="latex")


def test_all_formats_render_the_same_frame(results, tmp_path):
    """One code path, four renderers -- not four independent summaries."""
    summary = compute_ranking(results)
    csv_path = tmp_path / "r.csv"
    export_benchmark_results(results, str(csv_path), fmt="csv")
    from_csv = pd.read_csv(csv_path)
    assert len(from_csv) == len(summary)
    assert set(summary.index) == set(from_csv["oversampler"])


def test_markdown_does_not_hide_missing_values(tmp_path):
    """A blank cell would be ambiguous; nan must stay visible."""
    frame = pd.DataFrame(
        {
            "dataset": ["d"],
            "oversampler": ["A"],
            "metric": ["hassanat"],
            "hidden_ratio": [0.2],
            "run": [0],
            "error_rate": [np.nan],
        }
    )
    out = Path(tmp_path / "r.md")
    export_benchmark_results(frame, str(out), fmt="markdown")
    text = out.read_text(encoding="utf-8")
    assert "nan" in text
