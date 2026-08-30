"""Tests for human-readable report metadata blocks."""

from __future__ import annotations

import pandas as pd

from oversampleqa._report_metadata import (
    report_metadata_frame,
    report_metadata_html,
    report_metadata_markdown,
)


def _metadata_dict(frame: pd.DataFrame) -> dict[str, str]:
    return {row.field: row.value for row in frame.itertuples(index=False)}


def test_report_metadata_frame_summarises_reproducibility_columns(monkeypatch):
    monkeypatch.setattr(
        "oversampleqa._report_metadata._dependency_lock_hash", lambda: "abc123"
    )
    results = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(8)],
            "oversampler": ["SMOTE"] * 8,
            "metric": ["hassanat"] * 8,
            "hidden_ratio": [0.1] * 8,
            "reference": ["hidden_minority"] * 8,
            "run": list(range(8)),
            "split_seed": [100 + i for i in range(8)],
            "random_state": [None, pd.NA, 7, 7, 8, 8, 9, 9],
            "minority_label": [1] * 8,
            "oversampleqa_version": ["0.6.1"] * 8,
            "error_rate": [0.1] * 8,
        }
    )
    results.attrs["source"] = {
        "row_count": 42,
        "columns": ["dataset", "oversampler", "metric", "error_rate"],
    }

    metadata = _metadata_dict(report_metadata_frame(results))

    assert metadata["dependency_lock_hash"] == "abc123"
    assert metadata["result_rows"] == "8"
    assert metadata["datasets"].endswith("+2 more")
    assert metadata["oversamplers"] == "SMOTE"
    assert metadata["metrics"] == "hassanat"
    assert metadata["hidden_ratios"] == "0.1"
    assert metadata["reference"] == "hidden_minority"
    assert metadata["random_states"] == "7, 8, 9"
    assert metadata["source_rows"] == "42"
    assert metadata["source_columns"] == "dataset, oversampler, metric, error_rate"


def test_report_metadata_frame_handles_empty_results(monkeypatch):
    monkeypatch.setattr(
        "oversampleqa._report_metadata._dependency_lock_hash", lambda: None
    )

    metadata = _metadata_dict(report_metadata_frame(pd.DataFrame()))

    assert metadata["dependency_lock_hash"] == "unavailable"
    assert metadata["result_rows"] == "0"
    assert metadata["result_columns"] == "none"


def test_report_metadata_renderers_produce_tables():
    results = pd.DataFrame(
        {
            "dataset_name": ["toy"],
            "oversampler_name": ["RandomOverSampler"],
            "metric": ["euclidean"],
            "hidden_ratio": [0.2],
            "n_folds": [2],
            "n_repeats": [3],
        }
    )

    markdown = report_metadata_markdown(results)
    html = report_metadata_html(results)

    assert "| result_rows" in markdown
    assert "RandomOverSampler" in markdown
    assert "<table" in html
    assert "euclidean" in html


def test_report_metadata_keeps_unusual_scalar_values():
    results = pd.DataFrame({"random_state": [[1, 2], None]})

    metadata = _metadata_dict(report_metadata_frame(results))

    assert metadata["random_states"] == "[1, 2]"
