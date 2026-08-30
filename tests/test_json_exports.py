"""Strict JSON export regression tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa._json import strict_json_dumps
from oversampleqa.benchmark import export_benchmark_results
from oversampleqa.cli_enhanced import export_results, save_checkpoint


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant: {token}")


def _strict_loads(text: str):
    return json.loads(text, parse_constant=_reject_json_constant)


def test_strict_json_dumps_replaces_non_finite_values():
    text = strict_json_dumps(
        {
            "nan": np.nan,
            "inf": np.float64(np.inf),
            "array": np.array([1.0, -np.inf]),
            "missing": pd.NA,
            "flag": np.bool_(True),
        }
    )

    payload = _strict_loads(text)

    assert payload == {
        "nan": None,
        "inf": None,
        "array": [1.0, None],
        "missing": None,
        "flag": True,
    }


def test_cli_checkpoint_json_is_strict(tmp_path):
    save_checkpoint(
        tmp_path,
        {
            "status": "completed",
            "results": {
                "error_rate": np.nan,
                "interval": np.array([0.1, np.inf]),
            },
        },
    )

    payload = _strict_loads((tmp_path / ".oversampleqa_run.json").read_text())

    assert payload["results"]["error_rate"] is None
    assert payload["results"]["interval"] == [0.1, None]


def test_cli_validation_json_export_is_strict(tmp_path):
    export_results(
        {
            "error_rate": np.nan,
            "std": np.float64(np.inf),
            "rates": np.array([0.1, np.nan]),
        },
        ["json"],
        tmp_path,
    )

    payload = _strict_loads((tmp_path / "validation_results.json").read_text())

    assert payload["error_rate"] is None
    assert payload["std"] is None
    assert payload["rates"] == [0.1, None]

    metadata = _strict_loads(
        metadata_sidecar_path(tmp_path / "validation_results.json").read_text()
    )
    assert metadata["export_kind"] == "validation_results"
    assert metadata["data"]["values"]["error_rate"] is None


def test_benchmark_json_export_is_strict(tmp_path):
    frame = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "oversampler": ["A", "A"],
            "metric": ["hassanat", "hassanat"],
            "hidden_ratio": [0.2, 0.2],
            "run": [0, 1],
            "error_rate": [np.nan, np.inf],
        }
    )
    out = tmp_path / "benchmark.json"

    export_benchmark_results(frame, str(out), fmt="json")
    payload = _strict_loads(out.read_text(encoding="utf-8"))

    assert payload[0]["mean"] is None
