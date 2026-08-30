"""Strict JSON export regression tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa._json import strict_json_dumps
from oversampleqa.advanced_benchmark import (
    StatisticalBenchmark,
    _significant_pairwise,
)
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


def _one_dataset_two_samplers(errors_a: list[float], errors_b: list[float]) -> pd.DataFrame:
    """A minimal frame with the columns the pairwise analysis reads."""
    return pd.DataFrame(
        [
            {
                "dataset_name": "d",
                "metric": "hassanat",
                "oversampler_name": "A",
                "error_rates": errors_a,
            },
            {
                "dataset_name": "d",
                "metric": "hassanat",
                "oversampler_name": "B",
                "error_rates": errors_b,
            },
        ]
    )


def _pairwise_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    benchmark = StatisticalBenchmark.__new__(StatisticalBenchmark)
    benchmark.correction_method = "holm"
    return benchmark._add_statistical_analysis(frame)


def test_pairwise_columns_are_strict_json_when_a_fold_was_skipped():
    """A skipped fold used to make these two columns unparseable.

    Skipped folds are kept as ``nan`` deliberately, so Wilcoxon returns ``nan``
    and the plain encoder wrote a bare ``NaN`` token. These were the only export
    path still calling ``json.dumps`` directly, and no test read them back with
    a strict parser, so the columns shipped as invalid JSON.
    """
    analysed = _pairwise_analysis(
        _one_dataset_two_samplers([0.1, 0.2, np.nan, 0.4], [0.2, 0.3, 0.4, 0.5])
    )

    for column in ("pairwise_p_values", "pairwise_effect_sizes"):
        raw = analysed[column].iloc[0]
        assert "NaN" not in raw, f"{column} contains a bare NaN token: {raw}"
        assert _strict_loads(raw) == {"A_vs_B": None}


def test_pairwise_columns_keep_real_values_intact():
    """The null above must mean "no comparison", not "always null"."""
    analysed = _pairwise_analysis(
        _one_dataset_two_samplers(
            [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
            [0.50, 0.51, 0.52, 0.53, 0.54, 0.55],
        )
    )

    p_values = _strict_loads(analysed["pairwise_p_values"].iloc[0])
    effects = _strict_loads(analysed["pairwise_effect_sizes"].iloc[0])

    assert p_values["A_vs_B"] is not None and 0.0 <= p_values["A_vs_B"] <= 1.0
    assert effects["A_vs_B"] == -1.0


def test_skipped_fold_reports_no_significant_comparison():
    """``null`` must not become a crash or a false positive downstream.

    ``_significant_pairwise`` compares each p-value against alpha. ``nan < a``
    was quietly ``False``; ``None < a`` would raise. The existing ``is not
    None`` guard covers it, and this pins that the two agree.
    """
    analysed = _pairwise_analysis(
        _one_dataset_two_samplers([0.1, 0.2, np.nan, 0.4], [0.2, 0.3, 0.4, 0.5])
    )

    assert _significant_pairwise(analysed, 0.05) == []
