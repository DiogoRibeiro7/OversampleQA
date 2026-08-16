"""Tests for the ``oversampleqa fidelity`` subcommand.

The command exists so the fidelity/diversity split is reachable without writing
Python. Its job is to surface the one number the error rate cannot provide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from oversampleqa.cli_enhanced import cli


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    majority = rng.normal(0.0, 1.0, (400, 3))
    minority = rng.normal(2.2, 1.0, (120, 3))
    frame = pd.DataFrame(np.vstack([majority, minority]), columns=["f1", "f2", "f3"])
    frame["target"] = [0] * 400 + [1] * 120
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def test_fidelity_command_is_registered():
    result = CliRunner().invoke(cli, ["--help"])
    assert "fidelity" in result.output


def test_fidelity_runs_and_reports(dataset):
    result = CliRunner().invoke(cli, ["fidelity", str(dataset), "--target", "target"])
    assert result.exit_code == 0, result.output
    assert "Memorisation ratio" in result.output
    assert "Coverage" in result.output


def test_fidelity_writes_json(dataset, tmp_path):
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli, ["fidelity", str(dataset), "--target", "target", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {
        "error_rate",
        "precision",
        "coverage",
        "memorisation_distance_ratio",
    } <= set(payload)


def test_fidelity_separates_a_copying_sampler(dataset, tmp_path):
    """The reason the command exists.

    RandomOverSampler duplicates real points, so its error rate says nothing
    about synthesis quality. The memorisation ratio is what exposes it.
    """
    results = {}
    for sampler in ("SMOTE", "RandomOverSampler"):
        out = tmp_path / f"{sampler}.json"
        outcome = CliRunner().invoke(
            cli,
            [
                "fidelity",
                str(dataset),
                "--target",
                "target",
                "--oversampler",
                sampler,
                "-o",
                str(out),
            ],
        )
        assert outcome.exit_code == 0, outcome.output
        results[sampler] = json.loads(out.read_text(encoding="utf-8"))

    assert results["RandomOverSampler"]["memorisation_distance_ratio"] == pytest.approx(
        0.0
    )
    assert results["SMOTE"]["memorisation_distance_ratio"] > 0.1
    # The error rate alone does not separate them.
    gap = abs(
        results["SMOTE"]["error_rate"] - results["RandomOverSampler"]["error_rate"]
    )
    assert gap < 0.2


def test_fidelity_rejects_a_missing_target(dataset):
    result = CliRunner().invoke(
        cli, ["fidelity", str(dataset), "--target", "not_a_column"]
    )
    assert result.exit_code != 0
    assert "not found" in result.output
