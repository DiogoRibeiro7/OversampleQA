import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from oversampleqa.cli_enhanced import CLIConfig, cli, suggest_parameters
from oversampleqa.config_templates import generate_config_file


def build_dataset(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 4.2, 4.5, 4.8],
            "f2": [1.0, 1.1, 1.2, 3.5, 3.6, 3.8],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_cli_validate_basic(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    output_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "validate",
            str(dataset),
            "--target",
            "target",
            "--oversampler",
            "RandomOverSampler",
            "--no-resume",
            "--export",
            "json",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Validation completed" in result.output
    result_file = output_dir / "validation_results.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    assert data["dataset"].endswith("data.csv")


def test_cli_completion_instructions():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "bash"])
    assert result.exit_code == 0
    assert "Shell completion setup" in result.output
    assert "bash" in result.output


def test_config_load_save(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config = CLIConfig(config_path)
    config.data["defaults"]["metric"] = "euclidean"
    config.save_config()

    reloaded = CLIConfig(config_path)
    assert reloaded.data["defaults"]["metric"] == "euclidean"


def test_config_template_generation(tmp_path: Path):
    destination = generate_config_file("production", str(tmp_path / "production.yaml"))
    assert destination.exists()
    payload = yaml.safe_load(destination.read_text())
    assert payload["profiles"]["production"]["cache_results"] is True


def test_suggest_parameters_balanced_dataset():
    params = suggest_parameters(
        {"n_samples": 100, "n_features": 5, "imbalance_ratio": 0.4, "minority": 40, "majority": 60}
    )
    assert params["oversampler"] == "RandomOverSampler"
    assert params["metric"] in {"cosine", "euclidean"}


def test_benchmark_command(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--output", str(tmp_path / "bench")])
    assert result.exit_code == 0, result.output
    summary = tmp_path / "bench" / "benchmark_summary.csv"
    assert summary.exists()
