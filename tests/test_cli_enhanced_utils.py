import json
import platform
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from oversampleqa import cli_enhanced
from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa.cli_enhanced import (
    analyze_dataset,
    cli,
    explain_ratio,
    generate_recommendations,
    interpret_error_rate,
    load_checkpoint,
    save_checkpoint,
)


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(content.strip(), encoding="utf-8")
    return path


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


def test_manifest_runner_dispatches_validation_jobs(tmp_path, monkeypatch):
    manifest = tmp_path / "experiment.yaml"
    data = tmp_path / "data.csv"
    data.write_text("x,target\n0,0\n1,1\n", encoding="utf-8")
    manifest.write_text(
        """
version: 1
output: ignored-by-cli-override
defaults:
  target: target
  minority_label: 1
  metric: hassanat
  hidden_ratio: 0.1
  random_state: 42
  n_repeats: 5
  export: [json, markdown]
  resume: true
datasets:
  toy:
    path: data.csv
experiments:
  - name: SMOTE baseline
    dataset: toy
    oversampler: SMOTE
    calibrate: true
  - name: adasyn
    dataset: data.csv
    oversampler: ADASYN
    output: custom-output
    export: yaml
""".strip(),
        encoding="utf-8",
    )

    calls = []

    def fake_validation(**kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {"error_rate": 0.123}

    monkeypatch.setattr(cli_enhanced, "run_validation_with_progress", fake_validation)

    output = tmp_path / "runs"
    result = CliRunner().invoke(
        cli,
        ["run", str(manifest), "--output", str(output), "--no-resume"],
    )

    assert result.exit_code == 0, result.output
    assert "Manifest completed: 2 experiment(s)" in result.output
    assert len(calls) == 2

    first, second = calls
    assert first["dataset_path"] == data
    assert first["target"] == "target"
    assert first["minority_label"] == 1
    assert first["oversampler_name"] == "SMOTE"
    assert first["metric"] == "hassanat"
    assert first["hidden_ratio"] == 0.1
    assert first["random_state"] == 42
    assert first["n_repeats"] == 5
    assert first["calibrate"] is True
    assert first["export_formats"] == ("json", "markdown")
    assert first["resume"] is False
    assert first["output_dir"] == output / "SMOTE-baseline"

    assert second["dataset_path"] == data
    assert second["oversampler_name"] == "ADASYN"
    assert second["export_formats"] == ("yaml",)
    assert second["output_dir"] == output / "custom-output"

    summary_path = output / "manifest_summary.json"
    assert summary_path.exists()
    assert metadata_sidecar_path(summary_path).exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["n_experiments"] == 2
    assert summary_payload["experiments"][0]["output"].endswith("SMOTE-baseline")
    assert (output / "resolved_manifest.yaml").exists()


def test_manifest_runner_resolves_manifest_defaults(tmp_path, monkeypatch):
    manifest = tmp_path / "experiment.yaml"
    data = tmp_path / "data.csv"
    data.write_text("x,dataset_target\n0,0\n1,1\n", encoding="utf-8")
    _write_manifest(
        manifest,
        """
version: 1
output: manifest-runs
defaults:
  target: default_target
  minority_label: 1
  oversampler: SMOTE
  metric: euclidean
  hidden_ratio: 0.25
  random_state:
  n_repeats: 3
  export:
  resume: false
  mlflow: true
datasets:
  mapped:
    path: data.csv
    target: dataset_target
    minority_label: 2
  alias: data.csv
experiments:
  - dataset: mapped
    oversampler: BorderlineSMOTE
  - name: "!!!"
    dataset: alias
    oversampler: ADASYN
""",
    )

    calls = []

    def fake_validation(**kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {"error_rate": 0.25}

    monkeypatch.setattr(cli_enhanced, "run_validation_with_progress", fake_validation)

    summary = cli_enhanced.run_experiment_manifest(manifest)

    assert summary["n_experiments"] == 2
    assert len(calls) == 2
    first, second = calls
    assert first["dataset_path"] == data
    assert first["target"] == "dataset_target"
    assert first["minority_label"] == 2
    assert first["oversampler_name"] == "BorderlineSMOTE"
    assert first["metric"] == "euclidean"
    assert first["hidden_ratio"] == 0.25
    assert first["random_state"] is None
    assert first["n_repeats"] == 3
    assert first["export_formats"] == ()
    assert first["resume"] is False
    assert first["mlflow_override"] is True
    assert first["output_dir"] == tmp_path / "manifest-runs" / "experiment-1"
    assert second["dataset_path"] == data
    assert second["output_dir"] == tmp_path / "manifest-runs" / "experiment"


def test_manifest_runner_allows_absolute_experiment_output(tmp_path):
    data = tmp_path / "data.csv"
    # The dataset has to exist: resolution now checks every path up front.
    data.write_text("x,target\n0,0\n1,1\n", encoding="utf-8")
    absolute_output = tmp_path / "absolute-output"
    manifest = {
        "experiments": [
            {
                "dataset": str(data),
                "oversampler": "SMOTE",
                "output": str(absolute_output),
            }
        ]
    }

    output_root, jobs = cli_enhanced._resolved_manifest_experiments(
        manifest, tmp_path / "experiment.yaml", None, None
    )

    assert output_root == tmp_path / "oversampleqa_runs"
    assert jobs[0]["output_dir"] == absolute_output


def test_manifest_runner_rejects_unknown_fields(tmp_path):
    manifest = tmp_path / "bad.yaml"
    _write_manifest(
        manifest,
        """
version: 1
experiments:
  - name: broken
    dataset: data.csv
    oversampler: SMOTE
    typo: true
""",
    )

    result = CliRunner().invoke(cli, ["run", str(manifest)])

    assert result.exit_code != 0
    assert "Unknown field(s) in experiment 1: typo" in result.output


def test_manifest_runner_rejects_non_validation_experiments(tmp_path):
    manifest = tmp_path / "bad.yaml"
    _write_manifest(
        manifest,
        """
version: 1
experiments:
  - name: benchmark
    type: benchmark
    dataset: data.csv
    oversampler: SMOTE
""".strip(),
    )

    result = CliRunner().invoke(cli, ["run", str(manifest)])

    assert result.exit_code != 0
    assert "Only validation experiments are supported" in result.output


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not: [valid", "Failed to load manifest"),
        (
            """
- not
- a
- mapping
""",
            "Manifest field 'manifest' must be a mapping",
        ),
        (
            """
version: 2
experiments:
  - dataset: data.csv
    oversampler: SMOTE
""",
            "Unsupported manifest version 2",
        ),
        (
            """
unknown: true
experiments:
  - dataset: data.csv
    oversampler: SMOTE
""",
            "Unknown manifest field(s): unknown",
        ),
        (
            """
defaults:
  typo: true
experiments:
  - dataset: data.csv
    oversampler: SMOTE
""",
            "Unknown manifest default(s): typo",
        ),
        (
            """
datasets:
  toy:
    path: data.csv
    typo: true
experiments:
  - dataset: toy
    oversampler: SMOTE
""",
            "Unknown field(s) in dataset 'toy': typo",
        ),
        (
            """
datasets:
  toy:
    target: target
experiments:
  - dataset: toy
    oversampler: SMOTE
""",
            "Dataset 'toy' must define 'path'",
        ),
        (
            """
experiments: []
""",
            "Manifest field 'experiments' must be a non-empty list",
        ),
        (
            """
experiments:
  - oversampler: SMOTE
""",
            "Experiment 1 must define 'dataset'",
        ),
        (
            """
experiments:
  - dataset:
      - data.csv
    oversampler: SMOTE
""",
            "Experiment field 'dataset' must be a non-empty string",
        ),
        (
            """
datasets:
  toy:
    path: 123
experiments:
  - dataset: toy
    oversampler: SMOTE
""",
            "Manifest field 'dataset.path' must be a non-empty path",
        ),
        (
            """
experiments:
  - dataset: data.csv
    oversampler: SMOTE
    export: [json, 1]
""",
            "Manifest field 'export' must list strings",
        ),
        (
            """
experiments:
  - dataset: data.csv
    oversampler: SMOTE
    export:
      format: json
""",
            "Manifest field 'export' must be a string or list",
        ),
    ],
)
def test_manifest_runner_rejects_invalid_manifests(tmp_path, body, message):
    manifest = _write_manifest(tmp_path / "bad.yaml", body)

    result = CliRunner().invoke(cli, ["run", str(manifest)])

    assert result.exit_code != 0
    assert message in result.output


# --- manifest failure modes ---
#
# Each of these ran to completion, or failed uselessly, before the checks that
# back them existed. The manifest parser's stated job is to fail before a long
# experiment starts; these are the cases where it did not.


def _toy_manifest(tmp_path: Path, body: str) -> Path:
    """A manifest plus the dataset it references."""
    (tmp_path / "data.csv").write_text("x,target\n0,0\n1,1\n", encoding="utf-8")
    return _write_manifest(tmp_path / "m.yaml", body)


def _run_manifest(tmp_path: Path, manifest: Path, fake, monkeypatch, *args: str):
    monkeypatch.setattr(cli_enhanced, "run_validation_with_progress", fake)
    return CliRunner().invoke(
        cli, ["run", str(manifest), "--output", str(tmp_path / "runs"), *args]
    )


def _recording_fake(calls: list):
    def fake(**kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {"error_rate": 0.1}

    return fake


def test_manifest_rejects_unknown_dataset_reference(tmp_path, monkeypatch):
    """A typo in a declared name was silently taken as an inline path.

    It then surfaced as a missing file whose path the author never wrote, and
    only once that experiment was reached.
    """
    manifest = _toy_manifest(
        tmp_path,
        """
version: 1
datasets:
  toy: {path: data.csv}
experiments:
  - {name: one, dataset: toyy}
""",
    )
    calls: list = []
    result = _run_manifest(tmp_path, manifest, _recording_fake(calls), monkeypatch)

    assert result.exit_code == 1
    assert "Unknown dataset 'toyy'" in result.output
    assert "Did you mean 'toy'?" in result.output
    assert calls == []


def test_manifest_still_allows_an_undeclared_inline_path(tmp_path, monkeypatch):
    """The convenience the previous test constrains must survive it."""
    manifest = _toy_manifest(
        tmp_path,
        """
version: 1
datasets:
  toy: {path: data.csv}
experiments:
  - {name: one, dataset: data.csv}
""",
    )
    calls: list = []
    result = _run_manifest(tmp_path, manifest, _recording_fake(calls), monkeypatch)

    assert result.exit_code == 0, result.output
    assert calls[0]["dataset_path"] == tmp_path / "data.csv"


def test_manifest_checks_every_dataset_before_running_any(tmp_path, monkeypatch):
    """A bad path in the last experiment used to be found after the others ran."""
    manifest = _toy_manifest(
        tmp_path,
        """
version: 1
datasets:
  toy: {path: data.csv}
experiments:
  - {name: one, dataset: toy}
  - {name: two, dataset: toy}
  - {name: three, dataset: missing.csv}
""",
    )
    calls: list = []
    result = _run_manifest(tmp_path, manifest, _recording_fake(calls), monkeypatch)

    assert result.exit_code == 1
    assert "Dataset file(s) not found" in result.output
    assert "'three'" in result.output
    assert calls == [], "no experiment may start when a later dataset is missing"


def test_manifest_rejects_experiments_sharing_an_output_directory(tmp_path, monkeypatch):
    """Slugifying is lossy, so distinct names could collide and overwrite."""
    manifest = _toy_manifest(
        tmp_path,
        """
version: 1
datasets:
  toy: {path: data.csv}
experiments:
  - {name: "run 1", dataset: toy}
  - {name: "run/1", dataset: toy}
""",
    )
    calls: list = []
    result = _run_manifest(tmp_path, manifest, _recording_fake(calls), monkeypatch)

    assert result.exit_code == 1
    assert "both write to" in result.output
    assert calls == []


def test_manifest_reports_a_bad_scalar_as_a_cli_error(tmp_path, monkeypatch):
    """``int('positive')`` reached the user as a traceback with no output."""
    manifest = _toy_manifest(
        tmp_path,
        """
version: 1
datasets:
  toy: {path: data.csv}
experiments:
  - {name: one, dataset: toy, minority_label: positive}
""",
    )
    result = _run_manifest(tmp_path, manifest, _recording_fake([]), monkeypatch)

    assert result.exit_code == 1
    assert "minority_label" in result.output
    assert "'positive'" in result.output
    assert "Experiment 1 ('one')" in result.output


def test_manifest_records_completed_experiments_when_a_later_one_fails(
    tmp_path, monkeypatch
):
    """The summary was never written, so finished work left no record."""
    manifest = _toy_manifest(
        tmp_path,
        """
version: 1
datasets:
  toy: {path: data.csv}
experiments:
  - {name: one, dataset: toy}
  - {name: two, dataset: toy}
  - {name: three, dataset: toy}
""",
    )
    seen = {"n": 0}

    def fake(**kwargs):
        seen["n"] += 1
        if seen["n"] == 2:
            raise RuntimeError("oversampler blew up")
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {"error_rate": 0.1}

    result = _run_manifest(tmp_path, manifest, fake, monkeypatch)

    assert result.exit_code == 1
    assert "1 of 3 experiment(s) completed" in result.output

    summary_path = tmp_path / "runs" / "manifest_summary.json"
    assert summary_path.exists(), "completed work must be recorded before raising"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["n_experiments"] == 1
    assert payload["n_planned"] == 3

    statuses = [(e["name"], e["status"]) for e in payload["experiments"]]
    assert statuses == [("one", "completed"), ("two", "failed")]
    assert "oversampler blew up" in payload["experiments"][1]["reason"]
    assert seen["n"] == 2, "the run stops at the failure rather than continuing"


# --- doctor ---


def test_doctor_reports_versions_not_just_presence():
    """"pandas [OK]" reproduces nothing.

    A version difference in numpy or scikit-learn changes results, not merely
    whether the code runs, so a bug report needs the numbers.
    """
    facts = cli_enhanced.diagnostics()

    assert facts["oversampleqa"] and facts["oversampleqa"][0].isdigit()
    assert facts["python"] == platform.python_version()
    assert facts["platform"]
    assert set(facts["packages"]) == {
        distribution for _, distribution in cli_enhanced.DIAGNOSTIC_PACKAGES
    }
    for distribution, version in facts["packages"].items():
        assert version, f"{distribution} is a runtime dependency and must resolve"


def test_python_support_check_compares_numbers_not_strings(monkeypatch):
    """The old check read `sys.version.split()[0] >= "3.10"`.

    That is a string comparison, and "3.9" sorts after "3.10", so 3.7, 3.8 and
    3.9 all passed a check that exists to reject them -- it could not fail for
    any Python 3.
    """
    for version, supported in (
        ((3, 9, 18), False),
        ((3, 8, 10), False),
        ((3, 10, 0), True),
        ((3, 13, 5), True),
    ):
        monkeypatch.setattr(cli_enhanced.sys, "version_info", version)
        assert cli_enhanced.diagnostics()["python_supported"] is supported, version


def test_doctor_names_a_missing_dependency(monkeypatch):
    def fake(module, distribution):
        return None if distribution == "scipy" else "1.2.3"

    monkeypatch.setattr(cli_enhanced, "_dependency_version", fake)
    result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code == 0
    assert "scipy" in result.output
    assert "not installed" in result.output


def test_doctor_flags_an_unsupported_python(monkeypatch):
    monkeypatch.setattr(cli_enhanced.sys, "version_info", (3, 9, 18))
    result = CliRunner().invoke(cli, ["doctor"])

    assert "needs 3.10+" in result.output or "not supported" in result.output


def test_dependency_version_reports_a_missing_module_as_absent():
    """The real function, not a stand-in: its error paths are the point."""
    assert cli_enhanced._dependency_version("no_such_module_xyz", "no-such-dist") is None


def test_dependency_version_handles_an_importable_module_with_no_distribution():
    """Importable but not installed as a distribution.

    A vendored module, or one on PYTHONPATH from a source tree. It is present,
    so reporting it missing would be wrong, but there is no version to give.
    """
    assert cli_enhanced._dependency_version("json", "not-a-real-distribution") == "unknown"


def test_doctor_reports_a_healthy_environment():
    """The happy path, which the failure-path tests did not reach."""
    result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code == 0
    assert "All required components are present" in result.output
    assert "OversampleQA" in result.output
