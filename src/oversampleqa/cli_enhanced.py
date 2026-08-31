"""Enhanced command-line interface for OversampleQA."""

from __future__ import annotations

import copy
import json
import logging
import platform
import re
import sys
import textwrap
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import get_close_matches
from importlib import metadata
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import __version__ as _PACKAGE_VERSION
from ._export_metadata import write_export_metadata
from ._json import strict_json_dumps, write_json
from .advanced_benchmark import (
    StatisticalBenchmark,
    create_benchmark_report,
    format_statistical_summary,
)
from .benchmark import export_benchmark_results, load_standard_datasets
from .config_templates import CONFIG_TEMPLATES, generate_config_file
from .memory_efficient_validator import MemoryEfficientValidator
from .types import ValidationDetails
from .validator import validate_oversampling

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {
        "target": "target",
        "minority_label": 1,
        "oversampler": "SMOTE",
        "metric": "hassanat",
        "hidden_ratio": 0.1,
        "resume": True,
        "export": ["json"],
    },
    "profiles": {
        "quick": {"hidden_ratio": 0.1, "metric": "euclidean", "n_runs": 1},
        "thorough": {"hidden_ratio": 0.25, "metric": "hassanat", "n_runs": 10},
        "research": {"hidden_ratio": 0.5, "metric": "hassanat", "n_runs": 25},
    },
    "integrations": {
        "mlflow": {"enabled": False, "experiment_name": "OversampleQA"},
    },
}

CHECKPOINT_FILE = ".oversampleqa_run.json"
SUPPORTED_EXPORTS = {"json", "yaml", "markdown"}
KNOWN_PARAMS = {
    "target",
    "minority_label",
    "oversampler",
    "metric",
    "hidden_ratio",
    "export",
    "resume",
}
MANIFEST_VERSION = 1
MANIFEST_DEFAULT_KEYS = KNOWN_PARAMS | {
    "random_state",
    "n_repeats",
    "calibrate",
    "mlflow",
}
MANIFEST_DATASET_KEYS = {"path", "target", "minority_label"}
MANIFEST_EXPERIMENT_KEYS = MANIFEST_DEFAULT_KEYS | {
    "name",
    "dataset",
    "type",
    "output",
}


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


@dataclass
class CLIConfig:
    """Configuration management for the enhanced CLI."""

    # Optional as an *input*: __post_init__ always resolves it to a concrete
    # Path, so every read after construction is non-None.
    config_file: Path = None
    console: Console = field(default_factory=Console)
    data: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        default_path = Path.home() / ".oversampleqa" / "config.yaml"
        # Always concrete after this point; the Optional is an input convenience.
        self.config_file = (self.config_file or default_path).expanduser()
        self.data = self.load_config()

    def load_config(self) -> dict[str, Any]:
        """Load configuration from disk and merge with defaults.

        Returns:
            Merged configuration dictionary.
        """

        if not self.config_file.exists():
            return copy.deepcopy(DEFAULT_CONFIG)

        try:
            if self.config_file.suffix.lower() == ".json":
                raw = json.loads(self.config_file.read_text(encoding="utf-8"))
            else:
                raw = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigValidationError(f"Failed to load config: {exc}") from exc

        if raw is None:
            raw = {}

        self._validate_keys(raw)

        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.setdefault("profiles", {}).update(raw.get("profiles", {}))
        merged.setdefault("defaults", {}).update(raw.get("defaults", {}))
        merged.setdefault("integrations", {}).update(raw.get("integrations", {}))
        return merged

    def save_config(self, config: dict[str, Any] | None = None) -> None:
        """Persist configuration to disk.

        Args:
            config: Optional config data to persist, defaults to current data.
        """

        payload = copy.deepcopy(config or self.data)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with self.config_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def _validate_keys(self, config: dict[str, Any]) -> None:
        """Validate configuration keys and provide suggestions.

        Args:
            config: Configuration dictionary to validate.
        """

        def check_section(section: str, allowed: Iterable[str]) -> None:
            block = config.get(section, {})
            if not isinstance(block, dict):
                raise ConfigValidationError(f"Section '{section}' must be a mapping")
            for key in block:
                if key not in allowed:
                    suggestion = get_close_matches(key, allowed, n=1)
                    message = f"Unknown key '{key}' in section '{section}'"
                    if suggestion:
                        message += f". Did you mean '{suggestion[0]}'?"
                    raise ConfigValidationError(message)

        allowed_profile_keys = KNOWN_PARAMS | {
            "n_runs",
            "include_plots",
            "cache_results",
            "statistical_tests",
        }
        check_section("defaults", KNOWN_PARAMS)
        profiles = config.get("profiles", {})
        if isinstance(profiles, dict):
            for profile_name, params in profiles.items():
                if not isinstance(params, dict):
                    raise ConfigValidationError(
                        f"Profile '{profile_name}' must be a mapping"
                    )
                for key in params:
                    if key not in allowed_profile_keys:
                        suggestion = get_close_matches(key, allowed_profile_keys, n=1)
                        message = f"Unknown key '{key}' in profile '{profile_name}'"
                        if suggestion:
                            message += f". Did you mean '{suggestion[0]}'?"
                        raise ConfigValidationError(message)

    def resolve_defaults(self, profile: str | None = None) -> dict[str, Any]:
        """Return default parameters merged with optional profile.

        Args:
            profile: Optional profile name to apply.

        Returns:
            Merged defaults dictionary.
        """

        defaults = copy.deepcopy(DEFAULT_CONFIG["defaults"])
        defaults.update(self.data.get("defaults", {}))

        if profile:
            profile_data = self.get_profile(profile)
            defaults.update(profile_data)

        return defaults

    def get_profile(self, name: str) -> dict[str, Any]:
        """Return configuration profile parameters by name.

        Args:
            name: Profile name.

        Returns:
            Profile parameter mapping.
        """

        profiles = {**DEFAULT_CONFIG["profiles"], **self.data.get("profiles", {})}
        if name not in profiles:
            suggestion = get_close_matches(name, profiles.keys(), n=1)
            raise ConfigValidationError(
                f"Profile '{name}' not found."
                + (f" Did you mean '{suggestion[0]}'?" if suggestion else "")
            )
        return dict(profiles[name])

    def list_profiles(self) -> list[tuple[str, dict[str, Any]]]:
        """List all available profiles.

        Returns:
            List of ``(name, profile)`` pairs.
        """

        profiles = {**DEFAULT_CONFIG["profiles"], **self.data.get("profiles", {})}
        return sorted(profiles.items())


def load_dataset(
    dataset_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV/Parquet and split into features/target.

    Args:
        dataset_path: Dataset path.
        target_column: Target column name.

    Returns:
        Tuple of feature DataFrame and target Series.
    """

    if dataset_path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    if target_column not in df.columns:
        raise click.ClickException(
            f"Target column '{target_column}' not found in dataset."
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def load_checkpoint(output_dir: Path | None) -> dict[str, Any] | None:
    """Load a saved run checkpoint from the output directory.

    Args:
        output_dir: Output directory that may contain a checkpoint file.

    Returns:
        Parsed checkpoint payload or ``None`` if unavailable.
    """
    if not output_dir:
        return None
    checkpoint_path = output_dir / CHECKPOINT_FILE
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_checkpoint(output_dir: Path | None, payload: dict[str, Any]) -> None:
    """Save a run checkpoint to the output directory.

    Args:
        output_dir: Directory to write the checkpoint into.
        payload: Serializable results payload.
    """
    if not output_dir:
        return
    checkpoint_path = output_dir / CHECKPOINT_FILE
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint_path, payload)


def estimate_runtime(seconds: float) -> str:
    """Return a human-friendly runtime estimate.

    Args:
        seconds: Seconds to format.

    Returns:
        Formatted estimate string.
    """

    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if sec or not parts:
        parts.append(f"{sec}s")
    return " ~".join(parts)


def run_validation_with_progress(
    dataset_path: Path,
    target: str,
    minority_label: int,
    oversampler_name: str,
    metric: str,
    hidden_ratio: float,
    export_formats: Iterable[str],
    resume: bool,
    output_dir: Path | None,
    mlflow_override: bool,
    mlflow_config: dict[str, Any] | None,
    verbose: bool,
    random_state: int | None = 42,
    n_repeats: int = 1,
    calibrate: bool = False,
) -> dict[str, Any]:
    """Run validation with rich progress feedback.

    Args:
        dataset_path: Dataset path.
        target: Target column name.
        minority_label: Minority class label.
        oversampler_name: Oversampler class name.
        metric: Distance metric name.
        hidden_ratio: Fraction of majority to hide.
        random_state: Seed for the hold-out split.
        n_repeats: Number of independent hold-out splits.
        calibrate: Whether to compute the null calibration.
        export_formats: Formats to export.
        resume: Whether to reuse cached results.
        output_dir: Output directory for artifacts.
        mlflow_override: Force MLflow logging.
        mlflow_config: MLflow configuration dict.
        verbose: Whether to print results.

    Returns:
        Results dictionary.
    """

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(output_dir)
    if checkpoint and checkpoint.get("status") == "completed" and resume:
        console.print("[green]Using cached results from previous run.[/green]")
        return checkpoint["results"]

    stages = [
        "Loading dataset",
        "Analyzing class balance",
        "Fitting oversampler",
        "Validating samples",
        "Finalizing",
    ]

    results: dict[str, Any] = {}
    mlflow_settings = mlflow_config or {}
    mlflow_active = mlflow_override or bool(mlflow_settings.get("enabled"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task: TaskID = progress.add_task("Preparing validation", total=len(stages))

        progress.update(task, description=stages[0])
        X, y = load_dataset(dataset_path, target)
        progress.advance(task)

        n_samples, n_features = X.shape
        minority_count = int((y == minority_label).sum())
        majority_count = n_samples - minority_count

        progress.update(task, description=stages[1])
        imbalance_ratio = minority_count / max(majority_count, 1)
        runtime_estimate = estimate_runtime(
            n_samples * n_features * hidden_ratio / 3000 + 10
        )
        results.update(
            {
                "dataset": str(dataset_path),
                "n_samples": n_samples,
                "n_features": n_features,
                "minority_count": minority_count,
                "majority_count": majority_count,
                "imbalance_ratio": imbalance_ratio,
                "runtime_estimate": runtime_estimate,
            }
        )
        progress.advance(task)

        progress.update(task, description=stages[2])
        mod = __import__("imblearn.over_sampling", fromlist=[oversampler_name])
        oversampler_cls = getattr(mod, oversampler_name)
        oversampler = oversampler_cls()
        progress.advance(task)

        progress.update(task, description=stages[3])
        start = time.perf_counter()
        dispersion: dict[str, Any] = {}
        if n_samples > 20_000:
            validator = MemoryEfficientValidator()
            error_rate = validator.validate_oversampling(
                X.values,
                y.values,
                minority_label=minority_label,
                oversampler=oversampler,
                hidden_ratio=hidden_ratio,
                metric=metric,
                random_state=random_state,
            )
        elif n_repeats > 1:
            details = validate_oversampling(
                np.asarray(X.values),
                np.asarray(y.values),
                minority_label=minority_label,
                oversampler=oversampler,
                hidden_ratio=hidden_ratio,
                metric=metric,
                random_state=random_state,
                n_repeats=n_repeats,
                return_details=True,
            )
            # return_details=True always yields ValidationDetails; the runtime
            # check narrows the union without suppressing the type.
            if not isinstance(details, ValidationDetails):
                raise TypeError(
                    "validate_oversampling(return_details=True) must return "
                    f"ValidationDetails, got {type(details).__name__}"
                )
            error_rate = details.error_rate
            dispersion = {
                "n_repeats": details.n_repeats,
                "std": details.std,
                "interval": list(details.interval) if details.interval else None,
                "rates": list(details.rates),
            }
        else:
            error_rate = validate_oversampling(
                X.values,
                y.values,
                minority_label=minority_label,
                oversampler=oversampler,
                hidden_ratio=hidden_ratio,
                metric=metric,
                random_state=random_state,
            )
        elapsed = time.perf_counter() - start

        calibration: dict[str, Any] = {}
        if calibrate:
            from .inference import null_error_rate

            result = null_error_rate(
                np.asarray(X.values),
                np.asarray(y.values),
                minority_label,
                float(error_rate),
                hidden_ratio=hidden_ratio,
                metric=metric,
                random_state=random_state,
            )
            calibration = {
                "calibration": result.to_dict(),
                "calibration_reading": result.interpret(),
            }
        progress.advance(task)

        progress.update(task, description=stages[4])
        results.update(
            {
                "error_rate": float(error_rate),
                "metric": metric,
                "hidden_ratio": hidden_ratio,
                "random_state": random_state,
                **dispersion,
                **calibration,
                "oversampler": oversampler_name,
                "minority_label": minority_label,
                "elapsed_seconds": elapsed,
                "mlflow_experiment": mlflow_settings.get(
                    "experiment_name", "OversampleQA"
                ),
            }
        )
        progress.advance(task)

    if output_dir:
        save_checkpoint(
            output_dir,
            {
                "status": "completed",
                "results": results,
            },
        )

    export_results(results, export_formats, output_dir)
    if mlflow_active:
        integrate_with_mlflow(results, mlflow_settings)

    if verbose:
        display_results(results)

    return results


def export_results(
    results: dict[str, Any], formats: Iterable[str], output_dir: Path | None
) -> None:
    """Export results to requested formats.

    Args:
        results: Results dictionary.
        formats: Output formats.
        output_dir: Output directory, if any.
    """

    if not output_dir:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in SUPPORTED_EXPORTS:
            console.print(
                f"[yellow]Skipping unsupported export format '{fmt}'.[/yellow]"
            )
            continue
        if fmt_lower == "json":
            artifact = output_dir / "validation_results.json"
            write_json(artifact, results)
            write_export_metadata(
                artifact, export_kind="validation_results", data=results
            )
        elif fmt_lower == "yaml":
            artifact = output_dir / "validation_results.yaml"
            artifact.write_text(
                yaml.safe_dump(results, sort_keys=False), encoding="utf-8"
            )
            write_export_metadata(
                artifact, export_kind="validation_results", data=results
            )
        elif fmt_lower == "markdown":
            markdown = textwrap.dedent(
                f"""
                # OversampleQA Validation Report

                - Dataset: `{results["dataset"]}`
                - Samples: {results["n_samples"]}
                - Features: {results["n_features"]}
                - Minority Samples: {results["minority_count"]}
                - Majority Samples: {results["majority_count"]}
                - Imbalance Ratio: {results["imbalance_ratio"]:.3f}
                - Hidden Ratio: {results["hidden_ratio"]}
                - Metric: {results["metric"]}
                - Oversampler: {results["oversampler"]}
                - Error Rate: {results["error_rate"]:.3f}
                - Runtime: {results["elapsed_seconds"]:.2f}s
                - Estimated Runtime: {results["runtime_estimate"]}
                """
            ).strip()
            artifact = output_dir / "validation_results.md"
            artifact.write_text(markdown + "\n", encoding="utf-8")
            write_export_metadata(
                artifact, export_kind="validation_results", data=results
            )


def _slugify(value: str) -> str:
    """Return a filesystem-friendly name for a manifest experiment."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "experiment"


def _looks_like_path(value: str) -> bool:
    """Whether a dataset reference is meant as a path rather than a name."""
    return "/" in value or "\\" in value or Path(value).suffix != ""


def _coerce(value: Any, kind: Callable[[Any], Any], field: str, where: str) -> Any:
    """Convert a manifest scalar, reporting failure as a CLI error.

    ``int("positive")`` reaches the user as a bare traceback with no output at
    all, which says nothing about which experiment or which field was wrong.
    """
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"{where}: field {field!r} has value {value!r}, "
            f"which is not a valid {kind.__name__}"
        ) from exc


def _manifest_path(base_dir: Path, value: Any, field: str) -> Path:
    """Resolve a path from the manifest, relative to the manifest file."""
    if not isinstance(value, str) or not value:
        raise click.ClickException(f"Manifest field '{field}' must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    """Return ``value`` as a mapping or raise a Click-friendly error."""
    if not isinstance(value, dict):
        raise click.ClickException(f"Manifest field '{field}' must be a mapping")
    return dict(value)


def load_experiment_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate an experiment manifest.

    The first manifest version is deliberately narrow: it runs validation
    experiments and leaves fidelity/benchmark orchestration for later roadmap
    slices. Keeping this parser explicit makes unsupported manifest fields fail
    before a long experiment starts.
    """
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise click.ClickException(f"Failed to load manifest: {exc}") from exc

    manifest = _require_mapping(raw, "manifest")
    allowed_root = {"version", "output", "defaults", "datasets", "experiments"}
    unknown_root = sorted(set(manifest) - allowed_root)
    if unknown_root:
        raise click.ClickException(
            "Unknown manifest field(s): " + ", ".join(unknown_root)
        )

    version = manifest.get("version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise click.ClickException(
            f"Unsupported manifest version {version!r}; expected {MANIFEST_VERSION}"
        )

    defaults = _require_mapping(manifest.get("defaults", {}), "defaults")
    unknown_defaults = sorted(set(defaults) - MANIFEST_DEFAULT_KEYS)
    if unknown_defaults:
        raise click.ClickException(
            "Unknown manifest default(s): " + ", ".join(unknown_defaults)
        )

    datasets = _require_mapping(manifest.get("datasets", {}), "datasets")
    for name, spec in datasets.items():
        if isinstance(spec, str):
            continue
        dataset_spec = _require_mapping(spec, f"datasets.{name}")
        unknown_dataset = sorted(set(dataset_spec) - MANIFEST_DATASET_KEYS)
        if unknown_dataset:
            raise click.ClickException(
                f"Unknown field(s) in dataset '{name}': " + ", ".join(unknown_dataset)
            )
        if "path" not in dataset_spec:
            raise click.ClickException(f"Dataset '{name}' must define 'path'")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise click.ClickException(
            "Manifest field 'experiments' must be a non-empty list"
        )
    for index, experiment in enumerate(experiments, start=1):
        experiment_spec = _require_mapping(experiment, f"experiments[{index}]")
        unknown_experiment = sorted(set(experiment_spec) - MANIFEST_EXPERIMENT_KEYS)
        if unknown_experiment:
            raise click.ClickException(
                f"Unknown field(s) in experiment {index}: "
                + ", ".join(unknown_experiment)
            )
        if experiment_spec.get("type", "validation") != "validation":
            raise click.ClickException(
                "Only validation experiments are supported by manifest version 1"
            )
        if "dataset" not in experiment_spec:
            raise click.ClickException(f"Experiment {index} must define 'dataset'")

    return manifest


def _manifest_dataset_spec(
    manifest: dict[str, Any], experiment: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    """Resolve the dataset referenced by a manifest experiment."""
    datasets = _require_mapping(manifest.get("datasets", {}), "datasets")
    dataset_ref = experiment["dataset"]
    if not isinstance(dataset_ref, str) or not dataset_ref:
        raise click.ClickException(
            "Experiment field 'dataset' must be a non-empty string"
        )

    if dataset_ref in datasets:
        dataset = datasets[dataset_ref]
        if isinstance(dataset, str):
            dataset_spec = {"path": dataset}
        else:
            dataset_spec = _require_mapping(dataset, f"datasets.{dataset_ref}")
    else:
        # An undeclared reference is taken as an inline path, which is the
        # convenience the syntax is for -- but it also swallows a typo in a
        # declared name, reporting it much later as a missing file the author
        # never wrote. Only a reference that looks like a path gets that
        # benefit; a bare word is judged against the declared names instead.
        if datasets and not _looks_like_path(dataset_ref):
            hint = get_close_matches(dataset_ref, sorted(datasets), n=1)
            suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
            known = ", ".join(sorted(datasets)) or "none"
            raise click.ClickException(
                f"Unknown dataset {dataset_ref!r}; declared datasets are: "
                f"{known}.{suggestion} Use an explicit path (with a suffix or a "
                "directory separator) to reference a file that is not declared."
            )
        dataset_spec = {"path": dataset_ref}

    resolved = dict(dataset_spec)
    resolved["path"] = _manifest_path(manifest_dir, resolved["path"], "dataset.path")
    return resolved


def _normalise_exports(value: Any) -> tuple[str, ...]:
    """Return manifest export formats as a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        if not all(isinstance(item, str) for item in value):
            raise click.ClickException("Manifest field 'export' must list strings")
        return tuple(value)
    raise click.ClickException("Manifest field 'export' must be a string or list")


def _resolved_manifest_experiments(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_override: Path | None,
    resume_override: bool | None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Build validated, fully resolved validation jobs from a manifest."""
    manifest_dir = manifest_path.parent
    defaults = copy.deepcopy(DEFAULT_CONFIG["defaults"])
    defaults.update(_require_mapping(manifest.get("defaults", {}), "defaults"))

    configured_output = output_override or manifest.get("output", "oversampleqa_runs")
    output_root = _manifest_path(manifest_dir, str(configured_output), "output")

    jobs: list[dict[str, Any]] = []
    for index, raw_experiment in enumerate(manifest["experiments"], start=1):
        experiment = _require_mapping(raw_experiment, f"experiments[{index}]")
        dataset = _manifest_dataset_spec(manifest, experiment, manifest_dir)

        params = copy.deepcopy(defaults)
        params.update({key: value for key, value in dataset.items() if key != "path"})
        params.update(
            {
                key: value
                for key, value in experiment.items()
                if key not in {"dataset", "name", "type", "output"}
            }
        )

        name = str(experiment.get("name") or f"experiment-{index}")
        experiment_output = experiment.get("output")
        if experiment_output is None:
            output_dir = output_root / _slugify(name)
        else:
            output_dir = Path(str(experiment_output)).expanduser()
            if not output_dir.is_absolute():
                output_dir = output_root / output_dir

        resume = bool(params.get("resume", True))
        if resume_override is not None:
            resume = resume_override

        where = f"Experiment {index} ({name!r})"
        jobs.append(
            {
                "name": name,
                "dataset_path": dataset["path"],
                "target": params["target"],
                "minority_label": _coerce(
                    params["minority_label"], int, "minority_label", where
                ),
                "oversampler_name": str(params["oversampler"]),
                "metric": str(params["metric"]),
                "hidden_ratio": _coerce(
                    params["hidden_ratio"], float, "hidden_ratio", where
                ),
                "random_state": (
                    None
                    if params.get("random_state") is None
                    else _coerce(
                        params.get("random_state", 42), int, "random_state", where
                    )
                ),
                "n_repeats": _coerce(params.get("n_repeats", 1), int, "n_repeats", where),
                "calibrate": bool(params.get("calibrate", False)),
                "export_formats": _normalise_exports(params.get("export", [])),
                "resume": resume,
                "mlflow": bool(params.get("mlflow", False)),
                "output_dir": output_dir,
            }
        )

    _reject_colliding_outputs(jobs)
    _reject_missing_datasets(jobs)
    return output_root, jobs


def _reject_colliding_outputs(jobs: list[dict[str, Any]]) -> None:
    """Two experiments writing to one directory silently lose a result.

    Slugifying is lossy -- ``run 1`` and ``run/1`` both become ``run-1`` -- so
    distinct experiments could share an output directory, and the second
    overwrote the first with no warning. Naming is the author's to fix; the
    runner cannot guess which result was meant to survive.
    """
    seen: dict[Path, str] = {}
    for job in jobs:
        first = seen.get(job["output_dir"])
        if first is not None:
            raise click.ClickException(
                f"Experiments {first!r} and {job['name']!r} both write to "
                f"{job['output_dir']}; give one an explicit distinct 'output'."
            )
        seen[job["output_dir"]] = job["name"]


def _reject_missing_datasets(jobs: list[dict[str, Any]]) -> None:
    """Check every dataset before the first experiment starts.

    A path that does not exist used to surface only when its experiment was
    reached, so a typo in the last of five experiments was reported after the
    first four had already run -- exactly what parsing the manifest up front is
    supposed to prevent.
    """
    missing = [
        f"{job['name']!r} -> {job['dataset_path']}"
        for job in jobs
        if not job["dataset_path"].exists()
    ]
    if missing:
        raise click.ClickException(
            "Dataset file(s) not found:\n  " + "\n  ".join(missing)
        )


def run_experiment_manifest(
    manifest_path: Path,
    *,
    output_override: Path | None = None,
    resume_override: bool | None = None,
    mlflow_config: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run every validation experiment in a checked-in YAML manifest."""
    manifest = load_experiment_manifest(manifest_path)
    output_root, jobs = _resolved_manifest_experiments(
        manifest, manifest_path, output_override, resume_override
    )
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_payload = {
        "version": MANIFEST_VERSION,
        "experiments": [
            {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else list(value)
                    if key == "export_formats"
                    else value
                )
                for key, value in job.items()
            }
            for job in jobs
        ],
    }
    resolved_path = output_root / "resolved_manifest.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved_payload, sort_keys=False), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    failure: Exception | None = None
    for job in jobs:
        console.print(
            Panel.fit(f"Running manifest experiment: {job['name']}", style="bold blue")
        )
        try:
            results = run_validation_with_progress(
            dataset_path=job["dataset_path"],
            target=job["target"],
            minority_label=job["minority_label"],
            oversampler_name=job["oversampler_name"],
            metric=job["metric"],
            hidden_ratio=job["hidden_ratio"],
            random_state=job["random_state"],
            n_repeats=job["n_repeats"],
            calibrate=job["calibrate"],
            export_formats=job["export_formats"],
            resume=job["resume"],
            output_dir=job["output_dir"],
            mlflow_override=job["mlflow"],
                mlflow_config=mlflow_config or {},
                verbose=verbose,
            )
        except Exception as exc:
            # The run still stops here, but the experiments that did finish are
            # written down before it does. Losing the summary for four completed
            # experiments because the fifth failed discards hours of work and
            # leaves no record of what ran.
            failure = exc
            summaries.append(
                {
                    "name": job["name"],
                    "type": "validation",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "dataset": str(job["dataset_path"]),
                    "output": str(job["output_dir"]),
                    "oversampler": job["oversampler_name"],
                    "metric": job["metric"],
                    "hidden_ratio": job["hidden_ratio"],
                    "random_state": job["random_state"],
                    "n_repeats": job["n_repeats"],
                    "error_rate": None,
                }
            )
            break
        summaries.append(
            {
                "name": job["name"],
                "type": "validation",
                "status": "completed",
                "dataset": str(job["dataset_path"]),
                "output": str(job["output_dir"]),
                "oversampler": job["oversampler_name"],
                "metric": job["metric"],
                "hidden_ratio": job["hidden_ratio"],
                "random_state": job["random_state"],
                "n_repeats": job["n_repeats"],
                "error_rate": results.get("error_rate"),
            }
        )

    completed = [s for s in summaries if s["status"] == "completed"]
    summary = {
        "manifest": str(manifest_path),
        "manifest_version": MANIFEST_VERSION,
        "n_experiments": len(completed),
        "n_planned": len(jobs),
        "experiments": summaries,
    }
    summary_path = output_root / "manifest_summary.json"
    write_json(summary_path, summary)
    write_export_metadata(
        summary_path,
        export_kind="manifest_summary",
        data=summary,
        extra={"source": {"manifest": str(manifest_path)}},
    )
    console.print(f"[green]Manifest results stored in {output_root}[/green]")
    if failure is not None:
        raise click.ClickException(
            f"Experiment {summaries[-1]['name']!r} failed: {failure}. "
            f"{len(completed)} of {len(jobs)} experiment(s) completed; "
            f"see {summary_path}."
        ) from failure
    return summary


def integrate_with_mlflow(results: dict[str, Any], settings: dict[str, Any]) -> None:
    """Log results to MLflow if available.

    Args:
        results: Results dictionary.
        settings: MLflow settings dictionary.
    """

    try:
        import mlflow
    except ImportError:  # pragma: no cover - optional dependency
        console.print(
            "[yellow]MLflow integration requested but mlflow is not installed.[/yellow]"
        )
        return

    experiment = (
        results.get("mlflow_experiment")
        or settings.get("experiment_name")
        or "OversampleQA"
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="oversampleqa-validation"):
        mlflow.log_params(
            {
                "dataset": results["dataset"],
                "metric": results["metric"],
                "hidden_ratio": results["hidden_ratio"],
                "oversampler": results["oversampler"],
            }
        )
        mlflow.log_metrics({"error_rate": results["error_rate"]})
        mlflow.log_metric("elapsed_seconds", results["elapsed_seconds"])


def display_results(results: dict[str, Any]) -> None:
    """Pretty-print validation results.

    Args:
        results: Results dictionary.
    """

    table = Table(
        title="Validation Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Interpretation", style="yellow")

    error_rate = results["error_rate"]

    table.add_row(
        "Error Rate",
        f"{error_rate:.3f}",
        interpret_error_rate(error_rate),
    )
    if results.get("n_repeats", 1) > 1:
        std = results.get("std", float("nan"))
        interval = results.get("interval")
        spread = f"{error_rate:.3f} ± {std:.3f}"
        if interval:
            spread += f"  [{interval[0]:.3f}, {interval[1]:.3f}]"
        table.add_row(
            f"Across {results['n_repeats']} splits",
            spread,
            "Spread of the hold-out split, not a population CI",
        )
    if results.get("calibration"):
        cal = results["calibration"]
        table.add_row(
            "vs. ideal generator",
            f"null {cal['null_mean']:.3f} (z={cal['z_score']:.2f})",
            "Within null = indistinguishable from ideal",
        )
        table.add_row(
            "vs. worst case",
            f"ceiling {cal['ceiling_mean']:.3f}",
            "Rate a wrong-distribution generator would score",
        )
    table.add_row(
        "Imbalance Ratio",
        f"{results['imbalance_ratio']:.3f}",
        explain_ratio(results["imbalance_ratio"]),
    )
    table.add_row(
        "Estimated Runtime",
        results["runtime_estimate"],
        "Projected duration for similar runs",
    )
    table.add_row(
        "Actual Runtime",
        f"{results['elapsed_seconds']:.2f}s",
        "Measured wall-clock execution time",
    )
    console.print(table)

    console.print("\n[bold]Recommendations:[/bold]")
    for recommendation in generate_recommendations(
        error_rate, results["imbalance_ratio"]
    ):
        console.print(f"- {recommendation}")


def interpret_error_rate(error_rate: float) -> str:
    """Return a qualitative interpretation for the error rate.

    Args:
        error_rate: Validation error rate.

    Returns:
        Human-friendly interpretation string.
    """
    if error_rate < 0.1:
        return "Excellent result - synthetic samples closely match the minority distribution."
    if error_rate < 0.3:
        return "Acceptable result - monitor drift and consider tuning the hidden ratio."
    return "Risky result - investigate feature overlap with hidden majority samples."


def explain_ratio(ratio: float) -> str:
    """Return a qualitative explanation of the imbalance ratio.

    Args:
        ratio: Minority-to-majority ratio.

    Returns:
        Human-friendly explanation string.
    """
    if ratio < 0.1:
        return "Highly imbalanced - oversampling essential."
    if ratio < 0.3:
        return "Moderately imbalanced - suitable for standard oversamplers."
    return "Near-balanced - consider alternative validation strategies."


def generate_recommendations(error_rate: float, imbalance_ratio: float) -> list[str]:
    """Return actionable recommendations based on diagnostics.

    Args:
        error_rate: Validation error rate.
        imbalance_ratio: Minority-to-majority ratio.

    Returns:
        List of recommendation strings.
    """
    tips = []
    if error_rate > 0.3:
        tips.append("Evaluate advanced oversamplers such as BorderlineSMOTE or ADASYN.")
    if imbalance_ratio < 0.1:
        tips.append(
            "Experiment with higher hidden ratios to stress-test synthetic samples."
        )
    if error_rate < 0.1:
        tips.append("Proceed to downstream modelling with confidence.")
    tips.append("Store results with '--export markdown' for reporting.")
    return tips


def analyze_dataset(
    dataset_path: Path, target: str, minority_label: int
) -> dict[str, Any]:
    """Analyze dataset size, feature count, and class imbalance.

    Args:
        dataset_path: Dataset path.
        target: Target column name.
        minority_label: Minority class label.

    Returns:
        Summary statistics dict.
    """
    X, y = load_dataset(dataset_path, target)
    minority = int((y == minority_label).sum())
    majority = len(y) - minority
    ratio = minority / max(majority, 1)
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "minority": minority,
        "majority": majority,
        "imbalance_ratio": ratio,
    }


def suggest_parameters(dataset_info: dict[str, Any]) -> dict[str, Any]:
    """Suggest parameters based on dataset scale and imbalance.

    Args:
        dataset_info: Summary statistics dict.

    Returns:
        Suggested parameters for validation.
    """
    n_samples = dataset_info["n_samples"]
    n_features = dataset_info["n_features"]
    ratio = dataset_info["imbalance_ratio"]

    if n_samples > 20000:
        hidden_ratio = 0.1
    elif n_features > 50:
        hidden_ratio = 0.15
    else:
        hidden_ratio = 0.25

    if ratio < 0.1:
        oversampler = "SMOTE"
        metric = "hassanat"
    elif ratio < 0.2:
        oversampler = "ADASYN"
        metric = "euclidean"
    else:
        oversampler = "RandomOverSampler"
        metric = "cosine" if n_features > 30 else "euclidean"

    export = ["json", "markdown"] if n_samples < 10000 else ["json"]

    return {
        "hidden_ratio": hidden_ratio,
        "metric": metric,
        "oversampler": oversampler,
        "export": export,
    }


def show_dataset_table(info: dict[str, Any]) -> None:
    """Render a dataset summary table to the console.

    Args:
        info: Summary statistics dict.
    """
    table = Table(title="Dataset Overview")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Samples", str(info["n_samples"]))
    table.add_row("Features", str(info["n_features"]))
    table.add_row("Minority Samples", str(info["minority"]))
    table.add_row("Majority Samples", str(info["majority"]))
    table.add_row("Imbalance Ratio", f"{info['imbalance_ratio']:.3f}")
    console.print(table)


def guided_validation(dataset: Path, config: CLIConfig, profile: str | None) -> None:
    """Run an interactive validation workflow.

    Args:
        dataset: Dataset path.
        config: CLI configuration instance.
        profile: Optional profile name to apply.
    """
    console.print(Panel.fit("Interactive Validation Setup", style="bold blue"))
    defaults = config.resolve_defaults(profile)

    target = Prompt.ask("Target column", default=defaults["target"])
    minority_label = int(
        Prompt.ask("Minority class label", default=str(defaults["minority_label"]))
    )
    dataset_info = analyze_dataset(dataset, target, minority_label)
    show_dataset_table(dataset_info)

    suggested = suggest_parameters(dataset_info)
    console.print("\n[bold]Suggested parameters:[/bold]")
    for key, value in suggested.items():
        console.print(f"- {key}: {value}")

    metric = Prompt.ask("Distance metric", default=suggested["metric"])
    hidden_ratio = float(
        Prompt.ask("Hidden ratio", default=str(suggested["hidden_ratio"]))
    )
    oversampler = Prompt.ask("Oversampler", default=suggested["oversampler"])
    export = Prompt.ask(
        "Export formats (comma separated)",
        default=",".join(suggested["export"]),
    ).split(",")
    export = [fmt.strip() for fmt in export if fmt.strip()]

    output_dir_input = Prompt.ask(
        "Output directory (empty to skip exports)",
        default=str(Path.cwd() / "oversampleqa_outputs"),
    )
    output_dir = Path(output_dir_input).expanduser() if output_dir_input else None

    resume = Confirm.ask("Resume from previous results if available?", default=True)
    mlflow_enabled = Confirm.ask("Log results to MLflow if available?", default=False)

    if Confirm.ask("Proceed with validation?", default=True):
        run_validation_with_progress(
            dataset_path=dataset,
            target=target,
            minority_label=minority_label,
            oversampler_name=oversampler,
            metric=metric,
            hidden_ratio=hidden_ratio,
            export_formats=export,
            resume=resume,
            output_dir=output_dir,
            mlflow_override=mlflow_enabled,
            mlflow_config=config.data.get("integrations", {}).get("mlflow", {}),
            verbose=True,
        )


@click.group()
@click.version_option()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(path_type=Path),
    help="Configuration file path.",
)
@click.option("--profile", "-p", help="Configuration profile to use.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    profile: str | None,
    verbose: bool,
) -> None:
    """OversampleQA: Validate your oversampling methods with confidence!

    Args:
        ctx: Click context.
        config_path: Optional config path.
        profile: Optional profile name.
        verbose: Enable verbose output.
    """

    ctx.ensure_object(dict)
    config = CLIConfig(config_path, console=console)
    ctx.obj["config"] = config
    ctx.obj["profile"] = profile
    ctx.obj["verbose"] = verbose

    if profile:
        try:
            config.get_profile(profile)
            console.print(f"[green]Profile '{profile}' loaded.[/green]")
        except ConfigValidationError as exc:
            raise click.ClickException(str(exc)) from exc


@cli.command()
@click.argument("dataset", type=click.Path(path_type=Path, exists=True))
@click.option("--target", help="Target column name.")
@click.option("--minority-label", type=int, help="Minority class label.")
@click.option("--oversampler", help="Oversampler class (imbalanced-learn).")
@click.option("--metric", help="Distance metric to use.")
@click.option("--hidden-ratio", type=float, help="Hidden majority ratio.")
@click.option(
    "--random-state",
    type=int,
    help="Seed for the hold-out split. Changing it changes the result.",
)
@click.option(
    "--n-repeats",
    type=int,
    help="Independent hold-out splits; >1 reports mean and spread.",
)
@click.option(
    "--calibrate",
    is_flag=True,
    help="Compare the error rate against ideal and worst-case references.",
)
@click.option("--export", multiple=True, help="Export formats (json|yaml|markdown).")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to store outputs.",
)
@click.option(
    "--resume/--no-resume", default=None, help="Resume from previous runs if available."
)
@click.option("--interactive", "-i", is_flag=True, help="Interactive guided wizard.")
@click.option(
    "--mlflow",
    "mlflow_enabled",
    is_flag=True,
    help="Log results to MLflow if installed.",
)
@click.pass_context
def validate(
    ctx: click.Context,
    dataset: Path,
    target: str | None,
    minority_label: int | None,
    oversampler: str | None,
    metric: str | None,
    hidden_ratio: float | None,
    random_state: int | None,
    n_repeats: int | None,
    calibrate: bool,
    export: tuple[str, ...],
    output: Path | None,
    resume: bool | None,
    interactive: bool,
    mlflow_enabled: bool,
) -> None:
    """Validate oversampling on your dataset.

    Args:
        ctx: Click context.
        dataset: Dataset path.
        target: Target column name.
        minority_label: Minority class label.
        oversampler: Oversampler class name.
        metric: Distance metric name.
        hidden_ratio: Fraction of majority to hide.
        random_state: Seed for the hold-out split.
        n_repeats: Number of independent hold-out splits.
        calibrate: Whether to calibrate the rate against null and ceiling.
        export: Export formats.
        output: Output directory.
        resume: Resume from cached results.
        interactive: Run interactive wizard.
        mlflow_enabled: Log results to MLflow if available.
    """

    config: CLIConfig = ctx.obj["config"]
    profile: str | None = ctx.obj.get("profile")

    if interactive:
        guided_validation(dataset, config, profile)
        return

    defaults = config.resolve_defaults(profile)
    target = target or defaults["target"]
    minority_label = (
        minority_label if minority_label is not None else defaults["minority_label"]
    )
    oversampler = oversampler or defaults["oversampler"]
    metric = metric or defaults["metric"]
    hidden_ratio = (
        hidden_ratio if hidden_ratio is not None else defaults["hidden_ratio"]
    )
    random_state = (
        random_state if random_state is not None else defaults.get("random_state", 42)
    )
    n_repeats = n_repeats if n_repeats is not None else defaults.get("n_repeats", 1)
    export_formats = export or tuple(defaults.get("export", []))
    resume = defaults["resume"] if resume is None else resume

    results = run_validation_with_progress(
        dataset_path=dataset,
        target=target,
        minority_label=minority_label,
        oversampler_name=oversampler,
        metric=metric,
        hidden_ratio=hidden_ratio,
        random_state=random_state,
        n_repeats=n_repeats,
        calibrate=calibrate,
        export_formats=export_formats,
        resume=resume,
        output_dir=output,
        mlflow_override=mlflow_enabled,
        mlflow_config=config.data.get("integrations", {}).get("mlflow", {}),
        verbose=ctx.obj.get("verbose", False),
    )

    console.print(Panel.fit("Validation completed", style="bold green"))
    display_results(results)


@cli.command(name="run")
@click.argument("manifest", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Override the manifest output directory.",
)
@click.option(
    "--resume/--no-resume",
    default=None,
    help="Override per-experiment resume settings.",
)
@click.pass_context
def run_manifest(
    ctx: click.Context,
    manifest: Path,
    output: Path | None,
    resume: bool | None,
) -> None:
    """Run validation experiments from a YAML manifest.

    Args:
        ctx: Click context.
        manifest: Manifest path.
        output: Optional output directory override.
        resume: Optional resume override.
    """
    config: CLIConfig = ctx.obj["config"]
    summary = run_experiment_manifest(
        manifest_path=manifest,
        output_override=output,
        resume_override=resume,
        mlflow_config=config.data.get("integrations", {}).get("mlflow", {}),
        verbose=ctx.obj.get("verbose", False),
    )
    console.print(
        Panel.fit(
            f"Manifest completed: {summary['n_experiments']} experiment(s)",
            style="bold green",
        )
    )


@cli.command()
@click.argument("dataset", type=click.Path(path_type=Path, exists=True))
@click.option("--target", required=True, help="Target column name.")
@click.option("--minority-label", type=int, default=1, help="Minority class label.")
@click.option(
    "--oversampler", default="SMOTE", help="Oversampler class (imbalanced-learn)."
)
@click.option("--metric", default="hassanat", help="Distance metric to use.")
@click.option("--k", type=int, default=5, help="Neighbours for manifold estimates.")
@click.option(
    "--hidden-ratio", type=float, default=0.1, help="Fraction of majority to hide."
)
@click.option("--random-state", type=int, default=42, help="Seed for the hold-out.")
@click.option(
    "--utility",
    is_flag=True,
    help="Also fit models to measure downstream gain (much slower).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Write the report as JSON to this path.",
)
def fidelity(
    dataset: Path,
    target: str,
    minority_label: int,
    oversampler: str,
    metric: str,
    k: int,
    hidden_ratio: float,
    random_state: int,
    utility: bool,
    output: Path | None,
) -> None:
    """Measure fidelity and diversity, not just the error rate.

    The error rate is one scalar covering two failures that need opposite
    fixes: generating implausible points, and merely copying the training
    minority. This reports both axes.

    Args:
        dataset: Dataset path.
        target: Target column name.
        minority_label: Minority class label.
        oversampler: Oversampler class name.
        metric: Distance metric name.
        k: Neighbours for the manifold estimates.
        hidden_ratio: Fraction of majority to hide.
        random_state: Seed for the hold-out split.
        utility: Whether to measure downstream utility.
        output: Optional JSON output path.
    """
    from .fidelity import fidelity_report

    X, y = load_dataset(dataset, target)
    module = __import__("imblearn.over_sampling", fromlist=[oversampler])
    sampler = getattr(module, oversampler)()

    with console.status(f"Measuring fidelity for {oversampler}..."):
        report = fidelity_report(
            np.asarray(X.values, dtype=float),
            np.asarray(y.values),
            minority_label,
            sampler,
            metric=metric,
            k=k,
            hidden_ratio=hidden_ratio,
            random_state=random_state,
            include_utility=utility,
        )

    table = Table(
        title=f"Fidelity report - {oversampler}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Reads as", style="yellow")

    manifold = report.manifold
    table.add_row("Error rate", f"{report.error_rate:.3f}", "Lower is better")
    table.add_row(
        "Precision", f"{manifold.precision:.3f}", "Fidelity: are points plausible?"
    )
    table.add_row(
        "Recall", f"{manifold.recall:.3f}", "Diversity: is the real range covered?"
    )
    table.add_row("Density", f"{manifold.density:.3f}", "Fidelity, unsaturated")
    table.add_row("Coverage", f"{manifold.coverage:.3f}", "Diversity, robust")
    table.add_row(
        "Memorisation ratio",
        f"{report.memorisation.distance_ratio:.3f}",
        "Near 0 means copying training data",
    )
    table.add_row(
        "Boundary violations",
        f"{report.boundary.strict_rate:.3f}",
        "Points landing among majority neighbours",
    )
    if report.utility is not None:
        table.add_row(
            "Downstream gain",
            f"{report.utility.difference:+.4f}",
            f"{report.utility.scoring}, CI "
            f"[{report.utility.ci_lower:+.4f}, {report.utility.ci_upper:+.4f}]",
        )
    console.print(table)

    for note in report.interpret():
        console.print(f"[yellow]-[/yellow] {note}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(strict_json_dumps(report.to_dict()), encoding="utf-8")
        write_export_metadata(
            output, export_kind="fidelity_report", data=report.to_dict()
        )
        console.print(f"[green]Report written to {output}[/green]")


@cli.command()
@click.option(
    "--template", type=click.Choice(sorted(CONFIG_TEMPLATES)), default="production"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path.",
)
def template(template: str, output: Path) -> None:
    """Generate a configuration file from a named template.

    Args:
        template: Template name.
        output: Output file path.
    """

    destination = generate_config_file(template, str(output))
    console.print(f"[green]Template '{template}' written to {destination}[/green]")


@cli.command()
@click.pass_context
def profiles(ctx: click.Context) -> None:
    """List available configuration profiles.

    Args:
        ctx: Click context.
    """

    config: CLIConfig = ctx.obj["config"]
    rows = config.list_profiles()
    table = Table(title="Available Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Parameters", style="green")
    for name, params in rows:
        formatted = ", ".join(f"{k}={v}" for k, v in params.items())
        table.add_row(name, formatted)
    console.print(table)


@cli.command()
@click.argument(
    "shell", required=False, type=click.Choice(["bash", "zsh", "fish", "powershell"])
)
def completion(shell: str | None) -> None:
    """Provide shell completion installation instructions.

    Args:
        shell: Optional shell name.
    """

    shell = shell or "bash"
    script_name = "oversampleqa"
    env_var = f"_{script_name.upper().replace('-', '_')}_COMPLETE"
    instructions = {
        "bash": f'eval "$({env_var}=bash_source {script_name})"',
        "zsh": f'eval "$({env_var}=zsh_source {script_name})"',
        "fish": f"set -x {env_var} fish_source; {script_name} | source",
        "powershell": f"set-item env:{env_var} powershell_source; {script_name} | Out-String | Invoke-Expression",
    }
    console.print(Panel.fit("Shell completion setup", style="bold blue"))
    console.print(f"Selected shell: {shell}")
    console.print("Run the command below in your shell configuration:")
    console.print(f"[cyan]{instructions[shell]}[/cyan]")


def _run_statistical_benchmark(
    datasets: list[dict[str, Any]], output: Path, folds: int, repeats: int
) -> None:
    """Run the statistical benchmark engine and surface its results.

    Args:
        datasets: Dataset descriptors.
        output: Output directory for artifacts.
        folds: Number of CV folds.
        repeats: Number of CV repeats.
    """
    oversampler_names = ["SMOTE", "ADASYN"]
    oversampler_module = __import__(
        "imblearn.over_sampling", fromlist=oversampler_names
    )
    oversamplers = [getattr(oversampler_module, name)() for name in oversampler_names]

    with console.status("Running cross-validated statistical benchmark..."):
        engine = StatisticalBenchmark(n_folds=folds, n_repeats=repeats)
        frame = engine.run_comprehensive_benchmark(datasets, oversamplers)

    if frame.empty:
        console.print("[yellow]No benchmark results were produced.[/yellow]")
        return

    table = Table(
        title="Statistical Benchmark", show_header=True, header_style="bold magenta"
    )
    for column in ("Dataset", "Oversampler", "Metric", "Mean", "Std", "95% CI", "n"):
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(
            str(row["dataset_name"]),
            str(row["oversampler_name"]),
            str(row["metric"]),
            f"{row['mean_error']:.3f}",
            f"{row['std_error']:.3f}",
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
            str(int(row["n_observations"])),
        )
    console.print(table)

    output.mkdir(parents=True, exist_ok=True)
    statistics_path = output / "benchmark_statistics.csv"
    frame.to_csv(statistics_path, index=False)
    write_export_metadata(
        statistics_path,
        export_kind="statistical_benchmark_statistics",
        data=frame,
        extra={"benchmark_parameters": {"folds": folds, "repeats": repeats}},
    )
    summary_md = format_statistical_summary(frame)
    summary_path = output / "benchmark_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    write_export_metadata(
        summary_path,
        export_kind="statistical_benchmark_summary",
        data=frame,
        extra={"benchmark_parameters": {"folds": folds, "repeats": repeats}},
    )
    create_benchmark_report(frame, str(output / "benchmark_report.html"))
    console.print(
        f"[green]Statistical results stored in {output}[/green] "
        "(benchmark_statistics.csv, benchmark_summary.md, benchmark_report.html)"
    )


@cli.command()
@click.option("--include-openml", is_flag=True, help="Include OpenML datasets.")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=Path("benchmark_results")
)
@click.option(
    "--statistical",
    is_flag=True,
    help="Run cross-validated statistical benchmarking (CIs, p-values, effect sizes).",
)
@click.option(
    "--folds",
    type=int,
    default=5,
    show_default=True,
    help="CV folds (statistical mode).",
)
@click.option(
    "--repeats",
    type=int,
    default=5,
    show_default=True,
    help="CV repeats (statistical mode).",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    include_openml: bool,
    output: Path,
    statistical: bool,
    folds: int,
    repeats: int,
) -> None:
    """Run comprehensive benchmarking across datasets.

    Args:
        ctx: Click context.
        include_openml: Whether to include OpenML datasets.
        output: Output directory.
        statistical: Run cross-validated statistical benchmarking.
        folds: Number of CV folds for statistical mode.
        repeats: Number of CV repeats for statistical mode.
    """

    console.print(Panel.fit("Running comprehensive benchmark", style="bold green"))

    datasets = load_standard_datasets(include_openml=include_openml)

    if statistical:
        _run_statistical_benchmark(datasets, output, folds, repeats)
        return

    oversampler_names = ["SMOTE", "ADASYN"]
    hidden_ratios = [0.1, 0.25]
    oversampler_module = __import__(
        "imblearn.over_sampling", fromlist=oversampler_names
    )
    oversampler_classes = [
        getattr(oversampler_module, name) for name in oversampler_names
    ]

    total_steps = len(datasets) * len(oversampler_classes) * len(hidden_ratios)
    results: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing benchmark...", total=total_steps)
        for dataset in datasets:
            X, y = dataset["data"], dataset["target"]
            minority_label = dataset.get("minority_label", 1)
            dataset_name = dataset.get("name", "dataset")

            for oversampler_cls in oversampler_classes:
                for hidden_ratio in hidden_ratios:
                    progress.update(
                        task, description=f"{dataset_name} / {oversampler_cls.__name__}"
                    )
                    oversampler = oversampler_cls()
                    try:
                        error = validate_oversampling(
                            X,
                            y,
                            minority_label=minority_label,
                            oversampler=oversampler,
                            hidden_ratio=hidden_ratio,
                            metric="hassanat",
                        )
                    except ValueError:
                        progress.advance(task)
                        continue
                    results.append(
                        {
                            "dataset": dataset_name,
                            "oversampler": oversampler_cls.__name__,
                            "hidden_ratio": hidden_ratio,
                            "error_rate": error,
                        }
                    )
                    progress.advance(task)

    output.mkdir(parents=True, exist_ok=True)
    export_benchmark_results(
        pd.DataFrame(results), str(output / "benchmark_summary.csv")
    )
    console.print(f"[green]Benchmark results stored in {output}[/green]")


@cli.command()
@click.pass_context
def setup(ctx: click.Context) -> None:
    """Initial setup and configuration wizard.

    Args:
        ctx: Click context.
    """

    config: CLIConfig = ctx.obj["config"]
    console.print(Panel.fit("OversampleQA setup wizard", style="bold blue"))
    console.print("\n[bold]Let's configure OversampleQA for your needs![/bold]\n")

    use_case = Prompt.ask(
        "What's your primary use case?",
        choices=["research", "production", "education", "exploration"],
        default="exploration",
    )

    if use_case in CONFIG_TEMPLATES:
        template_params = CONFIG_TEMPLATES[use_case]["params"]
        config.data.setdefault("profiles", {})[use_case] = template_params
    if use_case == "production":
        config.data["defaults"]["export"] = ["json"]
        config.data["defaults"]["resume"] = True
    if use_case == "research":
        config.data["defaults"]["export"] = ["json", "markdown"]
        config.data["defaults"]["metric"] = "hassanat"

    config.save_config()
    console.print(
        "\n[green][OK] Setup complete! Run 'oversampleqa validate --help' to get started.[/green]"
    )


#: Distributions worth naming in a bug report, as (import name, PyPI name).
#: Version differences in these change results, not just whether code runs.
DIAGNOSTIC_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("imblearn", "imbalanced-learn"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
)


def _dependency_version(module: str, distribution: str) -> str | None:
    """Return an installed distribution's version, or None if absent."""
    try:
        __import__(module)
    except Exception:
        return None
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        # Importable but not installed as a distribution -- vendored, or on
        # PYTHONPATH from a source tree. Present, version unknown.
        return "unknown"


def diagnostics() -> dict[str, Any]:
    """Collect the environment facts a bug report needs.

    Separated from the rendering so it can be tested without parsing a table,
    and so anything else that needs the same facts does not reimplement them.
    """
    return {
        "oversampleqa": _PACKAGE_VERSION,
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 10),
        "platform": platform.platform(),
        "packages": {
            distribution: _dependency_version(module, distribution)
            for module, distribution in DIAGNOSTIC_PACKAGES
        },
    }


@cli.command()
def doctor() -> None:
    """Report the environment, for diagnosis and for bug reports.

    Prints versions rather than only pass/fail, because "pandas [OK]" does not
    reproduce anything -- and a version difference in numpy or scikit-learn
    changes results rather than merely whether the code runs.
    """

    console.print(Panel.fit("System diagnostics", style="bold yellow"))

    facts = diagnostics()
    table = Table(title="Diagnostic Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="white")
    table.add_column("Status", style="green")

    table.add_row("OversampleQA", facts["oversampleqa"], "[OK]")
    # `sys.version_info >= (3, 10)`, not a string comparison: the previous
    # check read `sys.version.split()[0] >= "3.10"`, and "3.9" sorts after
    # "3.10", so every unsupported Python -- 3.7, 3.8, 3.9 -- passed it. The
    # check could not fail for any Python 3.
    table.add_row(
        "Python",
        facts["python"],
        "[OK]" if facts["python_supported"] else "[X] needs 3.10+",
    )
    table.add_row("Platform", facts["platform"], "[OK]")
    for distribution, version in facts["packages"].items():
        table.add_row(
            distribution,
            version or "-",
            "[OK]" if version else "[X] not installed",
        )

    console.print(table)

    missing = [name for name, version in facts["packages"].items() if version is None]
    if missing or not facts["python_supported"]:
        if not facts["python_supported"]:
            console.print(
                f"[red]Python {facts['python']} is not supported; 3.10+ is "
                "required.[/red]"
            )
        if missing:
            console.print(
                "[red]Missing: " + ", ".join(missing) + ". Reinstall to fix.[/red]"
            )
    else:
        console.print("[green]All required components are present![/green]")
    console.print(
        "[dim]Paste this table into a bug report; it is what makes a result "
        "reproducible.[/dim]"
    )


def main() -> None:
    """Entry point for the enhanced CLI.

    Initializes logging and delegates to the Click CLI.
    """
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
