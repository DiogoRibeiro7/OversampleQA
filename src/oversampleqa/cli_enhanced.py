"""Enhanced command-line interface for OversampleQA."""

from __future__ import annotations

import copy
import json
import logging
import sys
import textwrap
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import get_close_matches
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
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        progress.advance(task)

        progress.update(task, description=stages[4])
        results.update(
            {
                "error_rate": float(error_rate),
                "metric": metric,
                "hidden_ratio": hidden_ratio,
                "random_state": random_state,
                **dispersion,
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
            (output_dir / "validation_results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
        elif fmt_lower == "yaml":
            (output_dir / "validation_results.yaml").write_text(
                yaml.safe_dump(results, sort_keys=False), encoding="utf-8"
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
            (output_dir / "validation_results.md").write_text(
                markdown + "\n", encoding="utf-8"
            )


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
        export_formats=export_formats,
        resume=resume,
        output_dir=output,
        mlflow_override=mlflow_enabled,
        mlflow_config=config.data.get("integrations", {}).get("mlflow", {}),
        verbose=ctx.obj.get("verbose", False),
    )

    console.print(Panel.fit("Validation completed", style="bold green"))
    display_results(results)


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
    frame.to_csv(output / "benchmark_statistics.csv", index=False)
    summary_md = format_statistical_summary(frame)
    (output / "benchmark_summary.md").write_text(summary_md, encoding="utf-8")
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


@cli.command()
def doctor() -> None:
    """Diagnose installation and configuration issues.

    Runs a minimal dependency check and reports status to the console.
    """

    console.print(Panel.fit("System diagnostics", style="bold yellow"))

    checks = [
        ("Python Version", sys.version.split()[0] >= "3.10"),
        ("pandas", _optional_import("pandas")),
        ("imbalanced-learn", _optional_import("imblearn")),
        ("rich", _optional_import("rich")),
    ]

    table = Table(title="Diagnostic Summary")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")

    for name, success in checks:
        table.add_row(name, "[OK]" if success else "[X]")

    console.print(table)
    if not all(success for _, success in checks):
        console.print(
            "[red]Some checks failed. Please reinstall missing dependencies.[/red]"
        )
    else:
        console.print("[green]All required components are present![/green]")


def _optional_import(module: str) -> bool:
    """Return True when a module can be imported.

    Args:
        module: Module name to import.

    Returns:
        ``True`` if import succeeds, otherwise ``False``.
    """
    try:
        __import__(module)
        return True
    except Exception:
        return False


def main() -> None:
    """Entry point for the enhanced CLI.

    Initializes logging and delegates to the Click CLI.
    """
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
