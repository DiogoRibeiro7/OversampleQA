"""Advanced benchmarking utilities with statistical analysis."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .validator import validate_oversampling


@dataclass
class BenchmarkResult:
    """Structured benchmark summary for a single (dataset, oversampler, metric)."""

    dataset_name: str
    oversampler_name: str
    metric: str
    error_rates: list[float]
    mean_error: float
    std_error: float
    confidence_interval: tuple[float, float]
    effect_size: float | None = None
    p_value: float | None = None
    recommended_samples: int | None = None


class StatisticalBenchmark:
    """Advanced benchmarking engine with statistical analysis."""

    def __init__(
        self,
        n_folds: int = 5,
        n_repeats: int = 5,
        confidence_level: float = 0.95,
        correction_method: str = "holm",
        random_state: int | None = 42,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.confidence_level = confidence_level
        self.correction_method = correction_method
        self.random_state = random_state

    def run_comprehensive_benchmark(
        self,
        datasets: Sequence[dict[str, Any]],
        oversamplers: Sequence[Any],
        metrics: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run repeated stratified benchmarking across datasets.

        Parameters
        ----------
        datasets:
            Sequence of dataset descriptors. Each entry should provide ``data``,
            ``target`` and optionally ``name`` and ``minority_label``.
        oversamplers:
            Sequence of initialised oversampler instances (will be cloned).
        metrics:
            Distance metrics to evaluate. Defaults to Hassanat, Euclidean, Mahalanobis.
        """

        metrics = tuple(metrics or ("hassanat", "euclidean", "mahalanobis"))

        all_results: list[BenchmarkResult] = []
        for dataset in datasets:
            all_results.extend(
                self._benchmark_single_dataset(dataset, oversamplers, metrics)
            )

        frame = pd.DataFrame([self._result_to_dict(r) for r in all_results])
        if frame.empty:
            return frame
        frame = self._add_statistical_analysis(frame)
        return frame

    def _benchmark_single_dataset(
        self,
        dataset: dict[str, Any],
        oversamplers: Sequence[Any],
        metrics: Sequence[str],
    ) -> list[BenchmarkResult]:
        """Run benchmark for a single dataset across oversamplers and metrics.

        Args:
            dataset: Dataset descriptor with ``data`` and ``target`` arrays.
            oversamplers: Oversamplers to evaluate.
            metrics: Distance metrics to test.

        Returns:
            List of BenchmarkResult entries for the dataset.
        """
        X, y = dataset["data"], dataset["target"]
        dataset_name = dataset.get("name", "dataset")
        minority_label = dataset.get("minority_label", 1)

        # basic preprocessing to avoid scaling issues for distance metrics
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results: list[BenchmarkResult] = []
        for oversampler in oversamplers:
            for metric in metrics:
                error_rates = self._cross_validated_errors(
                    X_scaled,
                    y,
                    minority_label=minority_label,
                    oversampler=oversampler,
                    metric=metric,
                )
                if not error_rates:
                    continue
                mean_error = float(np.mean(error_rates))
                std_error = (
                    float(np.std(error_rates, ddof=1)) if len(error_rates) > 1 else 0.0
                )
                ci_lower, ci_upper = self._confidence_interval(error_rates)
                power_samples = self._recommended_sample_size(error_rates)

                results.append(
                    BenchmarkResult(
                        dataset_name=dataset_name,
                        oversampler_name=oversampler.__class__.__name__,
                        metric=metric,
                        error_rates=error_rates,
                        mean_error=mean_error,
                        std_error=std_error,
                        confidence_interval=(ci_lower, ci_upper),
                        recommended_samples=power_samples,
                    )
                )
        return results

    def _cross_validated_errors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        minority_label: int,
        oversampler: Any,
        metric: str,
    ) -> list[float]:
        """Compute error rates across repeated stratified folds.

        Args:
            X: Feature matrix.
            y: Target labels.
            minority_label: Minority class label.
            oversampler: Oversampler instance to clone per fold.
            metric: Distance metric name.

        Returns:
            List of error rates for each evaluated fold.
        """
        errors: list[float] = []
        rng = np.random.default_rng(self.random_state)

        for _repeat in range(self.n_repeats):
            cv = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=(
                    None if self.random_state is None else rng.integers(0, 1_000_000)
                ),
            )
            for fold, (train_idx, _val_idx) in enumerate(cv.split(X, y)):
                # Only the training fold is used. validate_oversampling performs
                # its own hold-out internally, so the CV validation fold has no
                # role here -- the splitter is effectively a stratified
                # subsampler, and each "fold" is one subsample of the data
                # rather than a held-out evaluation. See docs/benchmarking.rst
                # for what the resulting intervals therefore describe.
                X_train = X[train_idx]
                y_train = y[train_idx]
                if len(np.unique(y_train)) < 2:
                    warnings.warn(
                        "Training fold lacks class diversity; skipping fold.",
                        stacklevel=2,
                    )
                    continue
                sampler = clone(oversampler)
                try:
                    error = validate_oversampling(
                        X_train,
                        y_train,
                        minority_label=minority_label,
                        oversampler=sampler,
                        hidden_ratio=0.1,
                        metric=metric,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    warnings.warn(
                        f"Validation failed for fold {fold}: {exc}", stacklevel=2
                    )
                    continue
                if np.isnan(error):
                    continue
                errors.append(float(error))
        return errors

    def _confidence_interval(self, values: Sequence[float]) -> tuple[float, float]:
        """Return a confidence interval for the provided values.

        Args:
            values: Sample values.

        Returns:
            Lower and upper bounds for the configured confidence level.
        """
        if len(values) < 2:
            return (
                float(values[0]) if values else 0.0,
                float(values[0]) if values else 0.0,
            )
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        alpha = 1 - self.confidence_level
        if len(arr) < 30:
            se = stats.sem(arr)
            t_val = stats.t.ppf(1 - alpha / 2, len(arr) - 1)
            margin = t_val * se
            return mean - margin, mean + margin
        lower = float(np.percentile(arr, alpha / 2 * 100))
        upper = float(np.percentile(arr, (1 - alpha / 2) * 100))
        return lower, upper

    def _recommended_sample_size(
        self, values: Sequence[float], target_power: float = 0.8
    ) -> int | None:
        """Estimate recommended sample size using a Cohen's d approximation.

        Args:
            values: Observed error rates.
            target_power: Desired statistical power.

        Returns:
            Estimated required sample size or ``None`` if not computable.
        """

        if len(values) < 2:
            return None
        arr = np.asarray(values, dtype=float)
        std = arr.std(ddof=1)
        if std == 0:
            return None
        d = abs(arr.mean()) / std
        if d == 0:
            return None
        # normal approximation for two-sided test
        alpha = 1 - self.confidence_level
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(target_power)
        n = ((z_alpha + z_beta) ** 2) * 2 / (d**2)
        return math.ceil(n)

    def _add_statistical_analysis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add pairwise p-values and effect sizes per dataset.

        Args:
            frame: Benchmark results dataframe.

        Returns:
            Dataframe with pairwise statistics columns populated.
        """
        frame = frame.copy()
        frame["pairwise_p_values"] = None
        frame["pairwise_effect_sizes"] = None

        for dataset_name in frame["dataset_name"].unique():
            mask = frame["dataset_name"] == dataset_name
            dataset_slice = frame.loc[mask]
            if len(dataset_slice) < 2:
                continue
            pvals = self._pairwise_statistical_tests(dataset_slice)
            effects = self._calculate_effect_sizes(dataset_slice)
            for idx in dataset_slice.index:
                frame.at[idx, "pairwise_p_values"] = json.dumps(pvals)
                frame.at[idx, "pairwise_effect_sizes"] = json.dumps(effects)
        return frame

    def _pairwise_statistical_tests(
        self, dataset_slice: pd.DataFrame
    ) -> dict[str, float]:
        """Compute pairwise Wilcoxon tests across oversamplers.

        Args:
            dataset_slice: Subset of results for a single dataset.

        Returns:
            Mapping of ``oversampler_a_vs_b`` to corrected p-values.
        """
        p_values: dict[str, float] = {}
        oversamplers = dataset_slice["oversampler_name"].unique()
        for i, os1 in enumerate(oversamplers):
            for os2 in oversamplers[i + 1 :]:
                errors1 = np.asarray(
                    dataset_slice.loc[
                        dataset_slice["oversampler_name"] == os1, "error_rates"
                    ].iloc[0],
                    dtype=float,
                )
                errors2 = np.asarray(
                    dataset_slice.loc[
                        dataset_slice["oversampler_name"] == os2, "error_rates"
                    ].iloc[0],
                    dtype=float,
                )
                try:
                    if (
                        len(errors1) == 0
                        or len(errors2) == 0
                        or len(errors1) != len(errors2)
                        or np.allclose(errors1, errors2)
                    ):
                        p_val = 1.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            _, p_val = stats.wilcoxon(errors1, errors2)
                except Exception:
                    p_val = 1.0
                key = f"{os1}_vs_{os2}"
                p_values[key] = p_val
        return self._apply_correction(p_values)

    def _calculate_effect_sizes(self, dataset_slice: pd.DataFrame) -> dict[str, float]:
        """Compute pairwise Cohen's d effect sizes.

        Args:
            dataset_slice: Subset of results for a single dataset.

        Returns:
            Mapping of ``oversampler_a_vs_b`` to Cohen's d.
        """
        effect_sizes: dict[str, float] = {}
        oversamplers = dataset_slice["oversampler_name"].unique()
        for i, os1 in enumerate(oversamplers):
            for os2 in oversamplers[i + 1 :]:
                errors1 = np.asarray(
                    dataset_slice.loc[
                        dataset_slice["oversampler_name"] == os1, "error_rates"
                    ].iloc[0],
                    dtype=float,
                )
                errors2 = np.asarray(
                    dataset_slice.loc[
                        dataset_slice["oversampler_name"] == os2, "error_rates"
                    ].iloc[0],
                    dtype=float,
                )
                pooled_std = self._pooled_std(errors1, errors2)
                if pooled_std == 0:
                    continue
                cohens_d = (errors1.mean() - errors2.mean()) / pooled_std
                effect_sizes[f"{os1}_vs_{os2}"] = float(cohens_d)
        return effect_sizes

    @staticmethod
    def _pooled_std(x: np.ndarray, y: np.ndarray) -> float:
        """Return pooled standard deviation for two samples.

        Args:
            x: Sample 1.
            y: Sample 2.

        Returns:
            Pooled standard deviation.
        """
        if len(x) < 2 or len(y) < 2:
            return 0.0
        n1, n2 = len(x), len(y)
        s1, s2 = x.var(ddof=1), y.var(ddof=1)
        pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        return math.sqrt(max(pooled, 0.0))

    def _apply_correction(self, p_values: dict[str, float]) -> dict[str, float]:
        """Apply multiple-comparison correction to p-values.

        Args:
            p_values: Raw p-values.

        Returns:
            Corrected p-values using the configured method.
        """
        if not p_values:
            return p_values
        if self.correction_method == "bonferroni":
            factor = len(p_values)
            return {k: min(v * factor, 1.0) for k, v in p_values.items()}
        # default Holm-Bonferroni
        sorted_items = sorted(p_values.items(), key=lambda item: item[1])
        corrected: dict[str, float] = {}
        m = len(sorted_items)
        for rank, (key, p_val) in enumerate(sorted_items, start=1):
            corrected_val = min((m - rank + 1) * p_val, 1.0)
            corrected[key] = corrected_val
        return corrected

    @staticmethod
    def _result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
        """Convert BenchmarkResult to a serializable dictionary.

        Args:
            result: Benchmark result structure.

        Returns:
            Dict suitable for DataFrame construction or serialization.
        """
        return {
            "dataset_name": result.dataset_name,
            "oversampler_name": result.oversampler_name,
            "metric": result.metric,
            "mean_error": result.mean_error,
            "std_error": result.std_error,
            "ci_lower": result.confidence_interval[0],
            "ci_upper": result.confidence_interval[1],
            "n_observations": len(result.error_rates),
            "error_rates": result.error_rates,
            "effect_size": result.effect_size,
            "p_value": result.p_value,
            "recommended_samples": result.recommended_samples,
        }


class DatasetRepository:
    """Repository for curated real-world and synthetic benchmarking datasets."""

    def __init__(self, cache_dir: str = ".oversampleqa_datasets") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_research_datasets(
        self,
        domains: Sequence[str] | None = None,
        max_samples: int = 10_000,
        include_openml: bool = False,
    ) -> list[dict[str, Any]]:
        """Load curated datasets for benchmarking.

        Args:
            domains: Domain names to load.
            max_samples: Maximum number of samples per dataset.
            include_openml: Whether to attempt OpenML downloads.

        Returns:
            List of dataset descriptors.
        """
        domains = tuple(domains or ("medical", "financial"))
        datasets: list[dict[str, Any]] = []
        for domain in domains:
            datasets.extend(self._load_domain(domain, max_samples, include_openml))
        return datasets

    def _load_domain(
        self, domain: str, max_samples: int, include_openml: bool
    ) -> list[dict[str, Any]]:
        """Load datasets for a single domain.

        Args:
            domain: Domain name.
            max_samples: Maximum number of samples per dataset.
            include_openml: Whether to attempt OpenML downloads.

        Returns:
            List of dataset descriptors.
        """
        domain = domain.lower()
        if domain == "medical":
            return self._load_medical(max_samples, include_openml)
        if domain == "financial":
            return self._load_financial(max_samples)
        return []

    def _load_medical(
        self, max_samples: int, include_openml: bool
    ) -> list[dict[str, Any]]:
        """Load medical datasets for benchmarking.

        Args:
            max_samples: Maximum number of samples per dataset.
            include_openml: Whether to attempt OpenML downloads.

        Returns:
            List of dataset descriptors.
        """
        from sklearn.datasets import load_breast_cancer

        cancer = load_breast_cancer()
        X, y = cancer.data[:max_samples], cancer.target[:max_samples]
        datasets = [
            {
                "name": "breast_cancer",
                "data": X,
                "target": y,
                "minority_label": 1,
            }
        ]
        if include_openml:
            try:
                from sklearn.datasets import fetch_openml

                diabetes = fetch_openml("diabetes", version=1, as_frame=False)
                Xd = diabetes.data[:max_samples]
                yd = (diabetes.target[:max_samples] == "tested_positive").astype(int)
                datasets.append(
                    {
                        "name": "diabetes_openml",
                        "data": Xd,
                        "target": yd,
                        "minority_label": 1,
                    }
                )
            except Exception as exc:  # pragma: no cover - network dependent
                warnings.warn(f"OpenML download failed: {exc}", stacklevel=2)
        return datasets

    def _load_financial(self, max_samples: int) -> list[dict[str, Any]]:
        """Load financial datasets for benchmarking.

        Args:
            max_samples: Maximum number of samples per dataset.

        Returns:
            List of dataset descriptors.
        """
        try:
            from imbalanced_datasets import creditcard  # optional
        except Exception:  # pragma: no cover
            creditcard = None

        datasets: list[dict[str, Any]] = []
        if creditcard is not None:  # pragma: no cover - optional dependency
            X, y = creditcard.load_data()
            datasets.append(
                {
                    "name": "creditcard",
                    "data": X[:max_samples],
                    "target": y[:max_samples],
                    "minority_label": 1,
                }
            )
        return datasets

    def create_synthetic_benchmark_suite(
        self, difficulty_levels: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Generate synthetic datasets for the requested difficulty levels.

        Args:
            difficulty_levels: Difficulty labels to generate.

        Returns:
            List of synthetic dataset descriptors.
        """
        difficulty_levels = tuple(
            difficulty_levels or ("easy", "medium", "hard", "extreme")
        )
        synthetic: list[dict[str, Any]] = []
        for difficulty in difficulty_levels:
            synthetic.extend(self._generate_difficulty(difficulty))
        return synthetic

    def _generate_difficulty(self, difficulty: str) -> list[dict[str, Any]]:
        """Create synthetic datasets for a single difficulty tier.

        Args:
            difficulty: Difficulty label.

        Returns:
            List of dataset descriptors for the difficulty tier.
        """
        from sklearn.datasets import make_classification

        difficulty = difficulty.lower()
        configs: list[dict[str, Any]]
        if difficulty == "easy":
            configs = [
                {
                    "n_samples": 600,
                    "n_features": 8,
                    "class_sep": 2.0,
                    "weights": [0.75, 0.25],
                },
                {
                    "n_samples": 800,
                    "n_features": 5,
                    "class_sep": 1.8,
                    "weights": [0.8, 0.2],
                },
            ]
        elif difficulty == "medium":
            configs = [
                {
                    "n_samples": 1000,
                    "n_features": 12,
                    "class_sep": 1.2,
                    "weights": [0.85, 0.15],
                },
            ]
        elif difficulty == "hard":
            configs = [
                {
                    "n_samples": 1500,
                    "n_features": 20,
                    "class_sep": 0.8,
                    "weights": [0.9, 0.1],
                },
            ]
        elif difficulty == "extreme":
            configs = [
                {
                    "n_samples": 2000,
                    "n_features": 40,
                    "class_sep": 0.4,
                    "weights": [0.97, 0.03],
                },
            ]
        else:
            configs = [
                {
                    "n_samples": 800,
                    "n_features": 10,
                    "class_sep": 1.0,
                    "weights": [0.8, 0.2],
                }
            ]

        datasets: list[dict[str, Any]] = []
        for idx, config in enumerate(configs):
            X, y = make_classification(
                random_state=42 + idx,
                n_informative=max(2, config["n_features"] // 2),
                n_redundant=config["n_features"] // 4,
                flip_y=0.02 if difficulty in {"hard", "extreme"} else 0.0,
                **config,
            )
            datasets.append(
                {
                    "name": f"{difficulty}_synthetic_{idx}",
                    "data": X,
                    "target": y,
                    "minority_label": 1,
                    "difficulty": difficulty,
                }
            )
        return datasets


def format_statistical_summary(
    results_df: pd.DataFrame, significance_level: float = 0.05
) -> str:
    """Render a Markdown summary of a statistical benchmark frame.

    The frame is expected to come from
    :meth:`StatisticalBenchmark.run_comprehensive_benchmark`. The summary lists,
    per dataset, the mean error, standard deviation and confidence interval for
    each oversampler/metric, followed by the statistically significant pairwise
    comparisons (corrected p-value below ``significance_level``).

    Args:
        results_df: Benchmark results dataframe.
        significance_level: Threshold below which a pairwise p-value is reported.

    Returns:
        A Markdown-formatted string.
    """
    if results_df.empty:
        return (
            "# OversampleQA Statistical Benchmark\n\nNo benchmark results available.\n"
        )

    lines = ["# OversampleQA Statistical Benchmark", ""]
    lines.append(
        "Confidence intervals use the configured confidence level. Pairwise "
        "p-values and effect sizes (Cohen's d) compare oversamplers on the same "
        "dataset and metric, corrected by the configured method."
    )
    lines.append("")

    for dataset_name, group in results_df.groupby("dataset_name", sort=True):
        lines.append(f"## Dataset: {dataset_name}")
        lines.append("")
        lines.append("| Oversampler | Metric | Mean error | Std | CI | n |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in group.iterrows():
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            lines.append(
                f"| {row['oversampler_name']} | {row['metric']} | "
                f"{row['mean_error']:.3f} | {row['std_error']:.3f} | {ci} | "
                f"{int(row['n_observations'])} |"
            )
        lines.append("")

        significant = _significant_pairwise(group, significance_level)
        if significant:
            lines.append(
                f"Significant pairwise differences (p < {significance_level}):"
            )
            for label, p_val, effect in significant:
                effect_str = f", d={effect:.2f}" if effect is not None else ""
                lines.append(f"- {label}: p={p_val:.4f}{effect_str}")
            lines.append("")

    return "\n".join(lines)


def _significant_pairwise(
    group: pd.DataFrame, significance_level: float
) -> list[tuple[str, float, float | None]]:
    """Extract significant pairwise comparisons from a per-dataset group.

    Args:
        group: Rows for a single dataset.
        significance_level: Threshold below which a p-value is reported.

    Returns:
        Sorted list of ``(comparison, p_value, effect_size)`` tuples.
    """
    if "pairwise_p_values" not in group.columns:
        return []
    raw_p = group["pairwise_p_values"].dropna()
    if raw_p.empty:
        return []
    p_values = json.loads(raw_p.iloc[0])
    effects: dict[str, Any] = {}
    if "pairwise_effect_sizes" in group.columns:
        raw_e = group["pairwise_effect_sizes"].dropna()
        if not raw_e.empty:
            effects = json.loads(raw_e.iloc[0])

    significant = [
        (label, float(p_val), effects.get(label))
        for label, p_val in p_values.items()
        if p_val is not None and p_val < significance_level
    ]
    significant.sort(key=lambda item: item[1])
    return significant


def create_benchmark_report(
    results_df: pd.DataFrame, output_path: str = "benchmark_report.html"
) -> Path:
    """Create a lightweight HTML report summarising benchmark statistics.

    Args:
        results_df: Benchmark results dataframe.
        output_path: Output HTML path.

    Returns:
        Path to the generated report.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if results_df.empty:
        html = "<html><body><h1>No benchmark results available.</h1></body></html>"
        output.write_text(html, encoding="utf-8")
        return output

    summary = results_df.copy()
    summary["ci"] = summary.apply(
        lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]", axis=1
    )
    summary["p_values"] = summary["pairwise_p_values"].fillna("{}")
    summary["effect_sizes"] = summary["pairwise_effect_sizes"].fillna("{}")

    table_html = summary[
        [
            "dataset_name",
            "oversampler_name",
            "metric",
            "mean_error",
            "std_error",
            "ci",
            "p_values",
            "effect_sizes",
            "recommended_samples",
        ]
    ].to_html(index=False, escape=False)

    html = f"""
    <html>
        <head>
            <title>OversampleQA Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 2rem; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ccc; padding: 0.5rem; }}
                th {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>OversampleQA Benchmark Report</h1>
            {table_html}
        </body>
    </html>
    """
    output.write_text(html, encoding="utf-8")
    return output
