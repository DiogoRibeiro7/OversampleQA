"""Advanced benchmarking utilities with statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import warnings

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
    error_rates: List[float]
    mean_error: float
    std_error: float
    confidence_interval: Tuple[float, float]
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    recommended_samples: Optional[int] = None


class StatisticalBenchmark:
    """Advanced benchmarking engine with statistical analysis."""

    def __init__(
        self,
        n_folds: int = 5,
        n_repeats: int = 5,
        confidence_level: float = 0.95,
        correction_method: str = "holm",
        random_state: Optional[int] = 42,
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
        datasets: Sequence[Dict[str, Any]],
        oversamplers: Sequence[Any],
        metrics: Optional[Sequence[str]] = None,
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

        all_results: List[BenchmarkResult] = []
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
        dataset: Dict[str, Any],
        oversamplers: Sequence[Any],
        metrics: Sequence[str],
    ) -> List[BenchmarkResult]:
        X, y = dataset["data"], dataset["target"]
        dataset_name = dataset.get("name", "dataset")
        minority_label = dataset.get("minority_label", 1)

        # basic preprocessing to avoid scaling issues for distance metrics
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results: List[BenchmarkResult] = []
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
                std_error = float(np.std(error_rates, ddof=1)) if len(error_rates) > 1 else 0.0
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
    ) -> List[float]:
        errors: List[float] = []
        rng = np.random.default_rng(self.random_state)

        for repeat in range(self.n_repeats):
            cv = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=None if self.random_state is None else rng.integers(0, 1_000_000),
            )
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                if len(np.unique(y_train)) < 2:
                    warnings.warn("Training fold lacks class diversity; skipping fold.")
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
                    warnings.warn(f"Validation failed for fold {fold}: {exc}")
                    continue
                if np.isnan(error):
                    continue
                errors.append(float(error))
        return errors

    def _confidence_interval(self, values: Sequence[float]) -> Tuple[float, float]:
        if len(values) < 2:
            return (float(values[0]) if values else 0.0, float(values[0]) if values else 0.0)
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

    def _recommended_sample_size(self, values: Sequence[float], target_power: float = 0.8) -> Optional[int]:
        """Rough sample size recommendation using Cohen's d approximation."""

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
        n = ((z_alpha + z_beta) ** 2) * 2 / (d ** 2)
        return int(math.ceil(n))

    def _add_statistical_analysis(self, frame: pd.DataFrame) -> pd.DataFrame:
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

    def _pairwise_statistical_tests(self, dataset_slice: pd.DataFrame) -> Dict[str, float]:
        p_values: Dict[str, float] = {}
        oversamplers = dataset_slice["oversampler_name"].unique()
        for i, os1 in enumerate(oversamplers):
            for os2 in oversamplers[i + 1 :]:
                errors1 = dataset_slice.loc[
                    dataset_slice["oversampler_name"] == os1, "error_rates"
                ].iloc[0]
                errors2 = dataset_slice.loc[
                    dataset_slice["oversampler_name"] == os2, "error_rates"
                ].iloc[0]
                try:
                    _, p_val = stats.wilcoxon(errors1, errors2)
                except Exception:
                    p_val = 1.0
                key = f"{os1}_vs_{os2}"
                p_values[key] = p_val
        return self._apply_correction(p_values)

    def _calculate_effect_sizes(self, dataset_slice: pd.DataFrame) -> Dict[str, float]:
        effect_sizes: Dict[str, float] = {}
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
        if len(x) < 2 or len(y) < 2:
            return 0.0
        n1, n2 = len(x), len(y)
        s1, s2 = x.var(ddof=1), y.var(ddof=1)
        pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        return math.sqrt(max(pooled, 0.0))

    def _apply_correction(self, p_values: Dict[str, float]) -> Dict[str, float]:
        if not p_values:
            return p_values
        if self.correction_method == "bonferroni":
            factor = len(p_values)
            return {k: min(v * factor, 1.0) for k, v in p_values.items()}
        # default Holm-Bonferroni
        sorted_items = sorted(p_values.items(), key=lambda item: item[1])
        corrected: Dict[str, float] = {}
        m = len(sorted_items)
        for rank, (key, p_val) in enumerate(sorted_items, start=1):
            corrected_val = min((m - rank + 1) * p_val, 1.0)
            corrected[key] = corrected_val
        return corrected

    @staticmethod
    def _result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
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
        domains: Optional[Sequence[str]] = None,
        max_samples: int = 10_000,
        include_openml: bool = False,
    ) -> List[Dict[str, Any]]:
        domains = tuple(domains or ("medical", "financial"))
        datasets: List[Dict[str, Any]] = []
        for domain in domains:
            datasets.extend(self._load_domain(domain, max_samples, include_openml))
        return datasets

    def _load_domain(
        self, domain: str, max_samples: int, include_openml: bool
    ) -> List[Dict[str, Any]]:
        domain = domain.lower()
        if domain == "medical":
            return self._load_medical(max_samples, include_openml)
        if domain == "financial":
            return self._load_financial(max_samples)
        return []

    def _load_medical(self, max_samples: int, include_openml: bool) -> List[Dict[str, Any]]:
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
                warnings.warn(f"OpenML download failed: {exc}")
        return datasets

    def _load_financial(self, max_samples: int) -> List[Dict[str, Any]]:
        try:
            from imbalanced_datasets import creditcard  # optional
        except Exception:  # pragma: no cover
            creditcard = None

        datasets: List[Dict[str, Any]] = []
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
        self, difficulty_levels: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        difficulty_levels = tuple(difficulty_levels or ("easy", "medium", "hard", "extreme"))
        synthetic: List[Dict[str, Any]] = []
        for difficulty in difficulty_levels:
            synthetic.extend(self._generate_difficulty(difficulty))
        return synthetic

    def _generate_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        from sklearn.datasets import make_classification

        difficulty = difficulty.lower()
        configs: List[Dict[str, Any]]
        if difficulty == "easy":
            configs = [
                {"n_samples": 600, "n_features": 8, "class_sep": 2.0, "weights": [0.75, 0.25]},
                {"n_samples": 800, "n_features": 5, "class_sep": 1.8, "weights": [0.8, 0.2]},
            ]
        elif difficulty == "medium":
            configs = [
                {"n_samples": 1000, "n_features": 12, "class_sep": 1.2, "weights": [0.85, 0.15]},
            ]
        elif difficulty == "hard":
            configs = [
                {"n_samples": 1500, "n_features": 20, "class_sep": 0.8, "weights": [0.9, 0.1]},
            ]
        elif difficulty == "extreme":
            configs = [
                {"n_samples": 2000, "n_features": 40, "class_sep": 0.4, "weights": [0.97, 0.03]},
            ]
        else:
            configs = [{"n_samples": 800, "n_features": 10, "class_sep": 1.0, "weights": [0.8, 0.2]}]

        datasets: List[Dict[str, Any]] = []
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


def create_benchmark_report(results_df: pd.DataFrame, output_path: str = "benchmark_report.html") -> Path:
    """Create a lightweight HTML report summarising benchmark statistics."""

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
