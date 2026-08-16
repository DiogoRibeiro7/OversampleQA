"""Benchmark utilities for oversampleqa."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ._provenance import openml_provenance, synthetic_provenance
from ._rng import RandomStateLike, as_generator
from .validator import validate_oversampling

logger = logging.getLogger(__name__)



def run_benchmark(
    datasets: list[dict],
    oversamplers: list,
    hidden_ratios: list[float] | None = None,
    n_runs: int = 10,
    distance_metric: str = "hassanat",
    random_state: RandomStateLike = None,
) -> pd.DataFrame:
    """Run validation across datasets and oversampling methods.

    Args:
        datasets: Dataset descriptors containing ``data`` and ``target``.
        oversamplers: Oversampler instances.
        hidden_ratios: Hidden ratios to evaluate.
        n_runs: Number of repetitions per configuration.
        distance_metric: Distance metric name.
        random_state: RNG seed for reproducibility.

    Returns:
        DataFrame with per-run error rates.
    """
    if hidden_ratios is None:
        hidden_ratios = [0.1, 0.25, 0.5]

    results = []
    rng = as_generator(random_state)

    logger.info("Starting benchmark with %d datasets", len(datasets))

    for data in datasets:
        X, y = data["data"], data["target"]
        minority_label = data.get("minority_label", 1)
        for oversampler in oversamplers:
            for ratio in hidden_ratios:
                for run in range(n_runs):
                    rs = rng.integers(0, 1_000_000)
                    oversampler.random_state = rs
                    # Vary the hold-out split per run as well. Reseeding only the
                    # oversampler left every run sharing one split, so the spread
                    # across runs omitted the largest source of variance.
                    split_seed = int(rng.integers(0, 2**31 - 1))
                    try:
                        error = validate_oversampling(
                            X,
                            y,
                            minority_label,
                            oversampler,
                            hidden_ratio=ratio,
                            metric=distance_metric,
                            random_state=split_seed,
                        )
                    except ValueError as exc:
                        # A dataset whose minority is too small to hold out from
                        # cannot support the estimand. Record it as a missing
                        # measurement and carry on, rather than aborting the whole
                        # sweep or -- worse -- recording a 0.0 that would read as a
                        # perfect score. compute_ranking reports these as n_missing.
                        warnings.warn(
                            f"Skipping {data.get('name', 'dataset')} with "
                            f"{oversampler.__class__.__name__} at hidden_ratio="
                            f"{ratio}: {exc}",
                            UserWarning,
                            stacklevel=2,
                        )
                        error = float("nan")
                    except Exception:
                        logger.exception("Validation failed for %s", oversampler)
                        raise
                    results.append(
                        {
                            "dataset": data.get("name", "dataset"),
                            "oversampler": oversampler.__class__.__name__,
                            "hidden_ratio": ratio,
                            "run": run,
                            "error_rate": error,
                        }
                    )
    return pd.DataFrame(results)


def load_standard_datasets(include_openml: bool = False) -> list[dict]:
    """Return a list of simple synthetic datasets for benchmarking.

    Parameters
    ----------
    include_openml:
        Whether to attempt downloading additional datasets from OpenML. The
        default is ``False`` to avoid slow network calls during tests.

    Returns
    -------
    list of dict
        Each entry contains ``name``, ``data``, ``target``,
        ``minority_label`` and ``provenance`` keys. The ``provenance`` value
        is a dict describing the dataset's ``source``, ``generator``,
        ``params``, ``url``, ``license`` and ``notes``.
    """

    from sklearn.datasets import (
        make_blobs,
        make_circles,
        make_classification,
        make_moons,
    )

    datasets: list[dict] = []

    if include_openml:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler

        openml_specs = [
            ("yeast-4", "class"),
            ("yeast-5", "class"),
            ("yeast-6", "class"),
            ("vehicle", "Class"),
        ]
        for name, target_col in openml_specs:
            try:  # pragma: no cover - network dependent
                ds = fetch_openml(name, version=1, as_frame=False)
                X = StandardScaler().fit_transform(ds.data)
                y = ds[target_col].astype(int)
                minority = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
                datasets.append(
                    {
                        "name": name,
                        "data": X,
                        "target": y,
                        "minority_label": minority,
                        "provenance": openml_provenance(
                            name,
                            1,
                            notes=(
                                "Downloaded from OpenML (version pinned to 1) and "
                                "standardized with StandardScaler."
                            ),
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to fetch %s: %s", name, exc)

    Xc, yc = make_classification(n_samples=200, weights=[0.9, 0.1], random_state=42)
    datasets.append(
        {
            "name": "classification",
            "data": Xc,
            "target": yc,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_classification",
                n_samples=200,
                weights=[0.9, 0.1],
                random_state=42,
            ),
        }
    )

    Xm, ym = make_moons(noise=0.2, random_state=42)
    datasets.append(
        {
            "name": "moons",
            "data": Xm,
            "target": ym,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_moons", noise=0.2, random_state=42
            ),
        }
    )

    Xr, yr = make_circles(noise=0.1, factor=0.5, random_state=42)
    datasets.append(
        {
            "name": "circles",
            "data": Xr,
            "target": yr,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_circles", noise=0.1, factor=0.5, random_state=42
            ),
        }
    )

    Xb, yb = make_blobs(
        n_samples=[90, 10],
        centers=[(-2, 0), (2, 0)],
        cluster_std=[1.0, 1.0],
        random_state=42,
    )
    datasets.append(
        {
            "name": "blobs",
            "data": Xb,
            "target": yb,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_blobs",
                n_samples=[90, 10],
                centers=[(-2, 0), (2, 0)],
                cluster_std=[1.0, 1.0],
                random_state=42,
            ),
        }
    )

    Xh, yh = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.95, 0.05],
        class_sep=0.5,
        random_state=7,
    )
    datasets.append(
        {
            "name": "hard_classification",
            "data": Xh,
            "target": yh,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_classification",
                n_samples=300,
                n_features=10,
                n_informative=5,
                n_redundant=2,
                weights=[0.95, 0.05],
                class_sep=0.5,
                random_state=7,
            ),
        }
    )

    Xe, ye = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],
        class_sep=2.0,
        random_state=21,
    )
    datasets.append(
        {
            "name": "easy_linear",
            "data": Xe,
            "target": ye,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_classification",
                n_samples=200,
                n_features=2,
                n_redundant=0,
                n_clusters_per_class=1,
                weights=[0.95, 0.05],
                class_sep=2.0,
                random_state=21,
            ),
        }
    )

    Xo, yo = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],
        class_sep=0.3,
        flip_y=0.03,
        random_state=22,
    )
    datasets.append(
        {
            "name": "overlap_classification",
            "data": Xo,
            "target": yo,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_classification",
                n_samples=200,
                n_features=2,
                n_redundant=0,
                n_clusters_per_class=1,
                weights=[0.95, 0.05],
                class_sep=0.3,
                flip_y=0.03,
                random_state=22,
            ),
        }
    )

    return datasets


def compute_ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Return mean, stddev and rank of oversamplers.

    Args:
        results: Benchmark results dataframe.

    Returns:
        Summary dataframe with ``mean``, ``std``, ``rank`` and ``n_missing``.

    Notes:
        ``validate_oversampling`` returns ``nan`` when a run produced no
        synthetic samples. Those runs are excluded from the mean and standard
        deviation rather than being counted as zero, and the number excluded is
        reported in ``n_missing`` so a mean computed from very few runs is
        visible rather than silent.
    """
    grouped = results.groupby("oversampler")["error_rate"]
    # pandas skips NaN by default; state it explicitly so the behaviour is a
    # decision rather than an accident.
    summary = grouped.agg(
        mean=lambda s: s.mean(skipna=True),
        std=lambda s: s.std(skipna=True),
    )
    summary["n_missing"] = grouped.apply(lambda s: int(s.isna().sum()))
    summary["rank"] = summary["mean"].rank(method="min")
    return summary


def export_benchmark_results(
    results: pd.DataFrame, output_path: str, fmt: str = "csv"
) -> None:
    """Export benchmark summary to CSV, JSON or Markdown.

    Args:
        results: Benchmark results dataframe.
        output_path: Destination path.
        fmt: Output format (csv, json, markdown).
    """
    summary = compute_ranking(results)
    fmt = fmt.lower()
    if fmt == "csv":
        summary.to_csv(output_path)
    elif fmt == "json":
        summary.reset_index().to_json(output_path, orient="records")
    elif fmt == "markdown":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary.to_csv(sep="|"))
    else:
        raise ValueError("fmt must be 'csv', 'json', or 'markdown'")
