"""Benchmark utilities for oversampleqa."""

from __future__ import annotations

import logging
import pathlib
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._export_metadata import write_export_metadata
from ._json import write_json
from ._provenance import openml_provenance, synthetic_provenance
from ._render import frame_to_html, frame_to_markdown
from ._rng import RandomStateLike, as_generator
from .validator import validate_oversampling

logger = logging.getLogger(__name__)

#: Column order of the long-format frame returned by :func:`run_benchmark`.
#: One row per (dataset, oversampler, metric, hidden_ratio, run).
_BENCHMARK_COLUMNS = (
    "dataset",
    "oversampler",
    "metric",
    "hidden_ratio",
    "run",
    "error_rate",
)



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
                            # The metric is part of what identifies a
                            # measurement, not just an argument to it. Without
                            # it, concatenating two sweeps run under different
                            # metrics gives a frame whose rows cannot be told
                            # apart -- and error rates are not comparable
                            # across metrics.
                            "metric": distance_metric,
                            "hidden_ratio": ratio,
                            "run": run,
                            "error_rate": error,
                        }
                    )
    # Fixed column order even when empty, so a caller that correctly handles
    # "no results" still gets a frame it can select columns from.
    frame = pd.DataFrame(results, columns=list(_BENCHMARK_COLUMNS))
    frame.attrs["dataset_provenance"] = {
        str(data.get("name", "dataset")): data["provenance"]
        for data in datasets
        if "provenance" in data
    }
    frame.attrs["benchmark_parameters"] = {
        "hidden_ratios": hidden_ratios,
        "n_runs": n_runs,
        "distance_metric": distance_metric,
        "random_state": repr(random_state),
    }
    return frame



def _imbalance(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    minority_label: int,
    keep: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Thin one class down to ``keep`` points, so the dataset is imbalanced.

    ``make_moons`` and ``make_circles`` return exactly balanced classes. A
    balanced dataset has nothing to oversample: SMOTE generates no synthetic
    points, ``validate_oversampling`` returns ``nan``, and the dataset
    contributes a missing measurement to every benchmark it appears in. Two of
    the seven built-in datasets behaved that way.

    Subsampling is random rather than positional, since these generators emit
    the classes in blocks and a head slice would take one whole class.
    """
    minority_idx = np.flatnonzero(y == minority_label)
    majority_idx = np.flatnonzero(y != minority_label)
    kept = rng.choice(minority_idx, size=min(keep, len(minority_idx)), replace=False)
    order = np.sort(np.concatenate([majority_idx, kept]))
    return X[order], y[order]


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

    Xc, yc = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    datasets.append(
        {
            "name": "classification",
            "data": Xc,
            "target": yc,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_classification",
                n_samples=1000,
                weights=[0.9, 0.1],
                random_state=42,
            ),
        }
    )

    Xm, ym = make_moons(n_samples=600, noise=0.2, random_state=42)
    Xm, ym = _imbalance(Xm, ym, 1, keep=60, rng=np.random.default_rng(42))
    datasets.append(
        {
            "name": "moons",
            "data": Xm,
            "target": ym,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_moons",
                n_samples=600,
                noise=0.2,
                random_state=42,
                minority_kept=60,
                subsample_seed=42,
            ),
        }
    )

    Xr, yr = make_circles(n_samples=600, noise=0.1, factor=0.5, random_state=42)
    Xr, yr = _imbalance(Xr, yr, 1, keep=60, rng=np.random.default_rng(42))
    datasets.append(
        {
            "name": "circles",
            "data": Xr,
            "target": yr,
            "minority_label": 1,
            "provenance": synthetic_provenance(
                "sklearn.datasets.make_circles",
                n_samples=600,
                noise=0.1,
                factor=0.5,
                random_state=42,
                minority_kept=60,
                subsample_seed=42,
            ),
        }
    )

    Xb, yb = make_blobs(
        n_samples=[450, 60],
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
                n_samples=[450, 60],
                centers=[(-2, 0), (2, 0)],
                cluster_std=[1.0, 1.0],
                random_state=42,
            ),
        }
    )

    Xh, yh = make_classification(
        n_samples=1200,
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
                n_samples=1200,
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
        n_samples=1200,
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
                n_samples=1200,
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
        n_samples=1200,
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
                n_samples=1200,
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


#: Columns that identify one comparable experiment. Error rates are only
#: commensurable within a fixed (dataset, hidden_ratio, metric).
_SPECIFICATION_COLUMNS = ("dataset", "hidden_ratio", "metric")


def compute_ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Rank oversamplers within each experiment, then aggregate the ranks.

    Error rates are not comparable across datasets, hold-out ratios or metrics:
    an easy dataset scores near 0.1 and a hard one near 0.9, and hassanat scores
    roughly twice euclidean on the same data. Pooling them and taking a mean
    asks a question with no answer.

    Ranking within each ``(dataset, hidden_ratio, metric)`` and averaging those
    ranks is the Demsar (2006) protocol, and the same logic underlying
    :func:`~oversampleqa.inference.friedman_nemenyi` -- so the ranking here and
    the significance test there answer the same question.

    Args:
        results: Long-format benchmark frame from :func:`run_benchmark`.

    Returns:
        Summary indexed by oversampler with ``mean_rank`` (lower is better),
        ``rank``, ``n_specifications``, and the pooled ``mean``, ``std`` and
        ``n_missing`` retained for reference.

    Warns:
        UserWarning: If oversamplers were ranked over different numbers of
            experiments. Mean ranks computed over different sets are not
            comparable, and the imbalance is usually caused by skipped runs.

    Notes:
        Averaging the raw error rate was not merely imprecise, it inverted
        results. Given a sampler that beats another on *every* dataset while
        having more of its runs skipped on the hard one, the pooled mean
        favours the loser -- Simpson's paradox, reachable here because the
        hold-out guards legitimately drop runs.

        ``nan`` runs are excluded rather than counted as zero, and the count is
        reported in ``n_missing``.
    """
    grouped = results.groupby("oversampler")["error_rate"]
    summary = grouped.agg(
        mean=lambda s: s.mean(skipna=True),
        std=lambda s: s.std(skipna=True),
    )
    summary["n_missing"] = grouped.apply(lambda s: int(s.isna().sum()))

    spec = [c for c in _SPECIFICATION_COLUMNS if c in results.columns]
    if not spec:
        # Nothing identifies separate experiments, so every row is already
        # comparable and the pooled mean is the only available ordering.
        summary["mean_rank"] = summary["mean"].rank(method="average")
        summary["n_specifications"] = 1
        summary["rank"] = summary["mean_rank"].rank(method="min")
        return summary

    # One score per (experiment, oversampler), then rank within the experiment.
    per_spec = results.groupby([*spec, "oversampler"])["error_rate"].mean()
    ranks = per_spec.groupby(level=list(range(len(spec)))).rank(method="average")

    mean_rank = ranks.groupby("oversampler").mean()
    counts = ranks.groupby("oversampler").count()
    summary["mean_rank"] = mean_rank
    summary["n_specifications"] = counts.astype("Int64")
    summary["rank"] = summary["mean_rank"].rank(method="min")

    if counts.nunique() > 1:
        warnings.warn(
            "Oversamplers were ranked over different numbers of experiments "
            f"({counts.to_dict()}). Mean ranks computed over different sets of "
            "experiments are not comparable; the imbalance usually means some "
            "runs were skipped. Check n_missing.",
            UserWarning,
            stacklevel=2,
        )
    return summary


def export_benchmark_results(
    results: pd.DataFrame, output_path: str, fmt: str = "csv"
) -> None:
    """Export benchmark summary to CSV, JSON or Markdown.

    Args:
        results: Benchmark results dataframe.
        output_path: Destination path.
        fmt: Output format: ``csv``, ``json``, ``markdown`` or ``html``.
            All four render the same ranking frame.

    Raises:
        ValueError: If ``fmt`` is not one of the four.
    """
    output = pathlib.Path(output_path)
    summary = compute_ranking(results)
    summary.attrs["source"] = {
        "row_count": len(results),
        "columns": [str(column) for column in results.columns],
        "attrs": dict(results.attrs),
    }
    fmt = fmt.lower()
    if fmt == "csv":
        summary.to_csv(output)
    elif fmt == "json":
        # nan becomes null. JSON has no NaN literal, and emitting one produces a
        # document that strict parsers reject; null at least round-trips.
        write_json(output, summary.reset_index().to_dict(orient="records"))
    elif fmt == "markdown":
        # This used to be `summary.to_csv(sep="|")`, which is not Markdown: no
        # header separator row and no edge pipes, so it rendered as one run-on
        # paragraph. The same bug was fixed in report.py; it survived here
        # because the renderer was duplicated rather than shared.
        output.write_text(frame_to_markdown(summary), encoding="utf-8")
    elif fmt == "html":
        output.write_text(frame_to_html(summary), encoding="utf-8")
    else:
        raise ValueError("fmt must be 'csv', 'json', 'markdown' or 'html'")

    write_export_metadata(output, export_kind="benchmark_summary", data=summary)
