"""Plotting helpers for oversampleqa."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .inference import FriedmanNemenyiResult

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.projections.polar import PolarAxes
from numpy.typing import NDArray
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

try:  # Optional dependency
    from umap import UMAP
except Exception as exc:  # pragma: no cover - optional import
    logger.warning("UMAP not available: %s", exc)
    UMAP = None


def plot_sample_distribution(
    majority: NDArray[np.floating],
    minority: NDArray[np.floating],
    synthetic: NDArray[np.floating],
    hidden_majority: NDArray[np.floating] | None = None,
    method: str = "pca",
    save_path: str | None = None,
) -> None:
    """Visualize sample distribution using PCA or UMAP.

    Parameters
    ----------
    majority, minority, synthetic : ndarray
        Arrays of majority, minority and synthetic samples.
    hidden_majority : ndarray, optional
        Hidden majority samples for reference.
    method : {{"pca", "umap"}}, default="pca"
        Dimensionality reduction method to use.
    save_path : str, optional
        If given, path to save the resulting plot. Otherwise the figure is
        closed and not displayed.
    """

    if method not in {"pca", "umap"}:
        raise ValueError("method must be 'pca' or 'umap'")

    X = np.vstack([majority, minority, synthetic])
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        if UMAP is None:
            raise ImportError("umap-learn is required for method='umap'")
        reducer = UMAP(n_components=2, random_state=42, n_jobs=1)

    comps = reducer.fit_transform(X)
    n_maj = len(majority)
    n_min = len(minority)

    plt.figure()
    plt.scatter(comps[:n_maj, 0], comps[:n_maj, 1], label="majority", alpha=0.5)
    plt.scatter(
        comps[n_maj : n_maj + n_min, 0],
        comps[n_maj : n_maj + n_min, 1],
        label="minority",
        alpha=0.5,
    )
    plt.scatter(
        comps[n_maj + n_min :, 0],
        comps[n_maj + n_min :, 1],
        label="synthetic",
        alpha=0.5,
    )

    if hidden_majority is not None:
        hid_comps = reducer.transform(hidden_majority)
        plt.scatter(
            hid_comps[:, 0], hid_comps[:, 1], label="hidden majority", marker="x"
        )

    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_error_comparison(
    benchmark_results: pd.DataFrame, save_path: str | None = None
) -> None:
    """Bar plot showing mean error rates for each oversampler.

    Args:
        benchmark_results: Benchmark results dataframe.
        save_path: Optional output image path.
    """
    summary = benchmark_results.groupby("oversampler")["error_rate"].mean()
    summary.plot(kind="bar")
    plt.ylabel("Mean error rate")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_error_boxplot(
    benchmark_results: pd.DataFrame, save_path: str | None = None
) -> None:
    """Boxplot of error rates for each oversampler.

    Args:
        benchmark_results: Benchmark results dataframe.
        save_path: Optional output image path.
    """
    benchmark_results.boxplot(column="error_rate", by="oversampler")
    plt.ylabel("Error rate")
    plt.title("Error rate distribution")
    plt.suptitle("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_error_heatmap(
    error_matrix: NDArray[np.integer],
    class_labels: list[int] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot heatmap of a multi-class error attribution matrix.

    Parameters
    ----------
    error_matrix : ndarray
        Matrix where ``matrix[i, j]`` counts synthetic samples generated for
        class ``i`` that are closest to hidden samples from class ``j``.
    class_labels : list of int, optional
        Labels for the classes corresponding to the rows/columns of the matrix.
        If not provided, integer indices are used.
    save_path : str, optional
        If given, path to save the resulting plot. Otherwise the figure is
        closed and not displayed.
    """

    labels = (
        class_labels if class_labels is not None else list(range(len(error_matrix)))
    )
    df = pd.DataFrame(error_matrix, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Hidden class")
    plt.ylabel("Synthetic class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_error_ranking(
    benchmark_results: pd.DataFrame, save_path: str | None = None
) -> None:
    """Line chart of mean error rate ranked by oversampler.

    Args:
        benchmark_results: Benchmark results dataframe.
        save_path: Optional output image path.
    """
    summary = (
        benchmark_results.groupby("oversampler")["error_rate"].mean().sort_values()
    )
    plt.figure()
    plt.plot(range(1, len(summary) + 1), summary.values, marker="o")
    plt.xticks(range(1, len(summary) + 1), summary.index, rotation=45, ha="right")
    plt.xlabel("Rank (lower is better)")
    plt.ylabel("Mean error rate")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_noise_sensitivity(results: pd.DataFrame, save_path: str | None = None) -> None:
    """Line plot showing error rate as label noise increases.

    Parameters
    ----------
    results : DataFrame
        Output of :func:`oversampleqa.metrics.noise_sensitivity_diagnostic`,
        expected to contain ``noise`` and ``error_rate`` columns.
    save_path : str, optional
        If given, path to save the resulting plot. Otherwise the figure is
        closed and not displayed.
    """

    plt.figure()
    sns.lineplot(data=results, x="noise", y="error_rate", marker="o")
    plt.xlabel("Label noise")
    plt.ylabel("Error rate")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_distance_histogram(
    dist_hidden: NDArray[np.floating],
    dist_minority: NDArray[np.floating],
    save_path: str | None = None,
) -> None:
    """Histogram of nearest distances to hidden majority and real minority samples.

    Parameters
    ----------
    dist_hidden, dist_minority : ndarray
        Distance matrices where rows correspond to synthetic samples and
        columns to hidden majority or real minority samples respectively.
    save_path : str, optional
        If given, path to save the resulting plot. Otherwise the figure is
        closed and not displayed.
    """

    hidden_nearest = dist_hidden.min(axis=1) if dist_hidden.size else np.array([])
    minority_nearest = dist_minority.min(axis=1) if dist_minority.size else np.array([])

    plt.figure()
    if hidden_nearest.size:
        sns.histplot(hidden_nearest, color="red", alpha=0.5, label="hidden")
    if minority_nearest.size:
        sns.histplot(minority_nearest, color="blue", alpha=0.5, label="minority")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.close()


def plot_class_balance(
    labels_before: NDArray[np.integer],
    labels_after: NDArray[np.integer],
    save_path: str | None = None,
) -> None:
    """Bar chart comparing class counts before and after oversampling.

    Parameters
    ----------
    labels_before, labels_after : ndarray
        Class labels prior to oversampling and after applying an oversampler.
    save_path : str, optional
        If given, path to save the resulting plot. Otherwise the figure is
        closed and not displayed.
    """

    counts_before = pd.Series(labels_before).value_counts().sort_index()
    counts_after = pd.Series(labels_after).value_counts().sort_index()
    df = pd.DataFrame({"before": counts_before, "after": counts_after})
    df.plot(kind="bar")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.close()


def plot_critical_difference(
    result: FriedmanNemenyiResult,
    save_path: str | None = None,
) -> None:
    """Draw a critical-difference diagram (Demsar 2006).

    Methods are placed on an axis by mean rank, best on the left. Methods whose
    ranks differ by less than the critical difference are joined by a bar,
    meaning the data does not separate them. The bar is the point of the
    diagram: it shows how much of the apparent ordering is noise.

    Args:
        result: Outcome of :func:`~oversampleqa.inference.friedman_nemenyi`.
        save_path: Where to write the figure. Closed without saving if omitted.
    """
    ranks = np.asarray(result.mean_ranks)
    names = list(result.method_names)
    order = np.argsort(ranks)

    fig, ax = plt.subplots(figsize=(8, 2 + 0.35 * len(names)))
    lo, hi = 0.5, len(names) + 0.5
    ax.set_xlim(hi, lo)  # rank 1 (best) on the left
    ax.set_ylim(0, len(names) + 2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_position(("data", len(names) + 1))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Mean rank (lower is better)")

    for row, idx in enumerate(order):
        y = len(names) - row
        ax.plot([ranks[idx], ranks[idx]], [y, len(names) + 1], color="0.4", lw=0.8)
        ax.plot(
            [ranks[idx], lo if row < len(names) / 2 else hi],
            [y, y],
            color="0.4",
            lw=0.8,
        )
        ha = "left" if row < len(names) / 2 else "right"
        ax.text(
            lo if row < len(names) / 2 else hi,
            y,
            f"  {names[idx]} ({ranks[idx]:.2f})  ",
            va="center",
            ha=ha,
        )

    # Bars joining groups that the critical difference cannot separate.
    cd = result.critical_difference
    bar_y = 0.6
    sorted_ranks = ranks[order]
    drawn: list[tuple[float, float]] = []
    for i in range(len(sorted_ranks)):
        j = i
        while j + 1 < len(sorted_ranks) and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i and not any(
            a <= sorted_ranks[i] and sorted_ranks[j] <= b for a, b in drawn
        ):
            ax.plot(
                [sorted_ranks[i] - 0.03, sorted_ranks[j] + 0.03],
                [bar_y, bar_y],
                color="0.1",
                lw=3,
                solid_capstyle="butt",
            )
            drawn.append((sorted_ranks[i], sorted_ranks[j]))
            bar_y += 0.35

    ax.set_title(
        f"Critical difference = {cd:.2f} "
        f"(alpha={result.alpha}, {result.n_datasets} datasets)\n"
        "Methods joined by a bar are not significantly different",
        fontsize=9,
        pad=28,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.close(fig)


# Radar axes for the fidelity suite, as (label, payload key, invert, bounded).
#
# Every axis is oriented so that outward means better. A radar chart whose
# spokes disagree on direction cannot be read at all: the eye judges the area of
# the polygon, so a larger polygon has to mean "better" on every spoke or it
# means nothing. `boundary_violation_strict` is therefore plotted as its
# complement rather than raw.
#
# `bounded` records whether the metric is confined to [0, 1] by construction.
# Density and the memorisation ratio are not -- density routinely exceeds 1 when
# synthetic points cluster more tightly than the real minority, and the
# memorisation ratio is a quotient of two median distances with no upper limit.
# Both are clipped for drawing, and the clipping is reported in the caption,
# because a value of 2.4 and a value of 1.0 otherwise render identically.
_FIDELITY_AXES: tuple[tuple[str, str, bool, bool], ...] = (
    ("precision", "precision", False, True),
    ("recall", "recall", False, True),
    ("coverage", "coverage", False, True),
    ("density", "density", False, False),
    ("diversity", "memorisation_distance_ratio", False, False),
    ("boundary safety", "boundary_violation_strict", True, True),
)


def _fidelity_axis_values(
    payload: Mapping[str, Any],
    selected: Sequence[tuple[str, str, bool, bool]],
    name: str = "",
) -> tuple[list[float], list[str]]:
    """Map one report onto the radial axes, returning values and clip notes.

    Split out of :func:`plot_fidelity_radar` so the orientation and clipping
    rules can be tested directly. Sealed inside the drawing call, the only
    assertion available is that a file appeared, which would not catch an
    inverted axis pointing the wrong way.

    Args:
        payload: Flat mapping of metric name to value.
        selected: Axis specs as ``(label, key, invert, bounded)``.
        name: Oversampler name, used only to label clip notes.

    Returns:
        The per-axis values in ``selected`` order, and a note for every value
        that had to be clipped to 1.0.
    """
    values: list[float] = []
    notes: list[str] = []
    for label, key, invert, bounded in selected:
        raw = float(payload.get(key, np.nan))
        # nan is preserved rather than coerced to 0.0: a zero would be
        # indistinguishable from a genuine measurement of total failure.
        value = 1.0 - raw if invert and not np.isnan(raw) else raw
        if not bounded and not np.isnan(value) and value > 1.0:
            notes.append(f"{name} {label}={value:.2f}".strip())
            value = 1.0
        values.append(value)
    return values, notes


def plot_fidelity_radar(
    reports: Mapping[str, Any],
    save_path: str | None = None,
    metrics: Sequence[str] | None = None,
) -> None:
    """Compare oversamplers across the fidelity suite on one radar chart.

    Outward is better on every axis. ``boundary safety`` is the complement of
    the strict violation rate for that reason; the raw rate is better when
    small, and mixing directions on one chart makes the area meaningless.

    The validation error rate is deliberately absent. It answers a different
    question -- whether synthetic points are confusable with held-out majority
    -- and putting it on the same polygon invites reading it as commensurable
    with the manifold metrics, which is the confusion
    :doc:`/fidelity` exists to prevent.

    Metrics that are ``nan`` because nothing could be measured are left as
    ``nan``, which draws a gap in the polygon. They are not coerced to zero: a
    zero here would be indistinguishable from a genuine measurement of total
    failure.

    Args:
        reports: Mapping of oversampler name to a
            :class:`~oversampleqa.fidelity.FidelityReport`, or to any mapping
            carrying the same keys.
        save_path: Where to write the figure. Closed without saving if omitted.
        metrics: Subset of axis labels to draw, in order. Defaults to all six.

    Raises:
        ValueError: If ``reports`` is empty, if ``metrics`` names an unknown
            axis, or if fewer than three axes are selected -- a radar chart
            with two spokes is a line, and with one is a point.
    """
    if not reports:
        raise ValueError("reports is empty; nothing to plot")

    known = {label: spec for spec in _FIDELITY_AXES for label in (spec[0],)}
    if metrics is None:
        selected = list(_FIDELITY_AXES)
    else:
        unknown = [m for m in metrics if m not in known]
        if unknown:
            raise ValueError(
                f"unknown metric(s) {unknown}; available: {sorted(known)}"
            )
        selected = [known[m] for m in metrics]
    if len(selected) < 3:
        raise ValueError(
            f"a radar chart needs at least 3 axes, got {len(selected)}"
        )

    labels = [spec[0] for spec in selected]
    angles = np.linspace(0.0, 2 * np.pi, len(selected), endpoint=False)
    closed = np.concatenate([angles, angles[:1]])  # close the polygon

    # subplot_kw={"polar": True} really does return a PolarAxes, but the stubs
    # only promise Axes, which has no set_rlabel_position.
    fig, base_ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax = cast(PolarAxes, base_ax)
    clipped: list[str] = []

    for name, report in reports.items():
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        values, notes = _fidelity_axis_values(payload, selected, name)
        clipped.extend(notes)
        series = np.concatenate([np.asarray(values, dtype=float), [values[0]]])
        ax.plot(closed, series, linewidth=1.8, label=name)
        ax.fill(closed, series, alpha=0.12)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    # Park the radial labels halfway between two spokes. Left at the default
    # they sit on the first axis, directly under the data lines.
    ax.set_rlabel_position(180.0 / len(selected))
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)
    ax.set_title("Fidelity profile (outward is better on every axis)", pad=24)

    if clipped:
        fig.text(
            0.5,
            0.015,
            "clipped at 1.0: " + ", ".join(clipped),
            ha="center",
            fontsize=7,
            style="italic",
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
