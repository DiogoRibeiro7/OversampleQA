"""Plotting helpers for oversampleqa."""

from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
        reducer = UMAP(n_components=2, random_state=42)

    comps = reducer.fit_transform(X)
    n_maj = len(majority)
    n_min = len(minority)

    plt.figure()
    plt.scatter(comps[:n_maj, 0], comps[:n_maj, 1], label="majority", alpha=0.5)
    plt.scatter(
        comps[n_maj:n_maj + n_min, 0],
        comps[n_maj:n_maj + n_min, 1],
        label="minority",
        alpha=0.5,
    )
    plt.scatter(
        comps[n_maj + n_min:, 0],
        comps[n_maj + n_min:, 1],
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
    else:
        plt.close()


def plot_error_comparison(
    benchmark_results: pd.DataFrame, save_path: str | None = None
) -> None:
    """Bar plot showing mean error rates for each oversampler."""
    summary = benchmark_results.groupby("oversampler")["error_rate"].mean()
    summary.plot(kind="bar")
    plt.ylabel("Mean error rate")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.close()


def plot_error_boxplot(
    benchmark_results: pd.DataFrame, save_path: str | None = None
) -> None:
    """Boxplot of error rates for each oversampler."""
    benchmark_results.boxplot(column="error_rate", by="oversampler")
    plt.ylabel("Error rate")
    plt.title("Error rate distribution")
    plt.suptitle("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
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

    labels = class_labels if class_labels is not None else list(range(len(error_matrix)))
    df = pd.DataFrame(error_matrix, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Hidden class")
    plt.ylabel("Synthetic class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.close()


def plot_error_ranking(
    benchmark_results: pd.DataFrame, save_path: str | None = None
) -> None:
    """Line chart of mean error rate ranked by oversampler."""
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
    else:
        plt.close()


def plot_noise_sensitivity(
    results: pd.DataFrame, save_path: str | None = None
) -> None:
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
    else:
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
    minority_nearest = (
        dist_minority.min(axis=1) if dist_minority.size else np.array([])
    )

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
