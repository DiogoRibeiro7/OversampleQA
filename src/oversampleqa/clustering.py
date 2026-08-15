"""Cluster-based diagnostics for synthetic samples.

This module implements utilities to cluster majority and synthetic
samples to detect overlap of synthetic data with majority regions.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cluster_based_diagnostics(
    majority: np.ndarray,
    synthetic: np.ndarray,
    n_clusters: int = 5,
    algorithm: str = "kmeans",
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int | None = None,
) -> tuple[np.ndarray, float]:
    """Flag synthetic samples that fall in majority-dominated clusters.

    Parameters
    ----------
    majority, synthetic : ndarray
        Arrays of majority and synthetic samples with shape ``(n_samples, n_features)``.
    n_clusters : int, default=5
        Number of clusters for the k-means algorithm.
    algorithm : {"kmeans", "dbscan"}, default="kmeans"
        Clustering algorithm to use.
    eps : float, default=0.5
        Neighborhood radius when using DBSCAN.
    min_samples : int, default=5
        Minimum samples per cluster for DBSCAN.
    random_state : int, optional
        Random state for k-means.

    Returns
    -------
    flagged : ndarray of bool
        Boolean mask indicating which synthetic samples are located in clusters
        dominated by majority data.
    overlap_score : float
        Silhouette score of the clustering which acts as a crude overlap metric.
    """

    if len(synthetic) == 0:
        return np.array([], dtype=bool), 0.0

    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    X = np.vstack([majority, synthetic])

    try:
        if algorithm == "kmeans":
            labels = KMeans(
                n_clusters=n_clusters, random_state=random_state
            ).fit_predict(X)
        elif algorithm == "dbscan":
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        else:
            raise ValueError("algorithm must be 'kmeans' or 'dbscan'")
    except Exception:  # pragma: no cover - defensive
        logger.exception("Clustering failed")
        raise

    maj_labels = labels[: len(majority)]
    synth_labels = labels[len(majority) :]

    flagged = np.zeros(len(synthetic), dtype=bool)
    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    for lbl in unique_labels:
        maj_mask = maj_labels == lbl
        synth_mask = synth_labels == lbl
        n_maj = maj_mask.sum()
        n_syn = synth_mask.sum()
        if n_syn == 0:
            continue
        ratio = n_maj / (n_maj + n_syn)
        if ratio > 0.5:
            flagged[synth_mask] = True

    if len(np.unique(labels)) > 1:
        try:
            score = silhouette_score(X, labels)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to compute silhouette score")
            score = 0.0
    else:
        score = 0.0

    return flagged, float(score)
