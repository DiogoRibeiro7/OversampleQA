from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.clustering import cluster_based_diagnostics


def test_cluster_based_diagnostics_kmeans():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        weights=[0.9, 0.1],
        random_state=0,
    )
    X_majority = X[y == 0]
    oversampler = SMOTE(random_state=0)
    X_res, y_res = oversampler.fit_resample(X, y)
    synthetic = X_res[len(X) :][y_res[len(X) :] == 1]
    flagged, score = cluster_based_diagnostics(X_majority, synthetic, n_clusters=2)
    assert flagged.dtype == bool
    assert len(flagged) == len(synthetic)
    assert 0.0 <= score <= 1.0


def test_cluster_based_diagnostics_dbscan():
    X, y = make_classification(
        n_samples=150,
        n_features=4,
        weights=[0.85, 0.15],
        random_state=1,
    )
    X_majority = X[y == 0]
    oversampler = SMOTE(random_state=1)
    X_res, y_res = oversampler.fit_resample(X, y)
    synthetic = X_res[len(X) :][y_res[len(X) :] == 1]
    flagged, score = cluster_based_diagnostics(
        X_majority,
        synthetic,
        algorithm="dbscan",
        eps=0.8,
        min_samples=3,
    )
    assert flagged.dtype == bool
    assert len(flagged) == len(synthetic)
    assert 0.0 <= score <= 1.0
