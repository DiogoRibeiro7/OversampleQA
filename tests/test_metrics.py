import numpy as np

from oversampleqa.metrics import (
    calculate_error_rate,
    check_model_fairness,
    confidence_ratio,
    local_density_divergence,
    minority_recall_loss,
    noise_sensitivity_diagnostic,
    umap_manifold_distance,
)


def test_calculate_error_rate():
    assert calculate_error_rate(5, 10) == 0.5


def test_confidence_ratio():
    assert confidence_ratio(2.0, 1.0) == 2.0


def test_local_density_divergence_zero():
    X = np.array([[0.0], [1.0], [2.0]])
    assert local_density_divergence(X, X) == 0.0


def test_minority_recall_loss():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    loss = minority_recall_loss(y_true, y_pred, minority_label=1)
    assert loss == 0.5


def test_umap_manifold_distance_runs():
    real = np.random.randn(20, 3)
    synth = real[:5] + 0.5
    dist = umap_manifold_distance(real, synth, random_state=0)
    assert dist >= 0.0


def test_check_model_fairness():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    protected = np.array([0, 0, 1, 1])
    diff = check_model_fairness(y_true, y_pred, protected, minority_label=1)
    assert diff >= 0.0


def test_noise_sensitivity_diagnostic(tmp_path):
    from imblearn.over_sampling import SMOTE
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=600, weights=[0.8, 0.2], random_state=0)
    df = noise_sensitivity_diagnostic(
        X,
        y,
        minority_label=1,
        oversampler=SMOTE(random_state=0),
        noise_levels=[0.0, 0.1],
        random_state=0,
    )
    assert list(df.columns) == ["noise", "error_rate", "n_flipped"]
    assert len(df) == 2
    # n_flipped joined the frame so the applied noise can be checked rather
    # than assumed: replacements used to be drawn from all classes, so a
    # selected point could keep its own label and binary data realised half
    # the requested level.
    assert df["n_flipped"].iloc[0] == 0
    assert df["n_flipped"].iloc[1] == int(len(y) * 0.1)
