from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

from oversampleqa.validator import validate_oversampling
from oversampleqa import noise_sensitivity_diagnostic
from oversampleqa.plotting import (
    plot_sample_distribution,
    plot_error_boxplot,
    plot_error_comparison,
    plot_error_ranking,
    plot_error_heatmap,
    plot_noise_sensitivity,
    plot_distance_histogram,
    plot_class_balance,
)


def test_plot_sample_distribution_runs(tmp_path):
    X, y = make_classification(n_samples=600, weights=[0.9, 0.1], random_state=0)
    oversampler = SMOTE(random_state=0)
    error = validate_oversampling(X, y, minority_label=1, oversampler=oversampler)
    assert 0.0 <= error <= 1.0
    maj = X[y == 0]
    mino = X[y == 1]
    synthetic = oversampler.fit_resample(X, y)[0][len(X):]
    plot_sample_distribution(
        maj,
        mino,
        synthetic,
        method="umap",
        save_path=tmp_path / "out.png",
    )


def test_plot_error_boxplot_and_ranking(tmp_path):
    df = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1", "d1"],
            "oversampler": ["A", "A", "B", "B"],
            "hidden_ratio": [0.1, 0.1, 0.1, 0.1],
            "run": [0, 1, 0, 1],
            "error_rate": [0.2, 0.3, 0.1, 0.15],
        }
    )
    box_path = tmp_path / "box.png"
    rank_path = tmp_path / "rank.png"
    plot_error_boxplot(df, save_path=box_path)
    plot_error_ranking(df, save_path=rank_path)
    assert box_path.exists() and rank_path.exists()


def test_plot_error_comparison(tmp_path):
    df = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1", "d1"],
            "oversampler": ["A", "A", "B", "B"],
            "hidden_ratio": [0.1, 0.1, 0.1, 0.1],
            "run": [0, 1, 0, 1],
            "error_rate": [0.2, 0.3, 0.1, 0.15],
        }
    )
    path = tmp_path / "comparison.png"
    plot_error_comparison(df, save_path=path)
    assert path.exists()


def test_plot_error_heatmap(tmp_path):
    matrix = np.array([[5, 1], [2, 3]])
    heat_path = tmp_path / "heat.png"
    plot_error_heatmap(matrix, class_labels=[0, 1], save_path=heat_path)
    assert heat_path.exists()


def test_plot_noise_sensitivity(tmp_path):
    X, y = make_classification(n_samples=600, weights=[0.9, 0.1], random_state=0)
    oversampler = SMOTE(random_state=0)
    results = noise_sensitivity_diagnostic(
        X, y, minority_label=1, oversampler=oversampler, noise_levels=[0.0, 0.1]
    )
    ns_path = tmp_path / "noise.png"
    plot_noise_sensitivity(results, save_path=ns_path)
    assert ns_path.exists()


def test_plot_distance_histogram(tmp_path):
    X, y = make_classification(n_samples=600, weights=[0.9, 0.1], random_state=0)
    oversampler = SMOTE(random_state=0)
    details = validate_oversampling(
        X, y, minority_label=1, oversampler=oversampler, return_details=True
    )
    assert 0.0 <= details.error_rate <= 1.0
    path = tmp_path / "hist.png"
    plot_distance_histogram(details.dist_hidden, details.dist_min, save_path=path)
    assert path.exists()


def test_plot_class_balance(tmp_path):
    X, y = make_classification(n_samples=600, weights=[0.9, 0.1], random_state=0)
    oversampler = SMOTE(random_state=0)
    X_res, y_res = oversampler.fit_resample(X, y)
    path = tmp_path / "balance.png"
    plot_class_balance(y, y_res, save_path=path)
    assert path.exists()
