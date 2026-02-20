"""Visualize sample distributions after oversampling."""

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.plotting import plot_sample_distribution


def main() -> None:
    X, y = make_classification(n_samples=300, weights=[0.9, 0.1], random_state=0)
    oversampler = SMOTE(random_state=0)
    X_res, y_res = oversampler.fit_resample(X, y)
    synthetic = X_res[len(X) :]
    majority = X[y == 0]
    minority = X[y == 1]
    plot_sample_distribution(
        majority,
        minority,
        synthetic,
        method="umap",
        save_path="distribution.png",
    )


if __name__ == "__main__":
    main()
