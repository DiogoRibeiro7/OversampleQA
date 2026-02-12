"""Diagnostics example for density divergence and noise sensitivity."""

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from oversampleqa.metrics import (
    local_density_divergence,
    noise_sensitivity_diagnostic,
)
from oversampleqa.validator import extract_synthetic_samples


def main() -> None:
    X, y = make_classification(
        n_samples=600,
        n_features=8,
        weights=[0.9, 0.1],
        random_state=42,
    )

    oversampler = SMOTE(random_state=42)
    X_res, y_res = oversampler.fit_resample(X, y)
    synthetic = extract_synthetic_samples(X, X_res, y_res, minority_label=1)

    density_gap = local_density_divergence(synthetic, X[y == 1], k=5)
    print(f"Local density divergence: {density_gap:.3f}")

    noise_df = noise_sensitivity_diagnostic(
        X,
        y,
        minority_label=1,
        oversampler=SMOTE(random_state=0),
        noise_levels=[0.0, 0.1, 0.2],
        hidden_ratio=0.1,
    )
    print(noise_df)


if __name__ == "__main__":
    main()
