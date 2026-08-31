import os
import sys
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath("src"))

from oversampleqa.plotting import (
    plot_class_balance,
    plot_distance_histogram,
    plot_error_heatmap,
)
from oversampleqa.validator import (
    validate_multiclass_oversampling,
    validate_oversampling,
)


def main() -> None:
    static_dir = Path("docs/_static")
    static_dir.mkdir(parents=True, exist_ok=True)

    # Distance histogram example
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        weights=[0.9, 0.1],
        random_state=42,
    )
    details = validate_oversampling(
        X,
        y,
        minority_label=1,
        oversampler=SMOTE(random_state=42),
        hidden_ratio=0.1,
        metric="hassanat",
        return_details=True,
    )
    plot_distance_histogram(
        details.dist_hidden,
        details.dist_min,
        save_path=str(static_dir / "distance_histogram.png"),
    )
    print(f"distance_histogram.png (error_rate={details.error_rate:.3f})")

    # Class balance example
    oversampler = SMOTE(random_state=0)
    _, y_res = oversampler.fit_resample(X, y)
    plot_class_balance(
        labels_before=y,
        labels_after=y_res,
        save_path=str(static_dir / "class_balance.png"),
    )
    print("class_balance.png")

    # Multiclass error heatmap
    Xm, ym = make_classification(
        n_samples=700,
        n_features=8,
        n_classes=3,
        n_informative=6,
        weights=[0.7, 0.2, 0.1],
        random_state=7,
    )
    _, matrix = validate_multiclass_oversampling(
        Xm,
        ym,
        oversampler=SMOTE(random_state=7),
        hidden_ratio=0.1,
        metric="hassanat",
        return_matrix=True,
    )
    plot_error_heatmap(
        error_matrix=matrix,
        class_labels=[0, 1, 2],
        save_path=str(static_dir / "multiclass_heatmap.png"),
    )
    print("multiclass_heatmap.png")


if __name__ == "__main__":
    main()
