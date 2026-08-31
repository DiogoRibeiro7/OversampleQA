"""
Multiclass Validation
=====================

Validate oversampling on a multiclass problem and read the resulting
confusion-style error matrix across classes.
"""

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.validator import validate_multiclass_oversampling


def main() -> None:
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_classes=3,
        n_informative=8,
        weights=[0.7, 0.2, 0.1],
        random_state=42,
    )

    oversampler = SMOTE(random_state=42)
    error_rates = validate_multiclass_oversampling(
        X=X,
        y=y,
        oversampler=oversampler,
        hidden_ratio=0.1,
        metric="hassanat",
    )
    print(error_rates)


if __name__ == "__main__":
    main()
