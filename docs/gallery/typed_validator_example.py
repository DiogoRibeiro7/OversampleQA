"""Typed validator example using ValidationConfig."""

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from oversampleqa.types import ValidationConfig
from oversampleqa.typed_validator import TypedValidator


def main() -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        weights=[0.85, 0.15],
        random_state=7,
    )

    config = ValidationConfig(hidden_ratio=0.1, metric="hassanat", return_details=False)
    validator = TypedValidator()
    result = validator.validate(
        X=X,
        y=y,
        minority_label=1,
        oversampler=SMOTE(random_state=7),
        config=config,
    )
    print(result)


if __name__ == "__main__":
    main()
