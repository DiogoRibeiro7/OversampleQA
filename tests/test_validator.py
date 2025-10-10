from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from oversampleqa.validator import (
    validate_oversampling,
    extract_synthetic_samples,
)


def test_validate_oversampling_runs():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    rate = validate_oversampling(
        X=X,
        y=y,
        minority_label=1,
        oversampler=SMOTE(random_state=42),
        hidden_ratio=0.2,
    )
    assert 0.0 <= rate <= 1.0


def test_validate_return_details():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        weights=[0.8, 0.2],
        random_state=0,
    )
    result = validate_oversampling(
        X=X,
        y=y,
        minority_label=1,
        oversampler=SMOTE(random_state=0),
        hidden_ratio=0.1,
        return_details=True,
    )
    rate, errors, dist_hidden, dist_min = result
    assert 0.0 <= rate <= 1.0
    assert isinstance(errors, int)
    assert dist_hidden.shape[0] == dist_min.shape[0]
    assert dist_hidden.shape[0] >= errors


def test_extract_synthetic_samples():
    X, y = make_classification(n_samples=100, weights=[0.8, 0.2], random_state=0)
    oversampler = SMOTE(random_state=0)
    X_res, y_res = oversampler.fit_resample(X, y)
    synth = extract_synthetic_samples(X, X_res, y_res, minority_label=1)
    assert len(synth) == len(X_res) - len(X)
