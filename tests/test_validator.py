import pytest
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.validator import (
    extract_synthetic_samples,
    validate_multiclass_oversampling,
    validate_oversampling,
)


def test_validate_oversampling_runs():
    X, y = make_classification(
        n_samples=600,
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


def test_validate_oversampling_minority_label_zero():
    X, y = make_classification(
        n_samples=600,
        n_features=5,
        weights=[0.1, 0.9],
        random_state=42,
    )
    rate = validate_oversampling(
        X=X,
        y=y,
        minority_label=0,
        oversampler=SMOTE(random_state=42),
        hidden_ratio=0.2,
    )
    assert 0.0 <= rate <= 1.0


def test_validate_return_details():
    X, y = make_classification(
        n_samples=600,
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
    assert 0.0 <= result.error_rate <= 1.0
    assert isinstance(result.n_errors, int)
    assert result.dist_hidden.shape[0] == result.dist_min.shape[0]
    assert result.dist_hidden.shape[0] >= result.n_errors
    assert result.n_synthetic == result.dist_hidden.shape[0]
    assert result.reference == "hidden_minority"


def test_extract_synthetic_samples():
    X, y = make_classification(n_samples=600, weights=[0.8, 0.2], random_state=0)
    oversampler = SMOTE(random_state=0)
    X_res, y_res = oversampler.fit_resample(X, y)
    synth = extract_synthetic_samples(X, X_res, y_res, minority_label=1)
    assert len(synth) == len(X_res) - len(X)


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5])
def test_validate_oversampling_rejects_out_of_range_hidden_ratio(bad_ratio):
    X, y = make_classification(
        n_samples=600, n_features=4, weights=[0.9, 0.1], random_state=0
    )
    with pytest.raises(ValueError, match="hidden_ratio must be in the open interval"):
        validate_oversampling(
            X=X,
            y=y,
            minority_label=1,
            oversampler=SMOTE(random_state=0),
            hidden_ratio=bad_ratio,
        )


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1])
def test_validate_multiclass_rejects_out_of_range_hidden_ratio(bad_ratio):
    X, y = make_classification(
        n_samples=600,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    with pytest.raises(ValueError, match="hidden_ratio must be in the open interval"):
        validate_multiclass_oversampling(
            X=X,
            y=y,
            oversampler=SMOTE(random_state=0),
            hidden_ratio=bad_ratio,
        )
