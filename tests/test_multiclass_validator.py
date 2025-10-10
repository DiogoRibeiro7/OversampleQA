from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from oversampleqa.validator import validate_multiclass_oversampling


def test_validate_multiclass_runs():
    X, y = make_classification(
        n_samples=300,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1],
        random_state=0,
    )
    result = validate_multiclass_oversampling(
        X,
        y,
        oversampler=SMOTE(random_state=0),
        hidden_ratio=0.2,
    )
    assert set(result.keys()) == {0, 1, 2}
    for rate in result.values():
        assert 0.0 <= rate <= 1.0


def test_validate_multiclass_matrix():
    X, y = make_classification(
        n_samples=200,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        random_state=1,
    )
    rates, matrix = validate_multiclass_oversampling(
        X,
        y,
        oversampler=SMOTE(random_state=1),
        hidden_ratio=0.1,
        return_matrix=True,
    )
    assert matrix.shape == (3, 3)
    assert set(rates.keys()) == {0, 1, 2}
