import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

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
        # nan is a legitimate value: the sampler generates nothing for the
        # largest class, and that class is unmeasured rather than perfect.
        assert np.isnan(rate) or 0.0 <= rate <= 1.0


def test_validate_multiclass_matrix():
    # 200 samples left class 2 with 2 held-out points, below min_hidden. The
    # fixture is grown rather than the guard lowered: a nearest-neighbour
    # comparison against 2 points is exactly what the guard exists to refuse.
    X, y = make_classification(
        n_samples=800,
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
