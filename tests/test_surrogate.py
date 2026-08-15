from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from oversampleqa.surrogate import evaluate_surrogate_models


def test_evaluate_surrogate_models_returns_metrics():
    X, y = make_classification(n_samples=100, weights=[0.8, 0.2], random_state=0)
    results = evaluate_surrogate_models(
        X,
        y,
        minority_label=1,
        oversampler=SMOTE(random_state=0),
        model=LogisticRegression(max_iter=1000),
        test_size=0.2,
        random_state=0,
    )
    assert set(results.keys()) == {"real_only", "real_plus_synth", "synth_only"}
    for metrics in results.values():
        assert all(key in metrics for key in ("f1", "recall", "precision"))
