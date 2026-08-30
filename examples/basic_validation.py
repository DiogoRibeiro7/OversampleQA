"""
Minimal Validation
==================

The smallest useful example: validate SMOTE on a synthetic imbalanced dataset
and print repeated hidden-majority error-rate diagnostics.
"""

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.validator import validate_oversampling

X, y = make_classification(n_samples=500, weights=[0.9, 0.1], random_state=0)
details = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,
    oversampler=SMOTE(random_state=0),
    hidden_ratio=0.1,
    metric="hassanat",
    random_state=0,
    n_repeats=5,
    return_details=True,
)
print(f"Mean error rate: {details.mean:.3f}")
print(f"Repeat std: {details.std:.3f}")
print(f"Repeat interval: {details.interval}")
