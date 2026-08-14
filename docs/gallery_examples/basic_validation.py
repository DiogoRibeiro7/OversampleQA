"""
Minimal Validation
==================

The smallest useful example: validate SMOTE on a synthetic imbalanced dataset
and print the hidden-majority error rate.
"""

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.validator import validate_oversampling


X, y = make_classification(n_samples=500, weights=[0.9, 0.1], random_state=0)
error = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,
    oversampler=SMOTE(random_state=0),
    hidden_ratio=0.1,
)
print(f"Error rate: {error:.3f}")
