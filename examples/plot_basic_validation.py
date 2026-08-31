"""
Basic Validation Example
========================

This example demonstrates the basic usage of OversampleQA
to validate SMOTE oversampling on a synthetic dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa import validate_oversampling

print(__doc__)

# Create an imbalanced dataset
X, y = make_classification(
    n_samples=500,
    n_features=2,  # 2D for easy visualization
    n_informative=2,
    n_redundant=0,
    weights=[0.9, 0.1],
    random_state=42,
)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Validate SMOTE oversampling
error_rate = validate_oversampling(
    X=X, y=y, minority_label=1, oversampler=SMOTE(random_state=42), hidden_ratio=0.1
)

print(f"Validation error rate: {error_rate:.3f}")

# Interpret results
if error_rate < 0.1:
    print("🟢 Result: Excellent - SMOTE generates realistic synthetic data")
elif error_rate < 0.3:
    print("🟡 Result: Moderate - Use SMOTE with caution")
else:
    print("🔴 Result: High risk - Consider alternative oversampling methods")

# Generate oversampled data for visualization
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Extract different sample types
majority = X[y == 0]
minority = X[y == 1]
synthetic = X_res[len(X) :]  # Synthetic samples are appended

# Create visualization
plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(majority[:, 0], majority[:, 1], c="blue", alpha=0.6, label="Majority")
plt.scatter(minority[:, 0], minority[:, 1], c="red", alpha=0.8, label="Minority")
plt.title("Original Imbalanced Data")
plt.legend()
plt.grid(True, alpha=0.3)

# After oversampling
plt.subplot(1, 2, 2)
plt.scatter(majority[:, 0], majority[:, 1], c="blue", alpha=0.6, label="Majority")
plt.scatter(minority[:, 0], minority[:, 1], c="red", alpha=0.8, label="Minority")
plt.scatter(
    synthetic[:, 0],
    synthetic[:, 1],
    c="orange",
    alpha=0.7,
    marker="^",
    label="Synthetic",
)
plt.title(f"After SMOTE (Error Rate: {error_rate:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nGenerated {len(synthetic)} synthetic samples")
print("Lower error rates indicate more realistic synthetic data")
