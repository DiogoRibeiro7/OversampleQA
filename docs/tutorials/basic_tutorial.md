<div class="cell markdown">

# Basic Validation Tutorial

This tutorial introduces the core concepts of OversampleQA validation.

</div>

<div class="cell code">

``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN

from oversampleqa import validate_oversampling

# Set random seed for reproducibility
np.random.seed(42)
```

</div>

<div class="cell markdown">

## 1. Create an Imbalanced Dataset

First, let's create a synthetic imbalanced dataset to work with.

</div>

<div class="cell code">

``` python
# Create imbalanced classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    weights=[0.9, 0.1],  # 90% majority, 10% minority
    random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.1f}:1")
```

</div>

<div class="cell markdown">

## 2. Basic Validation

Now let's validate SMOTE oversampling on this dataset.

</div>

<div class="cell code">

``` python
# Validate SMOTE oversampling across repeated hold-out splits
details = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,
    oversampler=SMOTE(random_state=42),
    hidden_ratio=0.1,
    metric="hassanat",
    random_state=42,
    n_repeats=10,
    return_details=True,
)

print(f"SMOTE mean error rate: {details.mean:.3f}")
print(f"Repeat standard deviation: {details.std:.3f}")
print(f"Repeat interval: {details.interval}")
```

</div>

<div class="cell markdown">

## 3. Compare Multiple Methods

Let's compare different oversampling methods.

</div>

<div class="cell code">

``` python
# Test multiple oversampling methods
methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

results = {}

for name, oversampler in methods.items():
    details = validate_oversampling(
        X, y, minority_label=1,
        oversampler=oversampler,
        hidden_ratio=0.1,
        metric="hassanat",
        random_state=42,
        n_repeats=10,
        return_details=True,
    )
    results[name] = details.mean
    print(f"{name}: mean={details.mean:.3f}, std={details.std:.3f}")

# Find the best method
best_method = min(results, key=results.get)
print(f"\nBest method: {best_method} (error rate: {results[best_method]:.3f})")
```

</div>

<div class="cell markdown">

## 4. Understanding the Results

The validation error rate tells us how many synthetic samples are more
similar to hidden majority samples than to real minority samples. This
helps assess the realism of synthetic data.

</div>

<div class="cell code">

``` python
# Visualize results
methods_list = list(results.keys())
error_rates = list(results.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(methods_list, error_rates, color=['skyblue', 'lightcoral'])
plt.ylabel('Validation Error Rate')
plt.title('Oversampling Method Comparison')
plt.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Excellent threshold')
plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
plt.legend()

# Add value labels on bars
for bar, value in zip(bars, error_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Lower error rates indicate fewer majority-like synthetic samples within this dataset and protocol.")
```

</div>
