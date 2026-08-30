# Quick Start Guide

This guide will get you started with OversampleQA in just a few minutes.

## Installation

Install the latest release from PyPI:

``` bash
pip install oversampleqa
```

For development, install it from source:

``` bash
git clone https://github.com/diogoribeiro7/OversampleQA.git
cd OversampleQA
poetry install
```

## Basic Usage

Here's a simple example of validating SMOTE oversampling. It fixes both the
dataset seed and the validation seed, then repeats the hold-out split so the
reported value is a small distribution rather than one lucky split:

``` python
from oversampleqa import validate_oversampling
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import numpy as np

# Create an imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.9, 0.1],  # 90% majority, 10% minority
    random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Validate SMOTE oversampling across repeated hold-out splits
details = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,  # Label of minority class
    oversampler=SMOTE(random_state=42),
    hidden_ratio=0.1,  # Hide 10% of majority samples
    metric="hassanat",
    random_state=42,
    n_repeats=10,
    return_details=True,
)

print(f"Mean validation error rate: {details.mean:.3f}")
print(f"Repeat standard deviation: {details.std:.3f}")
print(f"Repeat interval: {details.interval}")
```

## Concepts

OversampleQA validates synthetic samples by hiding a portion of the
majority class and comparing each generated sample to both hidden
majority and real minority examples. A synthetic point is counted as an
error when it is closer to the hidden majority than to the minority. The
resulting error rate is a signal of how often oversampling produces
majority-like artifacts.

## Reproducibility notes

- Fix random seeds in dataset generation, the validator, and oversamplers
  (e.g., `random_state=42` on both `SMOTE` and `validate_oversampling`).
- Keep `hidden_ratio` and `metric` fixed when comparing methods.
- Use `n_repeats` and report the mean plus spread, not a single split.
- Store the exact config used for each run. Benchmark reports also write an
  adjacent `*.metadata.json` sidecar with package, runtime, and data-shape
  metadata.

## Interpreting Results

The error rate tells you what fraction of synthetic samples are more similar
to hidden majority samples than to real minority samples. Interpret it within
one dataset, metric, `hidden_ratio`, reference mode, and repeat protocol. Do
not compare the raw scale across unrelated datasets; use the benchmark and
decision-guide pages for multi-dataset comparisons.

## Next Steps

- Read the [concepts](concepts.md) page for a deeper explanation
- Read the [user guide](user_guide.md) for detailed concepts
- Try the [tutorials](tutorials.md) for step-by-step examples
- Check the [API reference](api_reference.md) for all functions
