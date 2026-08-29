# OversampleQA Documentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**OversampleQA** is a validation toolkit for oversampling methods in imbalanced classification.

## Quick Start

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa import validate_oversampling

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)

error_rate = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,
    oversampler=SMOTE(random_state=42),
)

print(f"Error rate: {error_rate:.3f}")
```

## Installation

```bash
pip install oversampleqa
```

For development:

```bash
git clone https://github.com/diogoribeiro7/OversampleQA.git
cd OversampleQA
poetry install
```

## Contents

- [Installation](installation.md)
- [Quick start](quickstart.md)
- [Concepts](concepts.md)
- [User guide](user_guide.md)
- [API overview](api_landing.md)
- [API reference](api_reference.md)
- [Tutorials](tutorials.md)
- [Examples](examples.md)
- [FAQ](faq.md)
