Quick Start Guide
=================

This guide will get you started with OversampleQA in just a few minutes.

Installation
------------

OversampleQA is not published on PyPI. Install it from source:

.. code-block:: bash

   git clone https://github.com/diogoribeiro7/OversampleQA.git
   cd OversampleQA
   poetry install

Basic Usage
-----------

Here's a simple example of validating SMOTE oversampling:

.. code-block:: python

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

   # Validate SMOTE oversampling
   error_rate = validate_oversampling(
       X=X,
       y=y,
       minority_label=1,  # Label of minority class
       oversampler=SMOTE(random_state=42),
       hidden_ratio=0.1  # Hide 10% of majority samples
   )

   print(f"Validation error rate: {error_rate:.3f}")

Concepts
--------

OversampleQA validates synthetic samples by hiding a portion of the majority class and comparing each generated sample to both hidden majority and real minority examples. A synthetic point is counted as an error when it is closer to the hidden majority than to the minority. The resulting error rate is a signal of how often oversampling produces majority-like artifacts.

Reproducibility notes
---------------------

- Fix random seeds in dataset generation and oversamplers (e.g., ``random_state=42``).
- Keep ``hidden_ratio`` and ``metric`` fixed when comparing methods.
- Record versions for Python, NumPy, scikit-learn, and imbalanced-learn.
- Store the exact config used for each run.

Interpreting Results
--------------------

The error rate tells you what fraction of synthetic samples are more similar to hidden majority samples than to real minority samples:

* **< 0.1**: Excellent - Low risk, synthetic data looks realistic
* **0.1 - 0.3**: Moderate - Use with caution
* **> 0.3**: High risk - Consider alternative methods

Next Steps
----------

* Read the :doc:`concepts` page for a deeper explanation
* Read the :doc:`user_guide` for detailed concepts
* Try the :doc:`tutorials` for step-by-step examples
* Check the :doc:`api_reference` for all functions