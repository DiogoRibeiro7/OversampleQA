Advanced Tutorial
=================

This tutorial walks through comparing multiple oversamplers and validating robustness.

1. Create a dataset
------------------

.. code-block:: python

   from sklearn.datasets import make_classification

   X, y = make_classification(
       n_samples=1200,
       n_features=15,
       weights=[0.9, 0.1],
       random_state=0,
   )

2. Compare oversamplers
----------------------

.. code-block:: python

   from imblearn.over_sampling import SMOTE, ADASYN
   from oversampleqa.validator import validate_oversampling

   oversamplers = {
       "SMOTE": SMOTE(random_state=0),
       "ADASYN": ADASYN(random_state=0),
   }

   for name, sampler in oversamplers.items():
       rate = validate_oversampling(X, y, minority_label=1, oversampler=sampler)
       print(name, rate)

3. Validate across metrics
-------------------------

.. code-block:: python

   metrics = ["hassanat", "euclidean", "cosine"]
   for metric in metrics:
       rate = validate_oversampling(
           X, y, minority_label=1, oversampler=SMOTE(random_state=0), metric=metric
       )
       print(metric, rate)
