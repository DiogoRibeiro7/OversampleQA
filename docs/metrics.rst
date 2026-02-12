Metrics
=======

OversampleQA includes diagnostics beyond the base error rate.

Core metrics
------------

* ``calculate_error_rate``: fraction of synthetic samples that are closer to hidden majority than minority.
* ``confidence_ratio``: ratio of distances to minority vs majority.
* ``local_density_divergence``: compares local density around synthetic vs real data.
* ``minority_recall_loss``: 1 - recall for the minority class.
* ``umap_manifold_distance``: Wasserstein distance in UMAP space.
* ``check_model_fairness``: absolute gap in minority recall across protected groups.
* ``noise_sensitivity_diagnostic``: error rate across label noise levels.

Examples
--------

Noise sensitivity:

.. code-block:: python

   from oversampleqa.metrics import noise_sensitivity_diagnostic
   from imblearn.over_sampling import SMOTE

   df = noise_sensitivity_diagnostic(
       X, y,
       minority_label=1,
       oversampler=SMOTE(random_state=0),
       noise_levels=[0.0, 0.1, 0.2],
       hidden_ratio=0.1,
   )
   print(df)

UMAP manifold distance:

.. code-block:: python

   from oversampleqa.metrics import umap_manifold_distance

   d = umap_manifold_distance(real=X_minority, synthetic=X_syn, random_state=0)
   print(d)
