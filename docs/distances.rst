Distance Metrics
================

OversampleQA ships with a rich set of distance metrics for comparing synthetic and real samples.

Available metrics
-----------------

* ``hassanat``: scale-invariant, robust to feature magnitude.
* ``euclidean``: L2 distance for continuous features.
* ``manhattan``: L1 distance for continuous features.
* ``cosine``: cosine distance, useful for directional similarity.
* ``minkowski``: generalized Lp distance (pass ``p``).
* ``chebyshev``: maximum absolute coordinate difference.
* ``mahalanobis``: covariance-aware distance (pass ``cov_inv``).
* ``canberra``: fractional difference emphasizing small magnitudes.
* ``hamming``: binary vector distance.
* ``jaccard``: set overlap distance.
* ``braycurtis``: compositional distance.
* ``correlation``: 1 - correlation coefficient.
* ``energy``: energy distance for distributions.
* ``wasserstein``: 1D Wasserstein distance.
* ``hellinger``: probability distribution distance.
* ``jensen_shannon``: smoothed divergence-based distance.

Passing metric-specific parameters
----------------------------------

Some metrics accept additional arguments via ``metric_kwargs``:

.. code-block:: python

   from oversampleqa import validate_oversampling
   from imblearn.over_sampling import SMOTE
   import numpy as np

   cov = np.cov(X.T)
   cov_inv = np.linalg.pinv(cov)

   rate = validate_oversampling(
       X, y,
       minority_label=1,
       oversampler=SMOTE(random_state=0),
       metric="mahalanobis",
       metric_kwargs={"cov_inv": cov_inv},
   )

See :doc:`algorithms` for mathematical definitions.
