Distance Metrics
================

OversampleQA ships with a rich set of distance metrics for comparing synthetic and real samples.

Available metrics
-----------------

* ``hassanat``: scale-invariant, robust to feature magnitude. Each dimension
  contributes a value in ``[0, 1)``, so no single feature can dominate the sum.
  This is the package default.
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

Point metrics versus sample-based metrics
-----------------------------------------

.. warning::

   ``energy`` and ``wasserstein`` are **sample-based**, not point metrics. They
   are reachable through the same registry as the others, but they do not
   measure the same kind of quantity.

   Every other metric in the list treats its input as a single **point** in
   feature space and measures how far apart two points are. ``energy`` and
   ``wasserstein`` instead treat the input vector as a **set of scalar
   observations** drawn from a distribution, and measure how far apart two
   *distributions* are.

   The practical consequence: they are invariant to permutation of the input
   vector, whereas a point metric is not. ``euclidean([1, 2], [3, 4])``
   changes if you reorder the components; ``wasserstein`` does not. Choosing
   one of these as the validation metric therefore answers a different
   question from the one the rest of the package is asking.

``hellinger`` and ``jensen_shannon`` require non-negative inputs that can be
normalised to a probability vector. They raise ``ValueError`` on negative
input rather than silently taking absolute values, which would turn an invalid
call into a plausible-looking number.

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
