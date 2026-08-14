FAQ
===

What does the error rate mean?
------------------------------

It is the fraction of synthetic minority samples that are closer to hidden majority samples than to real
minority samples. Lower values generally indicate better synthetic data quality.

How do I choose the minority label?
-----------------------------------

Inspect your labels (for example with ``np.unique(y, return_counts=True)``) and set ``minority_label`` to the
class with fewer samples.

What if my dataset is multiclass?
---------------------------------

Use :func:`oversampleqa.validate_multiclass_oversampling` or the typed validator with a multiclass-aware
oversampler.

Why are there multiple distance metrics?
----------------------------------------

Different metrics capture different aspects of similarity. It is good practice to validate with at least two
metrics to confirm stability.

Error modes and edge cases
--------------------------

The following table documents how :func:`oversampleqa.validate_oversampling` and
:func:`oversampleqa.validate_multiclass_oversampling` behave at the boundaries.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Situation
     - Behavior
   * - ``minority_label`` not present in ``y``
     - Raises ``ValueError`` ("minority_label ... not found in y").
   * - ``hidden_ratio`` outside the open interval ``(0, 1)``
     - Raises ``ValueError`` up front before any work is done.
   * - Unknown ``metric`` name
     - Raises ``ValueError`` ("Unsupported metric ...").
   * - Binary helper called on multiclass data
     - Raises ``ValueError``; use :func:`oversampleqa.validate_multiclass_oversampling`.
   * - Too few minority samples for the oversampler (for example fewer than
       ``k_neighbors + 1`` for SMOTE)
     - The oversampler raises its own ``ValueError``; OversampleQA logs and
       re-raises it unchanged. Reduce the oversampler's ``k_neighbors`` or
       supply more minority samples.
   * - Single feature (one column)
     - Supported; validation runs normally.
   * - All-identical (degenerate) points
     - Runs without error; the error rate tends toward ``1.0`` because synthetic
       points are indistinguishable from the hidden majority. Treat this as a
       signal that the data carries no usable structure rather than a quality
       verdict.
   * - Mahalanobis metric with singular covariance
     - Handled gracefully (a pseudo-inverse is used); no exception is raised.
   * - Oversampler produces no synthetic samples
     - Returns an error rate of ``0.0`` (and empty detail arrays when
       ``return_details=True``).

Repro checklist
---------------

- Set ``random_state`` for dataset generation and oversamplers.
- Keep ``hidden_ratio`` and ``metric`` fixed for comparisons.
- Record dependency versions (Python, NumPy, scikit-learn, imbalanced-learn).
- Save the exact CLI or typed config used for each run.
- Use ``poetry.lock`` to pin versions when sharing results.