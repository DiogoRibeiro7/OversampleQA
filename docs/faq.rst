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

Repro checklist
---------------

- Set ``random_state`` for dataset generation and oversamplers.
- Keep ``hidden_ratio`` and ``metric`` fixed for comparisons.
- Record dependency versions (Python, NumPy, scikit-learn, imbalanced-learn).
- Save the exact CLI or typed config used for each run.
- Use ``poetry.lock`` to pin versions when sharing results.