Concepts
========

This page explains the core idea behind OversampleQA and how to interpret its outputs.

Hidden-majority validation
--------------------------

OversampleQA evaluates oversampling quality by hiding a portion of the majority class and comparing synthetic samples against two groups:

- The hidden majority samples (what the model should avoid imitating)
- The real minority samples (what the model should resemble)

Each synthetic sample is scored by its nearest-neighbor distance to both groups using a chosen metric. If a synthetic point is closer to the hidden majority than to the minority, it is counted as an error. The overall error rate is the fraction of synthetic samples that behave like majority examples.

Interpreting the error rate
---------------------------

The error rate is not a universal quality score. It depends on:

- The dataset geometry and class overlap
- The distance metric
- The hidden ratio
- The oversampler used

Lower values typically indicate that synthetic samples are more consistent with real minority data. When comparing methods, keep configuration fixed so results are comparable.

Binary vs multiclass
--------------------

For binary data, the error rate is computed directly from the hidden-majority comparison. For multiclass data, OversampleQA computes a confusion-style matrix where each row corresponds to synthetic samples of a class and each column corresponds to which hidden class those samples were closest to. Per-class error rates are derived from this matrix.

Choosing a metric
-----------------

Distance metrics control how “closeness” is measured. Hassanat is the default because it is robust to scaling and outliers, but Euclidean and Mahalanobis can be more appropriate for standardized continuous features. Use a metric that matches your feature types and normalization strategy.

What the error rate is compared against
---------------------------------------

Both sides of the comparison are held out by default. ``hidden_ratio`` of the
majority **and** of the minority are withheld before the oversampler is fitted,
and each synthetic point is scored against those two held-out sets. This is the
``reference="hidden_minority"`` estimand, and it is the same quantity
:func:`~oversampleqa.validate_multiclass_oversampling` measures, so binary and
multiclass results are comparable with each other.

The alternative, ``reference="train_minority"``, compares against the *whole*
minority class — the data the oversampler interpolated from. Held-out data on
one side and training data on the other biases the error rate toward zero by an
amount that depends on how densely packed the minority is, not on how good the
oversampler is. In the extreme, a sampler that merely duplicates real minority
points scores a perfect ``0.000``, because every copy sits at distance zero from
the minority set. That mode is retained only so pre-existing numbers can be
reproduced, and it warns when selected.

.. warning::

   The error rate is a **relative** quantity. Its scale depends on
   ``hidden_ratio``, on how dense the data is, and on dimensionality. A value of
   0.05 on one dataset and 0.05 on another do not mean the same thing, and the
   number has no absolute interpretation — there is no threshold above which an
   oversampler is "bad". Use it to compare oversamplers **on one dataset**, with
   every other parameter held fixed.

   Calibrating the error rate against a null model, which is what would make
   values comparable across datasets, is not yet implemented.

Minimum data requirements
-------------------------

Holding out a fraction of a small minority leaves very few points to compare
against, and a nearest-neighbour comparison against two or three points is close
to meaningless. ``validate_oversampling`` therefore raises when the held-out
minority would fall below ``min_hidden`` (default 5) rather than returning a
number that looks valid. Several of the small built-in benchmark datasets are
below this threshold at ``hidden_ratio=0.1``; :func:`~oversampleqa.run_benchmark`
records those as ``nan`` and reports the count in the ``n_missing`` column.

What this does not measure
--------------------------

OversampleQA measures similarity to hidden majority vs real minority samples. It does not replace full model evaluation, and it does not guarantee improved downstream performance. Use it as a diagnostic signal alongside standard validation.