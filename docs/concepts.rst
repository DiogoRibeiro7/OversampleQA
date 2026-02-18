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

What this does not measure
--------------------------

OversampleQA measures similarity to hidden majority vs real minority samples. It does not replace full model evaluation, and it does not guarantee improved downstream performance. Use it as a diagnostic signal alongside standard validation.