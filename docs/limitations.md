# Limitations

OversampleQA is designed to make oversampling failures visible. It does not
prove that an oversampler is useful for every downstream model, metric or
deployment setting.

## Diagnostic, Not Utility

The validation error rate measures whether synthetic points resemble held-out
majority data more than held-out minority data. A low error rate does not
guarantee better classifier precision, recall, calibration, fairness or
robustness. Always pair it with task-level model validation.

## Metric Dependence

Nearest-neighbor validation inherits the assumptions of the distance metric.
Changing the metric can change the conclusion, especially with mixed feature
types, high dimensionality, sparse vectors or correlated features. If two
reasonable metrics disagree, report the disagreement instead of selecting the
more convenient result.

## Small Minority Classes

Holding out a fraction of a small minority can leave too few reference points.
`validate_oversampling` raises when the held-out minority would fall below
`min_hidden` because returning a number would be misleading. For tiny minority
classes, prefer descriptive data audit work before ranking samplers.

## Class Overlap and Label Noise

When the true minority and majority distributions overlap, some real minority
points are also close to the majority. In that setting, a nonzero validation
error can reflect the dataset rather than a defective sampler. Use null
calibration to compare synthetic samples against real held-out minority points
under the same protocol.

## Duplicate-Producing Samplers

Samplers such as `RandomOverSampler` can copy real minority points. Copies sit
at distance zero from the training minority and can make nearest-neighbor
diagnostics look better than the synthetic diversity deserves. Use fidelity
diagnostics, especially memorisation ratio, before treating the error rate as a
quality score.

## High-Dimensional Data

Nearest-neighbor distances concentrate in high dimensions. The package warns in
some manifold diagnostics, but no warning can infer whether the representation
is meaningful for your domain. Reduce dimension, remove redundant features, or
use a domain-specific representation before interpreting small differences.

## Preprocessing Leakage

Fit preprocessing inside the same train/validation structure used by the
oversampler. Scaling, encoding, imputation, feature selection or dimensionality
reduction fitted on all rows can leak information from held-out points and make
the validation result too optimistic.

## Multiclass Attribution

Multiclass validation reports which hidden class synthetic points most resemble.
That is useful for identifying boundary confusion, but class-specific rates are
only as strong as the number and representativeness of held-out examples for
each class.

## Performance Measurements

The Performance workflow publishes timing and memory artifacts for trend
inspection, not merge gates. Shared CI runners and developer laptops are noisy
measurement environments. Compare artifacts only when the recorded environment
metadata is similar.

## Release Compatibility

Before 1.0, minor releases may change numerical behavior when the previous
behavior was wrong. Public API changes are guarded and documented, but old and
new scientific results are not automatically comparable. Read the changelog
before mixing results across package versions.
