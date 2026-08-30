# User Guide

This guide explains the core validation idea, how to configure it, and
how to read results.

## Overview

OversampleQA evaluates oversampling methods by hiding a portion of the
majority class, generating synthetic minority samples with your
oversampler, and checking whether those synthetic samples are closer to
the hidden majority than to the real minority. If they are, that is
counted as an error.

## Key parameters

- `minority_label`: the label for the minority class in your dataset.
- `hidden_ratio`: the fraction of majority samples to hide for
  validation.
- `metric`: the distance metric used to compare samples.

## Choosing a distance metric

Different metrics emphasize different properties of the data. For
example:

- `hassanat` is scale-invariant and robust to feature magnitude.
- `euclidean` is standard for continuous data.
- `mahalanobis` accounts for feature covariance when you pass an inverse
  covariance matrix.

See [Distance Metrics](distances.md) for the full list and
[Decision Guide](decision_guide.md) for practical metric selection.

## Interpreting the error rate

The error rate is the fraction of synthetic samples that are closer to hidden
majority samples than to real minority samples. Lower is better only within the
same dataset, metric, preprocessing and `hidden_ratio`.

Do not use fixed universal thresholds. Calibrate the result with
`null_error_rate` or `oversampleqa validate --calibrate`, then compare samplers
under the same configuration. If repeated runs overlap heavily, the diagnostic
does not distinguish the samplers.

## Reproducibility

For consistent results across runs and machines:

- Fix random seeds in both dataset generation and oversamplers (e.g.,
  `random_state=42`).
- Keep `hidden_ratio` and the distance metric fixed when comparing
  methods.
- Pin dependency versions using `poetry.lock` and record Python, NumPy,
  scikit-learn, and imbalanced-learn versions.
- Store the exact configuration used for each run (CLI config or typed
  config object).
- When using the enhanced CLI, keep `--resume` enabled so repeated runs
  reuse cached results when available.

## Recommended workflow

1.  Start with `hidden_ratio=0.1` and `metric="hassanat"`.
2.  Compare multiple oversamplers with the same metric.
3.  Inspect plots (PCA or UMAP) to validate intuition.
4.  Repeat with an alternative metric to confirm robustness.

## See also

- [Benchmarking](benchmarking.md)
- [Decision Guide](decision_guide.md)
- [Limitations](limitations.md)
- [Plotting](plotting.md)
- [Metrics](metrics.md)
- [Production Audit Workflow](production_audit.md)
