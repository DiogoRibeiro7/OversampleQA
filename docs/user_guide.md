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

See `distances` for a full list.

## Interpreting the error rate

The error rate is the fraction of synthetic samples that are closer to
hidden majority samples than to real minority samples. Lower is better.

- **\< 0.1**: excellent
- **0.1 - 0.3**: moderate
- **\> 0.3**: high risk

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
- [Plotting](plotting.md)
- [Metrics](metrics.md)
