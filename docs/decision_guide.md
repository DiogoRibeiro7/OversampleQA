# Decision Guide

OversampleQA answers a narrow question: do synthetic minority samples look
closer to held-out majority data than to held-out minority data under the chosen
metric? Use the result as a diagnostic, not as a replacement for model
validation.

## Choose the Metric First

The metric defines what "closer" means. Pick it before comparing samplers, keep
it fixed within a comparison, and report it with every result.

| Data shape | Start with | Cross-check with | Main caution |
|---|---|---|---|
| Continuous features on different scales | `hassanat` | `euclidean` after scaling | Hassanat reduces feature-magnitude dominance but does not remove feature engineering problems. |
| Standardized continuous features | `euclidean` | `hassanat` | Large redundant feature blocks can dominate nearest-neighbor behavior. |
| Correlated numeric features | `mahalanobis` | `euclidean` | Pass a stable `cov_inv`; a poorly estimated covariance matrix makes the result unstable. |
| Sparse or directional vectors | `cosine` | `euclidean` on normalized vectors | Zero vectors and constant vectors need explicit handling. |
| Binary indicator vectors | `hamming` or `jaccard` | domain-specific metric | Dense numeric metrics usually overstate small bit flips. |
| Probability vectors | `hellinger` or `jensen_shannon` | `hassanat` on transformed features | Inputs must be non-negative and interpretable as distributions. |

`energy` and `wasserstein` are sample-based distribution distances, not point
metrics. They are available for specialized checks, but they answer a different
question from nearest-neighbor validation.

## Interpret by Calibration

Avoid universal cutoffs. An error rate of `0.10` can be excellent on a noisy,
overlapping dataset and worrying on a clean, separated one.

Read a validation run in this order:

1. Compare against `~oversampleqa.null_error_rate` or
   `oversampleqa validate --calibrate`. This shows what real held-out minority
   points score under the same protocol.
2. Compare samplers on the same dataset with the same `hidden_ratio`, metric,
   random seeds and preprocessing.
3. Repeat the run with `n_repeats` and report the spread, not just one split.
4. Cross-check the winning sampler with a second metric that matches the data.
5. Run fidelity diagnostics when copying, low diversity or boundary violations
   are plausible failure modes.

## When Results Are Weak

Treat a result as weak evidence when any of these are true:

- The held-out minority count is close to `min_hidden`.
- Classes overlap heavily or labels are noisy.
- Most features are redundant, unscaled, high-cardinality categoricals, or
  mixed numeric/categorical encodings without a domain metric.
- The sampler duplicates training samples. Use the memorisation ratio from the
  fidelity report before trusting the error rate.
- Two samplers differ by less than the repeat-to-repeat spread.
- A sampler wins under one metric and loses under another equally plausible
  metric.

## Report Enough to Reproduce

A result should include:

- OversampleQA version.
- Dataset identifier and preprocessing steps.
- Minority label and class counts.
- Oversampler class and parameters, including its `random_state`.
- `hidden_ratio`, `min_hidden`, metric and `metric_kwargs`.
- Validator `random_state`, `n_repeats`, and whether
  `reseed_oversampler=True` was used.
- Null calibration, validation error rate, and repeat interval when available.
- Fidelity diagnostics for duplicate-producing or boundary-sensitive samplers.

See [Reproducibility](reproducibility.md) for the full repeatability checklist.
