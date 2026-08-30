# Decision Guide

OversampleQA answers one narrow question well: under a chosen distance metric,
do synthetic minority samples look closer to held-out majority data than to
held-out minority data? Use that signal as one diagnostic in an oversampling
decision, not as a replacement for model validation.

## Pick the Decision

Start from the decision you need to make, then choose the smallest diagnostic
stack that can support it.

| Decision | Minimum evidence | Use when | Do not conclude |
|---|---|---|---|
| Is one sampler obviously unsafe on this dataset? | Repeated `validate_oversampling`, calibrated by `null_error_rate` | You need a fast screen before model work | A low error rate means the sampler improves the model |
| Which of several samplers is safer on one dataset? | Same validation protocol for every sampler, plus `fidelity_report` | You are choosing between SMOTE variants or a duplicate baseline | Small mean differences matter when repeat intervals overlap |
| Are synthetic points distributionally close to real minority points? | `nn_two_sample_test`, `mst_two_sample_test`, or `cross_match_test` | You need a statistical check of sample similarity | A high p-value proves equality |
| Does oversampling add useful information? | `downstream_utility` or a leakage-safe model evaluation pipeline | The final decision depends on model performance | Geometry diagnostics alone are enough |
| Which sampler is best across datasets? | `StatisticalBenchmark` rankings and `fold_results()` skip accounting | You are writing a benchmark, paper, or package comparison | Raw error rates can be pooled across datasets |
| Can this run be audited later? | JSON/CSV/Markdown/HTML exports plus `*.metadata.json` sidecars | Results will be cited, reviewed, or used operationally | A screenshot or copied table is reproducible |

## Recommended Workflow

For exploratory work:

1. Choose a metric that matches the feature representation.
2. Run `validate_oversampling` with seeded sampler and validator randomness.
3. Set `n_repeats` above 1 and read the mean next to the repeat spread.
4. Compare only within the same dataset, preprocessing, `hidden_ratio`, metric,
   reference mode and seed protocol.

For a production or research decision:

1. Add `null_error_rate` or `oversampleqa validate --calibrate` so the observed
   rate is judged against real held-out minority points.
2. Run `fidelity_report` for the leading sampler and for any sampler that can
   duplicate training rows.
3. Check downstream utility with resampling inside each cross-validation fold.
4. Export the result tables and keep the adjacent `*.metadata.json` sidecars.
5. Write a short decision note: accepted, rejected or deferred, with the reason.

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
metrics. They are useful for specialized checks, but they answer a different
question from nearest-neighbor validation.

## Read the Signals Together

Avoid universal cutoffs. An error rate of `0.10` can be excellent on a noisy,
overlapping dataset and worrying on a clean, separated one.

| Signal | Strong reading | Weak or misleading reading |
|---|---|---|
| Validation error rate | Lower than alternatives under the same protocol and close to the calibrated null | Read as an absolute pass/fail threshold |
| Calibrated error rate | Observed rate is near the real-minority null and far from the majority ceiling | Calibration used different metric, split, or `hidden_ratio` |
| Repeat interval | Winner is separated by more than repeat-to-repeat variation | Intervals overlap heavily or only one split was run |
| Two-sample p-value | Low p-value flags a detectable distribution shift | High p-value treated as proof of equality, especially in high dimension |
| Fidelity metrics | Density/coverage are stable across `k`, memorisation is not near zero, boundary violations are low | Precision alone is high, or copying produces a deceptively good error rate |
| Downstream utility | Leakage-safe CV improves the task metric that matters | Accuracy improves on imbalanced data, or resampling happened before splitting |
| Benchmark ranking | Mean ranks are computed within comparable dataset/metric/ratio blocks and skips are counted | Raw means are averaged across unrelated datasets |

## Accept, Reject or Defer

Accept a sampler when the validation rate is close to the calibrated minority
reference, repeat variability is small enough for the decision, fidelity does
not show copying or boundary failures, and downstream evaluation improves the
application metric.

Reject a sampler when it looks majority-like, copies the training minority,
generates into boundary regions, deletes or reorders originals in an unsupported
way, or worsens the downstream model.

Defer the decision when evidence conflicts: the winner changes under an equally
valid metric, repeat intervals overlap, minority hold-outs are too small, most
folds skip, or high-dimensional diagnostics have too little power to separate
the candidates.

## When Results Are Weak

Treat a result as weak evidence when any of these are true:

- The held-out minority count is close to `min_hidden`.
- Classes overlap heavily or labels are noisy.
- Features are redundant, unscaled, high-cardinality categoricals, or mixed
  numeric/categorical encodings without a domain metric.
- The sampler duplicates training samples. Use the memorisation ratio from
  `fidelity_report` before trusting the error rate.
- Two samplers differ by less than the repeat-to-repeat spread.
- A sampler wins under one metric and loses under another equally plausible
  metric.
- Benchmark `fold_results()` shows skipped folds or unequal method coverage.

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
- Two-sample test name, sample counts, p-value and whether parent blocking was
  used.
- Fidelity diagnostics for duplicate-producing or boundary-sensitive samplers.
- Downstream model metric, splitter, scorer and pipeline construction.
- Benchmark rank, comparable block definition and skip counts when reporting
  multi-dataset comparisons.
- Export sidecar path and schema version for archived artifacts.

See [Reproducibility](reproducibility.md) for the full repeatability checklist
and [Production Audit Workflow](production_audit.md) for a concrete audit trail.
