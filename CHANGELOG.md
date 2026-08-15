# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut by hand: publish a GitHub release from a tag, which is also what
triggers the Zenodo archive and its DOI. Entries below the *Unreleased* section are
maintained manually.

## [Unreleased]

### Changed

- **`validate_oversampling` now measures a different quantity by default.** It
  compared each synthetic point's distance to the held-out majority against its
  distance to the *full* minority class — the data the oversampler interpolated
  from. Held-out data on one side and training data on the other biased the
  error rate toward zero by an amount driven by minority density rather than
  oversampler quality, and it did not match what
  `validate_multiclass_oversampling` measures.

  The new default, `reference="hidden_minority"`, also holds out part of the
  minority class and compares against that, making binary and multiclass the
  same estimand. **Error rates from before this release are not comparable to
  those after it.** Pass `reference="train_minority"` to reproduce old numbers;
  it emits a `FutureWarning` naming the bias.

  On a representative dataset the same SMOTE run moves from 0.010 to 0.234.

- `return_details=True` now returns a frozen `ValidationDetails` dataclass
  instead of a 4-tuple. It carries `n_synthetic`, `n_ties`, `duplication_rate`
  and `reference` alongside the previous fields. This is a breaking change on a
  pre-1.0 package.
- Ties are no longer counted as errors. The comparison is now strict (`<`), and
  points exactly equidistant from both reference sets are reported separately as
  `n_ties`, with a warning when they exceed 1% of synthetic points.
- `calculate_error_rate` returns `nan` when the denominator is zero, and
  `validate_oversampling` returns `nan` when no synthetic samples were produced.
  `0.0` was indistinguishable from a perfect score. `compute_ranking` excludes
  `nan` explicitly and reports how many runs were dropped in a new `n_missing`
  column.
- `run_benchmark` records `nan` and warns for datasets whose minority is too
  small to support the estimand, rather than aborting the whole sweep. Five of
  the seven built-in datasets are below the threshold at `hidden_ratio=0.1`.
- `MemoryEfficientValidator` and `TypedValidator` now share the estimand with
  `validate_oversampling` through `prepare_validation_split` and
  `score_nearest_distances`, instead of each carrying its own copy of the logic.

### Added

- `reference`, `minority_hidden_ratio` and `min_hidden` parameters on
  `validate_oversampling`. `min_hidden` (default 5) raises when the held-out
  minority would be too small for a nearest-neighbour comparison to mean
  anything, instead of returning a plausible-looking number.
- `duplication_rate` in `oversampleqa.metrics`: the fraction of synthetic points
  that are exact copies of real ones. `validate_oversampling` warns above 0.5.
  `RandomOverSampler` scores 1.0 — it only duplicates, so its error rate says
  nothing about synthesis quality, yet it previously scored a perfect 0.000 and
  ranked first.
- `ValidationDetails` and `ReferenceSet` exported from the package root.
- A cross-validator equivalence test covering all three validators. Its absence
  is why they had drifted apart.

### Fixed

- `extract_synthetic_samples` verifies that the oversampler preserved the
  original rows as a prefix, instead of assuming it. `SMOTEENN` and
  `SMOTETomek` delete originals, and because they can still return more rows
  than they were given, the old length check did not catch it — the function
  scored a mix of surviving originals and synthetic points and returned a
  plausible but wrong number. It now raises.

- **`hassanat_distance` did not implement the Hassanat (2014) distance.** It
  computed `1 - min(|a|, |b|) / max(|a|, |b|)` summed over dimensions: absolute
  values, and no unit shift. That function is not a metric — it scored `[-5]`
  and `[5]` as distance zero, violating identity of indiscernibles — and it was
  discontinuous at the origin, giving `[0]` and `[1e-9]` the maximum
  per-dimension distance. Any zero-centred feature was therefore scored close to
  randomly near its mean. It is now the definition from Hassanat (2014), with
  every per-dimension contribution bounded to `[0, 1)`.

  **`hassanat` is the package default metric**, so error rates computed before
  this release are **not comparable** to those computed after it. This affects
  `distance_matrix`, `validate_oversampling`, and
  `validate_multiclass_oversampling` under default settings.

  Regenerated artefacts:

  - `docs/benchmark_results.rst` — the pinned reference run. The headline cell
    (SMOTE / `classification` / hassanat) moved from `0.059` to `0.005`.
  - `docs/_static/distance_histogram.png` and
    `docs/_static/multiclass_heatmap.png` — both plotted with `metric="hassanat"`.

### Added

- `hassanat` now has a vectorised kernel in
  `OptimizedDistanceMatrix._vectorized_dispatch`. It was previously absent, so
  the default metric always fell through to a Python double loop calling the
  metric once per pair.
- Distance-metric audit: every metric in the registry is now checked against
  SciPy or an independent closed form, not merely against itself. SciPy is a
  new **dev-only** dependency for this.

### Changed

- `energy` and `wasserstein` are documented as **sample-based** metrics rather
  than point metrics, in their docstrings and in `docs/distances.rst`. They
  treat the input vector as a set of observations, not as a point in feature
  space, so they answer a different question from the rest of the registry.
- The reference benchmark in `docs/benchmark_results.rst` now seeds the
  oversamplers. Without that, the "pinned" run was not reproducible: the same
  configuration produced 0.003894, 0.003894 and 0.003186 on three consecutive
  trials, because `StatisticalBenchmark(random_state=...)` seeds the
  cross-validation splits but not the oversampler's own sampling.

## [0.2.0] - 2026-08-14

### Added

- `CHANGELOG.md` seeded, maintained manually alongside tagged releases.
- Reproducibility guide in the docs covering randomness sources, dataset
  provenance, and cache keying/invalidation.
- Documented public-API error modes and edge-case behavior in the FAQ.
- Provenance and license metadata on every built-in benchmark dataset
  (`load_standard_datasets` now attaches a `provenance` dict with source,
  generator, params, URL, license, and notes).
- Statistical benchmarking is now surfaced in the CLI: `oversampleqa benchmark
  --statistical` runs cross-validated benchmarking and prints a summary table
  plus writes a CSV, a Markdown summary, and an HTML report (confidence
  intervals, pairwise p-values, effect sizes).
- `format_statistical_summary` renders a statistical benchmark frame as Markdown
  and is exported from the package.
- Benchmark-results docs page with a reproducible reference run (pinned seed,
  configuration, and environment) and guidance on interpreting the output.
- Optional performance profiling script (`scripts/profile_performance.py`, `make
  profile`) that times the distance-matrix and validator hot paths and can save
  or check a JSON baseline to catch regressions.

### Changed

- `validate_oversampling` and `validate_multiclass_oversampling` now validate
  `hidden_ratio` up front and raise a clear `ValueError` when it is outside the
  open interval `(0, 1)`, instead of leaking a downstream scikit-learn error.

### Fixed

- Corrected the typed-validator and configuration examples in the docs: the
  `TypedValidator` example now uses the real `validate(...)` API, and
  `ValidationConfig` no longer shows a non-existent `minority_label` field.
- Documented the `setup` command and the global CLI options.

## [0.1.0] - 2025-10-10

Initial release.

### Added

- Hidden-majority validation of oversampling quality for binary and multiclass
  workflows, including per-class confusion-style error matrices.
- Memory-efficient validator with batched/streaming computation and a typed
  validator with Pydantic configuration and async support.
- 16 distance metrics across geometric, statistical, and probability families,
  with a registry and optimized, memory-aware distance-matrix computation.
- Diagnostic metrics beyond error rate: confidence ratio, local density
  divergence, minority recall loss, UMAP manifold distance, fairness checks,
  and noise sensitivity.
- Benchmarking utilities with dataset loaders, ranking, and export helpers,
  plus statistical benchmarking (k-fold CV, confidence intervals, effect sizes,
  p-values) and disk caching of results and distance matrices.
- Surrogate-model evaluation and cluster-overlap diagnostics.
- Report generation (Markdown/HTML) and a plotting suite.
- Rich CLI with profiles, templates, shell completion, and a `doctor`
  diagnostic, alongside a minimal legacy CLI.
- Plugin system for custom metrics and validators.
- Sphinx documentation, examples gallery, and tutorials.

[Unreleased]: https://github.com/DiogoRibeiro7/OversampleQA/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.1.0
