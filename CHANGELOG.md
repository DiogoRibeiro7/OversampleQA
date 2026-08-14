# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut by hand: publish a GitHub release from a tag, which is also what
triggers the Zenodo archive and its DOI. Entries below the *Unreleased* section are
maintained manually.

## [Unreleased]

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

[Unreleased]: https://github.com/DiogoRibeiro7/OversampleQA/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.1.0
