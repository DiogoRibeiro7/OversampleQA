# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated via [python-semantic-release](https://python-semantic-release.readthedocs.io/);
entries below the *Unreleased* section are generated from conventional commit messages.

## [Unreleased]

### Added

- `CHANGELOG.md` seeded and wired to the existing `python-semantic-release` configuration.
- Reproducibility guide in the docs covering randomness sources, dataset
  provenance, and cache keying/invalidation.

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
