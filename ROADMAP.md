# OversampleQA Roadmap

This roadmap summarizes the current scope, maturity, and next steps for the project.

## Project Goals

- Provide a practical, diagnostics-first toolkit for evaluating oversampling methods in imbalanced classification.
- Make synthetic sample quality measurable and comparable across methods, datasets, and metrics.
- Keep workflows reproducible with clear configuration, benchmarking, and exportable reports.
- Offer a fast path for practitioners and a deeper path for research use cases.
- Stay compatible with scikit-learn and imbalanced-learn conventions.

## Current State (v0.1.0)

Validation

- Hidden-majority error rate for binary and multiclass workflows (`validator.py`), with per-class confusion-style error matrices.
- Memory-efficient validator with batched / streaming computation (`memory_efficient_validator.py`).
- Typed validator with Pydantic config, async support, and session management (`typed_validator.py`).

Metrics and distances

- 16 distance metrics across geometric, statistical, and probability families (`distance.py`, `extended_distances.py`), with a registry and optimized, memory-aware matrix computation (`optimized_distance.py`).
- Diagnostic metrics beyond error rate: confidence ratio, local density divergence, minority recall loss, UMAP manifold distance, fairness checks, and noise sensitivity (`metrics.py`).

Benchmarking and reporting

- Benchmark runner with dataset loaders, ranking, and export helpers (`benchmark.py`).
- Statistical benchmarking with k-fold CV, confidence intervals, effect sizes, and p-values (`advanced_benchmark.py`).
- Disk caching of validation results and distance matrices (`caching.py`).
- Surrogate-model evaluation (real-only / real+synthetic / synthetic-only) and cluster-overlap diagnostics (`surrogate.py`, `clustering.py`).
- Report generation (Markdown/HTML) and a plotting suite (`report.py`, `plotting.py`).

Tooling and extensibility

- Rich CLI with profiles, templates, shell completion, and a `doctor` diagnostic (`cli_enhanced.py`), plus a minimal legacy CLI (`cli.py`).
- Plugin system for custom metrics and validators (`plugin_system.py`).
- 26 test files covering validators, metrics, distances, benchmarking, CLI, plotting, performance, and memory.
- Sphinx docs (22 sources, gallery), examples, and tutorials in-repo.
- CI on Python 3.10–3.12 with coverage upload, docs built with `-W`, weekly security scan, and a semantic-release pipeline (PyPI upload currently gated off).

## Milestones

Owner for all milestones: `diogoribeiro7`.

### v0.2.0 — May 15, 2026

Focus: release plumbing and reproducibility — close the gap between the configured tooling and an actual published release.

Scope:

- Add a `CHANGELOG.md` and wire it to the existing `python-semantic-release` config (currently `changelog_file` points at a file that does not exist).
- Enable and verify automated PyPI publishing: set `upload_to_pypi`/`PYPI_TOKEN` flow in `release.yml`, dry-run against TestPyPI first.
- Document reproducibility guidance (seeds, dataset provenance, cache invalidation) in `docs/`.
- Audit and document expected error modes / edge-case behavior for the public API (empty minority class, single feature, all-identical points, degenerate covariance for Mahalanobis).
- Tidy docs drift: confirm `concepts.rst`, `cli.rst`, and `configuration.rst` match the shipped CLI surface and config schema.

Definition of Done:

- `make lint typecheck` and `make coverage` pass locally; coverage stays above its current threshold.
- `make docs` builds cleanly with `-W` (warnings as errors).
- A tagged `v0.2.0` release produces artifacts and a `CHANGELOG.md` entry via CI.
- TestPyPI upload verified end-to-end.

### v0.3.0 — August 15, 2026

Focus: benchmarking depth and reporting confidence.

Scope:

- Expand the built-in dataset catalog with explicit licensing/provenance metadata.
- Surface the statistical benchmarking (CIs, effect sizes, p-values) in reports and CLI output, not just the API.
- Add optional performance/regression profiling runs to catch distance-matrix and validator slowdowns.
- Add a benchmark-results page in `docs/` with a reproducible reference run.

Definition of Done:

- Benchmark suite runs on at least 3 datasets without failures.
- Exported reports validated for JSON/YAML/Markdown, including statistical fields.
- A documented reference benchmark run is reproducible from a pinned seed and config.
- Release notes captured in `CHANGELOG.md`.

### v1.0.0 — December 15, 2026

Focus: API stability and release automation.

Scope:

- Backwards-compatibility policy and semantic-versioning guarantees.
- Freeze and document the public API surface (`__all__`) and the plugin/protocol APIs; publish at least one reference plugin.
- Public release checklist with automated verification (build, install-from-wheel smoke test, docs, security).
- Documentation audit and a deprecations plan for anything not promoted to stable.

Definition of Done:

- Backwards-compatibility policy documented and linked from the README.
- Full test suite and `make security` pass in CI.
- Tagged release with artifacts generated and published via CI.
- A clean-environment install-from-PyPI smoke test passes.

## Non-Goals

- End-to-end AutoML pipelines.
- Oversampler implementation (delegated to imbalanced-learn and similar libraries).
- Production model serving.
