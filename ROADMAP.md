# OversampleQA Roadmap

This roadmap summarizes the current scope, maturity, and next steps for the project.

## Project Goals

- Provide a practical, diagnostics-first toolkit for evaluating oversampling methods in imbalanced classification.
- Make synthetic sample quality measurable and comparable across methods, datasets, and metrics.
- Keep workflows reproducible with clear configuration, benchmarking, and exportable reports.
- Offer a fast path for practitioners and a deeper path for research use cases.
- Stay compatible with scikit-learn and imbalanced-learn conventions.

## Current State (v0.1.0)

- Core validation implemented for binary and multiclass workflows.
- Broad distance metric support with optimized and memory-efficient computation.
- Benchmarking utilities with dataset loaders and export helpers.
- Rich CLI with profiles, templates, shell completion, and diagnostics.
- Plugin system for custom metrics and validators.
- Sphinx docs, examples, and tutorials in-repo.

## Milestones

Owner for all milestones: `diogoribeiro7`.

### v0.2.0 — May 15, 2026

Focus: documentation quality and reproducibility.

Scope:

- Add concise CLI examples and configuration samples to docs.
- Add a short concepts page (reusing README content, expanded).
- Add a minimal tutorial notebook mirroring README quick start.
- Document reproducibility guidance (seeds, dataset provenance).
- Harden edge-case behavior and document expected error modes.

Definition of Done:

- `make lint typecheck` passes.
- `make coverage` passes locally.
- Docs build cleanly: `make docs`.
- Version bumped in `pyproject.toml` and tag created.

### v0.3.0 — August 15, 2026

Focus: benchmarking depth and reporting.

Scope:

- Expand benchmark suite with optional performance profiling runs.
- Improve reporting templates for research and production contexts.
- Add more built-in datasets with clear licensing metadata.
- Add a benchmark results README in `docs/`.

Definition of Done:

- Benchmark suite runs on at least 3 datasets without failures.
- Exported reports validated for JSON/YAML/Markdown.
- Release notes drafted in `CHANGELOG.md` (if reintroduced).

### v1.0.0 — December 15, 2026

Focus: API stability and release automation.

Scope:

- API stability commitments and semantic versioning guarantees.
- Formalize plugin APIs and publish reference plugins.
- Public release checklist with automated verification.
- Documentation audit and deprecations plan.

Definition of Done:

- Backwards compatibility policy documented.
- Full test suite passes and `make security` passes.
- Tagged release and release artifacts generated via CI.

## Non-Goals

- End-to-end AutoML pipelines.
- Oversampler implementation (delegated to imbalanced-learn and similar libraries).
- Production model serving.