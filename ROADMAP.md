# OversampleQA Roadmap

This roadmap reflects a repo review completed on June 19, 2026. The test suite is green (`103 passed`), but several release and quality gates still do not match the repo's stated maturity.

## Project Goals

- Provide a practical, diagnostics-first toolkit for evaluating oversampling methods in imbalanced classification.
- Make synthetic sample quality measurable and comparable across methods, datasets, and metrics.
- Keep workflows reproducible with clear configuration, benchmarking, and exportable reports.
- Offer a fast path for practitioners and a deeper path for research use cases.
- Stay compatible with scikit-learn and imbalanced-learn conventions.

## Current State (v0.1.0, reviewed June 19, 2026)

What is working:

- Core validation, benchmarking, reporting, plotting, and CLI flows are implemented.
- `poetry run pytest` passes across 241 tests.
- `make lint` (`ruff check src tests`) passes clean.
- `make typecheck` (`mypy src`) passes clean. The numerical core is fully strict;
  the CLI, plotting and benchmark modules carry scoped, documented relaxations
  at their untyped third-party boundaries.
- `make docs` passes with warnings treated as errors, enforced in CI.
- Developer tooling runs on Windows and Linux, and both are in the CI matrix.

What is not yet at release-ready quality:

- Async mode in `typed_validator.py` is not a working execution mode. It now
  raises a clear error rather than failing obscurely: the work is CPU-bound
  NumPy, so an event loop offers it no concurrency. Real parallelism over
  repeats and datasets is not implemented.
- The statistical claims in `advanced_benchmark.py` do not hold. See the
  findings below.

## Review Findings To Drive The Roadmap

### Bugs

- `src/oversampleqa/typed_validator.py` has latent runtime/type defects:
  - `_wald_confidence_interval()` annotates `Tuple[float, float]` but `Tuple` is not imported.
  - `ServiceRegistry.get()` raises `ConfigurationError`, but that symbol is not imported or defined in the module.
  - `validation_session()` uses `return` inside `finally`, which can suppress exceptions.
  - `ValidationMode.ASYNC` calls `run_until_complete()`, which is unsafe when already inside a running event loop.
- `docs/gallery/index.rst` and generated gallery pages cause unresolved-title and broken-cross-reference warnings, so the docs are not compatible with a `-W` build yet.
- `Makefile` commands are not portable to Windows. `clean` is broken in a PowerShell-first environment, and the docs target depends on Unix shell chaining.
- ~~`src/oversampleqa/caching.py` uses `@lru_cache` on an instance method backed by mutable temporary state (`self._distance_args`).~~ **Resolved.** Replaced with a lock-guarded LRU keyed on the content hash; the thread-safety contract is now stated and tested.

### Improvements

- Reduce the quality-gate gap between the README/roadmap promises and reality: lint, typecheck, docs, and release automation should all be verifiably green.
- Add explicit cross-platform maintenance for development tooling and CI, not just package runtime code.
- Add focused tests for async validation, service-registry error paths, and docs/gallery generation.
- Tighten plugin discovery and validation so custom extensions fail with actionable errors instead of permissive duck-typing.
- Turn docs and benchmark outputs into reproducible artifacts with pinned seeds, dataset provenance, and warning-free builds.

### New Features

- Surface statistical benchmark outputs directly in exported reports and CLI summaries.
- Publish a reference plugin that demonstrates the supported metric/validator extension contract.
- Add a documented benchmark catalog with dataset licenses, provenance, and pinned reference results.
- Add performance-regression tracking for distance matrices, validator throughput, and memory-efficient paths.

## Milestones

Owner for all milestones: `diogoribeiro7`.

### v0.2.0 — reset target: July 31, 2026

Focus: correctness and release-gate credibility.

Scope:

Delivered:

- Typed-validator correctness bugs fixed: the undefined `ConfigurationError`,
  the unimported `Tuple`, the `return` inside `finally`, and the async mode that
  threw under a running loop. The Wald interval was replaced with Wilson, which
  does not degenerate near a zero error rate. All four paths now have tests.
- `make lint` and `make typecheck` pass. Ruff had no configuration at all, so the
  project inherited whatever the installed version defaulted to; the rule set is
  now explicit, with every exemption carrying a reason.
- Docs build warning-clean under `-W`, enforced in CI. 33 orphaned autosummary
  stubs that duplicated the hand-written API pages were removed, along with the
  `exclude_patterns` entry that existed only to hide them.
- Tooling is cross-platform: `clean` is a Python script, docs invoke
  `sphinx-build` directly, and Windows is in the CI matrix.
- Caching was rebuilt: opt-in rather than constructed at import, no `lru_cache`
  on an instance method, no live object in the disk key, enforced eviction, and
  read-only returns.
- Tooling consolidated on ruff; black, isort and flake8 are gone.

Definition of Done:

- `poetry run pytest`, `make lint`, and `make typecheck` all pass on a clean
  checkout. **Met.**
- `make docs` succeeds with warnings treated as errors. **Met.**
- Typed validator async/session paths have dedicated tests. **Met.**
- The documented developer commands work on both Windows and Linux. **Met.**

### v0.3.0 — target: September 30, 2026

Focus: reproducible benchmarking and report depth.

Scope:

- Expand the built-in dataset catalog with explicit licensing and provenance metadata.
- Surface confidence intervals, effect sizes, and p-values in CLI output and exported reports, not only in the API.
- Add a reproducible benchmark-results page with pinned seeds/configuration.
- Add performance/regression checks for distance-matrix generation and memory-efficient validation.
- Export benchmark results in a more analysis-friendly shape across CSV/JSON/Markdown outputs.

Definition of Done:

- Benchmark suite runs on at least 3 documented datasets without failures.
- Statistical fields are visible in CLI summaries and exported reports.
- A reference benchmark run is reproducible from committed config and seed values.
- Performance baselines are captured and compared in CI or scheduled runs.

### v0.4.0 — target: November 30, 2026

Focus: extensibility and public API hardening.

Scope:

- Freeze and document the intended public API surface.
- Tighten plugin-system contracts and publish one reference plugin.
- Add better extension diagnostics for invalid plugins, missing methods, and registration collisions.
- Document compatibility guarantees and deprecation policy for pre-1.0 users.

Definition of Done:

- Public API and plugin contracts are documented and tested.
- A reference plugin is published in-repo or as a companion example package.
- Plugin loading failures produce actionable user-facing errors.

### v1.0.0 — target: January 29, 2027

Focus: stable release automation.

Scope:

- Finalize backwards-compatibility policy and semantic-versioning guarantees.
- Enable end-to-end automated release publishing with verified artifacts.
- Add a clean-environment install-and-smoke-test step for built distributions.
- Publish an explicit release checklist covering build, test, docs, and security verification.

Definition of Done:

- Tagged release artifacts are generated and published via CI.
- Full test suite, docs, lint, typecheck, and security checks pass in CI.
- A clean-environment install-from-wheel or install-from-PyPI smoke test passes.
- Release process is documented and repeatable.

## Non-Goals

- End-to-end AutoML pipelines.
- Oversampler implementation (delegated to imbalanced-learn and similar libraries).
- Production model serving.
