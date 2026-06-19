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
- `poetry run pytest` passes across 103 tests.
- Docs build to HTML successfully in permissive mode.

What is not yet at release-ready quality:

- `make lint` does not pass. The repo currently has a large backlog of `flake8`/`ruff` findings across both `src/` and `tests/`.
- `make typecheck` is not in a healthy state. The typed validator module contains unresolved symbols and async-control-flow issues that should be fixed before claiming strict typing support.
- `make docs` is not warning-clean. The Sphinx build currently succeeds with 66 warnings, mostly from gallery pages lacking resolvable titles/refs.
- The `Makefile` is Unix-centric even though development is happening on Windows: `clean` uses `rm`/`find`, and the docs command shells through `cd docs && poetry run make html`.
- Some advertised advanced surfaces are only partially real: async mode exists in `typed_validator.py`, but it is not production-safe yet.

## Review Findings To Drive The Roadmap

### Bugs

- `src/oversampleqa/typed_validator.py` has latent runtime/type defects:
  - `_wald_confidence_interval()` annotates `Tuple[float, float]` but `Tuple` is not imported.
  - `ServiceRegistry.get()` raises `ConfigurationError`, but that symbol is not imported or defined in the module.
  - `validation_session()` uses `return` inside `finally`, which can suppress exceptions.
  - `ValidationMode.ASYNC` calls `run_until_complete()`, which is unsafe when already inside a running event loop.
- `docs/gallery/index.rst` and generated gallery pages cause unresolved-title and broken-cross-reference warnings, so the docs are not compatible with a `-W` build yet.
- `Makefile` commands are not portable to Windows. `clean` is broken in a PowerShell-first environment, and the docs target depends on Unix shell chaining.
- `src/oversampleqa/caching.py` uses `@lru_cache` on an instance method backed by mutable temporary state (`self._distance_args`). That is brittle for long-lived processes and likely unsafe under concurrent use.

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

- Fix typed-validator correctness bugs in `src/oversampleqa/typed_validator.py`:
  - import/define missing symbols,
  - remove exception-silencing control flow,
  - make async mode safe under an existing event loop,
  - add tests that exercise those paths.
- Get `make lint` and `make typecheck` to pass, or narrow the configured rule set to what the project is actually prepared to enforce.
- Make docs warning-clean:
  - fix gallery title/reference generation,
  - ensure the docs command is run with warnings treated as errors in CI.
- Make developer tooling cross-platform:
  - replace Unix-only `clean`/docs shell commands,
  - verify workflow on Windows and Linux.
- Audit caching behavior in `src/oversampleqa/caching.py` and either harden the current design or simplify it to a safer disk-cache-only path.

Definition of Done:

- `poetry run pytest`, `make lint`, and `make typecheck` all pass on a clean checkout.
- `make docs` succeeds with warnings treated as errors.
- Typed validator async/session paths have dedicated tests.
- The documented developer commands work on both Windows and Linux.

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
