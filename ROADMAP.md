# OversampleQA Roadmap

Last revised August 16, 2026, after v0.5.0.

## Project Goals

- Provide a practical, diagnostics-first toolkit for evaluating oversampling methods in imbalanced classification.
- Make synthetic sample quality measurable and comparable across methods, datasets, and metrics.
- Keep workflows reproducible with clear configuration, benchmarking, and exportable reports.
- Offer a fast path for practitioners and a deeper path for research use cases.
- Stay compatible with scikit-learn and imbalanced-learn conventions.

## Current State (post-v0.5.0)

- 439 tests collected in the main suite, plus 13 in the reference plugin. CI
  runs the main suite on Linux across Python 3.10-3.12 and on Windows for
  Python 3.12, then installs and tests the reference plugin separately.
- `ruff check` and `mypy src` pass clean. The numerical core is fully strict;
  the CLI, plotting and benchmark modules carry scoped, documented relaxations
  at their untyped third-party boundaries.
- Docs build under `-W`, enforced in CI.
- `poetry.lock` is tracked, poetry is pinned, `poetry check --lock` gates
  installs, and the docs toolchain lives in the locked `docs` Poetry group, so
  tests and docs both run against governed dependencies.
- Public API frozen behind a snapshot test; deprecation policy documented and
  mechanised.
- Both benchmark dataset catalogs now attach the same provenance record shape:
  source, generator, params, URL, license and notes. Synthetic seeds, OpenML
  versions, bundled-data licenses and positional truncation are explicit.

### Distribution

**This package is not published to PyPI, and there is no plan to publish it.**
Releases are cut by hand and archived on Zenodo, which mints the DOI. Install
from the repository or from a Zenodo archive.

| Version | DOI |
|---|---|
| Concept (all versions) | [10.5281/zenodo.21940361](https://doi.org/10.5281/zenodo.21940361) |
| 0.5.0 | [10.5281/zenodo.21967099](https://doi.org/10.5281/zenodo.21967099) |
| 0.4.0 | [10.5281/zenodo.21965065](https://doi.org/10.5281/zenodo.21965065) |
| 0.3.0 | [10.5281/zenodo.21959782](https://doi.org/10.5281/zenodo.21959782) |
| 0.2.0 | [10.5281/zenodo.21940362](https://doi.org/10.5281/zenodo.21940362) |

## Delivered

### v0.2.0 — correctness and release-gate credibility

- Typed-validator defects fixed: undefined `ConfigurationError`, unimported
  `Tuple`, `return` inside `finally`, and the async mode that threw under a
  running loop. The Wald interval was replaced with Wilson, which does not
  degenerate near a zero error rate.
- Ruff had no configuration at all, so the project inherited whatever the
  installed version defaulted to. The rule set is now explicit, every exemption
  carrying a reason. Black, isort and flake8 removed.
- Docs build warning-clean under `-W`. 33 orphaned autosummary stubs that
  duplicated the hand-written API pages were removed.
- Tooling is cross-platform; Windows is in the CI matrix.
- Caching rebuilt: opt-in rather than constructed at import, no `lru_cache` on
  an instance method, no live object in the disk key, enforced eviction.

### v0.3.0 — statistical honesty and fidelity

- **`oversampleqa.inference`.** `null_error_rate` gives the error rate a
  reference scale by scoring real held-out minority points through the same
  pipeline. Three nearest-neighbour two-sample tests with permutation p-values.
  `error_rate_interval` uses a parent-block bootstrap, because synthetic points
  sharing a SMOTE parent are not independent trials — on 200 points from 40
  parents the interval is 2.3× wider than Wilson.
- `friedman_nemenyi` and `plot_critical_difference` implement the Demšar (2006)
  protocol, which is the benchmark's actual question and was absent.
- **`oversampleqa.fidelity`.** Separates "realistic" from "diverse", which one
  scalar cannot. `RandomOverSampler` scores an error rate comparable to SMOTE's
  while contributing no information at all; only the memorisation ratio
  separates them.
- Benjamini–Hochberg correction alongside Holm and Bonferroni.
- **The real Hassanat distance.** The built-in implementation had shipped for
  the project's entire history scoring `[-5]` and `[5]` as distance zero — it
  compared absolute values, so it violated the identity of indiscernibles and
  was not a metric. Every number it produced looked plausible.

### v0.4.0 — reporting

- `oversampleqa fidelity` subcommand and a fidelity block in reports.
- **The Markdown report was not Markdown.** It used `to_csv(sep="|")`, so it had
  no header separator row and no edge pipes and had never rendered as a table.
- `StatisticalBenchmark` returned a `(0, 0)` frame with no explanation when
  every fold was skipped; it now keeps its columns and warns once with the
  cause.

### v0.5.0 — extensibility and public API hardening

- Public API frozen behind a committed snapshot test, so drift shows up in
  review rather than at release. `docs/api_stability.md` documents what is
  public and what deprecation means pre-1.0.
- `deprecated()` decorator mechanising that policy.
- Entry-point plugin discovery, with a reference plugin in `examples/plugins/`
  installed and tested in CI.
- Plugin registration rejects name collisions, bad signatures, and metrics that
  fail an axiom smoke check — the lesson from the Hassanat defect, made
  structural. `register_validator` had been a bare dictionary assignment.
- `plot_fidelity_radar` for cross-sampler comparison.
- **`poetry.lock` was gitignored**, so CI re-resolved every dependency on every
  run and no two runs were guaranteed to test the same versions.

### Unreleased — provenance, dependency closure and export cleanup

- `advanced_benchmark.DatasetRepository` now carries dataset provenance in the
  same shape as `benchmark.load_standard_datasets`. The two catalogs no longer
  disagree about where a dataset came from, what generated it, which seed or
  OpenML version was used, or what license applies.
- The shared provenance helpers live in `oversampleqa._provenance`, with tests
  for required keys, licenses, synthetic seeds, OpenML version pinning, bundled
  data, and the fact that `max_samples` is a positional truncation rather than
  a random sample.
- The docs dependency stack moved out of `docs/requirements.txt` and into the
  locked Poetry `docs` group. The docs CI job now installs with
  `poetry install --with dev,docs`, so Sphinx and its extensions are governed by
  the same lock discipline as the test environment.
- The basic benchmark frame is being tightened into a fixed long-format schema:
  one row per `(dataset, oversampler, metric, hidden_ratio, run)`, with the
  metric named in the data rather than implied by the call site. Export paths
  share a table renderer so the Markdown bug cannot survive in one output path
  after being fixed in another.

## Open Work

Not scheduled against dates. This is a research toolkit maintained by one
person; the ordering below reflects priority, not a delivery commitment. Each
item should land with tests or documentation that would have caught the failure
mode it addresses.

### v0.5.1 — correctness closure before features

No new user-facing feature work should land until these items are resolved. The
goal is to bring the older benchmark, multiclass and inference layers up to the
same statistical standard as the corrected binary validator.

- **Null-calibration estimand consistency.** `null_error_rate()` must calibrate
  the same experiment as the default validator. Use disjoint minority splits for
  sampler training, the common minority reference and real null candidates, so
  observed synthetic candidates and real null candidates are scored against the
  same hidden-majority and held-out-minority reference sets. The ceiling should
  also avoid candidate/reference self-overlap.
- **Calibration interpretation fix.** `NullCalibration.interpret()` should
  distinguish values below the calibrated interval, inside it and above it. An
  observed score below `low` is not "within" `[low, high]`.
- **Multiclass parity with binary validation.** Multiclass validation should
  reuse the binary path's safety guarantees: verify original-row prefix
  preservation before extracting synthetic rows, enforce `min_hidden`, account
  for ties, and reject too-small hidden references instead of reporting
  plausible noise.
- **Multiclass missing measurements as `nan`.** Classes for which a sampler
  generates no synthetic points are unmeasured, not perfect. Report `nan` for
  those classes and compute macro summaries with explicit `nanmean` over
  genuinely evaluated target classes.
- **Benchmark ranking validity.** `compute_ranking()` should not average raw
  error rates across incomparable datasets, hidden ratios or metrics. Rank
  within each dataset/specification first, then aggregate ranks, matching the
  logic behind the Friedman/Nemenyi workflow.
- **Metric-scoped pairwise inference.** Advanced benchmark pairwise tests must
  group by `(dataset, metric)` before comparing oversamplers. P-values and
  effect sizes from one metric must never be attached to rows for another
  metric.
- **Paired effect sizes.** Replace independent-sample Cohen's d in paired
  benchmark comparisons with a paired standardized difference or rank-biserial
  effect size aligned with the Wilcoxon design.
- **Benchmark data fixes.** `DatasetRepository._load_medical()` should mark
  `load_breast_cancer()` minority label `0`, not `1`; OpenML loading should use
  the actual `fetch_openml(..., as_frame=False)` target interface; and OpenML
  tests should fail when requested OpenML datasets are all skipped.
- **Plugin metrics in validation.** Custom metric plugins discovered through
  entry points should be usable end to end by `distance_matrix()` and
  `validate_oversampling(..., metric=...)`, not only retrievable through
  `PluginManager.get_metric()`.
- **Metric-domain enforcement.** Validators should reject sample-level metrics
  such as `energy` and `wasserstein` for pointwise nearest-neighbour validation,
  and use declared metric domains to reject probability, boolean or
  non-negative metrics on incompatible data.
- **Inferential dependence caveats.** The nearest-neighbour, MST and cross-match
  permutation tests should either gain a block/per-fit resampling design for
  dependent synthetic samples or document their p-values as diagnostic
  approximations for SMOTE-like generators.
- **Advanced benchmark defaults.** Remove default Mahalanobis rows unless a
  covariance inverse is estimated for the relevant training/reference set, and
  remove `recommended_samples` until it is tied to a defined hypothesis and
  sampling unit.
- **Metric degeneracy tests.** Add deterministic metric-property tests for
  degenerate inputs: `cosine_distance(0, x)`, constant-vector correlation
  distance and other cases the randomized axiom smoke test can miss.
- **Noise sensitivity realised flips.** `noise_sensitivity_diagnostic()` should
  sample replacement labels excluding the original label, so a requested flip
  fraction corresponds to realised label changes.
- **CI quality gates.** Main CI should enforce `ruff check` and `mypy src`, not
  only tests and docs, because the roadmap and README claim those gates are
  clean.
- **Release-document consistency.** Fix the README citation version, update stale
  benchmarking docs, and check Markdown benchmark export rendering alongside the
  report renderer so release-facing docs and exports do not contradict the code.

### v0.6.0 — export and reporting trust

This release can start after the v0.5.1 correctness closure. It should improve
how results leave the package, without changing the statistical meaning again.

- **Fold-level benchmark export.** The statistical benchmark summary is already
  tidy by `(dataset, oversampler, metric)`, but the raw fold/repeat observations
  are still embedded as list-valued `error_rates`. Add an export with one row
  per `(dataset, oversampler, metric, repeat, fold)`, including the split seed,
  hidden ratio, error rate, skip status and skip reason.
- **Unified result schema.** Align the simple benchmark, statistical benchmark,
  validation report and `Report.to_frame()` outputs around named identifiers
  rather than positional assumptions. Dataset, oversampler, metric, seed,
  hidden ratio, reference mode, repeat/fold index and package version should be
  explicit wherever a row can leave the process.
- **Report metadata block.** Every Markdown, HTML, JSON and CSV export should
  carry or sit next to enough metadata to reproduce the run: OversampleQA
  version, Python version, dependency lock hash when available, platform,
  random seeds, metric parameters and dataset provenance.
- **Strict JSON outputs.** Audit every JSON-producing path for `NaN`,
  infinities and NumPy scalars. Machine-readable exports should be accepted by
  strict parsers without relying on pandas' permissive defaults.
- **Renderer consolidation.** Keep frame rendering in one internal module and
  route reports, benchmark exports and future CLI table exports through it.
  This prevents a second copy of the old `to_csv(sep="|")` bug.

### v0.6.0 — documentation debt

- **Benchmarking docs correction.** `docs/benchmarking.rst` still explains the
  old confidence-interval bug where the implementation switched from a
  Student-t interval to a percentile range at `n=30`. The code now uses a
  t-interval for the mean at every sample size; the docs should describe that
  behavior and keep the historical warning only as release-note context.
- **Concepts page calibration update.** `docs/concepts.rst` says error-rate
  calibration against a null model is not implemented, but
  `oversampleqa.inference` now provides `null_error_rate` and the CLI exposes
  `--calibrate`. Rewrite that section so new users do not learn an obsolete
  limitation.
- **Examples refresh.** Update gallery and tutorial examples so they use
  `n_repeats`, report duplication or memorisation when relevant, seed both the
  validator and the oversampler, and show the new export metadata instead of a
  point estimate alone.
- **Decision guide.** Add a short guide for choosing among error rate,
  calibrated error rate, two-sample tests, fidelity metrics, downstream utility
  and benchmark rankings. The current docs explain the pieces but leave too much
  assembly to the reader.

### v0.7.0 — user-facing features

- **Experiment manifest runner.** Add a YAML-driven command that can run
  validation, fidelity diagnostics and benchmarks from one checked-in manifest.
  The manifest should name datasets, target columns, samplers, metrics, seeds,
  hidden ratios, repeats, output formats and cache settings.
- **`oversampleqa compare` command.** Provide a first-class CLI workflow for
  comparing multiple oversamplers on one dataset. It should run the selected
  diagnostics, rank samplers, flag exact-copy behaviour, and write a report
  bundle without requiring users to compose several commands manually.
- **Report bundles.** Add an output directory format containing `report.html`,
  `summary.json`, `results.csv`, generated plots and a run manifest. This gives
  users one artefact they can archive, attach to an issue, or cite in a paper.
- **Quality gates.** Add configurable thresholds for CI-style use: maximum error
  rate, maximum memorisation, minimum downstream utility, maximum boundary
  violation and required confidence interval width. The CLI should exit
  non-zero with a clear reason when a gate fails.
- **Sampler recommendation summary.** Turn the existing diagnostics into a
  conservative recommendation table: best realism, best diversity, lowest
  memorisation, best downstream utility and "do not use" warnings when a sampler
  duplicates, deletes originals, or cannot support the requested estimand.
- **Dataset audit command.** Add `oversampleqa audit-data` to report class
  imbalance, minority size, feature types, missing values, duplicate rows,
  likely leakage columns, recommended `hidden_ratio`, `min_hidden`, and whether
  the dataset can support multiclass or fidelity diagnostics.
- **Plugin scaffold command.** Add a small generator for metric and validator
  plugins based on `examples/plugins/`, including pyproject entry points, tests
  and the axiom smoke-check harness.
- **Static comparison dashboard.** Generate a self-contained HTML dashboard for
  benchmark runs with sortable tables, fidelity radar plots, confidence
  intervals, skip reasons and provenance metadata. This should remain static
  HTML, not a hosted service.

### v0.8.0 — benchmark quality

- **Pinned benchmark catalog.** Define a small catalog of reference datasets
  that is stable enough to cite: generated datasets with fixed seeds, OpenML
  datasets with pinned versions, explicit licenses and documented expected
  failure modes.
- **Reference benchmark runs.** Store pinned benchmark outputs for a small
  sampler set and metric set. These should be documentation artefacts, not
  brittle CI gates, and should state the dependency lock and platform used.
- **Sampler capability matrix.** Record which oversamplers preserve original
  rows, expose `random_state`, support multiclass data, can delete samples and
  can produce exact duplicates. Use it in docs and warnings so unsupported
  samplers fail with actionable messages.
- **Statistical comparison polish.** Make the Friedman/Nemenyi workflow easier
  to use from benchmark outputs: validate balanced method/dataset designs,
  explain missing cells, and emit a ready-to-plot critical-difference table.
- **Skip accounting.** Promote skipped combinations from warnings into a
  structured result table. A benchmark with no usable folds should be easy to
  summarize programmatically.

### v0.9.0 — scalability and runtime behaviour

- **Process-level parallelism.** Add real parallelism across repeats, folds,
  datasets and sampler/metric combinations. `ValidationMode.ASYNC` raises
  rather than pretending: the work is CPU-bound NumPy, so an event loop offers
  no concurrency. Process-level parallelism is the honest version, with stable
  seeded streams per worker.
- **Streaming exports.** Large benchmark runs should not need to hold every fold
  result in memory. Add append-friendly CSV/JSONL writing with resume metadata
  and clear partial-run markers.
- **Cache process safety.** The cache is thread-safe through one instance but
  not process-safe on disk. Either document per-process cache directories in
  the parallel runner or add atomic writes and file locking.
- **Performance-regression tracking.** Keep the scheduled workflow non-blocking,
  but add historical trend artefacts and a local comparison command that can
  separate likely regressions from shared-runner noise.
- **Memory-budget reporting.** When batching kicks in, expose the chosen batch
  size, estimated peak memory, safety factor and metric multiplier in verbose
  CLI and report metadata.

### Research backlog

- **Distance metric review.** Revisit metrics that are sample-based rather than
  pointwise (`energy`, `wasserstein`) and decide whether they belong in the same
  registry, need a separate protocol, or should be flagged more prominently in
  APIs that expect point metrics.
- **Parent-aware inference integration.** `error_rate_interval` can use a
  parent-block bootstrap, but validators do not yet surface parent identifiers
  from oversamplers. Define a practical parent-tracking protocol where samplers
  expose enough information.
- **Calibration defaults.** Decide when the calibrated null/ceiling scale should
  be the default recommendation rather than an opt-in diagnostic. This needs
  examples across easy, overlapping, high-dimensional and tiny-minority regimes.
- **High-dimensional fidelity guidance.** The k-NN manifold metrics warn in high
  dimension; add empirical guidance on when density/coverage remain useful and
  when downstream utility or calibrated error rate is more defensible.
- **Multiclass fidelity.** The fidelity report currently rejects multiclass
  data. Define whether multiclass fidelity is one-vs-rest, per-class pairwise,
  or a separate confusion-style object.

### Maintenance backlog

- **Release metadata consistency.** The release-facing files (`README.md`,
  `CITATION.cff`, `docs/citation.rst`, `.zenodo.json` and `CHANGELOG.md`) should
  be checked as one unit before every archive so examples, version numbers, DOI
  guidance and citation snippets cannot drift.
- **Release checklist automation.** Add a local `scripts/release.py` check mode
  that verifies version strings, citation metadata, changelog links, DOI
  placeholders, docs build, API snapshot and clean wheel install before a
  GitHub release is published.
- **API-stability audit.** Before each minor release, compare the committed API
  snapshot with docs and examples. Any new export either needs documentation or
  an explicit decision that it is private.
- **Dependency policy.** Keep runtime dependencies conservative and documented.
  Any dependency used at import time belongs in the runtime group; docs-only and
  benchmark-only tools should stay out of the core install unless the import
  path requires them.
- **Issue templates and contribution notes.** Add prompts for benchmark
  reproducibility: seeds, dataset provenance, OversampleQA version, metric,
  hidden ratio, sampler configuration and whether calibration/fidelity was run.

### v1.0.0

Focus: a stable contract, not automated publishing.

- Finalise backwards-compatibility policy and semantic-versioning guarantees.
  Pre-1.0 the guarantee is best-effort; 1.0 is where that stops being true.
- A clean-environment install-and-smoke-test from a built wheel. Note this is
  install-from-wheel, **not** install-from-PyPI: nothing is published there.
- An explicit release checklist covering build, test, docs, and Zenodo
  archiving. `docs/citation.rst` holds the current version.

## Non-Goals

- **Publishing to PyPI.** Distribution is by repository and Zenodo archive.
- **Automated release publishing.** Releases are deliberately cut by hand; the
  release workflow was removed. A tag alone archives nothing — the Zenodo
  webhook fires on GitHub release publication.
- End-to-end AutoML pipelines.
- Oversampler implementation (delegated to imbalanced-learn and similar).
- Production model serving.
