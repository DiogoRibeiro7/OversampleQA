# OversampleQA Roadmap

Last revised August 29, 2026, after v0.6.1.

## Project Goals

- Provide a practical, diagnostics-first toolkit for evaluating oversampling methods in imbalanced classification.
- Make synthetic sample quality measurable and comparable across methods, datasets, and metrics.
- Keep workflows reproducible with clear configuration, benchmarking, and exportable reports.
- Offer a fast path for practitioners and a deeper path for research use cases.
- Stay compatible with scikit-learn and imbalanced-learn conventions.
- Raise the package toward scikit-learn-grade trust: stable APIs, rigorous
  numerical behaviour, reproducible examples, clear limits, and boring release
  mechanics.

## Current State (post-v0.6.1)

- 664 tests in the main suite, plus 13 in the reference plugin. CI runs the main
  suite on Linux across Python 3.10-3.12 and on Windows for Python 3.12, then
  installs and tests the reference plugin separately.
- `ruff check src tests` and `mypy src` are enforced by CI, not merely claimed.
  The numerical core is fully strict; the CLI, plotting and benchmark modules
  carry scoped, documented relaxations at their untyped third-party boundaries.
- Docs build with MkDocs in strict mode and deploy to GitHub Pages from the
  artifact that build produced, so what is published is what passed the gate.
- Documentation links are gated: MkDocs validates internal pages and anchors,
  and CI checks stable external targets such as the project docs, GitHub, PyPI
  and DOI links.
- `poetry.lock` is tracked, poetry is pinned, `poetry check --lock` gates
  installs, and the docs toolchain lives in the locked `docs` group, so tests
  and docs run against one governed dependency set.
- Public API frozen behind a snapshot test; deprecation policy documented and
  mechanised. `py.typed` ships, so the annotations reach downstream checkers.
- Sklearn integration is covered by focused CI tests for `cross_validate`,
  `GridSearchCV`, nested sampler parameters, sklearn `Pipeline` composition and
  common imbalanced-learn sampler paths.
- Core-path benchmark artifacts record runtime, peak traced memory,
  environment metadata and benchmark parameters for distance-matrix and
  validation paths. The scheduled Performance workflow uploads them for trend
  inspection without making noisy timing results a merge gate.
- Core trust documentation now covers metric choice, result interpretation
  limits and production audit workflows, while linking those decisions back to
  API stability and reproducibility guidance.
- Release-facing metadata is checked as one unit on every commit, and the check
  blocks a version bump until the previous release's DOI is recorded.

**0.6.0 was a correctness release.** The null calibration, benchmark ranking,
pairwise inference and permutation tests each reported a different quantity
before it, so results produced with 0.5.1 or earlier are not comparable with
results produced now. `CHANGELOG.md` states which numbers changed and by how
much.

### Distribution

Published to PyPI as [`oversampleqa`](https://pypi.org/project/oversampleqa/)
and archived on Zenodo. Releases are cut by hand: publishing a GitHub release
triggers the PyPI Trusted Publishing workflow and the Zenodo webhook together.
PyPI provides the installable package; Zenodo mints the DOI for citation.

Documentation: https://diogoribeiro7.github.io/OversampleQA/

| Version | DOI |
|---|---|
| Concept (all versions) | [10.5281/zenodo.21940361](https://doi.org/10.5281/zenodo.21940361) |
| 0.6.1 | [10.5281/zenodo.22159067](https://doi.org/10.5281/zenodo.22159067) |
| 0.6.0 | [10.5281/zenodo.21993371](https://doi.org/10.5281/zenodo.21993371) |
| 0.5.1 | none — see below |
| 0.5.0 | [10.5281/zenodo.21967099](https://doi.org/10.5281/zenodo.21967099) |
| 0.4.0 | [10.5281/zenodo.21965065](https://doi.org/10.5281/zenodo.21965065) |
| 0.3.0 | [10.5281/zenodo.21959782](https://doi.org/10.5281/zenodo.21959782) |
| 0.2.0 | [10.5281/zenodo.21940362](https://doi.org/10.5281/zenodo.21940362) |

**0.5.1 has no Zenodo record.** Zenodo archives by fetching the release tarball
from `codeload.github.com`; that request timed out during a GitHub outage on
2026-08-17, and redelivering the webhook returns `409` because Zenodo has
already seen the release. It cannot be archived after the fact. Cite the concept
DOI for that version. `tests/test_release_metadata.py` records the exception, so
the DOI check does not block later releases over a gap nobody can fill.

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

### v0.5.1 — provenance, dependency closure and PyPI

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

### v0.6.0 — correctness closure

Every item on the v0.5.1 correctness list, and it changed what several numbers
mean. The measurements below are from the codebase, not estimates.

- **Permutation tests rejected a true null every time on SMOTE output.** With
  both samples drawn from the *same* distribution and only the block structure
  differing: 100% false rejections at 0.05, median p 0.005. They were detecting
  the clustering SMOTE always produces. Passing `parents` subsamples one point
  per parent: 0% false rejections, median p 0.885.
- **The null calibrated a different experiment.** It scored against
  `fit_minority` (~90% of the minority) where the validator scores against the
  held-out 10% — a bar four times too low (0.033 against 0.133). Ceiling
  candidates could also be the points they were scored against: 8.8% of
  candidates, 64% of draws.
- **`compute_ranking` inverted results**, averaging error rates across
  incomparable experiments. Ranking is now within each
  (dataset, hidden_ratio, metric) and agrees with `friedman_nemenyi` exactly.
- **Pairwise statistics came from the wrong metric**, and the effect size
  discarded the pairing its p-value depended on (-0.573 against a paired -1.0).
- **Multiclass validation lacked the binary path's guarantees**: no prefix
  check, no `min_hidden`, ties to the lowest label index (38.6% of attributions
  on quantised data), and `0.0` for a class that was never measured.
- **`mahalanobis` without `cov_inv` was Euclidean** — one metric under
  another's name, and a default, so every benchmark carried duplicate rows.
- **Sample-level metrics were accepted pointwise**: `[0, 5]` and `[5, 0]` scored
  0.0 under wasserstein, and `energy` returned -5.0.
- **`cosine(0, x)` and `correlation(const, x)` returned 0.0**, calling distinct
  points identical — the shape of the original Hassanat defect, missed because
  the axiom check draws random vectors.
- **No built-in dataset could produce a measurement at the package's own
  defaults**: 42 of 42 benchmark rows were `nan`, now 0 of 42.
- **`noise_sensitivity_diagnostic` applied half the noise it reported** on
  binary data.
- Removed `recommended_samples`, which reported 1 on every row of every run.

### v0.6.1 — MkDocs documentation

- Documentation migrated from Sphinx/RST to MkDocs/Markdown.
- API reference pages now use mkdocstrings, keeping the docs close to the
  installed Python package surface.
- Repository, README and support links point readers at the published GitHub
  Pages documentation.
- The release DOI is recorded in `CITATION.cff`, README and
  `docs/citation.md`.

## Open Work

Not scheduled against dates. This is a research toolkit maintained by one
person; the ordering below reflects priority, not a delivery commitment. Each
item should land with tests or documentation that would have caught the failure
mode it addresses.

### Strategic target — scikit-learn-grade trust

OversampleQA should become a main tool people can rely on in serious imbalanced
classification work. That means the roadmap is now about trust as much as
capability: stable public contracts, sklearn-native workflows, defensible
statistics, scalable execution, strong documentation, and releases that users
can install and cite without ceremony.

- **Public API contract.** Keep the public API snapshot as a release gate, but
  expand it into an explicit compatibility contract: supported imports,
  signatures, default behaviours, warning classes, CLI commands and serialized
  output schemas. Every new public entry point should be documented or marked
  private before merge.
- **Sklearn-native integration.** Provide first-class helpers that fit
  scikit-learn and imbalanced-learn workflows: scorer-style callables,
  pipeline/grid-search examples, sampler comparison recipes, and compatibility
  tests against common `Pipeline`, `GridSearchCV` and imbalanced-learn sampler
  paths.
- **Scientific validation suite.** Add oracle datasets, adversarial samplers,
  property-based metric tests and calibration examples where expected behaviour
  is known. The aim is to prove not only that code runs, but that the reported
  quantities mean what the docs say they mean.
- **Limitations and decision guidance.** Document when each metric is strong,
  weak, misleading or undefined. Serious users need guidance for tiny minority
  classes, high-dimensional features, duplicate-producing samplers, multiclass
  data and noisy labels.
- **Performance discipline.** Add a benchmark harness with tracked runtime and
  memory profiles across sample count, dimensionality, metrics and samplers.
  Performance regressions should be visible before users discover them in long
  benchmark runs.
- **Distribution confidence.** Add clean-environment wheel install tests,
  minimum/latest dependency lanes where practical, strict link checks for docs,
  and post-release verification for PyPI, GitHub Pages and Zenodo.
- **Adoption-quality examples.** Ship recipes that look like real ML work:
  comparing SMOTE variants, checking a production dataset before oversampling,
  generating an audit bundle, and interpreting when a sampler should not be
  used.

### v0.7.0 — sklearn-grade foundation

Make the stable base stronger before adding large new workflows.

- **Compatibility tests for public imports and signatures.** Extend the API
  snapshot into tests that catch accidental signature, default-value and warning
  changes.
- **Wheel and installed-package smoke tests.** Build the wheel in CI, install it
  into a clean environment, import the documented public API, run one small
  validation, and verify CLI entry points.
- **Sklearn/imbalanced-learn integration tests.** Add focused tests proving the
  documented recipes work inside sklearn-style pipelines and with common
  imbalanced-learn samplers.
- **Docs link checking.** Add a CI gate for internal documentation links and
  selected external links that are stable enough to enforce.
- **Benchmark scaffold.** Introduce a lightweight benchmark harness and publish
  an initial baseline artifact for core validation paths.
- **Core trust docs.** Add or expand pages for API stability, choosing metrics,
  limitations, reproducibility, and production audit workflows.

### v0.8.0 — export and reporting trust

Improve how results leave the package, without changing statistical meaning
again. Two of the original items shipped in 0.6.0: the fold-level export
(`fold_results()`) and renderer consolidation (`oversampleqa._render`, which
exists because the `to_csv(sep="|")` bug had already been copied into a second
export path).

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

### v0.8.0 — documentation debt

The benchmarking-docs correction and the concepts-page calibration claim are
done; what remains is the material that helps a reader assemble the pieces.

- **Examples refresh.** Update gallery and tutorial examples so they use
  `n_repeats`, report duplication or memorisation when relevant, seed both the
  validator and the oversampler, and show the new export metadata instead of a
  point estimate alone.
- **Decision guide.** Add a short guide for choosing among error rate,
  calibrated error rate, two-sample tests, fidelity metrics, downstream utility
  and benchmark rankings. The current docs explain the pieces but leave too much
  assembly to the reader.

### v0.9.0 — user-facing features

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

### v0.10.0 — benchmark quality

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

### v0.11.0 — scalability and runtime behaviour

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

- **`umap-learn` is a hard runtime dependency**, pulling numba, llvmlite and
  pynndescent for a single optional plotting mode. Everyone who runs
  `pip install oversampleqa` pays that cost. Moving it to the `performance`
  extra would cut install weight substantially, but it breaks anyone calling
  `plot_sample_distribution(method="umap")`, so it belongs in a deliberate minor
  bump rather than a patch.
- **Docs build output should stay out of git.** The MkDocs site is generated in
  `site/` and published from CI artifacts. Keep generated docs output ignored so
  local builds do not create review noise.
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

Focus: a stable public contract and reproducible release process.

- Finalise backwards-compatibility policy and semantic-versioning guarantees.
  Pre-1.0 the guarantee is best-effort; 1.0 is where that stops being true.
- A clean-environment install-and-smoke-test from a built wheel and from PyPI
  after release publication.
- An explicit release checklist covering build, test, docs, PyPI publication and
  Zenodo archiving. `docs/citation.md` holds the current version.

## Non-Goals

- **Fully automated release creation.** Releases are deliberately initiated by
  hand. A tag alone publishes nothing; the PyPI workflow and Zenodo webhook fire
  only when a GitHub release is published.
- End-to-end AutoML pipelines.
- Oversampler implementation (delegated to imbalanced-learn and similar).
- Production model serving.
