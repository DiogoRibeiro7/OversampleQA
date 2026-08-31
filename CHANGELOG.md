# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut by hand: publish a GitHub release from a tag, which triggers
the PyPI publish workflow and the Zenodo archive. Entries below the *Unreleased*
section are maintained manually.

## [Unreleased]

### Added

- Python 3.13 is supported, tested and advertised. `python = ">=3.10"` has no
  upper bound, so `pip install oversampleqa` already succeeded on 3.13 -- but
  the classifiers stopped at 3.12 and nothing above 3.12 was ever run, so a
  3.13 user was told the package did not support them and a 3.13-only break
  would have reached PyPI unnoticed. The full matrix on `main` now includes it,
  and a test holds the classifiers equal to the versions CI runs.

### Fixed

- `oversampleqa doctor` rejects an unsupported Python. Its check read
  `sys.version.split()[0] >= "3.10"`, a string comparison in which `"3.9"`
  sorts after `"3.10"` -- so 3.7, 3.8 and 3.9 all passed a check that exists
  to reject them, and it could not fail for any Python 3.

### Changed

- `oversampleqa doctor` reports versions rather than only pass/fail, covering
  OversampleQA, Python, the platform, numpy, pandas, scikit-learn,
  imbalanced-learn, scipy and matplotlib. `pandas [OK]` reproduces nothing,
  and a version difference in numpy or scikit-learn changes results rather
  than merely whether the code runs. The facts are also available
  programmatically from `oversampleqa.cli_enhanced.diagnostics()`.

## [0.7.0] - 2026-08-31

The sklearn-grade foundation milestone: stable public contracts, checks that
run rather than merely exist, and reproducibility identifiers on every row that
can leave the process.

Most of this work landed before the changelog recorded it. Twenty-six commits
reached `main` after 0.6.1 with three of them documented here, so the entries
below were reconstructed from the pull requests. Nothing enforces changelog
coverage; the release-metadata tests check that a section exists for the
version, not that it describes the release.

### Added

- **Experiment manifests.** `oversampleqa run <manifest.yaml>` runs a set of
  validation experiments from a checked-in YAML file, resolving datasets,
  defaults and per-experiment overrides, and writing a resolved manifest and a
  summary beside the results.
- **Export metadata sidecars.** Every Markdown, HTML, JSON and CSV artifact is
  written next to an `<artifact>.metadata.json` carrying package and runtime
  versions, the dependency lock hash where available, row and column shape, and
  dataset provenance. Artifact shapes are unchanged; the metadata sits beside
  them.
- **Result reproducibility identifiers.** Simple benchmark rows carry the split
  seed, sampler seed, minority label, reference mode and package version;
  statistical summary and fold rows carry hidden ratio, reference mode,
  minority label, engine seed, fold and repeat settings and package version;
  `ValidationReport.to_frame` exposes the same core identifiers as direct
  columns alongside the existing `meta_*` ones.
- **Run metadata in reports.** Markdown and HTML benchmark reports open with a
  rendered metadata block, so a report read on its own says what produced it.
- **A shared result schema.** `oversampleqa._schema.IDENTIFIER_COLUMNS` names
  the identifier columns every result surface carries, and a test holds all
  four to it: `run_benchmark`, `fold_results`, the statistical summary and
  `ValidationReport.to_frame`. The surfaces had disagreed on two names --
  `dataset` against `dataset_name`, `oversampler` against `oversampler_name` --
  so `frame["dataset"]` worked or raised depending on which produced the frame.
  The canonical names were added alongside the originals; nothing was renamed
  or reordered. `RunMetadata` gains a `dataset` field, and
  `ValidationReport.to_frame` promotes `dataset_hash` out of the `meta_` block.
- **Public API signature snapshot.** `tests/api_signatures.json` pins the call
  contract of every exported name, so a changed default or a renamed parameter
  fails in review rather than in a user's code.
- **Warning contract snapshot.** `tests/warning_contract.json` records the
  category every `warnings.warn` site raises. Changing `warn_reference_bias`
  from `FutureWarning` to `UserWarning` moves no signature and removes no
  export, so nothing else would have caught it.
- **sklearn and imbalanced-learn integration tests.** `cross_validate` with a
  scorer, `GridSearchCV` over nested sampler parameters, the validator as a
  pipeline step, and SMOTE, BorderlineSMOTE and RandomOverSampler.
- **Installed-wheel smoke test in CI.** The wheel is built, installed into a
  clean virtualenv, and exercised through the documented public API and both
  console scripts. Source-tree tests pass happily on a wheel missing package
  data or a runtime dependency.
- **Documentation link gate.** `mkdocs build --strict` for internal links and
  an allowlist of external hosts checked in CI.
- **Benchmark scaffold and a committed baseline.** `scripts/benchmark_core_paths.py`
  plus `perf_baseline.json`, measured on a CI runner so the weekly comparison is
  like for like.
- **Core trust documentation.** Pages for API stability, choosing metrics,
  limitations, reproducibility and production audit workflows.
- **`.zenodo.json` declares `version`**, and a test holds it equal to the
  version in `pyproject.toml`. The omission left the deposit metadata unable to
  state its own version outside the GitHub-integration path.
- **Zenodo record links** to the documentation site (`isDocumentedBy`) and the
  PyPI project (`isIdenticalTo`), with a `notes` block pointing at the changelog
  and explaining the concept-versus-version DOI choice.

### Changed

- **Pull requests run one test job; `main` runs the full matrix.** The four
  test jobs were the whole cost of CI — the suite takes ~117s on Linux and
  ~148s on Windows against ~30s to install. A `ci-ok` job aggregates the
  others and is the single required status check, so branch protection is no
  longer a hidden copy of the matrix.
- **CI caches the Poetry environment**, not pip's. Poetry does not read pip's
  cache directory, so the previous keys did nothing measurable: 29s cached
  against 30s uncached.
- **NumPy annotations hardened** to explicit `NDArray[...]` aliases and
  `np.dtype[Any]`, fixing a latent quality-gate failure that surfaced whenever
  a lock refresh pulled newer NumPy typing rules.
- **Author affiliation** is recorded as Faculty of Media Arts and Design,
  Technical University of Porto in `CITATION.cff`, `.zenodo.json` and
  `AUTHORS.md`.
- **`CITATION.cff` carries the full descriptive title and matching keywords.**
  It held the bare name while both BibTeX blocks and the Zenodo record used the
  long form, so GitHub's *Cite this repository* button produced a different
  title from the one the README told people to paste. Both are now compared by
  test.

### Fixed

- **Strict JSON exports.** Validation checkpoints, validation and fidelity JSON,
  report JSON and benchmark JSON all route through a strict serializer, so bare
  `NaN` and `Infinity` tokens cannot reach a machine-readable export.
- **The pairwise statistics columns wrote invalid JSON.**
  `pairwise_p_values` and `pairwise_effect_sizes` called `json.dumps` directly.
  Skipped folds are kept as `nan` by design, so one skipped fold made both
  columns unparseable by a strict reader. A test now fails on any `json.dumps`
  in the package outside the helper that defines it.
- **Manifest runs validate before doing any work.** A typo'd dataset name was
  silently treated as a path; missing dataset files were found only when their
  experiment was reached, after earlier ones had run; two experiments whose
  names slugified alike shared an output directory and the second overwrote the
  first; a bad scalar raised a bare `ValueError` naming neither experiment nor
  field; and a failure part-way through discarded the summary for everything
  that had already completed.
- **The weekly performance workflow never checked for regressions.** Its
  comparison step was conditional on a baseline file that did not exist, so it
  was skipped on every run while the workflow reported success.
- **`test_stacklevel_points_at_the_caller` failed on Windows** over a
  drive-letter case difference (`c:` against `C:`), which said nothing about
  the stacklevel it asserts.

## [0.6.1] - 2026-08-29

### Changed

- Migrated the documentation build from Sphinx to MkDocs, with Material theme
  navigation and mkdocstrings API reference pages.
- Updated repository documentation links to point readers at the published
  GitHub Pages site.

## [0.6.0] - 2026-08-18

### Added

- `noise_sensitivity_diagnostic` reports `n_flipped`, the number of labels
  actually changed, so applied noise can be checked rather than assumed.
- `flip_labels` relabels selected points to a *different* class, uniform over
  the alternatives.

- **Documentation is published to GitHub Pages** at
  https://diogoribeiro7.github.io/OversampleQA/ on every push to `main`. The
  deploy job publishes the artifact the existing `-W` docs build produced
  rather than rebuilding, so what ships is the output that passed the gate. The
  artifact is uploaded on pull requests too, which proves it is publishable
  before merge; only deployment is restricted to `main`.
  The `documentation` URL in the package metadata pointed at the raw `.rst`
  source tree on GitHub and now points at the rendered site.
- `infer_minority_label` returns the least frequent label, and both dataset
  catalogs now derive `minority_label` from the data instead of declaring it.

### Removed

- **`recommended_samples`**, which answered a question nobody asks. Its effect
  size was `|mean| / std` — the effect for testing whether the error rate
  differs from *zero*. Error rates are never near zero relative to their spread,
  so the effect is always enormous and the column reported **1 on every row of
  every real run**. The sampling unit was confused too: the values are per-fold
  rates, but the formula carries the two-sample factor. It will come back when
  it is tied to a stated hypothesis — most usefully "how many folds to detect a
  difference *between two samplers*" — and a defined sampling unit.

### Fixed

- **The permutation tests rejected a true null every time on SMOTE output.**
  `nn_two_sample_test`, `mst_two_sample_test` and `cross_match_test` permute
  point labels, which assumes exchangeability — but points sharing a SMOTE
  parent move together, so the null was far too tight. Measured with synthetic
  and real drawn from the *same* distribution, differing only in the block
  structure: **100% false rejections at α=0.05**, median p 0.005. The tests were
  responding to the clustering SMOTE always produces, not to any difference in
  distribution, so they would call any SMOTE output "distinguishable from real"
  however good it was. Passing `parents` now subsamples one point per parent —
  genuinely exchangeable — across `n_subsamples` draws, combining p-values as
  twice the median (valid under arbitrary dependence). Same run block-aware: 0%
  false rejections, median p 0.885. The combination is conservative and power
  drops to the parent count, which is the honest sample size.
- **`noise_sensitivity_diagnostic` applied half the noise it reported.**
  Replacement labels were drawn from *all* classes, so a selected point could
  be "flipped" to the label it already had. Realised noise was
  `requested * (k - 1) / k` — on binary data, this package's main case, exactly
  half: a run labelled `noise=0.3` applied about 0.15, and the x-axis of every
  noise-sensitivity plot was overstated by 2×. Replacements now exclude the
  original label, so requested and realised agree exactly.
- **`cosine` reported a zero vector as identical to every vector.** With a zero
  norm the denominator vanishes and the angle is undefined; the code returned
  `0.0`, which says "identical direction". That is the identity of
  indiscernibles broken in the same way the original Hassanat defect broke it.
  It now raises. `cosine(x, x)` on a constant vector also returned
  **−2.22e-16** — a negative distance feeding a `nearest_hidden < nearest_min`
  comparison — and is now clamped to exactly zero.
- **`correlation` reported a constant vector as perfectly correlated.** A
  constant vector has zero variance, so the correlation is `nan`; the code
  caught the `nan` and returned `0.0`. `METRIC_DOMAINS` had documented the case
  as undefined all along — the code disagreed with the comment beside it. It now
  raises, as does a vector too short to have variance.
- Deterministic degeneracy tests for zero and constant vectors. The axiom smoke
  check draws random input, so it essentially never draws exactly these, which
  is how all three defects survived it.
- **`mahalanobis` was silently Euclidean.** With no `cov_inv` it fell back to an
  identity covariance — which *is* Euclidean distance, so it reported one metric
  under another's name. It was in the advanced benchmark's default metric list,
  so every default run produced a third set of rows byte-identical to the
  euclidean ones: euclidean counted twice in the per-metric rankings, and one
  pairwise comparison split into two for the multiple-comparison correction. It
  now raises, in both the scalar and vectorised paths, and is no longer a
  default in `run_comprehensive_benchmark` or the config templates. A test
  asserted the fallback; it asserted a bug, and now asserts the fix.
- **Sample-level metrics were accepted for pointwise validation.** `energy` and
  `wasserstein` compare two *distributions*; applied to a single pair of points
  they treat each point's own coordinates as the sample, so feature identity
  disappears — `[0, 5]` and `[5, 0]` score **0.0** under wasserstein against
  7.07 under euclidean, and `energy` returns **−5.0**, a negative distance fed
  straight into a `nearest_hidden < nearest_minority` comparison. Neither
  raised; both produced plausible error rates (0.53 and 0.78 against hassanat's
  0.50 on the same run). `validate_oversampling`,
  `validate_multiclass_oversampling` and `null_error_rate` now reject them,
  using the domains already declared in `METRIC_DOMAINS`. `distance_matrix`
  still computes them, since comparing whole samples is what they are for.
- **A registered metric plugin could not actually be used.** `distance_matrix`
  and every validator funnelling through it checked only the built-in table, so
  a plugin metric was rejected as unsupported by the exact functions it exists
  to be used by — while `PluginManager.get_metric` returned it happily.
  `resolve_metric` now consults built-ins and the plugin registry, per call
  rather than at import, since plugins register at runtime. The typed
  validator's config model rejected them too, before any validation could run.
  Unknown metric names now list the built-ins and say how to register a plugin.
- **Pairwise p-values and effect sizes were computed from the wrong metric.**
  `_add_statistical_analysis` grouped by dataset alone, so a dataset evaluated
  under several metrics gave a slice with several rows per oversampler; the
  lookups took `.iloc[0]`, ran the tests on whichever metric sorted first, and
  stamped that single result onto every row — including rows for the other
  metrics, whose error rates are not comparable with it. Grouping is now by
  (dataset, metric).
- **The effect size ignored the pairing its p-value depended on.** Wilcoxon
  signed-rank pairs the two samplers fold by fold; the reported effect was
  independent-samples Cohen's d, which pools both standard deviations and
  discards that pairing. Where folds move together — an awkward fold is awkward
  for both samplers — between-fold variance dominates the pooled deviation and
  the effect looks far smaller than the test says: measured −0.573 against a
  matched-pairs rank-biserial of −1.0. Replaced with the rank-biserial
  correlation, computed from the same signed ranks Wilcoxon sums.
- **`compute_ranking` averaged error rates across incomparable experiments.**
  Rates are commensurable only within a fixed (dataset, hidden_ratio, metric):
  an easy dataset scores near 0.1 and a hard one near 0.9, and hassanat scores
  roughly twice euclidean on the same data. Pooling them did not merely blur the
  ordering, it inverted it — given a sampler beating another on *every* dataset
  while having more runs skipped on the hard one, the pooled mean favoured the
  loser. Ranking is now done within each experiment and the ranks averaged,
  which is the Demšar protocol and matches `friedman_nemenyi` exactly, so the
  headline ordering and the significance test cannot disagree. Adds `mean_rank`
  and `n_specifications`, and warns when samplers were ranked over different
  numbers of experiments.
- **`DatasetRepository` declared the wrong minority class for
  `load_breast_cancer`.** It said `1`, but the minority is class 0 — 212
  malignant against 357 benign — so every benchmark using it oversampled the
  majority. `max_samples` makes a fixed answer impossible rather than merely
  wrong: the first 200 rows are 104 class-0 against 96 class-1, inverting which
  class is rarer.
- **`moons` and `circles` were exactly balanced.** A balanced dataset has
  nothing to oversample, so SMOTE produced no synthetic points and both
  contributed `nan` to every benchmark they appeared in. They are now
  imbalanced, with the subsample recorded in their provenance.
- **No built-in dataset could produce a measurement at the package's own
  defaults.** `hidden_ratio=0.1` against `min_hidden=5` needs 50 minority
  points; the largest built-in had 20, so `load_standard_datasets` returned
  `nan` for every row — the documented starting point could not demonstrate the
  documented workflow. The datasets are scaled up, keeping their imbalance
  ratios and seeds. A test now asserts the catalog yields no `nan` at defaults.

## [0.5.1] - 2026-08-17

### Added

- `macro_error_rate` averages per-class rates over the classes actually
  measured. A plain mean propagates the `nan` from an unmeasured class, and
  treating that `nan` as zero would read as a perfect score for the class that
  was never evaluated.
- `validate_multiclass_oversampling` accepts `min_hidden`.

- PyPI release preparation: package metadata now includes project URLs,
  classifiers and keywords; install docs use `pip install oversampleqa`; and
  `.github/workflows/publish.yml` builds and publishes distributions through
  PyPI Trusted Publishing when a GitHub release is published.
- `scripts/release.py` is now a local release-preparation check that runs the
  quality gates, builds the source distribution and wheel, and verifies package
  metadata. Uploading is handled by the GitHub release workflow rather than a
  local `twine upload` token.

- `tests/test_release_metadata.py` checks the release-facing metadata as one
  unit: the version across `pyproject.toml` (both blocks), `__init__.py`,
  `CITATION.cff` and the BibTeX in `README.md` and `docs/citation.rst`; the
  `date-released` against the dated CHANGELOG heading; that the current version
  has a recorded DOI; that the README badge uses the concept DOI while the
  "cite this exact release" line uses the version DOI; and that author,
  affiliation, ORCID and licence agree between `CITATION.cff` and
  `.zenodo.json`. It runs on every commit, because drift is introduced between
  releases and would otherwise surface only during one.
- CI enforces `ruff check src tests` and `mypy src`. Both were claimed clean by
  the README and the roadmap, and neither ran in CI.

- `StatisticalBenchmark.fold_results()` returns one row per attempted fold —
  dataset, oversampler, metric, repeat, fold, split seed, hidden ratio, error
  rate, skip status and skip reason. The summary frame reports a mean and
  interval per (dataset, oversampler, metric), which is enough to read a
  ranking and not enough to check one: it cannot be re-aggregated, plotted as a
  distribution, or given a different interval.
  Skipped folds are kept with `nan` and a reason rather than dropped, because a
  mean over three surviving folds of twenty-five is indistinguishable from a
  mean over twenty-five once the skips are gone. A combination whose folds all
  skip produces no summary row at all; the fold frame still shows every attempt.
  `split_seed` lets a single repeat be reproduced without rerunning the sweep.

- `run_benchmark` records the distance metric as a column. It identifies a
  measurement rather than merely parameterising one: without it, concatenating
  two sweeps run under different metrics gives a frame whose rows cannot be
  told apart, and error rates are not comparable across metrics — on one
  dataset hassanat scores roughly twice euclidean.
- `export_benchmark_results` gains an `html` format, so CSV, JSON, Markdown and
  HTML all render the same ranking frame through one code path.

- Every dataset from `advanced_benchmark.DatasetRepository` now carries a
  `provenance` record — source, generator, params, url, license and notes —
  matching what `benchmark.load_standard_datasets` already did. The two
  catalogs previously disagreed: one described where its data came from and the
  other returned bare arrays. The shared helpers live in
  `oversampleqa._provenance`, so there is one record shape rather than two, and
  a test asserts the catalogs agree.
  `max_samples` truncation is now disclosed as the positional slice it is, and
  `breast_cancer` is recorded as bundled real-world data with its UCI licence
  rather than lumped in with synthetic data.

### Fixed

- **Multiclass validation accepted a resampler that deletes original rows.**
  Synthetic points are identified positionally, so `SMOTEENN` and `SMOTETomek`
  produced a misaligned slice and plausible-looking numbers from it. The binary
  path has always refused this; the guard is now shared by both.
- **Multiclass validation enforced no minimum hidden-set size.** A class whose
  hold-out rounded to zero got an empty reference and silently stopped being a
  reference for any other class, so no synthetic point could ever be attributed
  to it.
- **Ties went to the lowest label index.** Attribution used a strict `<` against
  a running minimum over classes in label order, so a synthetic point
  equidistant from its own class and another was attributed to whichever sorted
  first. On quantised features this decided **38.6%** of attributions. A tie now
  goes to the point's own class, matching `score_nearest_distances`, where a tie
  is not evidence of an error.
- **A class the sampler generated nothing for scored `0.0`.** Nothing was
  measured, and `0.0` is the score of a perfect result. It is now `nan`.

- **The null calibrated a different experiment from the one it was calibrating.**
  `null_error_rate` scored held-out minority points against `fit_minority`,
  roughly 90% of the minority, while `validate_oversampling` scores synthetic
  points against the held-out 10%. A denser reference gives closer nearest
  neighbours and fewer errors, so the null sat at 0.033 where the same
  experiment gives 0.133 — a bar four times too low, against which ordinary
  samplers were reported as significantly worse than ideal. Calibration now
  splits the minority three ways: notional training data, a common reference
  shared by observed and null, and the real null candidates.
- **Ceiling candidates could be the very points they were scored against.**
  They were drawn from the full majority, which contains the hidden majority
  used as the reference; a candidate in the reference sits at distance zero
  from it and is an error by construction. Measured at 8.8% of candidates, with
  64% of draws affected. They are now drawn from the visible majority only.
- **`NullCalibration.interpret()` called a rate below the null interval
  "within" it.** The check was `observed <= high` with no lower bound. Below the
  interval is a distinct outcome — better than real held-out minority points
  score, which usually indicates memorisation rather than quality — and is now
  reported as such.

- **`mypy src` did not run at all.** `[tool.mypy] python_version = "3.10"` made
  mypy parse third-party stubs under 3.10, and numpy's stubs use PEP 695 `type`
  statements, so every run died with a syntax error before checking a single
  source file. Nothing noticed because CI never ran mypy. Removing the pin also
  surfaced two real `arg-type` errors in `plugin_system._register_module` that
  the older mypy had never reported. The 3.10 floor is enforced by the CI test
  matrix and ruff's `target-version`.
- The README cited version 0.3.0 in its BibTeX block and offered 0.3.0's DOI as
  "this exact release", two releases after the fact.

- A fold that produced `nan` — the sampler generated no synthetic points — was
  dropped with no warning at all, so it vanished from both the mean and the
  count backing the interval. It is now recorded as a skipped fold with that
  reason.
- `docs/benchmarking.rst` still documented the pre-0.3 confidence-interval
  defect as current behaviour, and named `_collect_fold_errors`, which does not
  exist. It now describes the t-interval used at every sample size and keeps
  the old behaviour as historical context.

- **`export_benchmark_results(fmt="markdown")` did not emit Markdown.** It used
  `to_csv(sep="|")` — no header separator row, no edge pipes — the same defect
  fixed in `report.py` for 0.4.0. It survived here because the renderer was
  duplicated rather than shared; both now come from `oversampleqa._render`.
- `run_benchmark` returned a `(0, 0)` frame when given no datasets, so a caller
  correctly handling "no results" still hit `KeyError` on any column access.
  The column order is now fixed even when empty.
- `docs/requirements.txt` pinned `sphinx==8.1.3` and `matplotlib==3.10.9` and
  was installed after `poetry install`, so the docs job built against two
  packages the lock did not govern. The docs toolchain moved into a locked
  poetry group and the file is gone.

## [0.5.0] - 2026-08-16

### Added

- **Entry-point plugin discovery.** `PluginManager.discover_entry_points()`
  registers metrics and validators advertised by any installed distribution
  under the `oversampleqa.metrics` and `oversampleqa.validators` groups. A
  plugin that fails to import or fails a check no longer blocks the rest: each
  failure warns with the entry point and the reason, since a plugin that
  silently fails to register looks exactly like one that was never installed.
  `strict=True` raises instead, for use in plugin test suites.
- **A reference plugin** in `examples/plugins/`: an installable package with one
  custom metric, one custom validator, its own README and 13 tests, wired into
  CI. Documented in `docs/plugins.rst`.
- `deprecated()` decorator, mechanising the policy `docs/api_stability.md`
  already promised: the warning names both the replacement and the removal
  version, which is the detail hand-written deprecations forget. Works on
  functions, methods and classes, appends a `.. deprecated::` note to the
  docstring, and raises against the caller's frame — Python hides
  `DeprecationWarning` outside `__main__` and per-module filters key on the
  reported location, so a warning that looks like it came from inside the
  package is invisible to whoever needs to act on it. Nothing is currently
  deprecated; a test asserts that, so the claim cannot rot.
- `plot_fidelity_radar` compares samplers across the fidelity suite on one
  radar chart. Every axis is oriented so outward is better — boundary safety is
  plotted as the complement of the violation rate — because a radar is read by
  the area of its polygon and axes that disagree on direction make that area
  meaningless. Density and diversity are unbounded above, so they are clipped at
  1.0 and any clipped value is named in a caption; `nan` stays `nan` and draws a
  gap rather than being flattened to zero.

### Fixed

- **`register_validator` silently overwrote on name collision.** It was a bare
  dictionary assignment, so two plugins claiming the same name left whichever
  loaded last in place and said nothing — and under entry-point discovery the
  load order is not something either author controls. It now refuses, as the
  metric path already did, and rejects a class with no callable `validate`.

## [0.4.0] - 2026-08-16

### Added

- `generate_report` accepts `fidelity_reports`, appending a fidelity and
  diversity section with the interpretation notes. A report carrying only the
  error rate omits the axis that usually decides which sampler to use.
- `frame_to_markdown` renders a DataFrame as a real Markdown table.
- `oversampleqa fidelity` subcommand, surfacing the fidelity/diversity split
  without writing Python. Reports precision, recall, density, coverage, the
  memorisation ratio and boundary violations, with `--utility` for downstream
  gain and `-o` for JSON.
- Gallery example `examples/fidelity_memorisation.py`, showing the case the
  fidelity module exists for: three of four metrics cannot separate SMOTE from
  RandomOverSampler, and only the memorisation ratio does.

### Fixed

- **The Markdown report was not Markdown.** It used `to_csv(sep="|")`, which
  produces no header separator row and no leading or trailing pipes, so it
  rendered as one run-on paragraph rather than a table in any viewer. Floats
  were also unrounded, putting values like `0.21000000000000002` into a
  document meant to be read.
- **`StatisticalBenchmark` returned an empty frame with no columns and no
  explanation when every fold was skipped.** A minority class small enough that
  splitting it into folds leaves too few points to hold out makes every fold
  raise; those were caught and warned per fold, then the combination was
  dropped silently. With enough datasets and samplers the per-fold warnings run
  to hundreds, and the return value said nothing. The frame now keeps its nine
  columns when empty — a `(0, 0)` frame raises `KeyError` on any column access,
  so a caller that correctly handles "no results" still broke — and one summary
  warning names the dropped combinations and the likely cause.

## [0.3.0] - 2026-08-16

### Added

- **`oversampleqa.fidelity`: separates "realistic" from "diverse".** The error
  rate is one scalar covering two failures that need opposite fixes —
  generating implausible points, and merely copying the training minority.
  `RandomOverSampler` is the clean case: it scores an error rate comparable to
  SMOTE's (0.242 vs 0.204) while contributing no information at all, and only
  the memorisation ratio separates them (0.000 vs 0.401).
- `precision_recall_density_coverage` implements the k-NN manifold estimators.
  Density and coverage are the more reliable pair: precision saturates at 1 and
  one real outlier with a large sphere can certify every synthetic point, while
  density counts spheres and keeps resolving past that.
- `sweep_k` recomputes across several `k`, because these metrics are k-sensitive
  and reporting one value would repeat the error rate's original sin.
- `memorisation_report`, whose headline is the ratio of median distance-to-training
  against median nearest-neighbour distance *within* the real minority. The
  denominator is the natural spacing of real data, so the number is scale-free.
- `boundary_violation_rate` in strict and graded form. It needs no hold-out, so
  it still reports when the minority is too small for the hold-out guard. It does
  not overlap `noise_sensitivity_diagnostic`, which measures response to injected
  label noise.
- `downstream_utility` (TSTR) via `imblearn.pipeline.Pipeline`, so the sampler
  runs **inside** each fold. A test builds the leaky version deliberately and
  asserts it scores higher, pinning the correct construction by evidence.
  Defaults to average precision, never accuracy.
- `fidelity_report` returning one dataclass with `to_dict()`, `to_frame()` and
  `interpret()`, and `docs/fidelity.rst` with the interpretation table.

- **`oversampleqa.inference`: the error rate now has a reference scale.**
  `null_error_rate` scores *real held-out minority points* through the identical
  pipeline, which is the rate an ideal generator — one drawing from the true
  minority distribution — would achieve. A ceiling from majority-drawn points
  bounds the other end. An observed rate is reported as a z-score, a percentile
  and a position on that scale, turning "0.13" into "0.13 against a null of
  0.11 ± 0.02, indistinguishable from ideal".
- Three nearest-neighbour two-sample tests, all with permutation p-values:
  Schilling–Henze (`nn_two_sample_test`), Friedman–Rafsky MST
  (`mst_two_sample_test`) and a greedy cross-match (`cross_match_test`). They
  answer whether synthetic points are distributionally indistinguishable from
  real ones. The pooled distance matrix is computed once and permutations only
  relabel. Any registered metric works, so `hassanat` composes with them.
- `error_rate_interval` with a **parent-block bootstrap**. A binomial interval
  assumes independent trials; synthetic points sharing a SMOTE parent are
  strongly dependent, so the effective sample size is nearer the parent count.
  On 200 points from 40 parents the block interval is **2.3× wider** than
  Wilson.
- `friedman_nemenyi` plus `plot_critical_difference`: the Demšar (2006) protocol
  for comparing methods across datasets, which is the benchmark's actual
  question and was absent.
- Benjamini–Hochberg (FDR) correction alongside Holm and Bonferroni.
- `oversampleqa validate --calibrate` prints observed, null and ceiling.
- `docs/inference.rst`, including an explicit *What these p-values do not tell
  you* section.


- `safety_factor` (default 0.8) on `OptimizedDistanceMatrix`: the fraction of
  the limit a batched computation plans against, leaving headroom for allocator
  overhead the analytic estimate does not model.
- A `performance` extra (`pip install 'oversampleqa[performance]'`) for `psutil`
  and `tqdm`. Without `psutil`, available memory is assumed to be 1 GB whatever
  the machine has — now logged once at INFO and documented, since it silently
  changed behaviour depending on an optional dependency.
- `scripts/profile_performance.py` records platform, Python, NumPy and CPU count
  alongside timings, warns when a baseline was recorded elsewhere, and uses a
  median of five runs rather than a single measurement.
- A scheduled, explicitly non-blocking performance workflow. Timing checks on
  shared runners produce false failures and train people to ignore red builds;
  the reasoning is stated in the workflow file.

- **`import oversampleqa` created a directory in the working directory.**
  `distance.py` built a `ValidationCache` at module scope and
  `ValidationCache.__init__` called `mkdir`, so merely importing the package
  created `.oversampleqa_cache` wherever the process happened to be — a home
  directory, a repo root, a container's `/` — and then accumulated pickled
  distance matrices there indefinitely. Nothing is constructed at import time
  now, and the directory is created lazily on first write.
- `lru_cache` on an instance method held a strong reference to `self` forever,
  was shared across every instance of the class, and was fed unhashable
  arguments through a mutable `_distance_args` side channel that two threads
  could interleave. On a cache hit the stored arguments were popped without
  being read, so a hit could return a matrix computed with a different
  `batch_size`. Replaced with a lock-guarded LRU keyed on the content hash.
- The joblib disk key hashed the `OptimizedDistanceMatrix` instance, so the key
  depended on optimizer state that cannot affect the result, and any
  locally-defined or `lambda` plugin metric hashed unstably. The optimizer is no
  longer part of the key.
- `memory_mb` was converted to `bytes_limit` and never read — the store grew
  without bound. Eviction is now enforced, oldest-first, against both a byte
  budget and an entry count, with `cache.size_bytes` and `cache.clear()`.
- Cached arrays were returned by reference, so one in-place operation downstream
  would silently corrupt every later hit. They are returned read-only.
- `batch_size` was dropped on the no-cache path of `compute_distance_matrix`, so
  an explicit batch size was ignored whenever caching was off.


- `random_state` on `validate_oversampling`, `validate_multiclass_oversampling`,
  `MemoryEfficientValidator`, `TypedValidator` and the benchmark runners,
  defaulting to `42`. It accepts an `int`, `Generator`, `SeedSequence` or `None`,
  normalised by a single shared helper in `oversampleqa._rng` so seeding cannot
  drift between validators again. Previously the seed governing the dominant
  source of variance was hard-coded and could not be set at all.
- `n_repeats` on `validate_oversampling`. Above 1 it draws independent hold-out
  splits and returns `rates`, `mean`, `std` and a percentile bootstrap
  `interval` on `ValidationDetails`. Repeat streams are spawned from a
  `SeedSequence` rather than derived as `seed + i`, which would correlate them.
- `reseed_oversampler` to give the sampler a fresh seed per repeat, so the
  dispersion covers the sampler's randomness as well as the split's.
- `stratify_by` on `validate_oversampling`, taking the hold-out fraction within
  each group so a split cannot miss a cluster entirely. Strata are never
  inferred.
- `--random-state` and `--n-repeats` on `oversampleqa validate`, both present in
  the `production` and `research` config templates. The rich summary shows
  mean ± sd and the interval when `n_repeats > 1`.


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


- `hassanat` now has a vectorised kernel in
  `OptimizedDistanceMatrix._vectorized_dispatch`. It was previously absent, so
  the default metric always fell through to a Python double loop calling the
  metric once per pair.
- Distance-metric audit: every metric in the registry is now checked against
  SciPy or an independent closed form, not merely against itself. SciPy is a
  new **dev-only** dependency for this.

### Changed

- **The Wald confidence interval is now Wilson.** Wald is symmetric about the
  estimate, so at an error rate near zero it collapses to zero width and can
  extend below zero. Error rates near zero are the common case here. The
  docstring records that both assume independent Bernoulli trials, which
  synthetic points sharing parent points are not.
- `ValidationMode.ASYNC` raises a `ConfigurationError` explaining that
  validation is CPU-bound NumPy, so an event loop offers it no concurrency, and
  directing async callers to `await validate_async(...)`.
- **Tooling consolidated on ruff.** `black`, `isort` and `flake8` are removed
  from dev dependencies and the pre-commit hooks; `ruff format` and
  `ruff check` cover the same ground in one tool with no rule conflicts.
- Ruff had no configuration at all, so the project inherited whatever the
  installed version defaulted to — which is how ~270 findings accumulated that
  nobody was expected to clear. The rule set is now explicit and every exemption
  carries a reason. `ruff check src tests` exits clean.
- `mypy src` exits clean. The numerical core is fully strict with no exemptions;
  `cli_enhanced`, `plotting` and the benchmark modules carry scoped per-code
  relaxations naming the untyped third-party boundary responsible, so a new
  class of error there still fails.
- The `Makefile` works on Windows: `clean` is now `scripts/clean.py` and `docs`
  invokes `sphinx-build` directly instead of a nested Unix-only `make`. Windows
  is in the CI matrix.
- Removed 33 orphaned autosummary stubs from `docs/api/` that duplicated the
  hand-written module pages, along with the `exclude_patterns` entry that
  existed only to keep them out of the build.


- **Caching is opt-in.** `distance_matrix` takes `cache=...`; without it nothing
  is written to disk. The default directory is the platform per-user cache
  directory (`platformdirs`, added as a runtime dependency with a
  `~/.cache/oversampleqa` fallback), never the working directory.
- `ValidationCache` documents its concurrency contract: safe through one
  instance, not process-safe.


- **The hold-out split now uses a `Generator` permutation instead of
  `train_test_split`.** This makes the binary and multiclass paths structurally
  identical and lets them accept a `Generator`, which scikit-learn's splitter
  does not. A consequence: the same seed selects different points than before,
  so **error rates from before this release are not reproducible with
  `random_state=42`**, even though 42 remains the default.
- `run_benchmark` varies the hold-out split per run. It previously reseeded only
  the oversampler, so every one of its `n_runs` repetitions shared a single
  hold-out split and the spread across runs omitted the largest variance source.
- `ValidationConfig.random_state` now defaults to `42` rather than `None`, and
  gained `n_repeats`.

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


- `energy` and `wasserstein` are documented as **sample-based** metrics rather
  than point metrics, in their docstrings and in `docs/distances.rst`. They
  treat the input vector as a set of observations, not as a point in feature
  space, so they answer a different question from the rest of the registry.
- The reference benchmark in `docs/benchmark_results.rst` now seeds the
  oversamplers. Without that, the "pinned" run was not reproducible: the same
  configuration produced 0.003894, 0.003894 and 0.003186 on three consecutive
  trials, because `StatisticalBenchmark(random_state=...)` seeds the
  cross-validation splits but not the oversampler's own sampling.

### Fixed

- **`wasserstein_1d_distance` did not compute the 1-D Wasserstein distance.**
  It advanced the CDF before adding each interval's contribution, crediting
  every interval with the value from *after* its right-hand jump. `[0, 1]` vs
  `[0, 3]` returned 0.5 where the true W₁ is 1.0. It now matches
  `scipy.stats.wasserstein_distance` on 300 random unequal-length trials. This
  also unblocked the vectorised kernel, which could not be registered while the
  scalar was wrong; `energy` is now the only metric without one.
- **`StatisticalBenchmark._confidence_interval` returned two different
  quantities.** Below 30 observations a Student-t interval for the mean; at 30
  or more the 2.5th–97.5th percentiles of the observations, which describe
  spread and do not narrow with more data. Both went into the same
  `ci_lower`/`ci_upper` columns, so on σ = 0.05 data the width jumped 4.7× from
  one extra observation. It is a t-interval for the mean at every size.
- **Holm correction was not monotone.** It scaled each p-value by its rank
  without a running maximum, so `[0.01, 0.02, 0.03]` corrected to
  `[0.03, 0.04, 0.03]` — the least significant comparison came out more
  significant than the middle one.
- `scipy` moved from a dev dependency to a runtime one. `advanced_benchmark`
  imports `scipy.stats` at module level, so the package could not be imported
  without it; declaring it test-only was wrong.

- **`ServiceRegistry.get` raised `ConfigurationError`, which was never defined
  or imported.** The error path itself raised `NameError`, replacing a clear
  "not registered" message with a confusing one. The hierarchy now lives in
  `oversampleqa.exceptions` (`OversampleQAError` and its subclasses), is
  exported from the package root, and `oversampleqa.types` re-exports it so
  existing imports keep working.
- `TypedValidator._wald_confidence_interval` annotated `Tuple` without importing
  it — a latent `NameError` surfacing only through `typing.get_type_hints`,
  Pydantic, or Sphinx autodoc.
- `validation_session` had `return` inside `finally`, which swallowed any
  exception raised inside the session body.
- `ValidationMode.ASYNC` called `run_until_complete`, which raises whenever an
  event loop is already running — Jupyter and every async host.


- **Memory estimation ignored the intermediates that actually drive peak use.**
  It counted only the `(n1, n2)` output, so the check passed and the kernel then
  allocated far more than the check permitted — bypassing the very batching
  logic meant to prevent it. Peak is now modelled per metric as
  `output × (flat + per_feature × d)`. Two terms are needed because the kernels
  split into families: `euclidean` peaks at ~3× the output regardless of `d`,
  while `hassanat` peaks at ~96× at d=16. The table is fitted to `tracemalloc`
  measurements, and two entries are re-verified by a test on every run.
- `_auto_batch_size` let every batch consume the entire memory limit, leaving no
  headroom for the result array that lives for the whole computation. It now
  reserves the output first and applies the metric's multiplier.


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

### Performance

- **Vectorised five more kernels.** `hamming`, `jaccard`, `hellinger` and
  `jensen_shannon` fell through to a Python double loop calling the metric once
  per pair. Measured at 200×400, d=20: hamming 105×, jaccard 128×, hellinger
  164×, jensen_shannon 53×. The default metric `hassanat` is 16× faster at
  500×5000 (22.6 s → 1.4 s).
- `energy` deliberately stays on `_pairwise`: broadcasting it needs an
  `(n1, n2, d, d)` intermediate, larger than the work it saves.
- `wasserstein` also stays on `_pairwise`, for a different reason. A correct
  kernel exists but the *scalar* implementation is wrong (its CDF walk drops the
  tail), so registering the kernel would make the two paths disagree and
  matching them would mean reproducing the bug.

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

[Unreleased]: https://github.com/DiogoRibeiro7/OversampleQA/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.6.1
[0.6.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.6.0
[0.5.1]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.5.1
[0.5.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.5.0
[0.4.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.4.0
[0.3.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.2.0
[0.1.0]: #010---2025-10-10
