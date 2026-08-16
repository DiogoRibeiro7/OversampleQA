# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut by hand: publish a GitHub release from a tag, which is also what
triggers the Zenodo archive and its DOI. Entries below the *Unreleased* section are
maintained manually.

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/DiogoRibeiro7/OversampleQA/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.5.0
[0.4.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.4.0
[0.3.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/OversampleQA/releases/tag/v0.1.0
