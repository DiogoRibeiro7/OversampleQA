# OversampleQA Roadmap

Last revised August 16, 2026, at v0.5.0.

## Project Goals

- Provide a practical, diagnostics-first toolkit for evaluating oversampling methods in imbalanced classification.
- Make synthetic sample quality measurable and comparable across methods, datasets, and metrics.
- Keep workflows reproducible with clear configuration, benchmarking, and exportable reports.
- Offer a fast path for practitioners and a deeper path for research use cases.
- Stay compatible with scikit-learn and imbalanced-learn conventions.

## Current State (v0.5.0)

- 427 tests in the main suite, plus 13 in the reference plugin, green on Linux
  and Windows across Python 3.10–3.12.
- `ruff check` and `mypy src` pass clean. The numerical core is fully strict;
  the CLI, plotting and benchmark modules carry scoped, documented relaxations
  at their untyped third-party boundaries.
- Docs build under `-W`, enforced in CI.
- `poetry.lock` is tracked, poetry is pinned, and `poetry check --lock` gates
  installs, so CI tests a fixed dependency set.
- Public API frozen behind a snapshot test; deprecation policy documented and
  mechanised.

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

## Open Work

Not scheduled against dates. This is a research toolkit maintained by one
person; the ordering below reflects priority, not a delivery commitment.

### Near-term

- **`docs/requirements.txt` pins `matplotlib==3.10.9`** and is installed after
  `poetry install`, so the docs job builds against a different matplotlib than
  the test jobs. Undermines the lockfile work.
- **Dataset provenance.** The built-in catalog has no licensing or provenance
  metadata. Datasets are fetched from OpenML at run time with no recorded
  version, so a "reproducible" benchmark is only as stable as OpenML.
- **Long-format benchmark export.** Results are one row per
  (dataset, oversampler, hidden_ratio, run) but statistics are spread across
  columns; a row per (dataset, oversampler, metric, repeat) would make grouping
  and joining trivial and remove bespoke reshaping from `report.py`.

### Later

- Real parallelism across repeats and datasets. `ValidationMode.ASYNC` raises
  rather than pretending: the work is CPU-bound NumPy, so an event loop offers
  no concurrency. Process-level parallelism is the honest version.
- Performance-regression tracking beyond the current scheduled, deliberately
  non-blocking workflow. Timing checks on shared runners produce false failures
  and train people to ignore red builds.
- A documented benchmark catalog with pinned reference results.

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
