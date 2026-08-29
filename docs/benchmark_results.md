# Benchmark Results

This page documents a small **reference benchmark run** that is
reproducible from a pinned seed and configuration. It is intended both
as a worked example of the statistical benchmarking API and as a
baseline you can re-run to check for regressions after upgrading
dependencies.

## Reproducing the reference run

The run below uses three of the built-in synthetic datasets, two
oversamplers, and two distance metrics, with five-fold cross-validation
repeated five times. The `~oversampleqa.StatisticalBenchmark` seed is
fixed at `42` and the built-in datasets are generated deterministically
(see `reproducibility`), so the structure of the results is stable
across runs.

``` python
from oversampleqa import StatisticalBenchmark, format_statistical_summary
from oversampleqa.benchmark import load_standard_datasets
from imblearn.over_sampling import SMOTE, RandomOverSampler

names = {"classification", "moons", "circles"}
datasets = [d for d in load_standard_datasets() if d["name"] in names]

engine = StatisticalBenchmark(n_folds=5, n_repeats=5, random_state=42)
frame = engine.run_comprehensive_benchmark(
    datasets,
    [RandomOverSampler(random_state=42), SMOTE(random_state=42)],
    metrics=["euclidean", "hassanat"],
)
print(format_statistical_summary(frame))
```

> [!IMPORTANT]
> Seed the oversamplers themselves, as above. `StatisticalBenchmark`'s
> `random_state` controls the cross-validation splits, not the
> oversampler's own sampling. Constructing `SMOTE()` without
> `random_state` draws from global NumPy state and the run is **not**
> reproducible: the same configuration produced mean errors of 0.003894,
> 0.003894 and 0.003186 on three consecutive trials.

The equivalent (whole-catalog) run from the CLI is:

``` bash
oversampleqa benchmark --statistical --folds 5 --repeats 5 -o benchmark_results
```

## Reference environment

The numbers below were produced with:

- Python 3.13
- numpy 2.3, scipy 1.15
- scikit-learn 1.6, imbalanced-learn 0.14

Exact error values can shift slightly with different dependency versions
because the oversamplers' internals change; the relative ordering and
the broad magnitudes are the stable signal.

## Reference output

``` text
## Dataset: circles

| Oversampler | Metric | Mean error | Std | CI | n |
| --- | --- | --- | --- | --- | --- |
| RandomOverSampler | euclidean | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| RandomOverSampler | hassanat | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| SMOTE | euclidean | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| SMOTE | hassanat | 0.000 | 0.000 | [0.000, 0.000] | 25 |

## Dataset: classification

| Oversampler | Metric | Mean error | Std | CI | n |
| --- | --- | --- | --- | --- | --- |
| RandomOverSampler | euclidean | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| RandomOverSampler | hassanat | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| SMOTE | euclidean | 0.001 | 0.002 | [-0.000, 0.002] | 25 |
| SMOTE | hassanat | 0.005 | 0.006 | [0.002, 0.007] | 25 |

## Dataset: moons

| Oversampler | Metric | Mean error | Std | CI | n |
| --- | --- | --- | --- | --- | --- |
| RandomOverSampler | euclidean | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| RandomOverSampler | hassanat | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| SMOTE | euclidean | 0.000 | 0.000 | [0.000, 0.000] | 25 |
| SMOTE | hassanat | 0.000 | 0.000 | [0.000, 0.000] | 25 |
```

## Interpreting the reference

Most of these synthetic datasets are easily separable, so the validation
error rate is effectively zero — the synthetic minority points are
nowhere near the hidden majority. The informative cell is **SMOTE on
the** `classification` **dataset with the Hassanat metric**, which
produces a measurable error rate of roughly `0.005` (mean 0.004602) with
a 95% confidence interval that excludes zero. That is the diagnostic
doing its job: it flags that, under this metric, a small but consistent
fraction of SMOTE's synthetic samples look majority-like.

> [!NOTE]
> These figures were regenerated after the Hassanat distance was
> corrected to match Hassanat (2014). The previous version of this page
> reported `0.059` for this cell, produced by an implementation that was
> not the Hassanat distance. The two numbers are **not comparable**.

Two takeaways:

- A near-zero error rate means the test data carries little overlap, not
  that an oversampler is necessarily "better"; compare methods on
  datasets where the error rate is non-trivial.
- The chosen metric matters. Euclidean reports ~0 on the same data where
  Hassanat surfaces a difference, so validating with more than one
  metric is good practice (see `concepts`).

## Artifacts

In `--statistical` mode the CLI writes three files to the output
directory:

- `benchmark_statistics.csv` — the full results frame (means, CIs,
  pairwise p-values and effect sizes).
- `benchmark_summary.md` — the Markdown summary shown above.
- `benchmark_report.html` — a standalone HTML report.
