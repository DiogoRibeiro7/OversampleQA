Benchmarking
============

OversampleQA provides lightweight benchmarking utilities to compare oversamplers across datasets.

Quick comparison
----------------

.. code-block:: python

   from oversampleqa.benchmark import load_standard_datasets, run_benchmark
   from imblearn.over_sampling import SMOTE, RandomOverSampler

   datasets = load_standard_datasets()[:2]
   oversamplers = [SMOTE(random_state=0), RandomOverSampler()]
   df = run_benchmark(datasets, oversamplers, hidden_ratios=[0.1], n_runs=3)
   print(df.head())

Advanced statistical benchmarking
---------------------------------

.. code-block:: python

   from oversampleqa.advanced_benchmark import StatisticalBenchmark, DatasetRepository
   from imblearn.over_sampling import SMOTE, RandomOverSampler

   repo = DatasetRepository()
   datasets = repo.create_synthetic_benchmark_suite(["easy", "hard"])
   bench = StatisticalBenchmark(n_folds=3, n_repeats=2, random_state=0)
   df = bench.run_comprehensive_benchmark(
       datasets, [SMOTE(random_state=0), RandomOverSampler()], metrics=["hassanat"]
   )
   print(df[["dataset_name", "oversampler_name", "mean_error"]])

Reports
-------

.. code-block:: python

   from oversampleqa.advanced_benchmark import create_benchmark_report

   report_path = create_benchmark_report(df, output_path="benchmark_report.html")
   print(report_path)

What the statistical benchmark's intervals cover
------------------------------------------------

:class:`~oversampleqa.StatisticalBenchmark` reports confidence intervals,
p-values and effect sizes. Be precise about what they describe.

Each observation in those statistics is **one cross-validation fold**.
``_fold_records`` runs ``n_repeats`` × ``n_folds`` stratified splits and calls
:func:`~oversampleqa.validate_oversampling` on each training fold. The variance
being summarised is therefore the variance of **fold composition** — which rows
land in each training fold — combined with the oversampler's own randomness if
it was constructed without a seed.

Since 0.3, the hold-out split inside each fold varies with the fold's seed.
Before that it was pinned at 42 for every fold, so the reported spread excluded
the single largest source of variance entirely.

``ci_lower`` and ``ci_upper`` are a Student-t confidence interval **for the
mean**, at every sample size. It narrows as :math:`\sqrt{n}`, and intervals are
comparable across configurations with different fold counts.

.. note::

   Before 0.3 this switched formula at 30 observations: a t-interval for the
   mean below, and the 2.5th–97.5th percentiles of the observations at or above.
   Those are different quantities, written into the same two columns. On
   normally distributed values with σ = 0.05 the reported width jumped from
   0.036 at n = 29 to 0.172 at n = 30 — a factor of 4.7 from one extra
   observation — and intervals could not be compared across configurations.
   Results produced before 0.3 carry that defect; results since do not.

Inspecting individual folds
---------------------------

The summary is one row per (dataset, oversampler, metric). That is enough to
read a ranking and not enough to check one: it cannot be re-aggregated, plotted
as a distribution, or given a different interval, and it does not say how many
folds actually contributed.

:meth:`~oversampleqa.StatisticalBenchmark.fold_results` returns one row per
attempted fold::

    bench = StatisticalBenchmark(n_folds=5, n_repeats=5)
    summary = bench.run_comprehensive_benchmark(datasets, samplers)
    folds = bench.fold_results()

with ``dataset_name``, ``oversampler_name``, ``metric``, ``repeat``, ``fold``,
``split_seed``, ``hidden_ratio``, ``error_rate``, ``skipped`` and
``skip_reason``.

Skipped folds are **kept**, with ``error_rate`` of ``nan`` and a stated reason.
This matters more than it sounds: a mean over three surviving folds out of
twenty-five is indistinguishable from a mean over twenty-five once the skips
are dropped. Count them before trusting an interval::

    contributing = (~folds["skipped"]).groupby(folds["oversampler_name"]).sum()

A combination whose folds *all* skip produces no summary row at all — only a
warning. The fold frame still shows every attempt and why each one failed.

``split_seed`` is the seed handed to the fold splitter for that repeat, so a
single repeat can be reproduced without rerunning the sweep.
