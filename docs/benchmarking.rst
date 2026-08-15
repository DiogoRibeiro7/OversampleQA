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
``_collect_fold_errors`` runs ``n_repeats`` × ``n_folds`` stratified splits and
calls :func:`~oversampleqa.validate_oversampling` on each training fold. The
variance being summarised is therefore the variance of **fold composition** —
which rows land in each training fold — combined with the oversampler's own
randomness if it was constructed without a seed.

Since 0.3, the hold-out split inside each fold varies with the fold's seed.
Before that it was pinned at 42 for every fold, so the reported spread excluded
the single largest source of variance entirely.

.. warning::

   ``_confidence_interval`` computes **two different quantities** depending on
   how many observations it receives:

   - fewer than 30: a Student-t interval for the **mean**, which narrows as
     :math:`\sqrt{n}`;
   - 30 or more: the 2.5th–97.5th **percentiles of the observations**, which
     describes the spread of individual folds and does *not* narrow with more
     data.

   Both are emitted into the same ``ci_lower`` / ``ci_upper`` columns. Crossing
   the boundary changes the reported width abruptly — on normally distributed
   values with σ = 0.05, the interval jumps from a width of 0.036 at n = 29 to
   0.172 at n = 30, a factor of 4.7, purely from adding one observation.

   Do not compare intervals across configurations with different observation
   counts, and do not read the ≥ 30 case as a confidence interval for the mean.
