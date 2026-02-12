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
