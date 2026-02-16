API Overview
============

This page groups the most commonly used APIs for quick discovery.

Validators
----------

.. autosummary::
   :nosignatures:

   oversampleqa.validate_oversampling
   oversampleqa.validate_multiclass_oversampling
   oversampleqa.extract_synthetic_samples
   oversampleqa.TypedValidator
   oversampleqa.MemoryEfficientValidator

Benchmarking
------------

.. autosummary::
   :nosignatures:

   oversampleqa.run_benchmark
   oversampleqa.load_standard_datasets
   oversampleqa.compute_ranking
   oversampleqa.StatisticalBenchmark
   oversampleqa.DatasetRepository
   oversampleqa.create_benchmark_report

Metrics
-------

.. autosummary::
   :nosignatures:

   oversampleqa.calculate_error_rate
   oversampleqa.confidence_ratio
   oversampleqa.local_density_divergence
   oversampleqa.minority_recall_loss
   oversampleqa.umap_manifold_distance
   oversampleqa.check_model_fairness
   oversampleqa.noise_sensitivity_diagnostic

Plotting
--------

.. autosummary::
   :nosignatures:

   oversampleqa.plot_sample_distribution
   oversampleqa.plot_error_comparison
   oversampleqa.plot_error_boxplot
   oversampleqa.plot_error_ranking
   oversampleqa.plot_error_heatmap
   oversampleqa.plot_noise_sensitivity
   oversampleqa.plot_distance_histogram
   oversampleqa.plot_class_balance

CLI
---

.. autosummary::
   :nosignatures:

   oversampleqa.cli.main
   oversampleqa.cli_enhanced.main
