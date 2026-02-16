Plotting
========

Visualization helpers are available in :mod:`oversampleqa.plotting`.

Sample distribution
-------------------

.. code-block:: python

   from oversampleqa.plotting import plot_sample_distribution

   plot_sample_distribution(
       majority=X_majority,
       minority=X_minority,
       synthetic=X_syn,
       method="pca",
       save_path="pca_plot.png",
   )

Error heatmap for multiclass validation
---------------------------------------

.. code-block:: python

   from oversampleqa.plotting import plot_error_heatmap

   plot_error_heatmap(error_matrix, class_labels=[0, 1, 2], save_path="heatmap.png")

Noise sensitivity curve
-----------------------

.. code-block:: python

   from oversampleqa.plotting import plot_noise_sensitivity

   plot_noise_sensitivity(results_df, save_path="noise_curve.png")
