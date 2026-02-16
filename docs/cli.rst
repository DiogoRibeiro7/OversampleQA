Command Line Interface
======================

OversampleQA includes a CLI for quick validation runs.

Basic usage
-----------

.. code-block:: bash

   oversampleqa-validate data.csv --target target --minority-label 1 --oversampler SMOTE

Options
-------

* ``--target``: name of the target column.
* ``--minority-label``: minority label value.
* ``--oversampler``: imbalanced-learn oversampler class name.
* ``--hidden-ratio``: fraction of majority samples to hide.
* ``--distance``: distance metric name.
* ``--out``: optional text report output path.
* ``--plot``: optional plot output path.

Examples
--------

Save a report and a PCA plot:

.. code-block:: bash

   oversampleqa-validate data.csv --target target --minority-label 1 --out report.txt --plot plot.png
