Command Line Interface
======================

OversampleQA ships two CLI entry points:

- ``oversampleqa``: enhanced CLI with profiles, templates, and diagnostics.
- ``oversampleqa-validate``: legacy minimal CLI for quick CSV validation.

Enhanced CLI
------------

Basic usage
~~~~~~~~~~~

.. code-block:: bash

   oversampleqa --help

Validate a dataset
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa validate data.csv \
     --target target \
     --minority-label 1 \
     --oversampler SMOTE \
     --metric hassanat \
     --hidden-ratio 0.1 \
     --export json \
     --output runs

Tiny example dataset
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   printf "x1,x2,target\n1.0,0.5,0\n1.1,0.4,0\n0.9,0.6,0\n2.0,1.9,1\n2.1,1.8,1\n" > tiny.csv

   oversampleqa validate tiny.csv \
     --target target \
     --minority-label 1 \
     --oversampler SMOTE \
     --metric euclidean \
     --hidden-ratio 0.2

Run interactively
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa validate data.csv --interactive

Configuration profiles
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa profiles
   oversampleqa validate data.csv --profile quick

Generate a config template
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa template --template production -o oversampleqa.yaml

Benchmark multiple datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa benchmark --output benchmark_results

Shell completion
~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa completion bash

Diagnostics
~~~~~~~~~~~

.. code-block:: bash

   oversampleqa doctor

Initial setup
~~~~~~~~~~~~~

Run the guided wizard to create a configuration file:

.. code-block:: bash

   oversampleqa setup

Global options
~~~~~~~~~~~~~~

These apply to any subcommand and come before it:

- ``--config/-c``: path to a configuration file (default ``~/.oversampleqa/config.yaml``).
- ``--profile/-p``: configuration profile to apply.
- ``--verbose/-v``: enable verbose output.
- ``--version``: print the version and exit.

Common options for ``validate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--target``: target column name in the dataset.
- ``--minority-label``: minority class label value.
- ``--oversampler``: imbalanced-learn oversampler class name.
- ``--metric``: distance metric to use.
- ``--hidden-ratio``: fraction of majority samples to hide.
- ``--export``: output formats (``json``, ``yaml``, ``markdown``).
- ``--output``: directory to store outputs.
- ``--resume/--no-resume``: reuse cached results when available.
- ``--interactive``: guided validation wizard.
- ``--mlflow``: log results to MLflow if installed.

Legacy CLI (minimal)
--------------------

.. code-block:: bash

   oversampleqa-validate data.csv --target target --minority-label 1 --oversampler SMOTE

Options
~~~~~~~

- ``--target``: name of the target column.
- ``--minority-label``: minority label value.
- ``--oversampler``: imbalanced-learn oversampler class name.
- ``--hidden-ratio``: fraction of majority samples to hide.
- ``--distance``: distance metric name.
- ``--out``: optional text report output path.
- ``--plot``: optional plot output path.