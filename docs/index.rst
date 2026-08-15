OversampleQA Documentation
==========================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/

**OversampleQA** is a validation toolkit for oversampling methods in imbalanced classification.

Quick Start
-----------

.. code-block:: python

   from oversampleqa import validate_oversampling
   from imblearn.over_sampling import SMOTE
   from sklearn.datasets import make_classification

   # Create imbalanced dataset
   X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)

   # Validate SMOTE oversampling
   error_rate = validate_oversampling(
       X=X, y=y, minority_label=1,
       oversampler=SMOTE(random_state=42)
   )

   print(f"Error rate: {error_rate:.3f}")

Installation
------------

OversampleQA is not published on PyPI. Install it from source:

.. code-block:: bash

   git clone https://github.com/diogoribeiro7/OversampleQA.git
   cd OversampleQA
   poetry install

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   concepts
   user_guide
   algorithms
   distances
   metrics
   benchmarking
   inference
   benchmark_results
   plotting
   visual_examples
   cli
   configuration
   reproducibility
   api_landing
   api_reference
   tutorials
   examples
   gallery/index
   contributing
   citation
   bibliography
   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`