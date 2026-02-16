Installation
============

OversampleQA works with Python 3.10+ and depends on NumPy, scikit-learn, and imbalanced-learn.

pip
---

.. code-block:: bash

   pip install oversampleqa

poetry
------

.. code-block:: bash

   poetry add oversampleqa

From source
-----------

.. code-block:: bash

   git clone https://github.com/your-org/oversampleqa.git
   cd oversampleqa
   poetry install

Optional dependencies
---------------------

UMAP-based plots and metrics require ``umap-learn``.

.. code-block:: bash

   pip install umap-learn

Troubleshooting
---------------

* If you see BLAS/LAPACK errors, install a compatible NumPy wheel for your platform.
* If you run into build issues on Windows, prefer installing via pip or poetry rather than source.
