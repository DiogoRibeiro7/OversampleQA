Installation
============

OversampleQA works with Python 3.10+ and depends on NumPy, scikit-learn, and imbalanced-learn.

OversampleQA is not published on PyPI, so install it from the repository.

From source
-----------

.. code-block:: bash

   git clone https://github.com/diogoribeiro7/OversampleQA.git
   cd OversampleQA
   poetry install

As a dependency
---------------

To depend on OversampleQA from another project, install it straight from git:

.. code-block:: bash

   pip install git+https://github.com/diogoribeiro7/OversampleQA.git

With poetry:

.. code-block:: bash

   poetry add git+https://github.com/diogoribeiro7/OversampleQA.git

Optional dependencies
---------------------

UMAP-based plots and metrics require ``umap-learn``.

.. code-block:: bash

   pip install umap-learn

Troubleshooting
---------------

* If you see BLAS/LAPACK errors, install a compatible NumPy wheel for your platform.
* If you run into build issues on Windows, make sure a recent ``pip`` and build toolchain are available before running ``poetry install``.
