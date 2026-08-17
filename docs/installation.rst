Installation
============

OversampleQA works with Python 3.10+ and depends on NumPy, scikit-learn, and imbalanced-learn.

From PyPI
---------

Install the latest release from PyPI:

.. code-block:: bash

   pip install oversampleqa

For the optional performance helpers:

.. code-block:: bash

   pip install "oversampleqa[performance]"

From source
-----------

Install from source when developing OversampleQA or testing unreleased changes:

.. code-block:: bash

   git clone https://github.com/diogoribeiro7/OversampleQA.git
   cd OversampleQA
   poetry install

Repository dependency
---------------------

To depend on the current repository state from another project, install it
straight from git:

.. code-block:: bash

   pip install git+https://github.com/diogoribeiro7/OversampleQA.git

With poetry:

.. code-block:: bash

   poetry add git+https://github.com/diogoribeiro7/OversampleQA.git

Troubleshooting
---------------

* If you see BLAS/LAPACK errors, install a compatible NumPy wheel for your platform.
* If you run into build issues on Windows, make sure a recent ``pip`` and build toolchain are available before running ``poetry install``.
