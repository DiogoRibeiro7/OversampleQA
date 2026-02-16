CLI Tutorial
============

This tutorial shows how to run validations from the command line.

Validate a CSV dataset
----------------------

.. code-block:: bash

   oversampleqa-validate data.csv --target target --minority-label 1 --oversampler SMOTE

Save outputs
------------

.. code-block:: bash

   oversampleqa-validate data.csv --target target --minority-label 1 --out report.txt --plot plot.png

Run the enhanced CLI
--------------------

.. code-block:: bash

   oversampleqa validate data.csv --target target --minority-label 1 --metric hassanat
