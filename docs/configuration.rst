Configuration
=============

OversampleQA can be configured via typed Python configs and via the enhanced CLI config file.

Typed validation configuration
------------------------------

.. code-block:: python

   from oversampleqa.types import ValidationConfig

   cfg = ValidationConfig(
       minority_label=1,
       hidden_ratio=0.1,
       metric="hassanat",
   )

Typed validator
---------------

.. code-block:: python

   from oversampleqa.typed_validator import TypedValidator
   from imblearn.over_sampling import SMOTE

   validator = TypedValidator(config=cfg, oversampler=SMOTE(random_state=0))
   result = validator.run(X, y)
   print(result.error_rate)

Enhanced CLI configuration
--------------------------

The enhanced CLI reads configuration from ``~/.oversampleqa/config.yaml`` by default. You can override it with ``--config`` and select a profile with ``--profile``.

Minimal example
~~~~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     target: target
     minority_label: 1
     oversampler: SMOTE
     metric: hassanat
     hidden_ratio: 0.1
     resume: true
     export:
       - json

Profiles example
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   profiles:
     quick:
       hidden_ratio: 0.1
       metric: euclidean
       n_runs: 1
     research:
       hidden_ratio: 0.25
       metric: hassanat
       n_runs: 10

Integrations
~~~~~~~~~~~~

.. code-block:: yaml

   integrations:
     mlflow:
       enabled: false
       experiment_name: OversampleQA

Generate a template
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   oversampleqa template --template production -o oversampleqa.yaml

Plugin system
-------------

You can register custom metrics and validators:

.. code-block:: python

   from oversampleqa.plugin_system import register_metric

   def my_metric(a, b):
       return 0.0

   register_metric("my_metric", my_metric)