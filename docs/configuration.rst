Configuration
=============

Typed validation configuration is available via ``ValidationConfig`` and ``TypedValidator``.

Basic configuration
-------------------

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

Plugin system
-------------

You can register custom metrics and validators:

.. code-block:: python

   from oversampleqa.plugin_system import register_metric

   def my_metric(a, b):
       return 0.0

   register_metric("my_metric", my_metric)
