Configuration
=============

OversampleQA can be configured via typed Python configs and via the enhanced CLI config file.

Typed validation configuration
------------------------------

.. code-block:: python

   from oversampleqa.types import ValidationConfig

   cfg = ValidationConfig(
       hidden_ratio=0.1,
       metric="hassanat",
       random_state=0,
   )

``minority_label`` and the oversampler are passed to ``validate`` rather than
stored on the config (see below).

Typed validator
---------------

.. code-block:: python

   from oversampleqa.typed_validator import TypedValidator
   from imblearn.over_sampling import SMOTE

   validator = TypedValidator()
   result = validator.validate(
       X,
       y,
       minority_label=1,
       oversampler=SMOTE(random_state=0),
       config=cfg,
   )
   print(result["error_rate"])

The config is optional; you can pass ``hidden_ratio``, ``metric``,
``return_details``, and ``random_state`` directly as keyword arguments instead.

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
Memory limits and the ``psutil`` fallback
-----------------------------------------

Distance-matrix computation is memory-aware: it estimates the peak footprint of
a computation and batches when it would not fit. The effective limit is
``min(memory_limit_gb, available)``, where ``available`` is read from
``psutil``.

.. warning::

   **Without ``psutil`` installed, available memory is assumed to be 1 GB**,
   whatever the machine actually has. Batching is then more conservative than it
   needs to be, and throughput differs from an otherwise identical environment
   that has ``psutil`` — which makes performance reports hard to compare. The
   fallback is logged once at ``INFO``.

Install the optional extra to get the real figure::

    pip install 'oversampleqa[performance]'

That also pulls in ``tqdm`` for progress bars.

Two further knobs on :class:`~oversampleqa.optimized_distance.OptimizedDistanceMatrix`:

``memory_limit_gb`` (default 4.0)
   Upper bound on what one computation may use.

``safety_factor`` (default 0.8)
   Fraction of the limit a batched computation plans against. The remainder is
   headroom for allocator overhead and transient copies that the analytic
   estimate does not model.

The estimate accounts for the intermediates each kernel allocates, not just the
output array. Kernels split into two families: ``euclidean`` and friends go
through BLAS and peak at a few multiples of the output regardless of the feature
dimension, while broadcasting kernels such as ``hassanat`` allocate
``(n1, n2, d)`` arrays and peak near ``6 × d`` times the output. Ignoring that
distinction is how an earlier version came to run the whole-input path when it
should have batched.
