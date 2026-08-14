Reproducibility
===============

OversampleQA is designed so that a validation or benchmark run can be repeated
exactly. This page explains the sources of randomness, how datasets are
provenanced, and how the on-disk cache is keyed and invalidated.

Sources of randomness
----------------------

There are two independent sources of randomness in a validation run:

1. **The hidden-majority split.** Internally, the validator hides a fraction of
   the majority class (``hidden_ratio``) and asks whether synthetic samples look
   more like the hidden majority or the real minority. This split is
   **deterministic**: :func:`~oversampleqa.validate_oversampling` fixes the split
   at ``random_state=42`` and
   :func:`~oversampleqa.validate_multiclass_oversampling` seeds its generator with
   ``numpy.random.default_rng(42)``. You do not configure it, and it does not
   vary between runs on the same input.

2. **The oversampler.** The oversampler is supplied by you, and its randomness is
   your responsibility. Always construct it with an explicit seed::

       from imblearn.over_sampling import SMOTE
       from oversampleqa import validate_oversampling

       error_rate = validate_oversampling(
           X=X, y=y, minority_label=1,
           oversampler=SMOTE(random_state=42),  # pin the oversampler seed
       )

   Without a seed, methods such as SMOTE, ADASYN, and BorderlineSMOTE will draw
   different synthetic samples on each run and the error rate will vary.

Because the hidden-majority split is fixed, **identical inputs plus an
identically-seeded oversampler produce an identical error rate.**

Stable input ordering
----------------------

The hashing and splitting operate on the arrays as given, so row order is part
of the input. If you load data from a source that does not guarantee a stable
order (for example a database query without ``ORDER BY``), sort the rows before
validating so that repeated runs see the same arrays.

Dataset provenance
------------------

The built-in dataset catalog in :func:`~oversampleqa.benchmark.load_standard_datasets`
is reproducible by construction:

- **Synthetic datasets** (``make_classification``, ``make_moons``,
  ``make_circles``, ``make_blobs``) are generated with fixed seeds, so they are
  byte-for-byte identical on every machine.
- **OpenML datasets** are optional (``include_openml=True``) and are fetched with
  a pinned dataset version (``fetch_openml(name, version=1)``). Pinning the
  version guards against an upstream dataset being silently replaced. Network
  fetches can still fail or be unavailable offline; failures are logged and the
  dataset is skipped rather than raising.

When you report results, record the OversampleQA version, the oversampler and
its seed, the metric, the ``hidden_ratio``, and (for benchmarks) the
``random_state`` passed to the runner. Together these fully determine the output.

Caching and invalidation
-------------------------

OversampleQA can cache distance matrices and validation results on disk
(default directory ``.oversampleqa_cache``) to avoid recomputation. Cache keys
are content-addressed with SHA256:

- A **dataset hash** combines each array's shape, dtype, and raw bytes
  (:meth:`~oversampleqa.caching.CacheManager.get_data_hash`).
- A **distance-matrix key** additionally folds in the metric name and the
  serialized metric keyword arguments.

This means the cache invalidates **automatically** whenever anything that would
change the result changes: the data values, their dtype or shape, the chosen
metric, or its parameters. There is no time-based expiry — a cache hit is only
ever returned for byte-identical inputs.

To force recomputation, delete the cache directory::

    rm -rf .oversampleqa_cache

Point the cache elsewhere by constructing the manager with a different
``cache_dir`` (for example a path unique to an experiment) so that concurrent
experiments do not share entries.

.. note::

   The cache key for a stored *validation result* is supplied by the caller as a
   parameters hash; make sure that hash includes every parameter that affects the
   result (oversampler identity and seed, ``hidden_ratio``, metric) so that two
   different configurations cannot collide on the same key.

Checklist for a reproducible run
--------------------------------

- Pin the oversampler seed (``random_state=...``).
- Record the OversampleQA version alongside results.
- Use a fixed ``random_state`` for the benchmark runner.
- Keep input row order stable.
- Clear or scope the cache directory when changing anything outside the hashed
  inputs (for example upgrading a dependency that changes oversampler output).
