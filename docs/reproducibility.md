# Reproducibility

OversampleQA is designed so that a validation or benchmark run can be
repeated exactly. This page explains the sources of randomness, how
datasets are provenanced, and how the on-disk cache is keyed and
invalidated.

## Sources of randomness

Every source of randomness in a run, and the knob that controls it:

| Source | Controlled by |
|----|----|
| Majority (and minority) hold-out split | `random_state` on the validator |
| Oversampler's own generation | `random_state` on the sampler instance; `reseed_oversampler=True` to vary it per repeat |
| Benchmark dataset generation | fixed seeds inside `~oversampleqa.benchmark.load_standard_datasets` |
| Benchmark CV folds | `random_state` on the runner |
| Cache keying | content hash of the inputs (see below) |

Pin the oversampler as well as the validator. Without a seed, SMOTE,
ADASYN and BorderlineSMOTE draw different synthetic samples on each run:

    from imblearn.over_sampling import SMOTE
    from oversampleqa import validate_oversampling

    error_rate = validate_oversampling(
        X=X, y=y, minority_label=1,
        oversampler=SMOTE(random_state=42),  # the sampler's own randomness
        random_state=42,                     # which points are hidden
    )

With both pinned, repeated runs on identical inputs return a
bit-identical error rate.

> [!NOTE]
> `random_state` accepts an `int`, a `numpy.random.Generator`, a
> `numpy.random.SeedSequence`, or `None`. Passing `None` draws fresh
> entropy and is deliberately **not** reproducible. The default is `42`.

## The seed is not a formality

Which majority points get hidden is the single largest driver of the
error rate. Changing only the seed, with the data and the oversampler's
own seed held fixed:

``` python
>>> validate_oversampling(X, y, 1, SMOTE(random_state=0), random_state=42)
0.2412
>>> validate_oversampling(X, y, 1, SMOTE(random_state=0), random_state=7)
0.4005
```

The same configuration, differing only in which 10% of the majority was
held out, gives error rates that differ by a factor of 1.7. A single run
therefore tells you very little on its own, which is what `n_repeats`
exists to address.

## Reporting a range instead of a point

`n_repeats` draws independent hold-out splits and reports the spread:

    details = validate_oversampling(
        X, y, minority_label=1,
        oversampler=SMOTE(random_state=0),
        n_repeats=20,
        return_details=True,
    )
    print(details.mean, details.std, details.interval)
    # 0.2920 0.0687 (0.2640, 0.3215)

Repeat streams are spawned from a `numpy.random.SeedSequence`. They are
**not** derived as `seed + i`, which produces correlated streams and
would understate the dispersion.

> [!WARNING]
> The reported interval is a percentile bootstrap over the per-repeat
> error rates. It describes the variability of the **hold-out split**,
> conditional on this dataset and on the oversampler's own seed.
>
> It is *not* a confidence interval for a population quantity, and by
> default it does not include the oversampler's own randomness at all.
> Pass `reseed_oversampler=True` to give the sampler a fresh seed per
> repeat; the dispersion then covers both sources together, which is a
> wider and different decomposition. Say which one you used when
> reporting.
>
> Synthetic points interpolated from shared parent points are not
> independent, so a binomial interval on a single run's error rate would
> be too narrow.

## Unrepresentative hold-outs

An unstratified hold-out can miss a cluster entirely when the majority
class has structure. Pass `stratify_by` with group labels aligned to `y`
to take the fraction within each group instead:

    validate_oversampling(
        X, y, minority_label=1, oversampler=SMOTE(random_state=0),
        stratify_by=cluster_ids,
    )

Strata are never inferred automatically — you know what grouping
matters.

## Stable input ordering

The hashing and splitting operate on the arrays as given, so row order
is part of the input. If you load data from a source that does not
guarantee a stable order (for example a database query without
`ORDER BY`), sort the rows before validating so that repeated runs see
the same arrays.

## Dataset provenance

Both dataset catalogs — `~oversampleqa.benchmark.load_standard_datasets`
and `~oversampleqa.advanced_benchmark.DatasetRepository` — attach a
`provenance` record to every dataset they return, with the same six
keys: `source`, `generator`, `params`, `url`, `license` and `notes`.

`license` is always present. `"Unknown"` is a legitimate value; omitting
the key is not, because an absent licence reads as "no restrictions" to
a hurried reader. `notes` carries anything needed to avoid misreading
the numbers — in particular, `max_samples` takes a *positional slice*,
not a random sample, and the record says so.

The catalog in `~oversampleqa.benchmark.load_standard_datasets` is
reproducible by construction:

- **Synthetic datasets** (`make_classification`, `make_moons`,
  `make_circles`, `make_blobs`) are generated with fixed seeds, so they
  are byte-for-byte identical on every machine.
- **OpenML datasets** are optional (`include_openml=True`) and are
  fetched with a pinned dataset version
  (`fetch_openml(name, version=1)`). Pinning the version guards against
  an upstream dataset being silently replaced. Network fetches can still
  fail or be unavailable offline; failures are logged and the dataset is
  skipped rather than raising.

When you report results, record the OversampleQA version, the
oversampler and its seed, the metric, the `hidden_ratio`, and (for
benchmarks) the `random_state` passed to the runner. Together these
fully determine the output.

## Caching and invalidation

Caching is **opt-in**. Nothing is cached, and no directory is created,
unless you construct a `~oversampleqa.caching.ValidationCache` and pass
it in:

    from oversampleqa.caching import ValidationCache
    from oversampleqa import distance_matrix

    cache = ValidationCache()            # per-user cache dir, created on first write
    D = distance_matrix(X1, X2, "hassanat", cache=cache)

Earlier versions built a cache at import time, which created
`.oversampleqa_cache` in the current working directory as a side effect
of `import oversampleqa`. That no longer happens, and the default
location is now the platform's per-user cache directory rather than the
working directory.

> [!NOTE]
> **Caching does not always pay.** The key is a content hash, which must
> read every input byte. For a BLAS-backed metric such as `euclidean`,
> hashing the inputs costs more than recomputing the result — on a
> 2000×10000 problem, 0.39 s to hash and store against 0.24 s to
> compute. For `hassanat` the same problem takes 26.7 s to compute
> against 0.32 s to hash and store, an 83× saving. Enable the cache for
> expensive metrics and repeated identical calls; leave it off
> otherwise.

Cache keys are content-addressed with SHA256:

- A **dataset hash** combines each array's shape, dtype, and raw bytes
  (`~oversampleqa.caching.CacheManager.get_data_hash`).
- A **distance-matrix key** additionally folds in the metric name and
  the serialized metric keyword arguments.

This means the cache invalidates **automatically** whenever anything
that would change the result changes: the data values, their dtype or
shape, the chosen metric, or its parameters. There is no time-based
expiry — a cache hit is only ever returned for byte-identical inputs.

`batch_size` is deliberately **not** part of the key. Batching splits
one computation into chunks and concatenates them, so it cannot change
the result; `tests/test_caching.py` pins that invariant for every
registered metric. The key also no longer includes the optimizer object,
which used to make it depend on internal state that cannot affect the
output and broke outright for locally-defined plugin metrics.

Cached arrays are returned **read-only**. A cache hit hands back the
stored array rather than a copy, so one in-place operation downstream
would otherwise corrupt every later hit silently; the write flag makes
that a loud `ValueError`. Call `.copy()` if you need to modify the
result.

### Thread and process safety

A single `ValidationCache` instance guards its own in-memory bookkeeping
with a lock, so concurrent use through one instance is safe. On-disk
writes are **not** atomic: two processes, or two instances sharing a
directory, can interleave and leave a truncated file. Give each process
its own `cache_dir`.

To force recomputation, delete the cache directory:

    rm -rf .oversampleqa_cache

Point the cache elsewhere by constructing the manager with a different
`cache_dir` (for example a path unique to an experiment) so that
concurrent experiments do not share entries.

> [!NOTE]
> The cache key for a stored *validation result* is supplied by the
> caller as a parameters hash; make sure that hash includes every
> parameter that affects the result (oversampler identity and seed,
> `hidden_ratio`, metric) so that two different configurations cannot
> collide on the same key.

## Checklist for a reproducible run

- Pin the validator seed (`random_state=...` on
  `validate_oversampling`).
- Pin the oversampler seed (`random_state=...` on the sampler).
- Report `n_repeats` and the spread, not just a point estimate from one
  split.
- Record the OversampleQA version alongside results.
- Use a fixed `random_state` for the benchmark runner.
- Keep input row order stable.
- Clear or scope the cache directory when changing anything outside the
  hashed inputs (for example upgrading a dependency that changes
  oversampler output).
