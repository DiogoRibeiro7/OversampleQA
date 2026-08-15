# API Stability

## What is public

Anything listed in `oversampleqa.__all__` and importable from the package root.
That set is snapshotted in `tests/api_surface.json` and guarded by
`tests/test_api_surface.py`, so an addition, removal, or change of kind fails CI
rather than surfacing at release.

Everything else is internal, including:

- modules prefixed with `_` (`_rng`),
- names prefixed with `_` (`_METRICS`, `_PEAK_MODEL`, `_pairwise`),
- anything reached only by importing a submodule directly.

Internal names can change in any release without notice. If you depend on one,
open an issue — it may be a sign it should be public.

## The pre-1.0 guarantee

The package is `0.x`. The guarantee is **best-effort, not semantic versioning**:
a minor release can change behaviour when the previous behaviour was wrong.
Several already have, and the changelog says so explicitly each time.

What is promised even pre-1.0:

- **Numeric changes are announced.** Anything that alters a computed result is
  called out in `CHANGELOG.md` with a note that old and new values are not
  comparable. A silently changed number is worse than a breaking change, because
  nothing tells you to re-check.
- **Removals get a deprecation period.** A name scheduled for removal emits a
  `DeprecationWarning` naming the replacement and the removal version, and
  survives at least **two minor releases**.
- **Exported results carry a schema version.** JSON from `ValidationReport`
  includes `schema_version`; consumers should refuse a major version they do not
  recognise rather than guess.

## Deprecation

A deprecated name:

1. emits `DeprecationWarning` on use, naming the replacement and the release in
   which it disappears;
2. keeps working unchanged for at least two minor releases;
3. is listed in the changelog under `Deprecated` when introduced and under
   `Removed` when dropped.

Currently deprecated:

| Name | Replacement | Removal |
|---|---|---|
| `reference="train_minority"` | `reference="hidden_minority"` | not before 0.5.0 |

`reference="train_minority"` emits a `FutureWarning` rather than a
`DeprecationWarning` because it is not merely an old spelling — it computes a
biased quantity, and the warning explains the bias. It remains available so
pre-0.3 numbers can be reproduced.

## Changing the API deliberately

When a change to the public surface is intended:

1. make the change,
2. regenerate the snapshot in the **same commit**:

   ```bash
   python -c "import json,sys; sys.path.insert(0,'tests'); \
   from test_api_surface import current_surface; \
   print(json.dumps(current_surface(), indent=2, sort_keys=True))" > tests/api_surface.json
   ```

3. record it in `CHANGELOG.md`.

The point of the snapshot is not to prevent change but to make it visible in the
diff, where a reviewer can weigh it.
