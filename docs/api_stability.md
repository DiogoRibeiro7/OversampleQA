# API Stability

## What is public

Anything listed in `oversampleqa.__all__` and importable from the package root.
That set is snapshotted in `tests/api_surface.json`, and stable call contracts
are snapshotted in `tests/api_signatures.json`. Both are guarded by
`tests/test_api_surface.py`, so an addition, removal, change of kind, or
parameter/default change fails CI rather than surfacing at release.

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

Point 1 is mechanised by `oversampleqa.deprecated`, so the wording does not have
to be reinvented at each site — and, in particular, so the removal version is
not the detail that gets forgotten:

```python
from oversampleqa import deprecated

@deprecated(removal_version="0.6.0", replacement="new_name")
def old_name(): ...
```

It works on functions, methods and classes, appends a `.. deprecated::` note to
the docstring so the change is visible in these docs, and raises the warning
against the **caller's** stack frame. That last point is not cosmetic: Python
hides `DeprecationWarning` by default outside `__main__`, and per-module filters
key on the reported location, so a warning that appears to originate inside
oversampleqa is invisible to the people who need to act on it.

Pass `category=FutureWarning` when the change alters results rather than
spelling; that category is shown to end users by default.

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
2. regenerate the snapshots in the **same commit**:

   ```bash
   python -c "import json,sys; sys.path.insert(0,'tests'); \
   from test_api_surface import current_surface; \
   print(json.dumps(current_surface(), indent=2, sort_keys=True))" > tests/api_surface.json
   python -c "import json,sys; sys.path.insert(0,'tests'); \
   from test_api_surface import current_signatures; \
   print(json.dumps(current_signatures(), indent=2, sort_keys=True))" > tests/api_signatures.json
   ```

3. record it in `CHANGELOG.md`.

The signature snapshot covers public objects where Python exposes a stable,
informative call signature. It records parameter names, ordering, keyword-only
markers, varargs, and defaults, while intentionally omitting annotations because
runtime annotation rendering differs across supported Python versions.
Exception classes, typing/protocol artefacts, and Pydantic-generated runtime
constructors are excluded because they either have no public constructor
contract or expose signatures that are generic or coupled to runtime
implementation details.

The point of the snapshots is not to prevent change but to make it visible in
the diff, where a reviewer can weigh it.
