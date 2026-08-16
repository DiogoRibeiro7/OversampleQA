# OversampleQA reference plugin

A minimal, installable package that adds one custom metric and one custom
validator to OversampleQA through entry points. Copy it as the starting point
for your own.

## What is here

| File | Purpose |
|---|---|
| `pyproject.toml` | Declares the two entry-point groups. This is the discovery mechanism. |
| `src/oversampleqa_example_plugin/metric.py` | `LorentzianDistance`, a real distance metric. |
| `src/oversampleqa_example_plugin/validator.py` | `MedianRatioValidator`, a validator on a different scale from the built-in error rate. |
| `tests/` | Tests, including the axiom check the host runs at registration. |

## Install and use

```bash
pip install -e examples/plugins
```

```python
from oversampleqa.plugin_system import plugin_manager

registered = plugin_manager.discover_entry_points()
print(registered)          # ['lorentzian', 'median_ratio']

metric = plugin_manager.get_metric("lorentzian")()
metric([1.0, 2.0], [1.0, 3.0])
```

Nothing imports `oversampleqa_example_plugin` explicitly. `discover_entry_points`
reads the metadata of every installed distribution, so installing the package is
what makes the plugin available.

## Declaring entry points

```toml
[project.entry-points."oversampleqa.metrics"]
lorentzian = "oversampleqa_example_plugin.metric:LorentzianDistance"

[project.entry-points."oversampleqa.validators"]
median_ratio = "oversampleqa_example_plugin.validator:MedianRatioValidator"
```

The key is the name the metric registers under. The value is
`module:object`. The object may be a class or a plain callable.

## What registration checks

Registration is not a dictionary assignment. Each of these raises `PluginError`
with a message saying what to do:

- **Name collisions.** A name already used by a built-in or another plugin is
  refused, never silently overridden. With entry-point discovery neither author
  controls load order, so "last one wins" would be a coin flip.
- **Signatures.** A metric must accept two positional arguments; a validator
  must have a callable `validate`.
- **Metric axioms.** `d(x, x) == 0`, `d(x, y) > 0` for distinct points,
  symmetry, non-negativity and finiteness, on random input.

The axiom check is worth dwelling on, because it is not hypothetical. The
project's own built-in Hassanat metric shipped for its entire history scoring
`[-5]` and `[5]` as distance zero — it compared absolute values, so it violated
the identity of indiscernibles, and nothing checked. Every number it produced
looked plausible. If your metric cannot pass this check, the numbers it feeds
into a validation run do not mean what the run reports.

If one plugin fails, the others still load. Each failure raises a warning naming
the entry point and the reason, because a plugin that quietly fails to register
looks exactly like one that was never installed.

## Design notes

**`LorentzianDistance` is provably a metric, and the axiom check does not
prove it.** The check tests identity, symmetry, non-negativity and finiteness —
not the triangle inequality. Passing it is necessary, not sufficient. The
triangle inequality is proved in the class docstring instead; if you write your
own metric, do the same rather than treating a green check as a proof.

**Pick a name no built-in uses.** The first draft of this example was called
`canberra`, which is already built in, so registration refused it — correctly,
and only end-to-end installation revealed it. `oversampleqa.distance._METRICS`
lists the taken names.

**`MedianRatioValidator` returns `nan`, not `0.0`, when it cannot measure.** No
synthetic points, a minority of fewer than two, or duplicate real points making
the scale undefined all yield `nan`. A `0.0` would be indistinguishable from a
genuine measurement of a sampler that copied its input exactly, and that
confusion is the single most common defect this project has had to fix.

**The two validators are not comparable.** The plugin protocol promises a float
and nothing more. `MedianRatioValidator` reports a scale-free distance ratio;
the built-in reports an error rate. Do not rank one against the other.

## Running the tests

```bash
pip install -e examples/plugins
pytest examples/plugins/tests
```
