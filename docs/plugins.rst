Writing a plugin
================

OversampleQA takes custom distance metrics and custom validators from installed
packages, discovered through entry points. A worked, installable example lives
in ``examples/plugins/`` — copy it rather than starting from scratch.

Entry-point discovery
---------------------

A plugin package advertises what it provides in its own ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."oversampleqa.metrics"]
   lorentzian = "my_package.metric:LorentzianDistance"

   [project.entry-points."oversampleqa.validators"]
   median_ratio = "my_package.validator:MedianRatioValidator"

The key is the name it registers under; the value is ``module:object``, and the
object may be a class or a plain callable.

Nothing needs to import the plugin:

.. code-block:: python

   from oversampleqa.plugin_system import plugin_manager

   registered = plugin_manager.discover_entry_points()
   metric = plugin_manager.get_metric("lorentzian")()

:meth:`~oversampleqa.plugin_system.PluginManager.discover_entry_points` reads
the metadata of every installed distribution, so installing the package is what
makes the plugin available.

Discovery is not called automatically at import. Scanning entry points on
``import oversampleqa`` would let a third-party package run code as a side
effect of importing this one, and would make import time depend on what else is
installed. Call it when you want it.

What registration checks
------------------------

Registration is not a dictionary assignment. Each of these raises
:class:`~oversampleqa.PluginError` with a message saying what to do about it:

**Name collisions.**
   A name already used by a built-in or another plugin is refused, never
   silently overridden. Under entry-point discovery neither author controls load
   order, so "last one wins" would be a coin flip. ``oversampleqa.distance``
   holds the built-in names.

**Signatures.**
   A metric must accept two positional arguments. A validator must have a
   callable ``validate``.

**Metric axioms.**
   ``d(x, x) == 0``, ``d(x, y) > 0`` for distinct points, symmetry,
   non-negativity and finiteness, checked on random input in the metric's
   declared domain.

The axiom check is not hypothetical. The built-in ``hassanat`` metric shipped
for this project's entire history scoring ``[-5]`` and ``[5]`` as distance zero,
because it compared absolute values. It was not a metric, every number it
produced looked plausible, and nothing checked.

Note what the check does **not** cover: the triangle inequality. Passing
registration is necessary, not sufficient. If you are writing a metric, prove
that step yourself — the example plugin's docstring shows what that looks like.

When a plugin fails
-------------------

A plugin that fails to import or fails a check does not prevent the others from
loading. Each failure raises a warning naming the entry point and the reason,
because a plugin that quietly fails to register looks exactly like one that was
never installed.

Pass ``strict=True`` to raise instead. Use it in your own test suite, where a
plugin that silently failed to load would make the suite pass for the wrong
reason.

.. code-block:: python

   plugin_manager.discover_entry_points(strict=True)

Returning "not measured"
------------------------

If your validator cannot produce a number — no synthetic points were generated,
the minority is too small, the scale is undefined — return ``nan``, not ``0.0``.
A zero is indistinguishable from a genuine measurement of a perfect score, and
that confusion has been the single most common defect in this project's history.
