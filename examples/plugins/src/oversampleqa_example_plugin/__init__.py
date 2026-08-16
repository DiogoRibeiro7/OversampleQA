"""A worked example of an OversampleQA plugin package.

Ships one custom metric and one custom validator, both advertised through entry
points so that :meth:`~oversampleqa.plugin_system.PluginManager.discover_entry_points`
finds them without this package being imported first.

The metric is deliberately a real metric. OversampleQA runs an axiom smoke check
at registration -- ``d(x, x) == 0``, ``d(x, y) > 0`` for distinct points,
symmetry, non-negativity, finiteness -- and refuses anything that fails. That
check exists because the project's own built-in Hassanat implementation shipped
for its entire history scoring ``[-5]`` and ``[5]`` as distance zero. An example
plugin that could not pass the check would be teaching the wrong lesson.
"""

from __future__ import annotations

from .metric import LorentzianDistance
from .validator import MedianRatioValidator

__all__ = ["LorentzianDistance", "MedianRatioValidator"]
__version__ = "0.1.0"
