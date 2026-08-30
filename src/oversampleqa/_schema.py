"""The identifier columns every result surface carries.

Four surfaces emit rows a caller can concatenate, join or store:
``run_benchmark``, ``StatisticalBenchmark.fold_results``, the statistical
summary frame, and ``ValidationReport.to_frame``. They grew separately and
named the same things differently -- ``dataset`` against ``dataset_name``,
``oversampler`` against ``oversampler_name`` -- so joining two of them meant
knowing which one produced which, and a caller writing ``frame["dataset"]``
worked or raised depending on where the frame came from.

These names are the contract. Every surface exposes all of them, whatever else
it also carries. The older names stay alongside, because renaming them would
break callers for no gain that adding the canonical name does not already give.
"""

from __future__ import annotations

#: Columns present on every result frame the package produces.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "dataset",
    "oversampler",
    "metric",
    "hidden_ratio",
    "reference",
    "minority_label",
    "oversampleqa_version",
)

__all__ = ["IDENTIFIER_COLUMNS"]
