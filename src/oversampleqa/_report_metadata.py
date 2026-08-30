"""Human-readable metadata blocks for report exports."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from typing import Any

import pandas as pd

from . import __version__ as _PACKAGE_VERSION
from ._export_metadata import _dependency_lock_hash
from ._render import frame_to_html, frame_to_markdown

_MAX_VALUES = 6
_FIELD_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("datasets", ("dataset", "dataset_name")),
    ("oversamplers", ("oversampler", "oversampler_name")),
    ("metrics", ("metric",)),
    ("hidden_ratios", ("hidden_ratio",)),
    ("reference", ("reference",)),
    ("minority_labels", ("minority_label",)),
    ("runs", ("run",)),
    ("repeats", ("repeat",)),
    ("folds", ("fold",)),
    ("split_seeds", ("split_seed",)),
    ("random_states", ("random_state", "oversampler_random_state")),
    ("n_folds", ("n_folds",)),
    ("n_repeats", ("n_repeats",)),
    ("result_versions", ("oversampleqa_version",)),
)


def report_metadata_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise run context in a small two-column frame."""
    lock_hash = _dependency_lock_hash()
    rows: list[dict[str, str]] = [
        {"field": "oversampleqa_version", "value": _PACKAGE_VERSION},
        {"field": "python_version", "value": platform.python_version()},
        {"field": "platform", "value": platform.platform()},
        {"field": "dependency_lock_hash", "value": lock_hash or "unavailable"},
        {"field": "result_rows", "value": str(len(results))},
        {
            "field": "result_columns",
            "value": _compact_sequence([str(column) for column in results.columns]),
        },
    ]

    for label, columns in _FIELD_COLUMNS:
        value = _compact_unique_values(results, columns)
        if value:
            rows.append({"field": label, "value": value})

    attrs = results.attrs
    source = attrs.get("source")
    if isinstance(source, dict):
        source_rows = source.get("row_count")
        if source_rows is not None:
            rows.append({"field": "source_rows", "value": str(source_rows)})
        source_columns = source.get("columns")
        if isinstance(source_columns, list):
            rows.append(
                {
                    "field": "source_columns",
                    "value": _compact_sequence([str(column) for column in source_columns]),
                }
            )

    return pd.DataFrame(rows, columns=["field", "value"])


def report_metadata_markdown(results: pd.DataFrame) -> str:
    """Render report metadata as Markdown."""
    return frame_to_markdown(report_metadata_frame(results))


def report_metadata_html(results: pd.DataFrame) -> str:
    """Render report metadata as HTML."""
    return frame_to_html(report_metadata_frame(results))


def _compact_unique_values(results: pd.DataFrame, columns: Sequence[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in results.columns:
            continue
        for value in results[column].tolist():
            if _is_missing(value):
                continue
            rendered = _format_value(value)
            if rendered not in seen:
                seen.add(rendered)
                values.append(rendered)
    return _compact_sequence(values)


def _compact_sequence(values: Sequence[str]) -> str:
    if not values:
        return "none"
    head = list(values[:_MAX_VALUES])
    if len(values) > _MAX_VALUES:
        head.append(f"+{len(values) - _MAX_VALUES} more")
    return ", ".join(head)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
