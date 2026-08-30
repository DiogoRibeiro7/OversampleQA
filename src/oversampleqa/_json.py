"""Strict JSON helpers for machine-readable exports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert Python, NumPy and pandas-like values to strict JSON values.

    The stdlib encoder emits bare ``NaN`` and ``Infinity`` tokens by default,
    but those are not valid JSON. Export paths use this helper before calling
    ``json.dumps(..., allow_nan=False)`` so non-finite numeric values become
    ``null`` instead.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if _is_missing_scalar(value):
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [json_safe(item) for item in value]

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def strict_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialize ``value`` as standards-compliant JSON."""
    return json.dumps(json_safe(value), indent=indent, allow_nan=False)


def write_json(path: str | Path, value: Any, *, indent: int | None = 2) -> None:
    """Write ``value`` as strict UTF-8 JSON."""
    Path(path).write_text(strict_json_dumps(value, indent=indent), encoding="utf-8")


def _is_missing_scalar(value: Any) -> bool:
    """Return whether ``value`` behaves like pandas' scalar missing values."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a runtime dependency
        return False

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, bool | np.bool_) and bool(result)
