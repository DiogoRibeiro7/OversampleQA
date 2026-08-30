"""Sidecar metadata for exported reports and result tables."""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import sklearn
from imblearn import __version__ as imblearn_version

from . import __version__ as oversampleqa_version
from ._json import json_safe, write_json

EXPORT_METADATA_SCHEMA_VERSION = "1.0"


def metadata_sidecar_path(artifact_path: str | Path) -> Path:
    """Return the metadata sidecar path for an exported artifact."""
    path = Path(artifact_path)
    return path.with_name(f"{path.name}.metadata.json")


def write_export_metadata(
    artifact_path: str | Path,
    *,
    export_kind: str,
    data: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write metadata describing an exported artifact and return its path."""
    artifact = Path(artifact_path)
    payload = export_metadata(
        artifact_path=artifact,
        export_kind=export_kind,
        data=data,
        extra=extra,
    )
    sidecar = metadata_sidecar_path(artifact)
    write_json(sidecar, payload)
    return sidecar


def export_metadata(
    *,
    artifact_path: str | Path,
    export_kind: str,
    data: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata sufficient to audit an exported artifact."""
    artifact = Path(artifact_path)
    payload: dict[str, Any] = {
        "schema_version": EXPORT_METADATA_SCHEMA_VERSION,
        "artifact": artifact.name,
        "artifact_format": artifact.suffix.lstrip(".") or None,
        "export_kind": export_kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "dependency_lock_hash": _dependency_lock_hash(),
    }
    summary = _data_summary(data)
    if summary:
        payload["data"] = summary
    if extra:
        payload.update(json_safe(extra))
    return cast(dict[str, Any], json_safe(payload))


def _environment() -> dict[str, Any]:
    """Return package and runtime versions needed to interpret an export."""
    return {
        "oversampleqa_version": oversampleqa_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "imblearn_version": imblearn_version,
    }


def _dependency_lock_hash(start: Path | None = None) -> str | None:
    """Return the nearest ``poetry.lock`` hash, if one is visible."""
    base = (start or Path.cwd()).resolve()
    for directory in (base, *base.parents):
        lockfile = directory / "poetry.lock"
        if lockfile.is_file():
            return hashlib.sha256(lockfile.read_bytes()).hexdigest()
    return None


def _data_summary(data: Any) -> dict[str, Any]:
    """Summarize exported data without duplicating the full artifact."""
    if isinstance(data, pd.DataFrame):
        summary: dict[str, Any] = {
            "row_count": len(data),
            "columns": [str(column) for column in data.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
        }
        if data.attrs:
            summary["attrs"] = json_safe(data.attrs)
        return summary
    if isinstance(data, dict):
        return {
            "keys": sorted(str(key) for key in data),
            "values": json_safe(data),
        }
    return {}
