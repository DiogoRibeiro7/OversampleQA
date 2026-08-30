"""One report object, and the metadata that makes it auditable.

Three surfaces had drifted apart: ``validate_oversampling`` returned a float or a
tuple, the inference layer returned its own dataclasses, and the fidelity suite
returned a third set. Every consumer -- CLI, reporting, plotting, benchmarks --
needed bespoke handling for each. :class:`ValidationReport` composes them, so a
consumer handles one shape.

Exported results outlive the code that produced them, so every export carries a
``schema_version`` and the metadata needed to reproduce the run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._json import json_safe as _json_safe
from ._json import strict_json_dumps

__all__ = [
    "SCHEMA_VERSION",
    "RunMetadata",
    "ValidationReport",
]

SCHEMA_VERSION = "1.0"
"""Version of the exported JSON structure.

Bump the minor part for additive changes and the major part when a field is
removed or changes meaning. Consumers should refuse a major version they do not
recognise rather than guess.
"""


def _dataset_hash(X: NDArray[np.floating], y: NDArray[np.integer]) -> str:
    """Content hash of the inputs, so a report names the data it describes."""
    hasher = hashlib.sha256()
    for array in (np.asarray(X), np.asarray(y)):
        hasher.update(str(array.shape).encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()[:16]


@dataclass(frozen=True)
class RunMetadata:
    """Everything needed to reproduce and audit a run.

    A number without its provenance is not a result. This records the package
    and dependency versions, the sampler and its parameters, the seed, and a
    hash of the data -- so a report exported today can be checked against a
    rerun in a year, and a mismatch localised to whichever of those changed.
    """

    oversampler: str = ""
    oversampler_params: dict[str, Any] = field(default_factory=dict)
    metric: str = "hassanat"
    hidden_ratio: float = 0.1
    reference: str = "hidden_minority"
    random_state: int | None = None
    n_repeats: int = 1
    dataset_hash: str = ""
    n_samples: int = 0
    n_features: int = 0
    minority_label: int | None = None
    oversampleqa_version: str = ""
    numpy_version: str = ""
    sklearn_version: str = ""
    imblearn_version: str = ""
    timestamp: str = ""

    @classmethod
    def capture(
        cls,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        oversampler: Any,
        *,
        minority_label: int | None = None,
        metric: str = "hassanat",
        hidden_ratio: float = 0.1,
        reference: str = "hidden_minority",
        random_state: int | None = None,
        n_repeats: int = 1,
    ) -> RunMetadata:
        """Collect metadata for a run about to happen, or just completed."""
        import sklearn
        from imblearn import __version__ as imblearn_version

        from . import __version__ as package_version

        params: dict[str, Any] = {}
        if hasattr(oversampler, "get_params"):
            params = {k: repr(v) for k, v in oversampler.get_params().items()}

        X_arr = np.asarray(X)
        return cls(
            oversampler=type(oversampler).__name__,
            oversampler_params=params,
            metric=metric,
            hidden_ratio=hidden_ratio,
            reference=reference,
            random_state=random_state,
            n_repeats=n_repeats,
            dataset_hash=_dataset_hash(X, y),
            n_samples=int(X_arr.shape[0]),
            n_features=int(X_arr.shape[1]) if X_arr.ndim > 1 else 1,
            minority_label=minority_label,
            oversampleqa_version=package_version,
            numpy_version=np.__version__,
            sklearn_version=sklearn.__version__,
            imblearn_version=imblearn_version,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping."""
        payload: dict[str, Any] = _json_safe(
            {
                "oversampler": self.oversampler,
                "oversampler_params": self.oversampler_params,
                "metric": self.metric,
                "hidden_ratio": self.hidden_ratio,
                "reference": self.reference,
                "random_state": self.random_state,
                "n_repeats": self.n_repeats,
                "dataset_hash": self.dataset_hash,
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "minority_label": self.minority_label,
                "oversampleqa_version": self.oversampleqa_version,
                "numpy_version": self.numpy_version,
                "sklearn_version": self.sklearn_version,
                "imblearn_version": self.imblearn_version,
                "timestamp": self.timestamp,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunMetadata:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(frozen=True)
class ValidationReport:
    """Everything known about one oversampler on one dataset.

    ``calibration``, ``inference`` and ``fidelity`` are optional because each
    costs real time: the calibration fits nothing but resamples repeatedly, the
    two-sample tests permute, and the fidelity suite can fit models. A report
    with only ``error_rate`` and ``details`` is the cheap default.
    """

    error_rate: float
    metadata: RunMetadata
    details: Any = None
    calibration: Any = None
    inference: Any = None
    fidelity: Any = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable mapping of the whole report.

        Non-finite floats become ``null``; see :func:`_json_safe`.
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "error_rate": _json_safe(self.error_rate),
            "metadata": self.metadata.to_dict(),
        }
        for name in ("details", "calibration", "inference", "fidelity"):
            component = getattr(self, name)
            if component is None:
                payload[name] = None
            elif hasattr(component, "to_dict"):
                payload[name] = _json_safe(component.to_dict())
            else:  # pragma: no cover - defensive
                payload[name] = _json_safe(component)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationReport:
        """Rebuild from :meth:`to_dict` output.

        Components come back as plain dicts rather than their original
        dataclasses: the export is the interchange format, and rehydrating each
        component type would couple this module to every one of them. Round
        trips are therefore compared on ``to_dict()``, which is what a consumer
        actually reads.
        """
        version = payload.get("schema_version", "0")
        if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"report schema version {version} is not compatible with "
                f"{SCHEMA_VERSION}; a major-version change means a field was "
                "removed or changed meaning, so this cannot be read safely"
            )
        return cls(
            error_rate=(
                float("nan")
                if payload.get("error_rate") is None
                else float(payload["error_rate"])
            ),
            metadata=RunMetadata.from_dict(payload.get("metadata", {})),
            details=payload.get("details"),
            calibration=payload.get("calibration"),
            inference=payload.get("inference"),
            fidelity=payload.get("fidelity"),
            schema_version=version,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON. ``allow_nan=False`` guarantees valid output."""
        return strict_json_dumps(self.to_dict(), indent=indent)

    def to_frame(self) -> pd.DataFrame:
        """Tidy one-row frame with every scalar flattened."""
        flat: dict[str, Any] = {
            "schema_version": self.schema_version,
            "error_rate": self.error_rate,
            "oversampler": self.metadata.oversampler,
            "metric": self.metadata.metric,
            "hidden_ratio": self.metadata.hidden_ratio,
            "reference": self.metadata.reference,
            "random_state": self.metadata.random_state,
            "n_repeats": self.metadata.n_repeats,
            "minority_label": self.metadata.minority_label,
            "oversampleqa_version": self.metadata.oversampleqa_version,
        }
        flat.update(
            {
                f"meta_{k}": v
                for k, v in self.metadata.to_dict().items()
                if not isinstance(v, (dict, list))
            }
        )
        for name in ("details", "calibration", "inference", "fidelity"):
            component = getattr(self, name)
            if component is None or not hasattr(component, "to_dict"):
                continue
            for key, value in component.to_dict().items():
                if not isinstance(value, (dict, list, tuple, np.ndarray)):
                    flat[f"{name}_{key}"] = value
        return pd.DataFrame([flat])

    def __rich__(self) -> str:
        """Compact CLI rendering."""
        lines = [
            f"[bold]OversampleQA report[/bold] (schema {self.schema_version})",
            f"  error rate     {self.error_rate:.4f}",
            f"  oversampler    {self.metadata.oversampler}",
            f"  metric         {self.metadata.metric}",
            f"  random_state   {self.metadata.random_state}",
            f"  dataset        {self.metadata.dataset_hash} "
            f"({self.metadata.n_samples}x{self.metadata.n_features})",
        ]
        if self.calibration is not None and hasattr(self.calibration, "interpret"):
            lines.append(f"  calibration    {self.calibration.interpret()}")
        if self.fidelity is not None and hasattr(self.fidelity, "interpret"):
            lines.extend(f"  fidelity       {n}" for n in self.fidelity.interpret())
        return "\n".join(lines)

    def with_components(self, **components: Any) -> ValidationReport:
        """Return a copy carrying additional components."""
        return replace(self, **components)
