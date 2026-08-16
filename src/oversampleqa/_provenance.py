"""Provenance metadata for benchmark datasets.

A benchmark result is only interpretable if you can say where its data came
from. These helpers build one consistent record for that, so the two catalogs --
:func:`~oversampleqa.benchmark.load_standard_datasets` and
:class:`~oversampleqa.advanced_benchmark.DatasetRepository` -- describe their
datasets the same way rather than each inventing a shape.

Every record carries the same six keys:

``source``
    ``"synthetic"``, ``"bundled"`` or ``"OpenML"``. The coarse question of
    whether the data is generated, shipped with scikit-learn, or downloaded.
``generator``
    The fully-qualified callable that produced or fetched it.
``params``
    Arguments passed, including any ``random_state``. For synthetic data this
    is sufficient to regenerate the dataset exactly.
``url``
    Where a human can read about it.
``license``
    Terms. "Unknown" is a legitimate value and is better than omitting the key,
    because omission reads as "no restrictions" to a hurried reader.
``notes``
    Anything a reader needs in order not to misread the numbers -- applied
    preprocessing, truncation, known caveats.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SKLEARN_LICENSE",
    "bundled_provenance",
    "openml_provenance",
    "synthetic_provenance",
]

SKLEARN_LICENSE = "BSD-3-Clause (scikit-learn synthetic generator)"

_SKLEARN_GENERATORS_URL = (
    "https://scikit-learn.org/stable/datasets/sample_generators.html"
)


def synthetic_provenance(generator: str, **params: Any) -> dict[str, Any]:
    """Build provenance for a scikit-learn synthetic dataset.

    Args:
        generator: Fully-qualified name of the generator used.
        **params: Generation parameters, including ``random_state``.

    Returns:
        A provenance record.
    """
    return {
        "source": "synthetic",
        "generator": generator,
        "params": params,
        "url": _SKLEARN_GENERATORS_URL,
        "license": SKLEARN_LICENSE,
        "notes": (
            "Generated deterministically from the fixed random_state; "
            "not real-world data."
        ),
    }


def bundled_provenance(
    generator: str,
    *,
    url: str,
    license: str,
    notes: str = "",
    **params: Any,
) -> dict[str, Any]:
    """Build provenance for a dataset shipped inside scikit-learn.

    Bundled data is reproducible in the sense that it does not change between
    runs, but it is real-world data with its own citation and terms, so it is
    recorded separately from synthetic data rather than lumped in with it.

    Args:
        generator: Fully-qualified loader, e.g. ``sklearn.datasets.load_breast_cancer``.
        url: Where the dataset is documented.
        license: Terms of use.
        notes: Caveats a reader needs, such as applied truncation.
        **params: Arguments passed to the loader.

    Returns:
        A provenance record.
    """
    return {
        "source": "bundled",
        "generator": generator,
        "params": params,
        "url": url,
        "license": license,
        "notes": notes,
    }


def openml_provenance(
    name: str,
    version: int,
    *,
    notes: str = "",
) -> dict[str, Any]:
    """Build provenance for a dataset fetched from OpenML.

    The version is part of the record because it is the only thing standing
    between a "reproducible" benchmark and an upstream dataset being silently
    replaced under it.

    Args:
        name: OpenML dataset name.
        version: Pinned OpenML version.
        notes: Preprocessing or caveats.

    Returns:
        A provenance record.
    """
    return {
        "source": "OpenML",
        "generator": "sklearn.datasets.fetch_openml",
        "params": {"name": name, "version": version},
        "url": f"https://www.openml.org/search?type=data&q={name}",
        "license": "Varies per dataset; see the OpenML page.",
        "notes": notes or f"Downloaded from OpenML with the version pinned to {version}.",
    }
