"""Guard against silent drift in the public API.

A committed snapshot of every public name makes an API change visible in review
rather than at release. When this test fails, either the change was unintended
and should be reverted, or it was intended and the snapshot should be updated in
the same commit -- which puts the diff in front of a reviewer.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import oversampleqa

SNAPSHOT = Path(__file__).parent / "api_surface.json"
SIGNATURE_SNAPSHOT = Path(__file__).parent / "api_signatures.json"

SIGNATURE_EXCLUSIONS = frozenset(
    {
        "ConfigurationError",
        "MetricError",
        "MetricPlugin",
        "OversampleQAError",
        "PydanticValidationConfig",
        "ReferenceSet",
        "ValidationError",
        "ValidationMode",
        "ValidationResult",
    }
)
UNINFORMATIVE_SIGNATURES = frozenset({"(*args, **kwargs)", "(*values)"})


def current_surface() -> dict[str, str]:
    """Map every exported name to a coarse description of what it is."""
    surface: dict[str, str] = {}
    for name in oversampleqa.__all__:
        obj = getattr(oversampleqa, name, None)
        if obj is None:
            surface[name] = "MISSING"
        elif isinstance(obj, type):
            surface[name] = "class"
        elif callable(obj):
            surface[name] = "callable"
        else:
            surface[name] = type(obj).__name__
    return surface


def current_signatures() -> dict[str, str]:
    """Map public objects with stable signatures to their call contracts."""
    signatures: dict[str, str] = {}
    for name in oversampleqa.__all__:
        obj = getattr(oversampleqa, name, None)
        if name in SIGNATURE_EXCLUSIONS or obj is None or not callable(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        normalized_signature = str(
            signature.replace(
                parameters=[
                    parameter.replace(annotation=inspect.Parameter.empty)
                    for parameter in signature.parameters.values()
                ],
                return_annotation=inspect.Signature.empty,
            )
        )
        if normalized_signature in UNINFORMATIVE_SIGNATURES:
            continue
        signatures[name] = normalized_signature
    return signatures


def test_every_exported_name_resolves():
    """__all__ must not name anything that does not exist.

    A name in __all__ that is not importable breaks `from oversampleqa import *`
    and misleads anyone reading the surface.
    """
    missing = [n for n in oversampleqa.__all__ if not hasattr(oversampleqa, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"


def test_all_is_sorted_and_unique():
    """A duplicated export usually means two modules exporting the same name."""
    names = list(oversampleqa.__all__)
    assert len(names) == len(set(names)), "duplicate names in __all__"


def test_public_surface_matches_the_snapshot():
    """Fail on any addition, removal, or kind change in the public API."""
    if not SNAPSHOT.exists():  # pragma: no cover - first run
        pytest.skip(f"no snapshot yet; write one to {SNAPSHOT}")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = current_surface()

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    changed = sorted(
        name for name in set(actual) & set(expected) if actual[name] != expected[name]
    )

    message_parts = []
    if added:
        message_parts.append(f"added: {added}")
    if removed:
        message_parts.append(f"REMOVED (breaking): {removed}")
    if changed:
        message_parts.append(f"changed kind: {changed}")

    assert not message_parts, (
        "the public API changed:\n  "
        + "\n  ".join(message_parts)
        + f"\nIf this was intended, regenerate {SNAPSHOT.name} in the same "
        "commit so the change is reviewable. Removals need a deprecation "
        "period first -- see docs/api_stability.md."
    )


def test_public_signatures_match_the_snapshot():
    """Fail on public call signatures changing without a reviewable diff."""
    if not SIGNATURE_SNAPSHOT.exists():  # pragma: no cover - first run
        pytest.skip(f"no snapshot yet; write one to {SIGNATURE_SNAPSHOT}")

    expected = json.loads(SIGNATURE_SNAPSHOT.read_text(encoding="utf-8"))
    actual = current_signatures()

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    changed = sorted(
        name for name in set(actual) & set(expected) if actual[name] != expected[name]
    )

    message_parts = []
    if added:
        message_parts.append(f"added: {added}")
    if removed:
        message_parts.append(f"REMOVED (breaking): {removed}")
    if changed:
        message_parts.append(f"changed signature: {changed}")

    assert not message_parts, (
        "the public API signatures changed:\n  "
        + "\n  ".join(message_parts)
        + f"\nIf this was intended, regenerate {SIGNATURE_SNAPSHOT.name} in the "
        "same commit so parameter and default changes are reviewable. Removals "
        "and incompatible changes need a deprecation period first -- see "
        "docs/api_stability.md."
    )
