"""Guard against silent drift in the public API.

A committed snapshot of every public name makes an API change visible in review
rather than at release. When this test fails, either the change was unintended
and should be reverted, or it was intended and the snapshot should be updated in
the same commit -- which puts the diff in front of a reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import oversampleqa

SNAPSHOT = Path(__file__).parent / "api_surface.json"


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
