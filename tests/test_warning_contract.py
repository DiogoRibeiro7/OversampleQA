"""Guard against silent drift in the warnings the package raises.

``test_api_surface.py`` snapshots public names and call signatures, which
catches a renamed function or a changed default. It does not catch a changed
*warning*, and callers depend on those too: anyone running
``warnings.simplefilter("error", FutureWarning)`` in their test suite, or
filtering ``ResourceWarning`` out of their logs, is relying on a contract
nothing checked.

Changing ``warn_reference_bias`` from ``FutureWarning`` to ``UserWarning``
would break that filtering silently -- no signature moves, no export
disappears, and every existing test still passes.

The categories are read statically with ``ast`` rather than by calling
anything, so the check needs no fixtures, cannot be defeated by a code path
that is hard to reach, and stays fast.
"""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "oversampleqa"
SNAPSHOT = Path(__file__).parent / "warning_contract.json"

#: Recorded when the category is a runtime value rather than a literal class.
#: ``deprecated`` takes the category as a parameter, which is deliberate and
#: not something this snapshot can or should pin to one class.
RUNTIME = "<runtime>"


def _is_warn_call(call: ast.Call) -> bool:
    """Whether a call node is ``warnings.warn(...)``."""
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "warn"
        and isinstance(func.value, ast.Name)
        and func.value.id == "warnings"
    )


def _category_of(call: ast.Call) -> str:
    """The warning category a ``warnings.warn`` call raises.

    ``warnings.warn`` defaults to ``UserWarning`` when no category is given,
    so an omitted argument is recorded as that rather than as absent -- the
    caller sees a UserWarning either way.
    """
    node: ast.expr | None = None
    if len(call.args) >= 2:
        node = call.args[1]
    for keyword in call.keywords:
        if keyword.arg == "category":
            node = keyword.value

    if node is None:
        return "UserWarning"
    if isinstance(node, ast.Name):
        return node.id if node.id.endswith("Warning") else RUNTIME
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr.endswith("Warning") else RUNTIME
    return RUNTIME


def _walk(node: ast.AST, scope: list[str], found: dict[str, list[str]], module: str) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            _walk(child, [*scope, child.name], found, module)
            continue
        if isinstance(child, ast.Call) and _is_warn_call(child):
            where = ".".join([module, *scope]) if scope else f"{module}.<module>"
            found.setdefault(where, []).append(_category_of(child))
        _walk(child, scope, found, module)


def current_contract() -> dict[str, list[str]]:
    """Map each warning site to the categories it raises."""
    found: dict[str, list[str]] = {}
    for path in sorted(SOURCE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _walk(tree, [], found, path.stem)
    return {key: sorted(value) for key, value in sorted(found.items())}


def test_warning_contract_matches_the_snapshot():
    """Fail on any warning added, removed, or given a different category.

    When this fails, either the change was unintended, or it was intended and
    the snapshot should be updated in the same commit -- which puts the diff in
    front of a reviewer, the same bargain test_api_surface.py makes.
    """
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = current_contract()

    added = sorted(set(current) - set(expected))
    removed = sorted(set(expected) - set(current))
    changed = {
        key: (expected[key], current[key])
        for key in sorted(set(expected) & set(current))
        if expected[key] != current[key]
    }

    assert not added, f"new warning site(s) not in the snapshot: {added}"
    assert not removed, f"warning site(s) gone from the source: {removed}"
    assert not changed, f"warning category changed: {changed}"


def test_every_recorded_category_is_a_real_warning_class():
    """A typo in a category name raises TypeError only when that line runs.

    ``warnings.warn(msg, UserWarnign)`` is valid Python and fails at runtime,
    possibly in a branch no test reaches.
    """
    for where, categories in current_contract().items():
        for category in categories:
            if category == RUNTIME:
                continue
            resolved = getattr(builtins, category, None)
            assert isinstance(resolved, type) and issubclass(resolved, Warning), (
                f"{where} raises {category!r}, which is not a built-in warning class"
            )
