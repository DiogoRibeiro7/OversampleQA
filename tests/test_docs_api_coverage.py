"""Every public module must have an API reference page.

Seven did not: `deprecation`, `estimator`, `exceptions`, `fidelity`,
`inference`, `plugin_contract` and `reports`. Their exports still appeared in
the rendered site, because `::: oversampleqa` documents the package's members,
so nothing looked obviously wrong -- but the module docstrings never rendered,
and in this codebase those carry the reasoning rather than a summary line.
`inference` had thirty-one lines of it going nowhere.

The roadmap asks for this comparison before every minor release. Doing it by
hand is how it came to be seven modules behind, so it is a test.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "oversampleqa"
API_DOCS = ROOT / "docs" / "api"

MKDOCSTRINGS_DIRECTIVE = re.compile(r":::\s*oversampleqa\.([a-zA-Z_]+)")


def public_modules() -> set[str]:
    """Module names a user can import, excluding private ones."""
    return {
        path.stem
        for path in SOURCE.glob("*.py")
        if not path.stem.startswith("_") and path.stem != "__init__"
    }


def documented_modules() -> set[str]:
    """Modules rendered by an mkdocstrings directive under docs/api."""
    found: set[str] = set()
    for path in API_DOCS.rglob("*.md"):
        found |= set(MKDOCSTRINGS_DIRECTIVE.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_public_module_has_an_api_page():
    missing = sorted(public_modules() - documented_modules())
    assert not missing, (
        f"public modules with no API reference page: {missing}. "
        f"Add docs/api/<name>.md containing '::: oversampleqa.<name>' and a "
        "nav entry in mkdocs.yml."
    )


def test_no_api_page_documents_a_module_that_no_longer_exists():
    """A page for a deleted module fails `mkdocs build --strict` at release."""
    stale = sorted(documented_modules() - public_modules())
    assert not stale, f"API pages for modules that do not exist: {stale}"


def test_every_api_page_is_in_the_navigation():
    """A page absent from the nav is unreachable, and mkdocs warns about it."""
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    orphans = sorted(
        path.name
        for path in API_DOCS.rglob("*.md")
        if f"api/{path.name}" not in nav
    )
    assert not orphans, f"API pages missing from the mkdocs nav: {orphans}"
