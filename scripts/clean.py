"""Remove build artefacts and caches.

Replaces the ``rm -rf`` / ``find -delete`` recipe in the Makefile, which only
worked on Unix while development happens on Windows.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Directories removed wholesale from the repository root.
ROOT_DIRECTORIES = (
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "docs/_build",
)

# Directory names removed wherever they appear.
NESTED_DIRECTORIES = ("__pycache__",)

# File patterns removed wherever they appear.
FILE_PATTERNS = ("*.pyc", "*.pyo", ".coverage")

# Never descend into these while searching.
SKIP = {".git", ".venv", "venv", "node_modules"}


def _iter_targets(root: Path):
    """Yield nested directories and files to remove, skipping SKIP roots."""
    for path in root.rglob("*"):
        if any(part in SKIP for part in path.parts):
            continue
        if path.is_dir() and path.name in NESTED_DIRECTORIES:
            yield path
        elif path.is_file() and any(path.match(p) for p in FILE_PATTERNS):
            yield path


def main() -> int:
    """Delete build artefacts. Returns a process exit code."""
    root = Path(__file__).resolve().parents[1]
    removed = 0

    for name in ROOT_DIRECTORIES:
        target = root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            print(f"removed {target.relative_to(root)}")
            removed += 1

    for name in root.glob("*.egg-info"):
        shutil.rmtree(name, ignore_errors=True)
        print(f"removed {name.relative_to(root)}")
        removed += 1

    # Collect first: deleting while walking would invalidate the iterator.
    for target in list(_iter_targets(root)):
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        removed += 1

    print(f"clean: removed {removed} item(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
