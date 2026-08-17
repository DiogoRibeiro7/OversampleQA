#!/usr/bin/env python3
"""Local release preparation checks.

Publishing is handled by ``.github/workflows/publish.yml`` when a GitHub
release is published. That workflow uses PyPI Trusted Publishing, so there is
no local API token or ``twine upload`` step here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a command from the repository root and fail fast."""
    print("+ " + " ".join(args))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(args, cwd=ROOT, check=True, env=merged_env)


def output(args: list[str]) -> str:
    """Return stripped command output."""
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_release_ready(*, skip_tests: bool, skip_clean_check: bool) -> None:
    """Run the local checks expected before publishing a GitHub release."""
    branch = output(["git", "branch", "--show-current"])
    if branch != "main":
        raise SystemExit("Release preparation must run on main.")

    if not skip_clean_check and output(["git", "status", "--porcelain"]):
        raise SystemExit("Working directory must be clean.")

    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    run_command(["poetry", "check", "--lock"])
    run_command(["poetry", "run", "ruff", "check", "src", "tests"])
    run_command(["poetry", "run", "mypy", "src"])
    if not skip_tests:
        run_command(
            ["poetry", "run", "pytest"],
            env={"OVERSAMPLEQA_PENDING_ZENODO_DOI": "1"},
        )
    run_command(["poetry", "build"])
    distributions = sorted(str(path.relative_to(ROOT)) for path in dist.iterdir())
    run_command(["python", "-m", "twine", "check", *distributions])

    print(
        "Release artefacts are ready in dist/. Publish a GitHub release to "
        "trigger PyPI Trusted Publishing and the Zenodo archive."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest. Use only after CI has already passed for this commit.",
    )
    parser.add_argument(
        "--skip-clean-check",
        action="store_true",
        help="Allow local dirty files while testing the release checks.",
    )
    args = parser.parse_args()

    try:
        ensure_release_ready(
            skip_tests=args.skip_tests,
            skip_clean_check=args.skip_clean_check,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
