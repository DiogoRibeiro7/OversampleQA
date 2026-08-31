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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Branches this may run from. `main` is protected and requires a pull
#: request, so the version bump necessarily lands on a branch first -- and
#: this check used to refuse the very branch the documented process creates,
#: making the checklist impossible to follow as written.
RELEASE_BRANCH_PREFIX = "release/"


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
    if branch != "main" and not branch.startswith(RELEASE_BRANCH_PREFIX):
        raise SystemExit(
            f"Release preparation must run on main or a "
            f"{RELEASE_BRANCH_PREFIX}* branch; this is {branch or 'a detached HEAD'!r}."
        )

    if not skip_clean_check and output(["git", "status", "--porcelain"]):
        raise SystemExit("Working directory must be clean.")

    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    run_command(["poetry", "check", "--lock"])
    run_command(["poetry", "run", "ruff", "check", "src", "tests", "scripts", "examples"])
    run_command(["poetry", "run", "mypy", "src"])
    if not skip_tests:
        run_command(["poetry", "run", "pytest"])
    run_command(["poetry", "build"])
    distributions = sorted(str(path.relative_to(ROOT)) for path in dist.iterdir())
    run_command(["python", "-m", "twine", "check", *distributions])

    # The same wheel check publish.yml runs. Without it a local pass says the
    # artefacts build, not that they will survive the publishing gate -- and a
    # wheel can be broken in ways the source tree never shows: missing package
    # data, an undeclared runtime dependency, a console script that only works
    # from an editable install.
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise SystemExit("poetry build produced no wheel to smoke test.")
    run_command(
        [
            "python",
            str(Path("scripts") / "smoke_installed_package.py"),
            str(wheels[0].relative_to(ROOT)),
        ]
    )

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
