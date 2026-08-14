"""Contributor onboarding automation script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def run(cmd: list[str], *, check: bool = True) -> None:
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def ensure_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        print(f"[ERROR] Required tool '{tool}' not found in PATH.")
        sys.exit(1)


def main() -> None:
    print("Starting contributor onboarding...")
    ensure_tool("poetry")
    ensure_tool("git")

    run(["poetry", "install", "--no-interaction"])
    run(["poetry", "run", "pre-commit", "install"])
    run(["poetry", "run", "pre-commit", "install", "--hook-type", "commit-msg"])

    default_branch = os.getenv("DEFAULT_BRANCH", "main")
    print(f"Environment ready. Remember to branch from '{default_branch}'.")


if __name__ == "__main__":
    main()
