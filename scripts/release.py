#!/usr/bin/env python3
"""Release automation script."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run command and check for errors."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


def main():
    """Main release process."""
    print("🚀 Starting release process...")

    # Check we're on main branch
    branch = run_command("git branch --show-current").strip()
    if branch != "main":
        print("❌ Must be on main branch for release")
        sys.exit(1)

    # Check working directory is clean
    status = run_command("git status --porcelain").strip()
    if status:
        print("❌ Working directory must be clean")
        sys.exit(1)

    # Run tests
    print("🧪 Running tests...")
    run_command("pytest tests/")

    # Build package
    print("📦 Building package...")
    run_command("python -m build")

    # Upload to PyPI (requires API token)
    print("📤 Uploading to PyPI...")
    run_command("python -m twine upload dist/*")

    print("✅ Release complete!")


if __name__ == "__main__":
    main()
