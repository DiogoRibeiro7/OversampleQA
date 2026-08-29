"""Smoke-test an OversampleQA wheel from a clean virtual environment.

This catches packaging mistakes that source-tree tests cannot see: missing
package data, broken console scripts, undeclared runtime dependencies and root
imports that only work because ``src/`` is on the developer machine.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SMOKE_CODE = r"""
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE

import oversampleqa
from oversampleqa import OversamplingValidator, validate_oversampling, validation_scorer

package_path = Path(oversampleqa.__file__).resolve()
if "src" in package_path.parts:
    raise SystemExit(f"imported from source tree instead of wheel: {package_path}")

rng = np.random.default_rng(7)
majority = rng.normal(loc=0.0, scale=0.4, size=(80, 2))
minority = rng.normal(loc=3.0, scale=0.4, size=(30, 2))
X = np.vstack([majority, minority])
y = np.array([0] * len(majority) + [1] * len(minority))

rate = validate_oversampling(
    X,
    y,
    minority_label=1,
    oversampler=SMOTE(random_state=0, k_neighbors=3),
    hidden_ratio=0.2,
    metric="euclidean",
    random_state=0,
)
if not 0.0 <= rate <= 1.0:
    raise SystemExit(f"invalid validation rate: {rate!r}")

validator = OversamplingValidator(
    SMOTE(random_state=1, k_neighbors=3),
    minority_label=1,
    hidden_ratio=0.2,
    metric="euclidean",
    random_state=0,
)
validator.fit(X, y)
score = validation_scorer(validator, X, y)
if score > 0.0:
    raise SystemExit(f"validation_scorer must follow greater-is-better: {score!r}")

print(
    f"installed smoke ok: oversampleqa {oversampleqa.__version__}, "
    f"rate={rate:.3f}, score={score:.3f}"
)
"""


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_script(venv_dir: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    return venv_dir / scripts_dir / f"{name}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install an OversampleQA wheel in a clean venv and smoke-test it."
    )
    parser.add_argument("wheel", type=Path, help="Path to the built wheel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"expected a wheel file, got {wheel}")

    with tempfile.TemporaryDirectory(prefix="oversampleqa-smoke-") as tmp:
        work_dir = Path(tmp)
        venv_dir = work_dir / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=work_dir)

        python = _venv_python(venv_dir)
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=work_dir,
        )
        _run([str(python), "-c", textwrap.dedent(SMOKE_CODE)], cwd=work_dir)

        for command in ("oversampleqa", "oversampleqa-validate"):
            _run([str(_venv_script(venv_dir, command)), "--help"], cwd=work_dir)


if __name__ == "__main__":
    main()
