# OversampleQA

[![CI](https://img.shields.io/github/actions/workflow/status/diogoribeiro7/OversampleQA/ci.yml?branch=main)](https://github.com/diogoribeiro7/OversampleQA/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/oversampleqa.svg)](https://pypi.org/project/oversampleqa/)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://diogoribeiro7.github.io/OversampleQA/)
[![License](https://img.shields.io/github/license/diogoribeiro7/OversampleQA)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21940361.svg)](https://doi.org/10.5281/zenodo.21940361)

A diagnostic toolkit to validate, audit, and benchmark oversampling methods for imbalanced classification.

**Documentation:** https://diogoribeiro7.github.io/OversampleQA/

## What It Does

- Validates oversampling quality with a hidden-majority error rate (binary and multiclass).
- Offers a broad set of distance metrics (Hassanat, Euclidean, Mahalanobis, etc.).
- Includes optimized and memory-efficient distance matrix computation.
- Supports benchmarking across datasets and oversamplers with exportable results.
- Provides a rich CLI with profiles, templates, shell completion, and diagnostics.
- Extensible via a plugin system for custom metrics and validators.

## Concepts

OversampleQA validates synthetic samples by hiding a portion of the majority class and asking whether generated points look more like the hidden majority or the real minority. Each synthetic sample is scored by its nearest-neighbor distance to both groups using a chosen metric. If a synthetic sample is closer to the hidden majority than to the minority, it is counted as an error. The resulting error rate is a direct signal of how often oversampling produces majority-like artifacts. Lower error rates suggest better minority fidelity, but the absolute value depends on the dataset, metric, and hidden ratio. For multiclass data, the same idea generalizes to a confusion-style error matrix across classes.

## Install

Python 3.10+ is required. Install the latest release from PyPI:

```bash
pip install oversampleqa
```

For the optional performance helpers:

```bash
pip install "oversampleqa[performance]"
```

For development or unreleased changes, install from source:

```bash
git clone https://github.com/diogoribeiro7/OversampleQA.git
cd OversampleQA
poetry install
```

To depend on the current repository state from another project:

```bash
pip install git+https://github.com/diogoribeiro7/OversampleQA.git
```

## Quick Start (Python)

```bash
python - <<'PY'
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from oversampleqa import validate_oversampling

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=10,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    random_state=42,
)

error_rate = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,
    oversampler=SMOTE(random_state=42),
    hidden_ratio=0.1,
    metric="hassanat",
)

print(f"SMOTE validation error rate: {error_rate:.3f}")
PY
```

## CLI

```bash
oversampleqa --help
```

```bash
oversampleqa validate data.csv \
  --target target \
  --minority-label 1 \
  --oversampler SMOTE \
  --metric hassanat \
  --hidden-ratio 0.1 \
  --export json \
  --output runs
```

```bash
oversampleqa profiles
oversampleqa template --template production -o oversampleqa.yaml
oversampleqa benchmark --output benchmark_results
oversampleqa doctor
```

Legacy minimal CLI (if you prefer a smaller surface):

```bash
oversampleqa-validate --help
```

## Configuration

The enhanced CLI loads configuration from `~/.oversampleqa/config.yaml` by default. You can override it with `--config` and select profiles with `--profile`.

## Examples And Docs

- Code samples live in `examples/` and `tutorials/`.
- MkDocs documentation sources are in `docs/`; the site configuration is `mkdocs.yml`.
- Published documentation: https://diogoribeiro7.github.io/OversampleQA/

Build docs:

```bash
make docs
```

## Development

```bash
# One-liner
make setup

# Or run onboarding helper
poetry run python scripts/onboard.py

# Manual steps
poetry install
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

## Quality Checks

```bash
# Lint and typecheck -- both pass clean
make lint typecheck
# Run tests with coverage
make coverage
# Security audit
make security
# Build docs the way CI does, with warnings as errors
make docs
# Full pre-commit suite
poetry run pre-commit run --all-files
```

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/) alone;
the enforced rule set lives in `pyproject.toml`. These commands work on Windows
and Linux alike.

## Citation

If you use OversampleQA in academic work, please cite it. Machine-readable metadata
lives in `CITATION.cff`, which GitHub renders as a "Cite this repository" button.

```bibtex
@software{ribeiro_oversampleqa,
  author  = {Ribeiro, Diogo},
  title   = {{OversampleQA: a diagnostic toolkit to validate, audit,
             and benchmark oversampling methods}},
  version = {0.8.0},
  year    = {2026},
  doi     = {10.5281/zenodo.21940361},
  url     = {https://doi.org/10.5281/zenodo.21940361}
}
```

The DOI above is the concept DOI: it always resolves to the newest archived
version. To cite this exact release instead, use `10.5281/zenodo.22215998`.
Every archived version's DOI is listed in `CITATION.cff`; see
[Citing OversampleQA](docs/citation.md) for the release and DOI sequence, and
for why 0.5.1 has no record of its own.

## License

MIT. See `LICENSE`.
