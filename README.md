
## Development

### Setup Development Environment

`ash
# One-liner
make setup

# Or run onboarding helper
poetry run python scripts/onboard.py

# Manual steps
poetry install
pre-commit install
pre-commit install --hook-type commit-msg
`

- Dev Containers: open the workspace in VS Code and choose **Reopen in Container** to use the provided .devcontainer setup.
- Local IDEs: the .vscode/ folder contains recommended settings for Ruff/Black integration.

### Quality Checks

`ash
# Fast lint + typecheck
make lint typecheck
# Run tests with coverage gate
make coverage
# Security audit
make security
# Full pre-commit suite
poetry run pre-commit run --all-files
`

