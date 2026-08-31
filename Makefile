.PHONY: help setup install hooks test lint typecheck format coverage security clean docs profile

# Every recipe below runs the same on Windows, macOS and Linux.
#
# `clean` used to call rm/find and `docs` used to `cd docs && make html`, neither
# of which works in PowerShell -- while development happens on Windows. Shell
# built-ins are replaced with Python equivalents, which are available wherever
# the project itself can run.

help:
	@echo "Available commands:"
	@echo "  setup       Install dependencies and git hooks"
	@echo "  install     Install package and dependencies"
	@echo "  hooks       Install pre-commit hooks"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  typecheck   Run mypy"
	@echo "  format      Format code"
	@echo "  coverage    Run tests with coverage"
	@echo "  security    Run security checks"
	@echo "  clean       Clean build artifacts"
	@echo "  docs        Build documentation (warnings are errors)"
	@echo "  profile     Run optional performance profiling"

setup: install hooks

install:
	poetry install --no-interaction

hooks:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg

test:
	poetry run pytest tests/ -v

coverage:
	poetry run pytest tests/ --cov=oversampleqa --cov-report=html

lint:
	poetry run ruff check src/ tests/ scripts/

typecheck:
	poetry run mypy src/oversampleqa

format:
	poetry run ruff format src/ tests/ examples/
	poetry run ruff check --fix src/ tests/ examples/

security:
	poetry run bandit -r src/
	poetry run safety check

clean:
	poetry run python scripts/clean.py

# --strict matches CI: a docs warning is a build failure.
docs:
	poetry run mkdocs build --strict

profile:
	poetry run python scripts/profile_performance.py
