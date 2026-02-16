.PHONY: help setup install hooks test lint typecheck format coverage security clean docs

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
	@echo "  docs        Build documentation"

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
	poetry run flake8 src/ tests/
	poetry run ruff check src/ tests/

typecheck:
	poetry run mypy src/oversampleqa

format:
	poetry run black src/ tests/ examples/
	poetry run isort src/ tests/ examples/

security:
	poetry run bandit -r src/
	poetry run safety check

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && poetry run make html