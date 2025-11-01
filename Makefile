.PHONY: help install test lint format clean docs

help:
	@echo "Available commands:"
	@echo "  install    Install package and dependencies"
	@echo "  test       Run tests"
	@echo "  lint       Run linting"
	@echo "  format     Format code"
	@echo "  clean      Clean build artifacts"
	@echo "  docs       Build documentation"

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=oversampleqa --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/oversampleqa

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

docs:
	cd docs && make html
