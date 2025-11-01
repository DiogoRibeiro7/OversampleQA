# Contributing to OversampleQA

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install in development mode: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/`
4. Commit changes: `git commit -m "feat: your change description"`
5. Push and create a pull request

## Code Standards

- Use Black for code formatting
- Follow PEP 8 guidelines
- Write tests for new functionality
- Update documentation as needed
- Follow conventional commit messages

## Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=oversampleqa

# Run specific test file
pytest tests/test_validator.py
```

## Questions?

Open an issue or start a discussion on GitHub.
