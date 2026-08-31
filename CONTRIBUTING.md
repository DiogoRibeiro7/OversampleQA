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

## Performance Profiling

An optional script profiles the hot paths (distance-matrix computation and the
validator) and can guard against regressions. It is not part of the default test
suite.

```bash
# Print a timing table
make profile            # or: python scripts/profile_performance.py

# Save a baseline, then check a later change against it
python scripts/profile_performance.py --save perf_baseline.json
python scripts/profile_performance.py --check perf_baseline.json --tolerance 1.5
```

The check exits non-zero if any benchmark is slower than `tolerance` times its
baseline, so it can run in CI or a pre-release gate.

`perf_baseline.json` is committed at the repository root and the weekly
Performance workflow checks against it. It was **measured on a GitHub
`ubuntu-latest` runner**, not on a developer machine, because that is where the
check runs -- a baseline recorded on your laptop compares your hardware against
CI's and reports differences that are not regressions.

To refresh it, run the Performance workflow (Actions -> Performance -> Run
workflow), download the `performance-profile` artifact, and commit its
`perf_current.json` as `perf_baseline.json`. Overwriting it from a local
`--save` run will produce a baseline that is not comparable to CI.

The script warns when the recorded environment differs from the current one, so
an incomparable comparison is visible rather than silently misleading. Expect
that warning to appear as GitHub updates its runner images.

## Changelog

A pull request that changes anything under `src/` must also add an entry under
`## [Unreleased]` in `CHANGELOG.md`. CI enforces this.

Documentation, CI and test-only pull requests are exempt automatically, because
the rule keys on `src/` rather than on judgement. If a source change genuinely
has no user-visible effect -- a pure refactor, say -- apply the `no-changelog`
label, which records the decision on the pull request instead of leaving it
implicit.

The rule exists because nine of the ten commits touching `src/` between 0.6.1
and 0.7.0 shipped without an entry, every feature in that milestone among them.
The release notes had to be reconstructed from pull request descriptions before
0.7.0 could be cut honestly. The release-metadata tests did not catch it: they
assert a changelog section *exists* for the version, not that it describes the
release.

## Questions?

Open an issue or start a discussion on GitHub.
