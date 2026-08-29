# Contributing

See the repository [contributing guide](https://github.com/diogoribeiro7/OversampleQA/blob/main/CONTRIBUTING.md) for development setup, review expectations, and pull request guidance.

## Local Checks

```bash
make lint typecheck
make test
make docs
```

## Performance Profiling

An optional script profiles the hot paths, including distance-matrix computation
and validation:

```bash
make profile
poetry run python scripts/profile_performance.py --save perf_baseline.json
poetry run python scripts/profile_performance.py --check perf_baseline.json --tolerance 1.5
```
