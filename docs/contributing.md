# Contributing

See the repository [contributing guide](https://github.com/diogoribeiro7/OversampleQA/blob/main/CONTRIBUTING.md) for development setup, review expectations, and pull request guidance.

## Local Checks

```bash
make lint typecheck
make test
make docs
```

## Performance Profiling

Optional scripts profile the hot paths, including distance-matrix computation
and validation:

```bash
make profile
poetry run python scripts/profile_performance.py --save perf_baseline.json
poetry run python scripts/profile_performance.py --check perf_baseline.json --tolerance 1.5
```

For a portable JSON artifact with runtime, peak traced memory and environment
metadata for the core validation paths:

```bash
poetry run python scripts/benchmark_core_paths.py --quick --output core_paths_current.json
```

The scheduled Performance workflow uploads both `perf_current.json` and
`core_paths_current.json`. Those files are comparison artifacts, not required
pull request gates.
