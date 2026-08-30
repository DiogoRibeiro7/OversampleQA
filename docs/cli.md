# Command Line Interface

OversampleQA ships two CLI entry points:

- `oversampleqa`: enhanced CLI with profiles, templates, and
  diagnostics.
- `oversampleqa-validate`: legacy minimal CLI for quick CSV validation.

## Enhanced CLI

### Basic usage

``` bash
oversampleqa --help
```

### Validate a dataset

``` bash
oversampleqa validate data.csv \
  --target target \
  --minority-label 1 \
  --oversampler SMOTE \
  --metric hassanat \
  --hidden-ratio 0.1 \
  --export json \
  --output runs
```

### Tiny example dataset

``` bash
printf "x1,x2,target\n1.0,0.5,0\n1.1,0.4,0\n0.9,0.6,0\n2.0,1.9,1\n2.1,1.8,1\n" > tiny.csv

oversampleqa validate tiny.csv \
  --target target \
  --minority-label 1 \
  --oversampler SMOTE \
  --metric euclidean \
  --hidden-ratio 0.2
```

### Run interactively

``` bash
oversampleqa validate data.csv --interactive
```

### Configuration profiles

``` bash
oversampleqa profiles
oversampleqa validate data.csv --profile quick
```

### Run a checked-in manifest

Use a manifest when a validation run should be repeatable by another person or
CI job:

``` yaml
version: 1
output: audit
defaults:
  target: target
  minority_label: 1
  metric: hassanat
  hidden_ratio: 0.1
  random_state: 42
  n_repeats: 10
  export: [json, markdown]
  resume: true
datasets:
  production_sample:
    path: data.csv
experiments:
  - name: smote-baseline
    dataset: production_sample
    oversampler: SMOTE
    calibrate: true
  - name: adasyn-check
    dataset: production_sample
    oversampler: ADASYN
```

``` bash
oversampleqa run oversampleqa-experiment.yaml
```

Each experiment writes normal validation exports under its own output
directory. The command also writes `resolved_manifest.yaml`,
`manifest_summary.json`, and `manifest_summary.json.metadata.json` at the
manifest output root.

### Generate a config template

``` bash
oversampleqa template --template production -o oversampleqa.yaml
```

### Benchmark multiple datasets

``` bash
oversampleqa benchmark --output benchmark_results
```

For cross-validated statistical benchmarking with confidence intervals,
pairwise p-values, and effect sizes, add `--statistical` (optionally
tuning `--folds` and `--repeats`):

``` bash
oversampleqa benchmark --statistical --folds 5 --repeats 5 -o benchmark_results
```

This prints a summary table and writes `benchmark_statistics.csv`,
`benchmark_summary.md`, and `benchmark_report.html` to the output
directory.

### Shell completion

``` bash
oversampleqa completion bash
```

### Diagnostics

``` bash
oversampleqa doctor
```

### Initial setup

Run the guided wizard to create a configuration file:

``` bash
oversampleqa setup
```

### Global options

These apply to any subcommand and come before it:

- `--config/-c`: path to a configuration file (default
  `~/.oversampleqa/config.yaml`).
- `--profile/-p`: configuration profile to apply.
- `--verbose/-v`: enable verbose output.
- `--version`: print the version and exit.

### Common options for `run`

- `--output`: override the manifest output directory.
- `--resume/--no-resume`: override per-experiment resume settings.

### Common options for `validate`

- `--target`: target column name in the dataset.
- `--minority-label`: minority class label value.
- `--oversampler`: imbalanced-learn oversampler class name.
- `--metric`: distance metric to use.
- `--hidden-ratio`: fraction of majority samples to hide.
- `--export`: output formats (`json`, `yaml`, `markdown`).
- `--output`: directory to store outputs.
- `--resume/--no-resume`: reuse cached results when available.
- `--interactive`: guided validation wizard.
- `--mlflow`: log results to MLflow if installed.

## Legacy CLI (minimal)

``` bash
oversampleqa-validate data.csv --target target --minority-label 1 --oversampler SMOTE
```

### Options

- `--target`: name of the target column.
- `--minority-label`: minority label value.
- `--oversampler`: imbalanced-learn oversampler class name.
- `--hidden-ratio`: fraction of majority samples to hide.
- `--distance`: distance metric name.
- `--out`: optional text report output path.
- `--plot`: optional plot output path.

## Fidelity report

The error rate is one scalar covering two failures that need opposite
fixes: generating implausible points, and merely copying the training
minority. The `fidelity` subcommand reports both axes.

``` bash
oversampleqa fidelity data.csv      --target target      --minority-label 1      --oversampler SMOTE      --metric hassanat      -o fidelity.json
```

Output includes precision, recall, density and coverage, the
memorisation ratio, and the boundary-violation rate. Add `--utility` to
also fit models and measure downstream gain — much slower, since it
trains a classifier per fold.

The memorisation ratio is the one to read first. Near zero means the
generator sits on top of its training data, and the error rate cannot
say anything about synthesis quality:

``` text
SMOTE              error=0.083  precision=1.000  memorisation=0.360
RandomOverSampler  error=0.115  precision=0.980  memorisation=0.000
```

See `fidelity` for the interpretation table.
