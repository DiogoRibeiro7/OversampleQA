# Advanced Tutorial

This tutorial walks through comparing multiple oversamplers and
validating robustness.

## Create a dataset

``` python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1200,
    n_features=15,
    weights=[0.9, 0.1],
    random_state=0,
)
```

## Compare oversamplers with repeated validation

``` python
from imblearn.over_sampling import ADASYN, SMOTE
from oversampleqa.validator import validate_oversampling

oversamplers = {
    "SMOTE": SMOTE(random_state=0),
    "ADASYN": ADASYN(random_state=0),
}

for name, sampler in oversamplers.items():
    details = validate_oversampling(
        X,
        y,
        minority_label=1,
        oversampler=sampler,
        hidden_ratio=0.1,
        metric="hassanat",
        random_state=42,
        n_repeats=10,
        return_details=True,
    )
    print(name, details.mean, details.std, details.interval)
```

## Run a statistical benchmark

Use `StatisticalBenchmark` when comparing several samplers or datasets. The
summary reports means and intervals; `fold_results()` exposes the repeat, fold,
seed, metric, hidden ratio, reference mode, and package version that identify
each measurement.

``` python
from oversampleqa import StatisticalBenchmark

datasets = [
    {
        "name": "classification_demo",
        "data": X,
        "target": y,
        "minority_label": 1,
    }
]

bench = StatisticalBenchmark(n_folds=3, n_repeats=3, random_state=42)
summary = bench.run_comprehensive_benchmark(
    datasets,
    list(oversamplers.values()),
    metrics=("hassanat", "euclidean"),
)

folds = bench.fold_results()
print(summary[["dataset_name", "oversampler_name", "metric", "mean_error", "ci_lower", "ci_upper"]])
print(folds[["repeat", "fold", "split_seed", "metric", "hidden_ratio", "oversampleqa_version"]].head())
```

## Validate across metrics

``` python
metrics = ["hassanat", "euclidean", "cosine"]
for metric in metrics:
    details = validate_oversampling(
        X,
        y,
        minority_label=1,
        oversampler=SMOTE(random_state=0),
        metric=metric,
        random_state=42,
        n_repeats=5,
        return_details=True,
    )
    print(metric, details.mean, details.std)
```
