# OversampleQA

> A diagnostic toolkit to validate, audit, and benchmark oversampling methods in imbalanced classification tasks.

---

## 🔍 Overview

**OversampleQA** is a validation library designed to assess the reliability of synthetic sampling methods like SMOTE, ADASYN, and their many variants. Instead of simply trusting evaluation metrics on oversampled data, this toolkit allows you to verify **how realistic and safe the synthetic data truly is**.

It implements a validation methodology introduced in the paper:

> *Stop Oversampling for Class Imbalance Learning: A Critical Review*
> Hassanat et al., 2022 ([arXiv:2202.03579](https://arxiv.org/abs/2202.03579))

---
## Installation

Install the package from PyPI:

```bash
pip install oversampleqa==0.1.0
```


## 📖 Documentation

Generate the API reference locally:

```bash
sphinx-build -b html docs docs/_build
```

Open `docs/_build/index.html` in your browser to explore the documentation, which details the implemented algorithms and includes a complete bibliography.


## ⚙️ Features

* 🔬 **Hassanat Distance-based validation** of synthetic samples
* 🧪 **Error estimation** of how many synthetic samples resemble hidden majority examples
* 📉 Support for **batch validation** across multiple datasets and methods
* 📊 Visualizations of minority, majority, and synthetic data points with PCA or UMAP
* 🔌 Compatible with `imbalanced-learn`, `smote-variants`, and custom samplers
* ✅ Optional CLI and markdown reporting
* 📝 Integrated logging and exception handling across modules
* 📈 Benchmark error boxplots, bar charts, ranking plots, multi-class error heatmaps, noise sensitivity trends, class balance comparisons, and distance histograms
* ⚖️ Fairness checks comparing recall across protected groups
* 🔬 UMAP-based manifold comparison between real and synthetic data
* 🎛 Noise sensitivity diagnostics across label perturbations
* 🧮 Surrogate model evaluation for real vs synthetic training modes
* 🔄 Multi-class oversampling validation
* 🛰 Cluster-based diagnostics to flag majority-overlapping synthetic samples
* 🧭 Extensive distance metrics including Minkowski, Chebyshev, Mahalanobis, Canberra, Jaccard, Bray-Curtis, correlation, energy, Wasserstein, Hellinger, and Jensen–Shannon
* 📐 Metrics such as confidence ratio, local density divergence, and minority recall loss

---

## 🧱 Core Modules

* `distance.py`: Implements Hassanat Distance and other metrics
* `validator.py`: Core validation logic using hidden majority samples
* `metrics.py`: Calculates error rates, confidence ratios, density divergence,
  and minority recall loss
* `benchmark.py`: Run comparisons across oversampling techniques
* `plotting.py`: Visualization of sample distributions via PCA or UMAP
* `report.py`: Structured output in markdown/JSON/HTML formats
* `clustering.py`: Cluster-based diagnostics to flag majority-overlapping synthetic samples

---

## 🚀 Quick Start

```bash
pip install oversampleqa
```

### Minimal Example

```python
from oversampleqa.validator import validate_oversampling
from oversampleqa.distance import hassanat_distance
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1500, weights=[0.9, 0.1], random_state=42)
error = validate_oversampling(
    X=X,
    y=y,
    minority_label=1,
    oversampler=SMOTE(),
    hidden_ratio=0.1
)
print(f"Error rate: {error:.3f}")
```

### CLI Example

You can also run validation directly from the command line using the
``oversampleqa-validate`` script:

```bash
oversampleqa-validate dataset.csv --target label --oversampler SMOTE
```

This command expects a CSV file with features and a target column named
``label``. It will print the estimated error rate after applying SMOTE.

### Built-in Example Datasets

For quick experimentation the package provides ``load_standard_datasets`` which
returns several toy datasets (classification, moons, circles, blobs, a
harder classification variant, a linearly separable set, and a high-overlap
variant) ready for benchmarking:

```python
from oversampleqa.benchmark import load_standard_datasets, run_benchmark
from imblearn.over_sampling import SMOTE

datasets = load_standard_datasets()
results = run_benchmark(datasets, [SMOTE(random_state=0)], hidden_ratios=[0.1], n_runs=1)
print(results.head())

# save a summary with rankings
from oversampleqa.benchmark import export_benchmark_results

export_benchmark_results(results, "summary.csv")
```

### Extracting Synthetic Samples

When working directly with oversampled data, you can isolate just the synthetic
points using ``extract_synthetic_samples``:

```python
from oversampleqa.validator import extract_synthetic_samples

X_res, y_res = SMOTE().fit_resample(X, y)
synthetic = extract_synthetic_samples(X, X_res, y_res, minority_label=1)
```

### Visualizing Synthetic Data

You can quickly visualize the distribution of real and synthetic samples using
``plot_sample_distribution``. Set ``method="umap"`` for a non-linear
projection:

```python
from oversampleqa.plotting import plot_sample_distribution

plot_sample_distribution(majority, minority, synthetic, method="umap")
```

### Surrogate Model Evaluation

Evaluate the impact of synthetic data on downstream models using
``evaluate_surrogate_models``:

```python
from oversampleqa.surrogate import evaluate_surrogate_models
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

results = evaluate_surrogate_models(
    X,
    y,
    minority_label=1,
    oversampler=SMOTE(random_state=0),
    model=LogisticRegression(max_iter=1000),
)
print(results)
```

### Multi-class Validation

To assess oversampling when dealing with more than two classes use
``validate_multiclass_oversampling``:

```python
from oversampleqa.validator import validate_multiclass_oversampling
from imblearn.over_sampling import SMOTE

rates = validate_multiclass_oversampling(
    X,
    y,
    oversampler=SMOTE(random_state=0),
    hidden_ratio=0.1,
)
print(rates)
```

---

## 📦 When to Use This Package

* You use **oversampling** and want to verify it's not generating unsafe or unrealistic data
* You're evaluating models for **high-risk applications** (medical, fraud, safety)
* You need a **benchmarking pipeline** for synthetic data reliability
* You want **better trust and interpretability** for minority class predictions

---

## 🧠 Citing the Underlying Method

If you use this toolkit in academic or applied work, cite:

```
@article{hassanat2022stop,
  title={Stop Oversampling for Class Imbalance Learning: A Critical Review},
  author={Hassanat, Ahmad B. and Tarawneh, Ahmad S. and Altarawneh, Ghada A. and Almuhaimeed, Abdullah},
  journal={arXiv preprint arXiv:2202.03579},
  year={2022}
}
```

---

## 🛠 Roadmap (Completed Features)
All roadmap tasks have been implemented as of version 0.1.0.

* ✅ Surrogate model accuracy validation
* ✅ Cluster-based overlap checks
* ✅ Support for multi-class imbalance
* ✅ Auto-reports with markdown + charts
* ✅ Benchmarking utilities with ranking charts and boxplots
* ✅ Fairness checks and manifold comparison diagnostics
* ✅ Noise sensitivity diagnostics
* ✅ Extended CLI with output and plot flags

---

## Testing and Coverage

Run tests with coverage enabled using:

```bash
poetry run pytest --cov=oversampleqa --cov-report=term-missing
```

A GitHub Actions workflow runs the same checks on each commit.

## 👥 Contributing

We welcome contributions. Please open issues or submit PRs to:

* Add new validation metrics
* Improve benchmarking speed
* Extend visualization support


## 📚 Bibliography

See [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) for references to algorithms and
techniques used throughout this package.
Refer to [CITATION.cff](CITATION.cff) for citation details.

---

## 📄 License

MIT License. See `LICENSE` file.

---

## 👤 Author

Maintained by [Diogo Ribeiro](https://orcid.org/0009-0001-2022-7072)  
Professional: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)  
Personal: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)
