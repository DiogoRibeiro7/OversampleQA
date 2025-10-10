# oversampleqa — Detailed Roadmap

This document outlines the development of `oversampleqa`, a validation and benchmarking library for imbalanced classification workflows. The package focuses not just on function-by-function implementation but also on high-level goals, target use cases, and diagnostic coverage.

---

## 🎯 Package Focus

A validation and benchmarking library for models and preprocessing methods applied to **imbalanced classification problems**, with special focus on:

* ✅ Oversampling/undersampling validation
* ✅ Error analysis of synthetic samples
* ✅ Sensitivity diagnostics for imbalance mitigation
* ✅ Distributional checks (e.g., overlap with hidden majority)
* ✅ Human-understandable reports and plots
* ✅ Modular, extensible design compatible with scikit-learn and imbalanced-learn

---

## 🧱 Modules & Responsibilities

| Module         | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `distance.py`  | Implement multiple distance metrics (Hassanat, Euclidean, Mahalanobis) |
| `validator.py` | Validate oversampling using hidden majority comparison                 |
| `metrics.py`   | Compute error rates, overlaps, density divergence                      |
| `benchmark.py` | Run batched evaluations across datasets and oversampling methods       |
| `plotting.py`  | 2D/3D visualizations (PCA, UMAP) of synthetic vs real examples         |
| `report.py`    | Export JSON/Markdown/HTML summary reports                              |
| `cli.py`       | (Optional) Command-line interface to run validations and exports       |

---

## 📦 Phase 1 — Core Functionality

### 1.1. `distance.py`

* ✅ `hassanat_distance()` implementation
* ✅ Add `euclidean_distance()`, `manhattan_distance()`, `cosine_distance()`
* ✅ General-purpose `distance_matrix(X1, X2, metric="hassanat")`

### 1.2. `validator.py`

* ✅ `validate_oversampling(X, y, oversampler, minority_label, hidden_ratio)`
* ✅ Extract synthetic samples from post-resample
* ✅ Compare synthetic to hidden majority vs real minority
* ✅ Support multiple distance metrics
* ✅ Return error counts and full similarity matrix (optional)

### 1.3. `metrics.py`

* ✅ Calculate error rate: # majority-like synthetic / total synthetic
* ✅ Confidence ratio: dist\_min / dist\_maj
* ✅ Local density divergence
* ✅ Minority-class recall loss (on synthetic data)

---

## 📊 Phase 2 — Benchmarking and Reporting

### 2.1. `benchmark.py`

* ✅ Load and normalize common datasets (Yeast4/5/6, Vehicle3)
* ✅ Batch run all oversamplers on each dataset
* ✅ Repeat with 10%, 25%, 50% hidden majority
* ✅ Export CSV, JSON, Markdown summaries with error, rank, stddev

### 2.2. `plotting.py`

* ✅ PCA/UMAP 2D scatter plot of:

  * Majority
  * Minority
  * Hidden Majority
  * Synthetic samples
* ✅ Boxplot of error rates across oversamplers
* ✅ Rank vs error line chart

### 2.3. `report.py`

* ✅ Write markdown and HTML reports with:

  * Table of metrics per sampler
  * Charts and plots (as images)
  * Summary rank table

### 2.4. Additional distance metrics

* ✅ Implement Hellinger distance for probability vectors
* ✅ Implement Jensen–Shannon distance for probability vectors

---

## 🧠 Phase 3 — Advanced Diagnostics

### 3.1. Surrogate Model Evaluation

* ✅ Train model on:

  * Real-only
  * Real + synthetic
  * Synthetic-only
* ✅ Compare F1/Recall/Precision on held-out test set
* ✅ Report shift in generalization

### 3.2. Cluster-Based Diagnostics

* ✅ Cluster majority + synthetic via k-means / DBSCAN
* ✅ Flag synthetic examples in high-density majority regions
* ✅ Add overlap score via silhouette or centroid distance

### 3.3. Multi-Class Extension

* ✅ Generalize validation logic to handle >2 classes
* ✅ Matrix of error rates per (synthetic class, true proximity class)

---

## 🧪 Optional Modules / Extensions

* ✅ **Manifold comparison** using UMAP projections
* ✅ **Model fairness checker** (minority group calibration after oversampling)
* ✅ **Clustering diagnostics** as alternative to distance validation
* ✅ **Noise sensitivity diagnostics** (effect of label noise on synthetic risk)

---

## 🎯 Target Audience

* Researchers experimenting with **imbalanced classification** techniques
* ML engineers working on **fraud, medical, security, safety-critical systems**
* Developers using **`imbalanced-learn`**, **`smote-variants`**, or **AutoML** pipelines
* Anyone concerned with **trust, generalization, and validity of synthetic data**

---

## 📁 Phase 4 — CLI and Packaging

### 4.1. CLI Interface

* ✅ `oversampleqa-validate path/to.csv --oversampler=SMOTE --hidden=0.25`
* ✅ Optional flags: `--out report.md`, `--plot pca.png`, `--distance hassanat`

### 4.2. PyPI Support

* ✅ Create `pyproject.toml` using Poetry
* ✅ Add CLI entry point
* ✅ Register package on PyPI
* ✅ Provide installation instructions and versioning in README

All roadmap phases are complete as of version 0.1.0.

---

## 🛡 Phase 5 — Reliability and Logging

* ✅ Add consistent logging to validator, benchmark, clustering, and surrogate modules
* ✅ Wrap oversampler and clustering calls in try/except blocks
* ✅ Log errors in the CLI and propagate exceptions cleanly
* ✅ Document logging setup in the README

All reliability tasks are complete.

---

## 📈 Long-Term Vision

* Provide the **first real diagnostic tool** for oversampling that goes beyond metrics
* Promote **transparency and interpretability** in imbalanced learning workflows
* Help prevent overfitting to fake data in sensitive, real-world applications
