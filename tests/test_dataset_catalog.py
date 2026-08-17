"""The built-in datasets must be usable for the thing they exist for.

Three defects, all of which made a dataset contribute nothing:

* ``DatasetRepository`` declared ``minority_label=1`` for ``load_breast_cancer``,
  whose minority is class 0 (212 malignant against 357 benign) -- so the
  benchmark oversampled the majority. ``max_samples`` then inverts it again: the
  first 200 rows are 104 class-0 against 96 class-1.
* ``make_moons`` and ``make_circles`` return exactly balanced classes. There is
  nothing to oversample, so SMOTE generated no points and the error rate was
  ``nan``.
* Every built-in dataset's minority was too small for the package's own default
  ``hidden_ratio=0.1`` against ``min_hidden=5``, which needs 50 minority points.
  The largest had 20. The whole catalog returned ``nan``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer

from oversampleqa.advanced_benchmark import DatasetRepository
from oversampleqa.benchmark import load_standard_datasets, run_benchmark
from oversampleqa.validator import infer_minority_label

DEFAULT_HIDDEN_RATIO = 0.1
MIN_HIDDEN = 5


@pytest.fixture(scope="module")
def catalog():
    return load_standard_datasets()


def test_declared_minority_label_matches_the_data(catalog):
    for dataset in catalog:
        y = np.asarray(dataset["target"]).astype(int)
        assert dataset["minority_label"] == infer_minority_label(y), dataset["name"]


def test_no_dataset_is_balanced(catalog):
    """A balanced dataset has nothing to oversample."""
    for dataset in catalog:
        y = np.asarray(dataset["target"]).astype(int)
        counts = np.bincount(y, minlength=2)
        assert counts[0] != counts[1], f"{dataset['name']} is balanced"


def test_every_dataset_supports_the_default_holdout(catalog):
    """Needing 50 minority points, the largest built-in had 20."""
    for dataset in catalog:
        y = np.asarray(dataset["target"]).astype(int)
        n_minority = int(np.sum(y == dataset["minority_label"]))
        held_out = int(n_minority * DEFAULT_HIDDEN_RATIO)
        assert held_out >= MIN_HIDDEN, (
            f"{dataset['name']}: {n_minority} minority points hold out "
            f"{held_out}, below min_hidden={MIN_HIDDEN}"
        )


def test_the_catalog_produces_measurements_not_nan(catalog):
    """The whole point. Every row was nan at default settings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_benchmark(catalog, [SMOTE(random_state=0)], n_runs=1)
    assert not results.empty
    assert results["error_rate"].notna().all(), (
        "these datasets are the documented starting point; a nan here means "
        "the catalog cannot demonstrate the package's own default workflow"
    )


def test_difficulty_ordering_is_visible(catalog):
    """A catalog whose datasets all score alike would not be a benchmark."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_benchmark(catalog, [SMOTE(random_state=0)], n_runs=1)
    per_dataset = results.groupby("dataset")["error_rate"].mean()
    assert per_dataset["easy_linear"] < per_dataset["hard_classification"]


# --- infer_minority_label ---


def test_infers_the_least_frequent_label():
    y = np.array([0] * 90 + [1] * 10)
    assert infer_minority_label(y) == 1


def test_infers_a_non_binary_minority():
    y = np.array([0] * 50 + [1] * 30 + [2] * 5)
    assert infer_minority_label(y) == 2


def test_breast_cancer_minority_is_class_zero():
    """The declared value was 1, so the benchmark oversampled the majority."""
    assert infer_minority_label(load_breast_cancer().target) == 0


def test_truncation_can_invert_the_minority():
    """Which is why no hardcoded label can be right at every max_samples."""
    target = load_breast_cancer().target
    assert infer_minority_label(target) == 0
    assert infer_minority_label(target[:200]) == 1


def test_ties_resolve_deterministically():
    y = np.array([0] * 10 + [1] * 10)
    assert infer_minority_label(y) == infer_minority_label(y) == 0


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        infer_minority_label(np.array([], dtype=int))


# --- DatasetRepository ---


@pytest.mark.parametrize("max_samples", [200, 10_000])
def test_repository_labels_track_the_truncated_data(max_samples):
    repo = DatasetRepository()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datasets = repo.load_research_datasets(max_samples=max_samples)
    assert datasets
    for dataset in datasets:
        y = np.asarray(dataset["target"]).astype(int)
        assert dataset["minority_label"] == infer_minority_label(y), (
            f"{dataset['name']} at max_samples={max_samples}"
        )


def test_synthetic_suite_labels_match_the_data():
    repo = DatasetRepository()
    for dataset in repo.create_synthetic_benchmark_suite(["easy", "hard"]):
        y = np.asarray(dataset["target"]).astype(int)
        assert dataset["minority_label"] == infer_minority_label(y), dataset["name"]
