"""Tests for fold-level benchmark export.

The summary frame reports a mean and interval per (dataset, oversampler,
metric). That is enough to read a ranking and not enough to check one, because
it does not say how many folds contributed or why the rest did not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification

from oversampleqa.advanced_benchmark import _FOLD_COLUMNS, StatisticalBenchmark


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(
        n_samples=900,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        random_state=0,
    )
    return [{"name": "d", "data": X, "target": y, "minority_label": 1}]


@pytest.fixture(scope="module")
def run(dataset):
    bench = StatisticalBenchmark(n_folds=3, n_repeats=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = bench.run_comprehensive_benchmark(
            dataset,
            [SMOTE(random_state=0), RandomOverSampler(random_state=0)],
            metrics=["hassanat"],
        )
    return bench, summary


def _tiny_dataset():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (120, 3)), rng.normal(3, 1, (9, 3))])
    y = np.array([0] * 120 + [1] * 9)
    return [{"name": "tiny", "data": X, "target": y, "minority_label": 1}]


def test_columns_are_fixed(run):
    bench, _ = run
    assert list(bench.fold_results().columns) == list(_FOLD_COLUMNS)


def test_one_row_per_repeat_and_fold(run):
    bench, _ = run
    folds = bench.fold_results()
    # 2 oversamplers x 1 metric x 2 repeats x 3 folds
    assert len(folds) == 12
    key = ["dataset_name", "oversampler_name", "metric", "repeat", "fold"]
    assert not folds.duplicated(key).any()


def test_fold_means_reconcile_with_the_summary(run):
    """If these disagree, one of the two frames is lying."""
    bench, summary = run
    folds = bench.fold_results()
    from_folds = (
        folds[~folds["skipped"]].groupby("oversampler_name")["error_rate"].mean()
    )
    from_summary = summary.set_index("oversampler_name")["mean_error"]
    for name, value in from_summary.items():
        assert from_folds[name] == pytest.approx(value)


def test_split_seed_is_constant_within_a_repeat(run):
    """One splitter per repeat; the seed identifies it."""
    bench, _ = run
    folds = bench.fold_results()
    per_repeat = folds.groupby("repeat")["split_seed"].nunique()
    assert (per_repeat == 1).all()


def test_split_seed_differs_across_repeats(run):
    """Repeats drawing the same split would not be repeats."""
    bench, _ = run
    folds = bench.fold_results()
    seeds = folds.groupby("repeat")["split_seed"].first()
    assert seeds.nunique() == len(seeds)


def test_hidden_ratio_is_recorded(run):
    bench, _ = run
    assert set(bench.fold_results()["hidden_ratio"]) == {0.1}


def test_successful_folds_carry_no_skip_reason(run):
    bench, _ = run
    folds = bench.fold_results()
    survived = folds[~folds["skipped"]]
    assert (survived["skip_reason"] == "").all()
    assert survived["error_rate"].notna().all()


def test_skipped_folds_are_kept_with_a_reason():
    """The case that used to vanish entirely.

    A minority too small to hold out from produces no summary row at all. The
    fold frame still shows every attempt and why each failed.
    """
    bench = StatisticalBenchmark(n_folds=3, n_repeats=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = bench.run_comprehensive_benchmark(
            _tiny_dataset(), [RandomOverSampler(random_state=0)], metrics=["hassanat"]
        )
    folds = bench.fold_results()
    assert summary.empty
    assert len(folds) == 6
    assert folds["skipped"].all()
    assert folds["error_rate"].isna().all()
    assert folds["skip_reason"].str.len().gt(0).all()
    assert "min_hidden" in folds["skip_reason"].iloc[0]


def test_contributing_fold_count_is_recoverable():
    """A mean over 3 of 25 folds looks identical to one over 25 once skips go."""
    bench = StatisticalBenchmark(n_folds=3, n_repeats=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench.run_comprehensive_benchmark(
            _tiny_dataset(), [RandomOverSampler(random_state=0)], metrics=["hassanat"]
        )
    folds = bench.fold_results()
    assert int((~folds["skipped"]).sum()) == 0
    assert int(folds["skipped"].sum()) == 6


def test_empty_before_any_run_but_keeps_columns():
    """A (0, 0) frame raises KeyError on any column access."""
    folds = StatisticalBenchmark().fold_results()
    assert folds.empty
    assert list(folds.columns) == list(_FOLD_COLUMNS)
    assert folds["error_rate"].tolist() == []


def test_a_reused_engine_does_not_accumulate_folds(dataset):
    bench = StatisticalBenchmark(n_folds=3, n_repeats=1, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench.run_comprehensive_benchmark(
            dataset, [SMOTE(random_state=0)], metrics=["hassanat"]
        )
        first = len(bench.fold_results())
        bench.run_comprehensive_benchmark(
            dataset, [SMOTE(random_state=0)], metrics=["hassanat"]
        )
        second = len(bench.fold_results())
    assert first == second


def test_summary_is_unchanged_by_the_addition(run):
    """Adding the fold frame must not alter what the summary reports."""
    _, summary = run
    assert {"dataset_name", "oversampler_name", "metric", "mean_error"} <= set(
        summary.columns
    )
    assert len(summary) == 2
