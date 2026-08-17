"""Pairwise statistics must belong to the row they are attached to.

`_add_statistical_analysis` grouped by dataset alone. With several metrics per
dataset the slice held several rows per oversampler, the lookups took
``.iloc[0]``, and so the tests ran on whichever metric sorted first -- then
stamped that result onto every row, including rows for the other metrics. Error
rates are not comparable across metrics, so those p-values described a
comparison the row did not represent.

The effect size was independent-samples Cohen's d beside a paired Wilcoxon
test. On fold errors that move together it understates the effect the test is
detecting: measured -0.573 where the matched-pairs rank-biserial is -1.0.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification

from oversampleqa.advanced_benchmark import StatisticalBenchmark


@pytest.fixture(scope="module")
def multi_metric_run():
    X, y = make_classification(
        n_samples=900,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        random_state=0,
    )
    datasets = [{"name": "d", "data": X, "target": y, "minority_label": 1}]
    bench = StatisticalBenchmark(n_folds=3, n_repeats=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = bench.run_comprehensive_benchmark(
            datasets,
            [SMOTE(random_state=0), RandomOverSampler(random_state=0)],
            metrics=["hassanat", "euclidean"],
        )
    return bench, frame


def _effects(frame, metric):
    row = frame[frame["metric"] == metric].iloc[0]
    return json.loads(row["pairwise_effect_sizes"])


def test_effect_sizes_differ_between_metrics(multi_metric_run):
    """They were identical, because both came from one metric's data."""
    _, frame = multi_metric_run
    assert _effects(frame, "hassanat") != _effects(frame, "euclidean")


def test_every_row_of_a_metric_carries_that_metric_statistics(multi_metric_run):
    _, frame = multi_metric_run
    for metric in ("hassanat", "euclidean"):
        rows = frame[frame["metric"] == metric]
        payloads = {row["pairwise_effect_sizes"] for _, row in rows.iterrows()}
        assert len(payloads) == 1, "rows of one metric disagree"


def test_effect_size_matches_the_rows_own_error_rates(multi_metric_run):
    """Recomputed from the row's data, not taken on trust."""
    bench, frame = multi_metric_run
    for metric in ("hassanat", "euclidean"):
        rows = frame[frame["metric"] == metric]
        smote = np.asarray(
            rows.loc[rows["oversampler_name"] == "SMOTE", "error_rates"].iloc[0],
            dtype=float,
        )
        ros = np.asarray(
            rows.loc[
                rows["oversampler_name"] == "RandomOverSampler", "error_rates"
            ].iloc[0],
            dtype=float,
        )
        expected = bench._rank_biserial(smote, ros)
        recorded = _effects(frame, metric)["SMOTE_vs_RandomOverSampler"]
        assert recorded == pytest.approx(expected)


# --- rank-biserial ---


@pytest.fixture
def bench():
    return StatisticalBenchmark()


def test_uniformly_worse_gives_plus_one(bench):
    x = np.array([0.5, 0.6, 0.7, 0.8])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    assert bench._rank_biserial(x, y) == pytest.approx(1.0)


def test_uniformly_better_gives_minus_one(bench):
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.5, 0.6, 0.7, 0.8])
    assert bench._rank_biserial(x, y) == pytest.approx(-1.0)


def test_sign_says_which_sampler_is_better(bench):
    """Positive means the first had higher error, so positive favours the second."""
    worse_first = bench._rank_biserial(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
    assert worse_first > 0


def test_bounded_to_plus_minus_one(bench):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rng.normal(size=12), rng.normal(size=12)
        value = bench._rank_biserial(x, y)
        assert value is not None
        assert -1.0 <= value <= 1.0


def test_identical_samples_have_no_effect_to_report(bench):
    """None, not 0.0 -- there is no effect, rather than an effect of exactly nil."""
    x = np.array([0.3, 0.4, 0.5])
    assert bench._rank_biserial(x, x.copy()) is None


def test_zero_differences_are_dropped_like_wilcoxon_does(bench):
    x = np.array([0.5, 0.5, 0.9])
    y = np.array([0.5, 0.5, 0.1])
    assert bench._rank_biserial(x, y) == pytest.approx(1.0)


def test_unpairable_samples_return_none(bench):
    assert bench._rank_biserial(np.array([0.1, 0.2]), np.array([0.1])) is None
    assert bench._rank_biserial(np.array([]), np.array([])) is None


def test_paired_measure_exceeds_pooled_when_folds_move_together(bench):
    """The reason to change it.

    Both samplers find the same folds hard, so between-fold variance dominates
    the pooled deviation while the paired difference is consistent. Cohen's d
    reports a modest effect; the paired measure reports the near-total
    consistency the Wilcoxon test is actually responding to.
    """
    fold_difficulty = np.array([0.10, 0.35, 0.60, 0.85, 0.20, 0.70])
    x = fold_difficulty + 0.03
    y = fold_difficulty
    pooled = bench._pooled_std(x, y)
    cohens_d = (x.mean() - y.mean()) / pooled
    rank_biserial = bench._rank_biserial(x, y)
    assert abs(cohens_d) < 0.2
    assert rank_biserial == pytest.approx(1.0)
