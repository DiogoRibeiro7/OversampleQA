"""Ranking must not average error rates across incomparable experiments.

Error rates are commensurable only within a fixed (dataset, hidden_ratio,
metric). An easy dataset scores near 0.1 and a hard one near 0.9; hassanat
scores roughly twice euclidean on the same data. Pooling them and taking a mean
asks a question with no answer -- and it does not merely blur the ordering, it
inverts it, because the hold-out guards legitimately drop runs and leave the
pooled mean weighted toward whichever experiments survived.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from oversampleqa.benchmark import compute_ranking
from oversampleqa.inference import friedman_nemenyi

SPEC = ["dataset", "hidden_ratio", "metric"]


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "dataset": d,
                "oversampler": s,
                "metric": m,
                "hidden_ratio": h,
                "run": i,
                "error_rate": v,
            }
            for d, s, m, h, i, v in rows
        ]
    )


def _simpsons_case():
    """A beats B on every dataset, but B's pooled mean is lower.

    B's runs on the hard dataset are mostly skipped, so its pooled mean is
    dominated by the easy dataset. This is reachable in practice: the minority
    hold-out guard drops runs exactly like this.
    """
    rows = []
    rows += [("easy", "A", "hassanat", 0.1, i, 0.10) for i in range(8)]
    rows += [("easy", "B", "hassanat", 0.1, i, 0.15) for i in range(8)]
    rows += [("hard", "A", "hassanat", 0.1, i, 0.70) for i in range(8)]
    rows += [("hard", "B", "hassanat", 0.1, i, 0.75) for i in range(2)]
    rows += [("hard", "B", "hassanat", 0.1, i, float("nan")) for i in range(2, 8)]
    return _frame(rows)


def test_pooled_mean_still_favours_the_loser():
    """Pins the premise. If this stops holding, the test below proves nothing."""
    df = _simpsons_case()
    pooled = df.groupby("oversampler")["error_rate"].mean()
    assert pooled["B"] < pooled["A"]


def test_winner_on_every_dataset_ranks_first():
    df = _simpsons_case()
    per_dataset = df.groupby(["dataset", "oversampler"])["error_rate"].mean()
    for dataset in ("easy", "hard"):
        assert per_dataset[(dataset, "A")] < per_dataset[(dataset, "B")]

    summary = compute_ranking(df)
    assert summary.loc["A", "rank"] == 1.0
    assert summary.loc["A", "mean_rank"] < summary.loc["B", "mean_rank"]


def test_rank_is_not_driven_by_the_pooled_mean():
    df = _simpsons_case()
    summary = compute_ranking(df)
    assert summary.loc["A", "mean"] > summary.loc["B", "mean"]
    assert summary.loc["A", "rank"] < summary.loc["B", "rank"]


def test_mean_rank_agrees_with_friedman_nemenyi():
    """The ranking and the significance test must answer the same question.

    Demsar's protocol is what `friedman_nemenyi` implements; if the headline
    ordering came from somewhere else, a reader could see one method ranked
    first and the test declare another the winner.
    """
    rng = np.random.default_rng(0)
    rows = []
    for dataset in ("d1", "d2", "d3", "d4", "d5"):
        for metric in ("hassanat", "euclidean"):
            for name, centre in (("A", 0.2), ("B", 0.35), ("C", 0.5)):
                for i in range(3):
                    rows.append(
                        (
                            dataset,
                            name,
                            metric,
                            0.1,
                            i,
                            float(centre + rng.normal(0, 0.02)),
                        )
                    )
    df = _frame(rows)
    summary = compute_ranking(df)

    wide = df.groupby([*SPEC, "oversampler"])["error_rate"].mean().unstack("oversampler")
    result = friedman_nemenyi(wide.to_numpy(), list(wide.columns))
    for i, name in enumerate(wide.columns):
        assert summary.loc[name, "mean_rank"] == pytest.approx(result.mean_ranks[i])


def test_metric_defines_a_separate_experiment():
    """Error rates are not comparable across metrics.

    A sampler that wins under both metrics must rank first even when the two
    metrics sit on wildly different scales.
    """
    rows = []
    rows += [("d", "A", "hassanat", 0.1, i, 0.80) for i in range(4)]
    rows += [("d", "B", "hassanat", 0.1, i, 0.90) for i in range(4)]
    rows += [("d", "A", "euclidean", 0.1, i, 0.05) for i in range(4)]
    rows += [("d", "B", "euclidean", 0.1, i, 0.10) for i in range(4)]
    summary = compute_ranking(_frame(rows))
    assert summary.loc["A", "rank"] == 1.0
    assert summary.loc["A", "n_specifications"] == 2


def test_unequal_experiment_coverage_warns():
    """Mean ranks over different experiment sets are not comparable."""
    rows = [
        ("d1", "A", "m", 0.1, 0, 0.2),
        ("d1", "B", "m", 0.1, 0, 0.3),
        ("d2", "A", "m", 0.1, 0, 0.5),
    ]
    with pytest.warns(UserWarning, match="different numbers of experiments"):
        compute_ranking(_frame(rows))


def test_equal_coverage_does_not_warn():
    rows = [
        ("d1", "A", "m", 0.1, 0, 0.2),
        ("d1", "B", "m", 0.1, 0, 0.3),
        ("d2", "A", "m", 0.1, 0, 0.5),
        ("d2", "B", "m", 0.1, 0, 0.6),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_ranking(_frame(rows))


def test_missing_runs_are_still_reported():
    summary = compute_ranking(_simpsons_case())
    assert summary.loc["B", "n_missing"] == 6
    assert summary.loc["A", "n_missing"] == 0


def test_ties_share_a_rank():
    rows = [
        ("d1", "A", "m", 0.1, 0, 0.3),
        ("d1", "B", "m", 0.1, 0, 0.3),
        ("d2", "A", "m", 0.1, 0, 0.4),
        ("d2", "B", "m", 0.1, 0, 0.4),
    ]
    summary = compute_ranking(_frame(rows))
    assert summary.loc["A", "mean_rank"] == summary.loc["B", "mean_rank"]


def test_frame_without_specification_columns_still_ranks():
    """Nothing identifies separate experiments, so pooling is all there is."""
    df = pd.DataFrame(
        {
            "oversampler": ["A", "A", "B", "B"],
            "error_rate": [0.1, 0.2, 0.4, 0.5],
        }
    )
    summary = compute_ranking(df)
    assert summary.loc["A", "rank"] == 1.0
    assert summary.loc["A", "n_specifications"] == 1
