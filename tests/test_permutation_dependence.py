"""Permutation tests must not treat SMOTE's children as independent points.

Permuting labels assumes exchangeability. Points sharing a SMOTE parent lie on
segments from the same minority point, so they move together, and the
permutation null is far too tight. Measured under H0 -- synthetic and real drawn
from the *same* distribution, differing only in that the synthetic sample is
block-structured:

    false rejections at 0.05   ignoring blocks: 100%   block-aware: 0%
    median p                   ignoring blocks: 0.005  block-aware: 0.885

The naive test rejects the null every time it is true. It was responding to the
clustering, which SMOTE always produces, rather than to any difference in
distribution -- so it would call any SMOTE output "distinguishable from real"
however good it was.

`error_rate_interval` already handled this dependence with a parent-block
bootstrap; these tests did not.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa.inference import (
    _combine_blocked,
    _one_per_parent,
    cross_match_test,
    mst_two_sample_test,
    nn_two_sample_test,
)

TESTS = [nn_two_sample_test, mst_two_sample_test, cross_match_test]


def _blocked_null(seed: int, n_parents: int = 20, per_parent: int = 5):
    """Same distribution for both samples; synthetic is clustered by parent."""
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 1, (100, 3))
    centres = rng.normal(0, 1, (n_parents, 3))
    synthetic = np.repeat(centres, per_parent, axis=0) + rng.normal(
        0, 0.15, (n_parents * per_parent, 3)
    )
    parents = np.repeat(np.arange(n_parents), per_parent)
    return synthetic, real, parents


# --- _one_per_parent ---


def test_picks_exactly_one_point_per_parent():
    rng = np.random.default_rng(0)
    parents = np.repeat(np.arange(15), 4)
    idx = _one_per_parent(parents, rng)
    assert len(idx) == 15
    assert len(set(parents[idx])) == 15


def test_selected_points_share_no_parent():
    """The whole point: the subsample is exchangeable."""
    rng = np.random.default_rng(1)
    parents = np.repeat(np.arange(10), 7)
    idx = _one_per_parent(parents, rng)
    assert len(set(parents[idx])) == len(idx)


def test_selection_varies_between_draws():
    rng = np.random.default_rng(2)
    parents = np.repeat(np.arange(30), 8)
    draws = {tuple(_one_per_parent(parents, rng)) for _ in range(10)}
    assert len(draws) > 1, "every draw picked the same representatives"


def test_uneven_block_sizes_are_handled():
    rng = np.random.default_rng(3)
    parents = np.array([0, 0, 0, 1, 2, 2])
    idx = _one_per_parent(parents, rng)
    assert sorted(parents[idx]) == [0, 1, 2]


# --- the dependence fix ---


@pytest.mark.parametrize("test", TESTS, ids=lambda f: f.__name__)
def test_blocked_result_is_labelled_as_such(test):
    synthetic, real, parents = _blocked_null(0)
    result = test(
        synthetic, real, n_permutations=99, parents=parents, n_subsamples=3,
        random_state=0,
    )
    assert result.name.endswith("_blocked")


@pytest.mark.parametrize("test", TESTS, ids=lambda f: f.__name__)
def test_naive_test_rejects_a_true_null_on_dependent_data(test):
    """Pins the defect. If this stops holding, the fix below proves nothing."""
    synthetic, real, _ = _blocked_null(0)
    naive = test(synthetic, real, n_permutations=199, random_state=0)
    assert naive.p_value <= 0.05


@pytest.mark.parametrize("test", TESTS, ids=lambda f: f.__name__)
def test_blocking_does_not_reject_that_true_null(test):
    synthetic, real, parents = _blocked_null(0)
    blocked = test(
        synthetic, real, n_permutations=199, parents=parents, n_subsamples=5,
        random_state=0,
    )
    assert blocked.p_value > 0.05


def test_blocking_raises_the_p_value_across_seeds():
    """Not one lucky seed."""
    naive, blocked = [], []
    for seed in range(8):
        synthetic, real, parents = _blocked_null(seed)
        naive.append(
            nn_two_sample_test(synthetic, real, n_permutations=99, random_state=seed).p_value
        )
        blocked.append(
            nn_two_sample_test(
                synthetic, real, n_permutations=99, parents=parents,
                n_subsamples=3, random_state=seed,
            ).p_value
        )
    assert np.mean(np.asarray(naive) <= 0.05) > 0.5
    assert np.mean(np.asarray(blocked) <= 0.05) == 0.0


def test_parents_none_keeps_the_original_behaviour():
    synthetic, real, _ = _blocked_null(0)
    a = nn_two_sample_test(synthetic, real, n_permutations=99, random_state=4)
    b = nn_two_sample_test(
        synthetic, real, n_permutations=99, parents=None, random_state=4
    )
    assert a.p_value == b.p_value
    assert a.name == b.name


def test_blocked_test_is_deterministic():
    synthetic, real, parents = _blocked_null(0)
    kwargs = {
        "n_permutations": 99,
        "parents": parents,
        "n_subsamples": 3,
        "random_state": 11,
    }
    assert (
        nn_two_sample_test(synthetic, real, **kwargs).p_value
        == nn_two_sample_test(synthetic, real, **kwargs).p_value
    )


def test_at_least_one_subsample_is_required():
    synthetic, real, parents = _blocked_null(0)
    with pytest.raises(ValueError, match="at least 1"):
        nn_two_sample_test(
            synthetic, real, n_permutations=9, parents=parents, n_subsamples=0,
            random_state=0,
        )


def test_combination_is_capped_at_one():
    """Twice the median can exceed 1; a p-value cannot."""

    class _Result:
        name, statistic, p_value = "t", 1.0, 0.9
        n_synthetic = n_real = n_permutations = 1
        null_statistics: tuple[float, ...] = ()
        asymptotic_p_value = None

    combined = _combine_blocked(
        lambda _sub: _Result(),
        np.zeros((6, 2)),
        np.repeat(np.arange(3), 2),
        3,
        np.random.default_rng(0),
    )
    assert combined.p_value == 1.0
