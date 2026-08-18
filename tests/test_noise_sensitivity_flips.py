"""A requested noise level must be the realised one.

`noise_sensitivity_diagnostic` drew replacement labels from *all* classes, so a
selected point could be "flipped" to the label it already had. The realised
noise was `requested * (k - 1) / k`:

    classes  requested  realised  ratio
          2       0.30    0.1570  0.523
          3       0.30    0.2003  0.668
          5       0.30    0.2393  0.798

On binary data -- this package's main case -- half the requested noise was
applied, so the x-axis of every noise-sensitivity plot was overstated by 2x.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.metrics import flip_labels, noise_sensitivity_diagnostic


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        random_state=0,
    )
    return X, y


# --- flip_labels ---


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_every_selected_label_actually_changes(k):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(k), 200)
    labels = np.unique(y)
    idx = rng.choice(len(y), 400, replace=False)
    flipped = flip_labels(y, idx, labels, rng)
    assert np.all(flipped[idx] != y[idx])


@pytest.mark.parametrize("k", [2, 3, 5])
def test_unselected_labels_are_untouched(k):
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(k), 200)
    labels = np.unique(y)
    idx = rng.choice(len(y), 300, replace=False)
    mask = np.ones(len(y), bool)
    mask[idx] = False
    flipped = flip_labels(y, idx, labels, rng)
    assert np.array_equal(flipped[mask], y[mask])


def test_the_input_array_is_not_modified():
    rng = np.random.default_rng(2)
    y = np.array([0, 1, 0, 1, 0, 1])
    before = y.copy()
    flip_labels(y, np.array([0, 1, 2]), np.unique(y), rng)
    assert np.array_equal(y, before)


def test_replacements_are_spread_over_the_alternatives():
    """Uniform over the k - 1 others, not always the same one."""
    rng = np.random.default_rng(3)
    y = np.zeros(3000, dtype=int)
    labels = np.array([0, 1, 2, 3])
    flipped = flip_labels(y, np.arange(3000), labels, rng)
    counts = np.bincount(flipped, minlength=4)
    assert counts[0] == 0
    for label in (1, 2, 3):
        assert 800 < counts[label] < 1200, counts


def test_no_indices_is_a_no_op():
    rng = np.random.default_rng(4)
    y = np.array([0, 1, 0, 1])
    result = flip_labels(y, np.array([], dtype=int), np.unique(y), rng)
    assert np.array_equal(result, y)


# --- the diagnostic ---


def test_realised_noise_equals_the_requested_level(binary_data):
    X, y = binary_data
    levels = [0.0, 0.1, 0.2, 0.3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = noise_sensitivity_diagnostic(
            X, y, 1, SMOTE(random_state=0), noise_levels=levels, random_state=0
        )
    for _, row in df.iterrows():
        assert row["n_flipped"] == int(len(y) * row["noise"])


def test_the_frame_reports_how_many_labels_moved(binary_data):
    """So the applied noise can be checked rather than assumed."""
    X, y = binary_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = noise_sensitivity_diagnostic(
            X, y, 1, SMOTE(random_state=0), noise_levels=[0.2], random_state=0
        )
    assert "n_flipped" in df.columns
    assert df["n_flipped"].iloc[0] == 200


def test_zero_noise_flips_nothing(binary_data):
    X, y = binary_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = noise_sensitivity_diagnostic(
            X, y, 1, SMOTE(random_state=0), noise_levels=[0.0], random_state=0
        )
    assert df["n_flipped"].iloc[0] == 0


def test_a_single_class_is_refused():
    """With one class there is no other label, so no level differs from zero."""
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="at least two classes"):
        noise_sensitivity_diagnostic(X, y, 0, SMOTE(), noise_levels=[0.1])


def test_is_deterministic(binary_data):
    X, y = binary_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = noise_sensitivity_diagnostic(
            X, y, 1, SMOTE(random_state=0), noise_levels=[0.2], random_state=7
        )
        b = noise_sensitivity_diagnostic(
            X, y, 1, SMOTE(random_state=0), noise_levels=[0.2], random_state=7
        )
    assert a["error_rate"].iloc[0] == b["error_rate"].iloc[0]
