"""Tests for the fidelity and diversity suite.

The error rate conflates two opposite failures -- generating implausible points,
and merely copying the training data. ``RandomOverSampler`` is the clean case:
it scores well on a single scalar while contributing no information at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification

from oversampleqa.exceptions import ValidationError
from oversampleqa.fidelity import (
    boundary_violation_rate,
    fidelity_report,
    memorisation_report,
    precision_recall_density_coverage,
    sweep_k,
)


@pytest.fixture
def imbalanced():
    return make_classification(
        n_samples=800,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        random_state=0,
    )


# --- manifold metrics, against known answers ------------------------------


def test_identical_distributions_score_near_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (200, 3))
    b = rng.normal(0, 1, (200, 3))
    m = precision_recall_density_coverage(a, b, k=5, metric="euclidean")
    assert m.precision > 0.9
    assert m.recall > 0.9
    assert m.coverage > 0.9


def test_disjoint_distributions_score_zero():
    rng = np.random.default_rng(0)
    near = rng.normal(0, 1, (200, 3))
    far = rng.normal(50, 1, (200, 3))
    m = precision_recall_density_coverage(far, near, k=5, metric="euclidean")
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.coverage == 0.0
    assert m.density == 0.0


def test_density_distinguishes_what_precision_saturates_on():
    """Precision saturates at 1; density keeps resolving beyond it.

    Both configurations below are fully inside the real manifold, so precision
    is 1.0 for each and cannot tell them apart. Density counts *how many* real
    spheres contain each point, so it separates points sitting on top of real
    data (many overlapping spheres, > 1) from points piled in one spot (few
    spheres, < 1) -- even though that pile is technically "inside".
    """
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (100, 2))

    on_real = real.copy()
    piled = rng.normal(0, 0.05, (100, 2))

    spread_out = precision_recall_density_coverage(
        on_real, real, k=5, metric="euclidean"
    )
    concentrated = precision_recall_density_coverage(
        piled, real, k=5, metric="euclidean"
    )

    assert spread_out.precision == pytest.approx(1.0)
    assert concentrated.precision == pytest.approx(1.0)
    assert spread_out.density > 1.0
    assert concentrated.density < 1.0
    assert spread_out.coverage > concentrated.coverage


def test_manifold_rejects_too_few_real_points():
    rng = np.random.default_rng(0)
    with pytest.raises(ValidationError, match="needs at least"):
        precision_recall_density_coverage(
            rng.normal(size=(20, 3)), rng.normal(size=(3, 3)), k=5
        )


def test_manifold_rejects_empty_synthetic():
    rng = np.random.default_rng(0)
    with pytest.raises(ValidationError, match="empty"):
        precision_recall_density_coverage(np.empty((0, 3)), rng.normal(size=(30, 3)))


def test_high_dimensional_input_warns():
    """These metrics concentrate in high dimension; that must not be silent."""
    rng = np.random.default_rng(0)
    synthetic = rng.normal(size=(30, 40))
    real = rng.normal(size=(30, 40))
    with pytest.warns(UserWarning, match="concentrate in high dimension"):
        precision_recall_density_coverage(synthetic, real, k=5, metric="euclidean")


def test_sweep_k_reports_every_usable_k():
    rng = np.random.default_rng(0)
    synthetic = rng.normal(size=(80, 3))
    real = rng.normal(size=(80, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = sweep_k(synthetic, real, ks=(3, 5, 10), metric="euclidean")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["k"]) == [3, 5, 10]


def test_sweep_k_skips_unusable_k():
    rng = np.random.default_rng(0)
    synthetic = rng.normal(size=(20, 3))
    real = rng.normal(size=(8, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = sweep_k(synthetic, real, ks=(3, 5, 50), metric="euclidean")
    assert 50 not in list(frame["k"])


def test_sweep_k_raises_when_nothing_is_usable():
    rng = np.random.default_rng(0)
    with pytest.raises(ValidationError, match="none of k"):
        sweep_k(rng.normal(size=(20, 3)), rng.normal(size=(4, 3)), ks=(10, 20))


# --- memorisation ---------------------------------------------------------


def test_duplicates_score_a_zero_ratio():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(150, 3))
    report = memorisation_report(train.copy(), train, metric="euclidean")
    assert report.distance_ratio == pytest.approx(0.0)
    assert report.exact_duplicate_rate == pytest.approx(1.0)
    assert "copying" in report.interpret()


def test_genuinely_new_points_score_near_one():
    """A ratio near 1 means synthetic points sit as far from training data as
    real points sit from each other."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3))
    fresh = rng.normal(size=(200, 3))
    report = memorisation_report(fresh, train, metric="euclidean")
    assert 0.5 < report.distance_ratio < 2.0
    assert report.exact_duplicate_rate == 0.0


@pytest.mark.parametrize("metric", ["euclidean", "hassanat", "manhattan"])
def test_duplicate_detection_is_metric_independent(metric):
    """euclidean goes through a BLAS gram trick that loses the last bits to
    cancellation, returning ~1e-8 rather than 0 for identical points, while
    direct formulas return exact zeros. An equality test would report a
    different duplicate rate per metric for the same data.
    """
    rng = np.random.default_rng(0)
    train = rng.normal(size=(120, 3))
    report = memorisation_report(train.copy(), train, metric=metric)
    assert report.exact_duplicate_rate == pytest.approx(1.0)


def test_memorisation_rejects_tiny_training_sets():
    rng = np.random.default_rng(0)
    with pytest.raises(ValidationError, match="at least 2"):
        memorisation_report(rng.normal(size=(10, 3)), rng.normal(size=(1, 3)))


# --- boundary violations --------------------------------------------------


def test_points_in_majority_territory_are_flagged(imbalanced):
    X, y = imbalanced
    majority_points = X[y == 0][:50]
    report = boundary_violation_rate(majority_points, X, y, 1, k=5, metric="euclidean")
    assert report.strict_rate > 0.8
    assert report.graded_rate > 0.8


def test_points_in_minority_territory_are_not_flagged(imbalanced):
    X, y = imbalanced
    minority_points = X[y == 1][:50]
    report = boundary_violation_rate(minority_points, X, y, 1, k=5, metric="euclidean")
    assert report.strict_rate < 0.2


def test_graded_rate_is_more_sensitive_than_strict(imbalanced):
    """Strict needs all k neighbours to be majority; graded catches drift."""
    X, y = imbalanced
    rng = np.random.default_rng(0)
    boundary_points = X[y == 1][:40] + rng.normal(0, 0.5, (40, X.shape[1]))
    report = boundary_violation_rate(boundary_points, X, y, 1, k=5, metric="euclidean")
    assert report.graded_rate >= report.strict_rate


# --- the headline argument ------------------------------------------------


def test_random_oversampler_looks_fine_until_memorisation_is_measured(imbalanced):
    """The whole reason this module exists.

    RandomOverSampler only duplicates, so it adds no information whatsoever.
    Its error rate is comparable to SMOTE's, and only the memorisation ratio
    separates them.
    """
    X, y = imbalanced
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smote = fidelity_report(X, y, 1, SMOTE(random_state=0), metric="hassanat")
        ros = fidelity_report(
            X, y, 1, RandomOverSampler(random_state=0), metric="hassanat"
        )

    # The error rate does not separate them.
    assert abs(smote.error_rate - ros.error_rate) < 0.15
    # Memorisation does, decisively.
    assert ros.memorisation.distance_ratio == pytest.approx(0.0)
    assert smote.memorisation.distance_ratio > 0.2
    assert any("Memorisation" in note for note in ros.interpret())
    assert not any("Memorisation" in note for note in smote.interpret())


def test_report_flattens_and_frames(imbalanced):
    X, y = imbalanced
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = fidelity_report(X, y, 1, SMOTE(random_state=0))
    payload = report.to_dict()
    assert {
        "error_rate",
        "precision",
        "coverage",
        "memorisation_distance_ratio",
    } <= set(payload)
    frame = report.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1


def test_report_rejects_multiclass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 3))
    y = np.repeat([0, 1, 2], 30)
    with pytest.raises(ValidationError, match="binary"):
        fidelity_report(X, y, 1, SMOTE(random_state=0))


# --- downstream utility, and the leakage it must avoid --------------------


@pytest.mark.slow
def test_resampling_before_the_split_inflates_the_score(imbalanced):
    """Pin the leakage by demonstrating it, not by asserting a comment.

    Oversampling before cross-validation lets synthetic points derived from
    validation-fold minority samples reach the training folds. A SMOTE point
    interpolated from a validation point is nearly that point, so the model is
    scored on data it effectively trained on.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = imbalanced
    estimator = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=1)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Wrong: resample everything, then cross-validate.
    X_leaky, y_leaky = SMOTE(random_state=0).fit_resample(X, y)
    leaky = cross_val_score(
        estimator, X_leaky, y_leaky, cv=splitter, scoring="average_precision"
    ).mean()

    # Right: the sampler lives inside the pipeline, so it runs per fold.
    honest = cross_val_score(
        ImbPipeline([("s", SMOTE(random_state=0)), ("m", estimator)]),
        X,
        y,
        cv=splitter,
        scoring="average_precision",
    ).mean()

    assert leaky > honest, (
        f"expected the leaky construction to score higher; got leaky={leaky:.4f} "
        f"honest={honest:.4f}"
    )


@pytest.mark.slow
def test_downstream_utility_reports_a_paired_difference(imbalanced):
    from oversampleqa.fidelity import downstream_utility

    X, y = imbalanced
    report = downstream_utility(X, y, SMOTE(random_state=0), cv=3, n_resamples=200)
    assert report.scoring == "average_precision"
    assert report.ci_lower <= report.difference <= report.ci_upper
    assert len(report.fold_differences) == 3
