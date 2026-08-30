"""Integration tests for sklearn and imbalanced-learn workflows.

These tests guard the public estimator against regressions that unit-level
``get_params`` and ``score`` tests would not catch: scorer invocation from
sklearn model-selection utilities, nested sampler parameter routing, and
pipeline composition around preprocessing.
"""

from __future__ import annotations

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oversampleqa import (
    OversamplingValidator,
    validate_oversampling,
    validation_scorer,
)


@pytest.fixture
def sklearn_data():
    """Enough minority samples for validation inside 3-fold CV test folds."""
    return make_classification(
        n_samples=900,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.75, 0.25],
        random_state=0,
    )


@pytest.fixture
def cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def test_validator_cross_validates_with_sklearn_scorer(sklearn_data, cv):
    X, y = sklearn_data
    validator = OversamplingValidator(
        SMOTE(random_state=0),
        hidden_ratio=0.2,
        metric="euclidean",
        random_state=0,
    )

    result = cross_validate(
        validator,
        X,
        y,
        cv=cv,
        scoring=validation_scorer,
        error_score="raise",
    )

    assert result["test_score"].shape == (3,)
    assert np.all(np.isfinite(result["test_score"]))
    assert np.all(result["test_score"] <= 0.0)


def test_grid_search_tunes_nested_sampler_parameters(sklearn_data, cv):
    X, y = sklearn_data
    search = GridSearchCV(
        OversamplingValidator(
            SMOTE(random_state=0),
            hidden_ratio=0.2,
            metric="euclidean",
            random_state=0,
        ),
        {"oversampler__k_neighbors": [3, 5]},
        scoring=validation_scorer,
        cv=cv,
        error_score="raise",
    )

    search.fit(X, y)

    assert search.best_params_["oversampler__k_neighbors"] in {3, 5}
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert search.best_estimator_.oversampler.k_neighbors == search.best_params_[
        "oversampler__k_neighbors"
    ]


def test_validator_composes_as_final_step_in_sklearn_pipeline(sklearn_data):
    X, y = sklearn_data
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "validate",
                OversamplingValidator(
                    SMOTE(random_state=0),
                    hidden_ratio=0.2,
                    metric="euclidean",
                    random_state=0,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)

    validator = pipeline.named_steps["validate"]
    assert 0.0 <= validator.error_rate_ <= 1.0
    assert pipeline.score(X, y) == pytest.approx(-validator.error_rate_)


@pytest.mark.parametrize(
    "sampler",
    [
        SMOTE(k_neighbors=3, random_state=0),
        BorderlineSMOTE(k_neighbors=3, random_state=0),
        RandomOverSampler(random_state=0),
    ],
    ids=["smote", "borderline-smote", "random-over-sampler"],
)
def test_common_imbalanced_learn_samplers_validate(sklearn_data, sampler):
    X, y = sklearn_data

    error_rate = validate_oversampling(
        X,
        y,
        minority_label=1,
        oversampler=sampler,
        hidden_ratio=0.2,
        metric="euclidean",
        random_state=0,
    )

    assert 0.0 <= error_rate <= 1.0
