import asyncio

import numpy as np
import pytest
from imblearn.over_sampling import RandomOverSampler

from oversampleqa.typed_validator import TypedValidator, validation_session, registry
from oversampleqa.types import ValidationConfig, ValidationError, ValidationMode
from oversampleqa.plugin_system import plugin_manager, register_metric


def sample_data():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)
    return X, y


def test_typed_validator_standard():
    X, y = sample_data()
    validator = TypedValidator()
    result = validator.validate(
        X,
        y,
        minority_label=1,
        oversampler=RandomOverSampler(),
        hidden_ratio=0.2,
        metric="hassanat",
    )
    assert 0.0 <= result["error_rate"] <= 1.0


def test_typed_validator_invalid_input():
    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0, 0], dtype=int)
    validator = TypedValidator()
    with pytest.raises(ValidationError):
        validator.validate(
            X,
            y,
            minority_label=1,
            oversampler=RandomOverSampler(),
            hidden_ratio=0.2,
        )


def test_validation_session_async():
    X, y = sample_data()
    config = ValidationConfig(hidden_ratio=0.2)

    async def run():
        async with validation_session(config) as validator:
            result = await validator.validate_async(
                X, y, minority_label=1, oversampler=RandomOverSampler(), config=config
            )
            assert "error_rate" in result

    asyncio.run(run())


def test_plugin_registration():
    @register_metric("unit_distance")
    class UnitMetric:
        def __call__(self, x1, x2, **kwargs):
            return 1.0

    metric_cls = plugin_manager.get_metric("unit_distance")
    assert metric_cls()([0.0], [1.0]) == 1.0
