import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

from oversampleqa.memory_efficient_validator import MemoryEfficientValidator


def test_memory_efficient_streaming_return_details(tmp_path):
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        weights=[0.8, 0.2],
        random_state=0,
    )
    validator = MemoryEfficientValidator(memory_limit_gb=1e-6, temp_dir=str(tmp_path))
    rate, errors, dist_hidden, dist_min = validator.validate_oversampling(
        X=X,
        y=y,
        minority_label=1,
        oversampler=SMOTE(random_state=0),
        hidden_ratio=0.2,
        metric="euclidean",
        return_details=True,
    )
    assert 0.0 <= rate <= 1.0
    assert isinstance(errors, int)
    assert dist_hidden.shape[0] == dist_min.shape[0]
    validator.cleanup()
    assert validator._stream_dirs == []
