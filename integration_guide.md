# Integration Guide: Adding New Distance Metrics to oversampleqa

This guide provides step-by-step instructions for integrating the extended distance metrics into the `oversampleqa` package and validating their correctness.

## 📋 Prerequisites

- Existing `oversampleqa` package structure
- Python 3.10+
 - Required dependencies: `numpy`, `scikit-learn`, `pytest`

## 🔧 Step 1: Update the Distance Module

### 1.1 Modify `src/oversampleqa/distance.py`

Add the new distance functions to your existing distance module:

```python
# Add these imports at the top
import warnings

# Add all the new distance functions (minkowski_distance, chebyshev_distance, etc.)
# ... (copy from the extended_distances code above)

# Update the _METRICS dictionary
_METRICS = {
    "hassanat": hassanat_distance,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "cosine": cosine_distance,
    # New metrics
    "minkowski": minkowski_distance,
    "chebyshev": chebyshev_distance,
    "mahalanobis": mahalanobis_distance,
    "canberra": canberra_distance,
    "hamming": hamming_distance,
    "jaccard": jaccard_distance,
    "braycurtis": braycurtis_distance,
    "correlation": correlation_distance,
    "energy": energy_distance,
    "wasserstein": wasserstein_1d_distance,
}
```

### 1.2 Update Dependencies

No additional dependencies are required for these metrics.

## 🧪 Step 2: Create Comprehensive Tests

### 2.1 Create `tests/test_extended_distances.py`

Copy the test file from the artifacts above and implement the actual test logic:

```python
import numpy as np
import pytest
from oversampleqa.distance import (
    minkowski_distance, chebyshev_distance, mahalanobis_distance,
    canberra_distance, hamming_distance, jaccard_distance,
    braycurtis_distance, correlation_distance, energy_distance,
    wasserstein_1d_distance, distance_matrix, _METRICS
)

class TestNewDistanceMetrics:
    """Test all new distance metrics."""
    
    def test_minkowski_equivalence(self):
        """Test Minkowski distance equivalence to known metrics."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])
        
        # p=1 should equal Manhattan
        from oversampleqa.distance import manhattan_distance
        dist_p1 = minkowski_distance(x1, x2, p=1)
        manhattan_dist = manhattan_distance(x1, x2)
        assert np.isclose(dist_p1, manhattan_dist)
        
        # p=2 should equal Euclidean
        from oversampleqa.distance import euclidean_distance
        dist_p2 = minkowski_distance(x1, x2, p=2)
        euclidean_dist = euclidean_distance(x1, x2)
        assert np.isclose(dist_p2, euclidean_dist)
    
    # ... implement other test methods
```

### 2.2 Run Tests

```bash
cd oversampleqa
python -m pytest tests/test_extended_distances.py -v
```

## ✅ Step 3: Validation Strategy

### 3.1 Mathematical Property Validation

Create a validation script to check mathematical properties:

```python
# validation_script.py
from oversampleqa.distance import _METRICS
import numpy as np

def validate_metric_properties(metric_name, metric_func, test_vectors):
    """Validate mathematical properties of a distance metric."""
    results = {"passed": True, "errors": []}
    
    for i, x in enumerate(test_vectors):
        for j, y in enumerate(test_vectors):
            try:
                dist_xy = metric_func(x, y)
                dist_yx = metric_func(y, x)
                
                # Non-negativity
                if dist_xy < 0:
                    results["errors"].append(f"{metric_name}: Negative distance {dist_xy}")
                    results["passed"] = False
                
                # Identity
                if i == j and not np.isclose(dist_xy, 0, atol=1e-10):
                    results["errors"].append(f"{metric_name}: Non-zero self-distance {dist_xy}")
                    results["passed"] = False
                
                # Symmetry
                if not np.isclose(dist_xy, dist_yx, rtol=1e-10):
                    results["errors"].append(f"{metric_name}: Asymmetric distances {dist_xy} vs {dist_yx}")
                    results["passed"] = False
                    
            except Exception as e:
                results["errors"].append(f"{metric_name}: Exception {str(e)}")
                results["passed"] = False
    
    return results

# Generate test vectors
np.random.seed(42)
test_vectors = [
    np.zeros(5),
    np.ones(5),
    np.random.randn(5),
    np.random.exponential(1, 5),
    np.array([1, 0, 0, 0, 0]),
]

# Validate all metrics
for name, func in _METRICS.items():
    if func is not None:
        results = validate_metric_properties(name, func, test_vectors)
        status = "✅ PASS" if results["passed"] else "❌ FAIL"
        print(f"{name:15s} {status}")
        if not results["passed"]:
            for error in results["errors"][:3]:  # Show first 3 errors
                print(f"  {error}")
```

### 3.2 Performance Validation

Test computational performance:

```python
# performance_test.py
import time
import numpy as np
from oversampleqa.distance import distance_matrix, _METRICS

def benchmark_metrics(n_samples=100, n_features=10):
    """Benchmark all distance metrics."""
    np.random.seed(42)
    X1 = np.random.randn(n_samples, n_features)
    X2 = np.random.randn(n_samples, n_features)
    
    results = {}
    
    for metric_name in _METRICS:
        if _METRICS[metric_name] is None:
            continue
            
        start_time = time.time()
        try:
            # Compute smaller matrix for timing
            matrix = distance_matrix(X1[:20], X2[:20], metric=metric_name)
            elapsed = time.time() - start_time
            results[metric_name] = elapsed
            print(f"{metric_name:15s}: {elapsed:.4f}s")
        except Exception as e:
            print(f"{metric_name:15s}: ERROR - {str(e)}")
            results[metric_name] = float('inf')
    
    return results

# Run benchmark
print("Performance Benchmark (20x20 distance matrix):")
benchmark_results = benchmark_metrics()
```

## 🔬 Step 4: Empirical Validation

### 4.1 Test with Real Datasets

Validate on imbalanced classification datasets:

```python
# empirical_validation.py
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from oversampleqa.validator import validate_oversampling
import pandas as pd

def compare_metrics_on_dataset():
    """Compare validation results across different distance metrics."""
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, weights=[0.8, 0.2], 
        n_informative=15, n_redundant=5, random_state=42
    )
    
    metrics_to_test = [
        "hassanat", "euclidean", "manhattan", "cosine",
        "minkowski", "chebyshev", "canberra", "correlation"
    ]
    
    results = []
    
    for metric in metrics_to_test:
        try:
            error_rate = validate_oversampling(
                X, y, minority_label=1, 
                oversampler=SMOTE(random_state=42),
                metric=metric,
                hidden_ratio=0.1
            )
            results.append({"metric": metric, "error_rate": error_rate})
            print(f"{metric:15s}: {error_rate:.4f}")
        except Exception as e:
            print(f"{metric:15s}: ERROR - {str(e)}")
    
    return pd.DataFrame(results)

# Run comparison
print("Validation Error Rates by Distance Metric:")
comparison_results = compare_metrics_on_dataset()
```

### 4.2 Consistency Checks

Check that results are consistent and reasonable:

```python
def consistency_checks(results_df):
    """Perform consistency checks on validation results."""
    error_rates = results_df["error_rate"].values
    
    # All error rates should be between 0 and 1
    assert all(0 <= rate <= 1 for rate in error_rates), "Error rates outside [0,1]"
    
    # Error rates shouldn't all be identical (would indicate bug)
    unique_rates = len(set(error_rates))
    if unique_rates == 1:
        print("⚠️  Warning: All metrics gave identical results")
    
    # No error rate should be exactly 1.0 (perfect failure is suspicious)
    if any(rate == 1.0 for rate in error_rates):
        print("⚠️  Warning: Some metrics have 100% error rate")
    
    # Results should be somewhat correlated but not identical
    import numpy as np
    if len(error_rates) > 2:
        std_dev = np.std(error_rates)
        if std_dev < 0.001:
            print("⚠️  Warning: Very low variance between metrics")
        elif std_dev > 0.5:
            print("⚠️  Warning: Very high variance between metrics")
    
    print("✅ Consistency checks completed")

# Run consistency checks
consistency_checks(comparison_results)
```

## 📊 Step 5: Documentation and Examples

### 5.1 Update Documentation

Add metric descriptions to the README or documentation:

```markdown
## Available Distance Metrics

| Metric | Use Case | Characteristics |
|--------|----------|----------------|
| `hassanat` | Original validation methodology | Handles zero values well |
| `euclidean` | General purpose | Standard L2 norm |
| `manhattan` | High-dimensional data | L1 norm, less sensitive to outliers |
| `cosine` | Sparse/text data | Angle-based, magnitude-independent |
| `minkowski` | Configurable norm | Generalization of euclidean/manhattan |
| `chebyshev` | Worst-case analysis | Maximum difference across dimensions |
| `mahalanobis` | Correlated features | Accounts for covariance structure |
| `canberra` | Different feature scales | Weighted Manhattan distance |
| `correlation` | Linear relationships | Based on Pearson correlation |
| `hamming` | Categorical data | Counts differing positions |
| `jaccard` | Binary/set data | Set intersection/union ratio |
```

### 5.2 Create Usage Examples

```python
# examples/distance_comparison.py
from oversampleqa.validator import validate_oversampling
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def compare_distance_metrics():
    """Example comparing different distance metrics."""
    X, y = make_classification(n_samples=500, weights=[0.8, 0.2], random_state=42)
    
    metrics = ["hassanat", "euclidean", "manhattan", "canberra", "correlation"]
    error_rates = []
    
    for metric in metrics:
        error = validate_oversampling(
            X, y, minority_label=1, 
            oversampler=SMOTE(random_state=42),
            metric=metric
        )
        error_rates.append(error)
        print(f"{metric}: {error:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, error_rates)
    plt.ylabel("Validation Error Rate")
    plt.title("SMOTE Validation Error by Distance Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("distance_comparison.png")
    plt.show()

if __name__ == "__main__":
    compare_distance_metrics()
```

## 🚀 Step 6: Final Integration

### 6.1 Update Main Package

Update `src/oversampleqa/__init__.py`:

```python
# Add new imports if desired
from .distance import (
    hassanat_distance, euclidean_distance, manhattan_distance, cosine_distance,
    minkowski_distance, chebyshev_distance, mahalanobis_distance,
    canberra_distance, correlation_distance, distance_matrix
)

__all__ = [
    # ... existing exports
    "minkowski_distance", "chebyshev_distance", "mahalanobis_distance",
    "canberra_distance", "correlation_distance",
    # ... other exports
]
```

### 6.2 Run Full Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=oversampleqa --cov-report=html

# Run performance tests (optional)
python -m pytest tests/ -v -m "slow"
```

## ✨ Best Practices

1. **Start Small**: Implement and test 2-3 metrics first before adding all
2. **Document Everything**: Each metric should have clear docstrings explaining use cases
3. **Performance Aware**: Some metrics (like Mahalanobis) may be slow on large datasets
4. **Validate Thoroughly**: Use both synthetic and real datasets for validation
5. **Error Handling**: Ensure graceful handling of edge cases (NaN, infinity, etc.)
6. **Backwards Compatibility**: Don't break existing API when adding new metrics

## 🔍 Troubleshooting

- **Common Issues:**

- **Import errors**: Ensure all dependencies are installed correctly
- **Performance problems**: Some metrics are inherently slower; document this
- **Numerical instability**: Add proper handling for edge cases (zeros, very large/small values)
- **Inconsistent results**: Verify mathematical properties and test with known datasets

**Debugging Tips:**

```python
# Test individual metric on simple data
import numpy as np
from oversampleqa.distance import minkowski_distance

x1 = np.array([1.0, 2.0])
x2 = np.array([3.0, 4.0])
print(f"Distance: {minkowski_distance(x1, x2, p=2)}")  # Should be 2*sqrt(2) ≈ 2.83
```

This comprehensive approach ensures that your extended distance metrics are robust, well-tested, and properly integrated into the `oversampleqa` package.