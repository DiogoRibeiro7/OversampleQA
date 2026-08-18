from oversampleqa.advanced_benchmark import DatasetRepository


def test_load_domain_unknown_returns_empty():
    repo = DatasetRepository()
    assert repo._load_domain("unknown", max_samples=10, include_openml=False) == []


def test_load_financial_optional_dependency():
    repo = DatasetRepository()
    data = repo._load_financial(max_samples=10)
    assert isinstance(data, list)
