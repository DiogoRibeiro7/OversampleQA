"""Both dataset catalogs must describe where their data came from.

`benchmark.load_standard_datasets` has carried provenance for a while;
`advanced_benchmark.DatasetRepository` carried none, so the two disagreed. These
tests hold them to the same record.
"""

from __future__ import annotations

import pytest

from oversampleqa._provenance import (
    bundled_provenance,
    openml_provenance,
    synthetic_provenance,
)
from oversampleqa.advanced_benchmark import DatasetRepository
from oversampleqa.benchmark import load_standard_datasets

REQUIRED = {"source", "generator", "params", "url", "license", "notes"}


def _all_repository_datasets():
    repo = DatasetRepository()
    return repo.load_research_datasets(max_samples=200) + (
        repo.create_synthetic_benchmark_suite(["easy", "medium", "hard", "extreme"])
    )


def test_repository_datasets_all_carry_provenance():
    datasets = _all_repository_datasets()
    assert datasets, "fixture produced no datasets"
    for dataset in datasets:
        provenance = dataset.get("provenance")
        assert provenance is not None, f"{dataset['name']} has no provenance"
        assert set(provenance) >= REQUIRED, f"{dataset['name']} is missing keys"


def test_standard_datasets_all_carry_provenance():
    for dataset in load_standard_datasets():
        assert set(dataset["provenance"]) >= REQUIRED, dataset["name"]


def test_both_catalogs_use_the_same_record_shape():
    """The point of the shared helper: one shape, not two."""
    a = {frozenset(d["provenance"]) for d in _all_repository_datasets()}
    b = {frozenset(d["provenance"]) for d in load_standard_datasets()}
    assert a == b


def test_licence_is_never_silently_omitted():
    """An absent licence key reads as 'no restrictions' to a hurried reader."""
    for dataset in _all_repository_datasets() + load_standard_datasets():
        assert dataset["provenance"]["license"].strip()


def test_synthetic_records_capture_the_seed():
    """Params must be sufficient to regenerate the data exactly."""
    synthetic = [
        d
        for d in _all_repository_datasets()
        if d["provenance"]["source"] == "synthetic"
    ]
    assert synthetic
    for dataset in synthetic:
        assert "random_state" in dataset["provenance"]["params"]


def test_synthetic_seeds_differ_across_tiers_datasets():
    """Two datasets generated from the same seed would not be two datasets."""
    seeds = [
        (d["name"], d["provenance"]["params"]["random_state"])
        for d in _all_repository_datasets()
        if d["provenance"]["source"] == "synthetic"
    ]
    per_tier: dict[str, list[int]] = {}
    for name, seed in seeds:
        per_tier.setdefault(name.rsplit("_", 1)[0], []).append(seed)
    for tier, values in per_tier.items():
        assert len(values) == len(set(values)), f"{tier} reuses a seed"


def test_bundled_data_is_not_labelled_synthetic():
    """breast_cancer is real patient data with its own citation and terms."""
    cancer = next(
        d for d in _all_repository_datasets() if d["name"] == "breast_cancer"
    )
    assert cancer["provenance"]["source"] == "bundled"
    assert "UCI" in cancer["provenance"]["license"]


def test_truncation_is_disclosed():
    """max_samples takes a positional slice, which is not a random sample."""
    cancer = next(
        d for d in _all_repository_datasets() if d["name"] == "breast_cancer"
    )
    assert "not a random sample" in cancer["provenance"]["notes"]


def test_openml_record_pins_a_version():
    """The only thing between a reproducible benchmark and a silent swap."""
    record = openml_provenance("diabetes", 1)
    assert record["params"]["version"] == 1
    assert record["source"] == "OpenML"


def test_synthetic_helper_records_its_params():
    record = synthetic_provenance("sklearn.datasets.make_moons", noise=0.3, random_state=7)
    assert record["params"] == {"noise": 0.3, "random_state": 7}
    assert record["source"] == "synthetic"


def test_bundled_helper_requires_a_licence():
    with pytest.raises(TypeError):
        bundled_provenance("x", url="u")  # type: ignore[call-arg]
