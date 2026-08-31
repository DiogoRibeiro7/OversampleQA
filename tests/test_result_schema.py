"""Every result surface exposes the same identifier columns.

Four surfaces emit rows a caller can concatenate, join or store, and they grew
separately: ``dataset`` against ``dataset_name``, ``oversampler`` against
``oversampler_name``. Joining two of them meant knowing which one produced
which, and ``frame["dataset"]`` worked or raised depending on where the frame
came from.

``IDENTIFIER_COLUMNS`` is the contract. This pins that each surface honours it,
because a shared schema nothing checks is a convention, and conventions drift
exactly as these four did.
"""

from __future__ import annotations

from oversampleqa._schema import IDENTIFIER_COLUMNS
from oversampleqa.advanced_benchmark import BenchmarkResult, StatisticalBenchmark
from oversampleqa.benchmark import _BENCHMARK_COLUMNS
from oversampleqa.reports import RunMetadata, ValidationReport


def _missing(columns) -> list[str]:
    return [name for name in IDENTIFIER_COLUMNS if name not in set(columns)]


def test_simple_benchmark_frame_declares_every_identifier():
    assert _missing(_BENCHMARK_COLUMNS) == []


def test_fold_frame_declares_every_identifier_even_when_empty():
    """Empty of rows but not of columns, so column access works either way."""
    benchmark = StatisticalBenchmark.__new__(StatisticalBenchmark)
    benchmark._fold_records_all = []

    frame = benchmark.fold_results()

    assert _missing(frame.columns) == []
    assert frame.empty


def test_validation_report_frame_declares_every_identifier():
    report = ValidationReport(
        error_rate=0.25,
        metadata=RunMetadata(oversampler="SMOTE", metric="hassanat"),
    )

    frame = report.to_frame()

    assert _missing(frame.columns) == []
    assert len(frame) == 1


def test_validation_report_carries_the_dataset_hash_as_a_column():
    """A row nobody can trace back to its data is not much of a record.

    The validator sees arrays, so ``dataset`` is empty unless the caller names
    one. ``dataset_hash`` is the identity that is always there, and it used to
    be reachable only through the ``meta_`` prefix.
    """
    report = ValidationReport(
        error_rate=0.25,
        metadata=RunMetadata(oversampler="SMOTE", dataset_hash="abc123"),
    )

    frame = report.to_frame()

    assert frame["dataset_hash"].iloc[0] == "abc123"
    assert frame["dataset"].iloc[0] == ""


def test_a_named_dataset_reaches_the_validation_frame():
    report = ValidationReport(
        error_rate=0.25,
        metadata=RunMetadata(oversampler="SMOTE", dataset="creditcard"),
    )

    assert report.to_frame()["dataset"].iloc[0] == "creditcard"


def test_the_canonical_names_agree_with_the_originals_they_alias():
    """`dataset` must not drift from `dataset_name`; they are one value."""
    record = StatisticalBenchmark._result_to_dict(
        BenchmarkResult(
            dataset_name="d",
            oversampler_name="SMOTE",
            metric="hassanat",
            hidden_ratio=0.1,
            reference="hidden_minority",
            minority_label=1,
            random_state=0,
            n_folds=5,
            n_repeats=1,
            oversampleqa_version="0.6.1",
            error_rates=[0.1, 0.2],
            mean_error=0.15,
            std_error=0.05,
            confidence_interval=(0.05, 0.25),
        )
    )

    assert record["dataset"] == record["dataset_name"] == "d"
    assert record["oversampler"] == record["oversampler_name"] == "SMOTE"
    assert _missing(record) == []
