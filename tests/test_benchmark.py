from imblearn.over_sampling import SMOTE

from oversampleqa.benchmark import (
    _BENCHMARK_COLUMNS,
    compute_ranking,
    export_benchmark_results,
    load_standard_datasets,
    run_benchmark,
)


def test_run_benchmark_basic():
    datasets = load_standard_datasets()
    assert len(datasets) >= 7
    oversamplers = [SMOTE(random_state=0)]
    df = run_benchmark(datasets, oversamplers, hidden_ratios=[0.1], n_runs=1)
    assert not df.empty
    # Pinned against the declared schema rather than a literal, so the two
    # cannot drift apart. `metric` joined the frame because it identifies a
    # measurement: error rates are not comparable across metrics, so rows from
    # two sweeps are indistinguishable without it.
    assert list(df.columns) == list(_BENCHMARK_COLUMNS)


def test_export_and_ranking(tmp_path):
    df = run_benchmark(
        load_standard_datasets()[:1],
        [SMOTE(random_state=0)],
        hidden_ratios=[0.1],
        n_runs=1,
    )
    summary = compute_ranking(df)
    assert "rank" in summary.columns
    csv_path = tmp_path / "out.csv"
    export_benchmark_results(df, csv_path, fmt="csv")
    assert csv_path.exists()
    json_path = tmp_path / "out.json"
    export_benchmark_results(df, json_path, fmt="json")
    assert json_path.exists()
    md_path = tmp_path / "out.md"
    export_benchmark_results(df, md_path, fmt="markdown")
    assert md_path.exists()


def test_load_standard_datasets_with_openml():
    datasets = load_standard_datasets(include_openml=True)
    assert len(datasets) >= 7


def test_standard_datasets_have_provenance():
    required = {"source", "generator", "params", "url", "license", "notes"}
    for dataset in load_standard_datasets():
        provenance = dataset.get("provenance")
        assert provenance is not None, f"{dataset['name']} missing provenance"
        assert required <= set(provenance), f"{dataset['name']} provenance incomplete"
        assert provenance["source"] == "synthetic"
        assert provenance["license"]
