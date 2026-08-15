import pandas as pd
from imblearn.over_sampling import SMOTE

from oversampleqa.benchmark import load_standard_datasets, run_benchmark
from oversampleqa.report import generate_report


def test_generate_report_markdown(tmp_path):
    df = pd.DataFrame(
        {
            "dataset": ["d"],
            "oversampler": ["SMOTE"],
            "hidden_ratio": [0.1],
            "run": [0],
            "error_rate": [0.2],
        }
    )
    path = tmp_path / "report.md"
    content = generate_report(df, output_path=path)
    assert path.exists()
    assert isinstance(content, str)
    # plots should be created next to the report
    assert (tmp_path / "report_box.png").exists()
    assert (tmp_path / "report_rank.png").exists()


def test_benchmark_to_report(tmp_path):
    datasets = load_standard_datasets()[:1]
    df = run_benchmark(datasets, [SMOTE(random_state=0)], hidden_ratios=[0.1], n_runs=1)
    path = tmp_path / "bench_report.md"
    generate_report(df, output_path=path)
    assert path.exists()
