"""
Benchmarking Oversamplers
=========================

Run a benchmark across the built-in standard datasets and render a report.
"""

import json
from pathlib import Path

from imblearn.over_sampling import ADASYN, SMOTE

from oversampleqa.benchmark import load_standard_datasets, run_benchmark
from oversampleqa.report import generate_report


def main() -> None:
    datasets = load_standard_datasets()[:2]
    oversamplers = [SMOTE(random_state=0), ADASYN(random_state=0)]
    results = run_benchmark(
        datasets,
        oversamplers,
        hidden_ratios=[0.1, 0.25],
        n_runs=3,
        random_state=0,
    )
    print(results)

    report_path = Path("benchmark_report.md")
    generate_report(results, output_path=str(report_path), include_plots=False)

    metadata_path = report_path.with_name(f"{report_path.name}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"Wrote report: {report_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"OversampleQA version: {metadata['environment']['oversampleqa_version']}")
    print(f"Benchmark rows audited: {metadata['source']['row_count']}")


if __name__ == "__main__":
    main()
