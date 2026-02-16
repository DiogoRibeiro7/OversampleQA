"""
Benchmark Example
=================

Run a simple benchmark using oversampleqa.
"""

from imblearn.over_sampling import SMOTE

from oversampleqa.benchmark import load_standard_datasets, run_benchmark
from oversampleqa.report import generate_report


def main() -> None:
    datasets = load_standard_datasets()
    oversamplers = [SMOTE(random_state=0)]
    results = run_benchmark(datasets, oversamplers, hidden_ratios=[0.1], n_runs=1)
    print(results)
    generate_report(results, output_path="benchmark_report.md")


if __name__ == "__main__":
    main()
