"""Simple report generation for oversampleqa."""

from __future__ import annotations

import pandas as pd

from .benchmark import compute_ranking
from .plotting import plot_error_boxplot, plot_error_ranking


def generate_report(
    benchmark_results: pd.DataFrame,
    output_format: str = "markdown",
    output_path: str | None = None,
    include_plots: bool = True,
) -> str:
    """Generate a basic text report from benchmark results.

    Args:
        benchmark_results: Benchmark results dataframe.
        output_format: Output format (markdown or html).
        output_path: Optional output file path.
        include_plots: Whether to include plot artifacts.

    Returns:
        Rendered report content as a string.
    """
    if output_format not in {"markdown", "html"}:
        raise ValueError("output_format must be 'markdown' or 'html'")

    summary = compute_ranking(benchmark_results)
    if output_format == "markdown":
        content = summary.to_csv(sep="|")
    else:
        content = summary.to_html()

    if include_plots and output_path:
        base = str(output_path).rsplit(".", 1)[0]
        box_path = base + "_box.png"
        rank_path = base + "_rank.png"
        plot_error_boxplot(benchmark_results, save_path=box_path)
        plot_error_ranking(benchmark_results, save_path=rank_path)
        if output_format == "markdown":
            content += f"\n![boxplot]({box_path})\n![ranking]({rank_path})\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content
