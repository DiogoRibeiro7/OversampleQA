"""Report generation for oversampleqa."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._export_metadata import write_export_metadata
from ._render import frame_to_html, frame_to_markdown
from ._report_metadata import report_metadata_html, report_metadata_markdown
from .benchmark import compute_ranking
from .plotting import plot_error_boxplot, plot_error_ranking

__all__ = ["frame_to_html", "frame_to_markdown", "generate_report"]


def _fidelity_section(reports: dict[str, Any], output_format: str) -> str:
    """Render the fidelity block for one or more samplers.

    The error rate cannot distinguish an implausible generator from one that
    copies its training data, so a report carrying only the error rate is
    missing the axis that usually decides which sampler to use.
    """
    rows = []
    notes: list[str] = []
    for name, report in reports.items():
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        rows.append(
            {
                "oversampler": name,
                "error_rate": payload.get("error_rate", float("nan")),
                "precision": payload.get("precision", float("nan")),
                "recall": payload.get("recall", float("nan")),
                "density": payload.get("density", float("nan")),
                "coverage": payload.get("coverage", float("nan")),
                "memorisation": payload.get(
                    "memorisation_distance_ratio", float("nan")
                ),
                "boundary": payload.get("boundary_violation_strict", float("nan")),
            }
        )
        if hasattr(report, "interpret"):
            notes.extend(f"**{name}**: {note}" for note in report.interpret())

    frame = pd.DataFrame(rows)
    if output_format == "html":
        body = str(frame.to_html(index=False, float_format=lambda v: f"{v:.4f}"))
        if notes:
            body += "<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>"
        return "<h2>Fidelity and diversity</h2>" + body

    # A heading immediately after a table does not render as a heading; the
    # blank line is required.
    parts = ["", "", "## Fidelity and diversity", "", frame_to_markdown(frame), ""]
    parts.extend(f"- {note}" for note in notes)
    parts.append(
        "\n_A memorisation ratio near zero means the generator sits on top of "
        "its training data; the error rate says nothing about synthesis quality "
        "in that case._"
    )
    return "\n".join(parts)


def generate_report(
    benchmark_results: pd.DataFrame,
    output_format: str = "markdown",
    output_path: str | None = None,
    include_plots: bool = True,
    fidelity_reports: dict[str, Any] | None = None,
) -> str:
    """Generate a report from benchmark results.

    Args:
        benchmark_results: Benchmark results dataframe.
        output_format: Output format (``markdown`` or ``html``).
        output_path: Optional output file path.
        include_plots: Whether to include plot artifacts.
        fidelity_reports: Optional mapping of oversampler name to
            :class:`~oversampleqa.fidelity.FidelityReport`. When given, a
            fidelity section is appended covering the axis the error rate
            cannot express.

    Returns:
        Rendered report content as a string.

    Raises:
        ValueError: If ``output_format`` is not recognised.
    """
    if output_format not in {"markdown", "html"}:
        raise ValueError("output_format must be 'markdown' or 'html'")

    summary = compute_ranking(benchmark_results)
    if output_format == "markdown":
        content = "\n".join(
            [
                "# OversampleQA Report",
                "",
                "## Run metadata",
                "",
                report_metadata_markdown(benchmark_results),
                "",
                "## Ranking",
                "",
                frame_to_markdown(summary),
            ]
        )
    else:
        content = (
            "<h1>OversampleQA Report</h1><h2>Run metadata</h2>"
            + report_metadata_html(benchmark_results)
            + "<h2>Ranking</h2>"
            + summary.to_html()
        )

    if fidelity_reports:
        content += _fidelity_section(fidelity_reports, output_format)

    if include_plots and output_path:
        base = str(output_path).rsplit(".", 1)[0]
        box_path = base + "_box.png"
        rank_path = base + "_rank.png"
        plot_error_boxplot(benchmark_results, save_path=box_path)
        plot_error_ranking(benchmark_results, save_path=rank_path)
        if output_format == "markdown":
            content += f"\n\n![boxplot]({box_path})\n![ranking]({rank_path})\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        write_export_metadata(
            output_path,
            export_kind="benchmark_report",
            data=summary,
            extra={
                "source": {
                    "row_count": len(benchmark_results),
                    "columns": [str(column) for column in benchmark_results.columns],
                    "attrs": dict(benchmark_results.attrs),
                }
            },
        )
    return content
