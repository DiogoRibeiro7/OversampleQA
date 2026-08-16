"""Report generation for oversampleqa."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .benchmark import compute_ranking
from .plotting import plot_error_boxplot, plot_error_ranking

__all__ = ["frame_to_markdown", "generate_report"]


def frame_to_markdown(frame: pd.DataFrame, *, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Written out rather than delegated to ``DataFrame.to_markdown``, which needs
    ``tabulate``. That is installed here only as a transitive dependency of
    something else, and depending on a package nobody declared is how a working
    install becomes a broken one after an unrelated upgrade.

    The previous implementation used ``to_csv(sep="|")``, which is not Markdown:
    it has no header separator row and no leading or trailing pipes, so it
    rendered as one run-on paragraph rather than a table.

    Args:
        frame: Frame to render. The index becomes the first column when it is
            named, since ``compute_ranking`` returns the oversampler there.
        float_format: Format applied to floating-point cells. Raw repr leaks
            values like ``0.21000000000000002`` into a document meant to be read.

    Returns:
        A Markdown table, or a note when the frame is empty.
    """
    if frame.empty:
        return "_No results._"

    display = frame.reset_index() if frame.index.name else frame.copy()

    def render(value: Any) -> str:
        if isinstance(value, float):
            return float_format.format(value)
        return str(value)

    headers = [str(c) for c in display.columns]
    rows = [[render(v) for v in row] for row in display.itertuples(index=False)]

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        padded = [c.ljust(w) for c, w in zip(cells, widths, strict=True)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(headers), separator, *(line(r) for r in rows)])


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
        body = frame.to_html(index=False, float_format=lambda v: f"{v:.4f}")
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
            ["# OversampleQA Report", "", "## Ranking", "", frame_to_markdown(summary)]
        )
    else:
        content = "<h1>OversampleQA Report</h1><h2>Ranking</h2>" + summary.to_html()

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
    return content
