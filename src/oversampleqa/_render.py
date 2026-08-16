"""Frame rendering shared by the report and export paths.

Lives here rather than in ``report`` because ``report`` imports from
``benchmark``, so ``benchmark`` importing back from ``report`` would be a
cycle -- and both need to render a frame as Markdown. Two copies is how the
``to_csv(sep="|")`` bug survived in one of them after being fixed in the
other.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["frame_to_html", "frame_to_markdown"]


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


def frame_to_html(frame: pd.DataFrame, *, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as an HTML table.

    Args:
        frame: Frame to render. A named index becomes the first column.
        float_format: Format applied to floating-point cells.

    Returns:
        An HTML table, or a note when the frame is empty.
    """
    if frame.empty:
        return "<p><em>No results.</em></p>"
    display = frame.reset_index() if frame.index.name else frame
    return display.to_html(index=False, float_format=float_format.format)
