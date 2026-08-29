"""Tests for report rendering.

The Markdown path previously used ``to_csv(sep="|")``, which is not Markdown:
no header separator row, no leading or trailing pipes, and unrounded floats. It
rendered as one run-on paragraph in any viewer.
"""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

from oversampleqa._export_metadata import metadata_sidecar_path
from oversampleqa.report import frame_to_markdown, generate_report


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "dataset": ["a"] * 4,
            "oversampler": ["SMOTE", "SMOTE", "ROS", "ROS"],
            "hidden_ratio": [0.1] * 4,
            "run": [0, 1, 0, 1],
            "error_rate": [0.10, 0.12, 0.20, 0.22],
        }
    )


def _is_markdown_table(text: str) -> bool:
    """A GFM table needs piped rows and a header separator."""
    lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return False
    separator = lines[1]
    return set(separator.replace("|", "").replace(" ", "")) <= {"-", ":"}


def test_markdown_output_is_actually_a_table(results):
    content = generate_report(results, output_format="markdown", include_plots=False)
    assert _is_markdown_table(content)


def test_floats_are_rounded(results):
    """Raw repr leaks values like 0.21000000000000002 into a read document."""
    content = generate_report(results, output_format="markdown", include_plots=False)
    assert "0.21000000000000002" not in content
    assert "0.2100" in content


def test_every_row_is_piped(results):
    content = generate_report(results, output_format="markdown", include_plots=False)
    table_lines = [
        line
        for line in content.splitlines()
        if line.strip() and not line.startswith("#") and "|" in line
    ]
    assert all(line.startswith("|") and line.endswith("|") for line in table_lines)


def test_empty_frame_renders_a_note():
    assert frame_to_markdown(pd.DataFrame()) == "_No results._"


def test_named_index_becomes_a_column():
    """compute_ranking puts the oversampler in the index; it must still show."""
    frame = pd.DataFrame({"mean": [0.1, 0.2]}, index=pd.Index(["a", "b"], name="key"))
    rendered = frame_to_markdown(frame)
    assert "key" in rendered
    assert "| a" in rendered


def test_unknown_format_raises(results):
    with pytest.raises(ValueError, match="markdown"):
        generate_report(results, output_format="latex", include_plots=False)


def test_fidelity_section_is_appended(results):
    fidelity = {
        "SMOTE": {
            "error_rate": 0.20,
            "precision": 0.94,
            "memorisation_distance_ratio": 0.40,
        },
        "ROS": {
            "error_rate": 0.24,
            "precision": 0.95,
            "memorisation_distance_ratio": 0.00,
        },
    }
    content = generate_report(
        results,
        output_format="markdown",
        include_plots=False,
        fidelity_reports=fidelity,
    )
    assert "## Fidelity and diversity" in content
    assert "memorisation" in content


def test_fidelity_heading_has_a_blank_line_before_it(results):
    """A heading immediately after a table does not render as a heading."""
    content = generate_report(
        results,
        output_format="markdown",
        include_plots=False,
        fidelity_reports={"SMOTE": {"error_rate": 0.2}},
    )
    lines = content.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("## Fidelity"))
    assert lines[index - 1].strip() == ""


def test_fidelity_notes_are_included(results):
    """The interpretation is the point; a table of numbers is not a diagnosis."""
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.datasets import make_classification

    from oversampleqa.fidelity import fidelity_report

    X, y = make_classification(
        n_samples=700,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = fidelity_report(X, y, 1, RandomOverSampler(random_state=0))

    content = generate_report(
        results,
        output_format="markdown",
        include_plots=False,
        fidelity_reports={"RandomOverSampler": report},
    )
    assert "Memorisation" in content


def test_html_output_includes_fidelity(results):
    content = generate_report(
        results,
        output_format="html",
        include_plots=False,
        fidelity_reports={"SMOTE": {"error_rate": 0.2}},
    )
    assert "<h2>Fidelity and diversity</h2>" in content
    assert "<table" in content


def test_report_without_fidelity_is_unchanged(results):
    """The parameter is optional; omitting it must not alter the ranking output."""
    plain = generate_report(results, output_format="markdown", include_plots=False)
    assert "Fidelity" not in plain


def test_report_export_writes_metadata_sidecar(results, tmp_path):
    out = tmp_path / "report.md"
    generate_report(
        results,
        output_format="markdown",
        output_path=str(out),
        include_plots=False,
    )

    metadata = json.loads(metadata_sidecar_path(out).read_text(encoding="utf-8"))

    assert metadata["export_kind"] == "benchmark_report"
    assert metadata["data"]["columns"]
    assert metadata["source"]["row_count"] == len(results)
