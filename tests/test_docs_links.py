"""Tests for documentation link checking helpers."""

from __future__ import annotations

from scripts.check_docs_links import (
    documentation_files,
    iter_checked_urls,
    strip_url_punctuation,
)


def test_strip_url_punctuation_handles_markdown_and_bibtex_suffixes():
    assert (
        strip_url_punctuation("https://doi.org/10.5281/zenodo.21940361}")
        == "https://doi.org/10.5281/zenodo.21940361"
    )
    assert (
        strip_url_punctuation("https://github.com/diogoribeiro7/OversampleQA).")
        == "https://github.com/diogoribeiro7/OversampleQA"
    )


def test_iter_checked_urls_skips_git_dependency_specs(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "\n".join(
            [
                "pip install git+https://github.com/diogoribeiro7/OversampleQA.git",
                "[project](https://github.com/diogoribeiro7/OversampleQA)",
                "[unsupported](https://example.com/not-checked)",
            ]
        ),
        encoding="utf-8",
    )

    assert iter_checked_urls(doc) == {
        "https://github.com/diogoribeiro7/OversampleQA"
    }


def test_documentation_files_include_citation_metadata(tmp_path):
    readme = tmp_path / "README.md"
    citation = tmp_path / "CITATION.cff"
    docs = tmp_path / "docs"
    nested = docs / "page.md"
    readme.write_text("", encoding="utf-8")
    citation.write_text("", encoding="utf-8")
    docs.mkdir()
    nested.write_text("", encoding="utf-8")

    assert set(documentation_files((readme, citation, docs))) == {
        citation,
        nested,
        readme,
    }
