"""Tests for documentation link checking helpers."""

from __future__ import annotations

from urllib.error import HTTPError

from scripts import check_docs_links
from scripts.check_docs_links import (
    check_url,
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


# --- distinguishing a dead link from an unreachable host ---
#
# The checker reported both as failures. On 2026-08-31 doi.org returned five
# timeouts and a 502 for six perfectly valid DOIs, which read exactly like six
# dead links -- and a report that cries wolf is skipped as reliably as a red
# build that means nothing.


def _fixed_response(monkeypatch, result):
    """Make request_url return a status, or raise, without any network."""

    def fake(url, timeout):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(check_docs_links, "request_url", fake)


def test_a_404_is_reported_as_gone(monkeypatch):
    _fixed_response(monkeypatch, 404)
    assert check_url("https://example.com/x", retries=1, timeout=1.0) == (
        "gone",
        "HTTP 404",
    )


def test_a_410_is_reported_as_gone(monkeypatch):
    _fixed_response(monkeypatch, 410)
    kind, _ = check_url("https://example.com/x", retries=1, timeout=1.0)
    assert kind == "gone"


def test_a_502_is_reported_as_unavailable(monkeypatch):
    """The host answered badly; that is not evidence the link is wrong."""
    _fixed_response(monkeypatch, 502)
    kind, _ = check_url("https://example.com/x", retries=0, timeout=1.0)
    assert kind == "unavailable"


def test_a_timeout_is_reported_as_unavailable(monkeypatch):
    _fixed_response(monkeypatch, TimeoutError("The read operation timed out"))
    kind, detail = check_url("https://example.com/x", retries=0, timeout=1.0)
    assert kind == "unavailable"
    assert "timed out" in detail


def test_a_403_is_unavailable_rather_than_gone(monkeypatch):
    """Hosts refuse robots; that says nothing about whether the page exists."""
    _fixed_response(
        monkeypatch, HTTPError("https://example.com/x", 403, "Forbidden", {}, None)
    )
    kind, _ = check_url("https://example.com/x", retries=0, timeout=1.0)
    assert kind == "unavailable"


def test_a_reachable_url_returns_none(monkeypatch):
    _fixed_response(monkeypatch, 200)
    assert check_url("https://example.com/x", retries=1, timeout=1.0) is None


def test_a_gone_url_is_not_retried(monkeypatch):
    """A 404 will not become a 200 on the second attempt.

    Retrying it only slows the run down, and the sleep between attempts is
    paid for every dead link.
    """
    attempts = []

    def fake(url, timeout):
        attempts.append(url)
        return 404

    monkeypatch.setattr(check_docs_links, "request_url", fake)
    check_url("https://example.com/x", retries=3, timeout=1.0)

    assert len(attempts) == 1


def test_an_unavailable_url_is_retried(monkeypatch):
    """A transient failure is exactly what a retry is for."""
    attempts = []

    def fake(url, timeout):
        attempts.append(url)
        raise TimeoutError("nope")

    monkeypatch.setattr(check_docs_links, "request_url", fake)
    check_url("https://example.com/x", retries=2, timeout=1.0)

    assert len(attempts) == 3
