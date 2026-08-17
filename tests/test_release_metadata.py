"""Release-facing metadata must agree across every file that states it.

The version, the DOIs, the author and the licence are written down in six
places. Nothing checked that they matched, and they did not: two consecutive
releases shipped without their DOI reaching ``CITATION.cff``, and the README
went on citing 0.3.0 and its version DOI while the package was at 0.5.0.

A checklist did not prevent that, twice. This is the check instead. It runs on
every commit, not only at release time, because drift is introduced between
releases and only becomes visible during one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CONCEPT_DOI = "10.5281/zenodo.21940361"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _toml_version(text: str, section: str) -> str:
    """Read ``version`` from one TOML section, without a TOML parser.

    ``tomllib`` is 3.11+, and CI runs 3.10. ``tomli`` exists in the lock only as
    a transitive dependency of something else, and depending on a package nobody
    declared is how a working install breaks after an unrelated upgrade. Two
    scalar fields do not justify either.
    """
    after = text.split(f"[{section}]", 1)
    assert len(after) == 2, f"no [{section}] section"
    block = after[1].split("\n[", 1)[0]
    found = re.search(r'^version = "(.*?)"', block, re.M)
    assert found, f"no version in [{section}]"
    return found.group(1)


@pytest.fixture(scope="module")
def pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def zenodo() -> dict:
    return json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def citation_doc() -> str:
    return (ROOT / "docs" / "citation.rst").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def changelog() -> str:
    return (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def version(pyproject) -> str:
    return _toml_version(pyproject, "tool.poetry")


def _version_dois(citation: dict) -> dict[str, str]:
    """Map release version to its DOI, from the CITATION.cff identifiers."""
    found = {}
    for entry in citation.get("identifiers", []):
        match = re.search(r"version (\d+\.\d+\.\d+)", entry.get("description", ""))
        if match:
            found[match.group(1)] = entry["value"]
    return found


# --- the version, in every place it is written ---


def test_version_is_wellformed(version):
    assert VERSION_RE.match(version), version


def test_commitizen_version_matches(pyproject, version):
    assert _toml_version(pyproject, "tool.commitizen") == version


def test_package_version_matches(version):
    text = (ROOT / "src" / "oversampleqa" / "__init__.py").read_text(encoding="utf-8")
    found = re.search(r'__version__ = "(.*?)"', text)
    assert found and found.group(1) == version


def test_citation_version_matches(citation, version):
    assert str(citation["version"]) == version


def test_changelog_has_a_section_for_the_current_version(changelog, version):
    assert f"## [{version}]" in changelog, (
        f"CHANGELOG has no section for {version}; the version was bumped "
        "without promoting the Unreleased block."
    )


def test_changelog_date_matches_citation(changelog, citation, version):
    found = re.search(rf"## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})", changelog)
    assert found, f"no dated CHANGELOG heading for {version}"
    assert found.group(1) == str(citation["date-released"])


def test_readme_bibtex_version_matches(readme, version):
    """The README's BibTeX block said 0.3.0 while the package was at 0.5.0."""
    found = re.search(r"version = \{(.*?)\}", readme)
    assert found, "README has no BibTeX version field"
    assert found.group(1) == version


def test_citation_doc_bibtex_version_matches(citation_doc, version):
    found = re.search(r"version = \{(.*?)\}", citation_doc)
    assert found, "docs/citation.rst has no BibTeX version field"
    assert found.group(1) == version


# --- DOIs ---


def test_concept_doi_is_the_same_everywhere(readme, citation, citation_doc):
    assert CONCEPT_DOI in readme
    assert CONCEPT_DOI in citation_doc
    assert CONCEPT_DOI in {i["value"] for i in citation["identifiers"]}


def test_readme_badge_points_at_the_concept_doi(readme):
    """A badge on a version DOI would go stale at every release."""
    badge = re.search(r"\[!\[DOI\]\((.*?)\)\]\((.*?)\)", readme)
    assert badge, "no DOI badge in README"
    assert CONCEPT_DOI in badge.group(1)
    assert CONCEPT_DOI in badge.group(2)


def _released_versions(changelog: str) -> list[str]:
    """Versions with a dated CHANGELOG heading, newest first."""
    return re.findall(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}", changelog, re.M)


def test_every_previous_release_has_a_recorded_doi(citation, changelog, version):
    """Step 5 of the release checklist, missed on both 0.4.0 and 0.5.0.

    Deliberately excludes the version being prepared. A version DOI is minted by
    Zenodo *after* the GitHub release is published, so between the version bump
    and the archive there is a window where the current version legitimately has
    no DOI -- and requiring one there turns CI red for the whole of it.

    Checking the previous release instead needs no flag and cannot be forgotten:
    the next version bump is blocked until the last one's DOI is recorded. That
    catches the omission one release later rather than never, which is what
    actually happened twice.
    """
    dois = _version_dois(citation)
    assert dois, "CITATION.cff records no version DOIs at all"

    # Archiving began at 0.2.0; 0.1.0 predates the Zenodo integration and has no
    # DOI to record. The floor is derived from the earliest DOI actually present
    # rather than hardcoded, so it stays correct without being maintained.
    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split("."))

    floor = min(parts(v) for v in dois)
    previous = [
        v
        for v in _released_versions(changelog)
        if v != version and parts(v) >= floor
    ]
    assert previous, "no earlier archived release found in the CHANGELOG"
    missing = [v for v in previous if v not in dois]
    assert not missing, (
        f"CITATION.cff records no DOI for {missing}. Copy it from the Zenodo "
        "record page into the identifiers block."
    )


def test_every_recorded_doi_is_distinct(citation):
    values = [i["value"] for i in citation["identifiers"]]
    assert len(values) == len(set(values))


def test_no_recorded_version_doi_equals_the_concept_doi(citation):
    """They resolve differently; conflating them misdirects a citation."""
    assert CONCEPT_DOI not in _version_dois(citation).values()


def test_readme_cites_the_current_version_doi_once_it_exists(readme, citation, version):
    """It offered 0.3.0's DOI as 'this exact release' at 0.5.0.

    Conditional on the DOI existing, for the same reason as above: during the
    pre-archive window there is nothing to cite yet.
    """
    dois = _version_dois(citation)
    if version not in dois:
        pytest.skip(f"{version} has no DOI yet; Zenodo mints it after release")
    assert dois[version] in readme, (
        f"README should offer {dois[version]} as the current version DOI"
    )


def test_citation_doc_cites_the_current_version_doi_once_it_exists(
    citation_doc, citation, version
):
    dois = _version_dois(citation)
    if version not in dois:
        pytest.skip(f"{version} has no DOI yet; Zenodo mints it after release")
    assert dois[version] in citation_doc


def test_recorded_versions_all_appear_in_the_changelog(citation, changelog):
    for released in _version_dois(citation):
        assert f"## [{released}]" in changelog, released


# --- .zenodo.json ---


def test_zenodo_declares_no_version(zenodo):
    """Zenodo takes the version from the git tag.

    A version here would be a seventh place to forget to update.
    """
    assert "version" not in zenodo


def test_zenodo_author_matches_citation(zenodo, citation):
    author = citation["authors"][0]
    creator = zenodo["creators"][0]
    assert creator["name"] == f"{author['family-names']}, {author['given-names']}"
    assert creator["affiliation"] == author["affiliation"]
    assert creator["orcid"] in author["orcid"]


def test_zenodo_licence_matches_citation(zenodo, citation):
    assert zenodo["license"].lower() == str(citation["license"]).lower()


def test_zenodo_links_to_the_same_repository_as_citation(zenodo, citation):
    """Deliberately not asserting a concept DOI here.

    Zenodo derives the version/concept relations itself from the deposition
    chain. Writing one into ``related_identifiers`` by hand would be a
    duplicate of something the service already knows, and a seventh place to
    keep in step.
    """
    related = json.dumps(zenodo.get("related_identifiers", []))
    assert str(citation["repository-code"]).rstrip("/") in related


# --- packaging ---


def test_py_typed_marker_is_present():
    """PEP 561: without this file, every downstream type checker ignores us.

    The package is annotated throughout and checked under mypy strict, and
    exports Protocols for callers to implement. None of that reaches a consumer
    unless the marker ships, and the failure is silent -- their checker simply
    treats the package as untyped.
    """
    assert (ROOT / "src" / "oversampleqa" / "py.typed").is_file()


def test_package_declares_a_pypi_readme(pyproject):
    """A missing readme renders the PyPI project page blank."""
    assert 'readme = "README.md"' in pyproject
