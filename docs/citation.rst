Citing OversampleQA
===================

OversampleQA ships machine-readable citation metadata in two files at the
repository root:

``CITATION.cff``
   Citation File Format metadata. GitHub renders it as the *Cite this
   repository* button and converts it to BibTeX and APA on demand.

``.zenodo.json``
   Deposit metadata for Zenodo. When a GitHub release is archived, Zenodo reads
   this file from the release tarball and uses it instead of the defaults it
   would otherwise infer from the repository.

Keep the two in sync: title, authors, ORCID, license, and keywords should say
the same thing in both.

BibTeX
------

.. code-block:: bibtex

   @software{ribeiro_oversampleqa,
     author  = {Ribeiro, Diogo},
     title   = {{OversampleQA: a diagnostic toolkit to validate, audit,
                and benchmark oversampling methods}},
     version = {0.5.0},
     year    = {2026},
     doi     = {10.5281/zenodo.21940361},
     url     = {https://doi.org/10.5281/zenodo.21940361}
   }

Which DOI to cite
-----------------

Zenodo mints two kinds of DOI:

Concept DOI — ``10.5281/zenodo.21940361``
   Always resolves to the newest archived version. Zenodo labels it *Cite all
   versions* on the record page. This is what the README badge and
   ``CITATION.cff`` point at, because it never goes stale.

Version DOI — ``10.5281/zenodo.21967099`` for 0.5.0
   Points at one specific release. Use it in a paper, where the reader needs the
   exact code that produced the results.

Publishing and archiving a release
----------------------------------

The GitHub-Zenodo link is already enabled for this repository, so publishing a
GitHub release archives the source on Zenodo. The PyPI distribution is uploaded
by ``.github/workflows/publish.yml`` from the same GitHub release, using PyPI
Trusted Publishing rather than a stored API token.

Before the first PyPI release, configure the project on PyPI as a trusted
publisher:

* owner: ``diogoribeiro7``
* repository: ``OversampleQA``
* workflow: ``publish.yml``
* environment: ``pypi``

1. Update ``version`` and ``date-released`` in ``CITATION.cff``, the version in
   ``pyproject.toml``, ``[tool.commitizen]``, and ``src/oversampleqa/__init__.py``,
   and promote the *Unreleased* changelog section. Commit.
2. Run the local release checks::

      python scripts/release.py

3. Tag the release and push the tag::

      git tag -s vX.Y.Z -m "vX.Y.Z"
      git push origin vX.Y.Z

4. Publish a GitHub release for that tag. The PyPI workflow and the Zenodo
   webhook both fire on release publication, so pushing a tag alone publishes
   nothing.
5. Confirm the PyPI workflow uploaded both the source distribution and wheel.
6. Zenodo creates the record and mints a fresh version DOI within a few minutes.
   The concept DOI stays the same. Confirm the metadata came from
   ``.zenodo.json`` by opening the new record.
7. Add the new version DOI to the ``identifiers`` block in ``CITATION.cff``. The
   README badge and BibTeX entry use the concept DOI and need no change.

Should the integration ever need re-enabling, it is done by hand at
https://zenodo.org/account/settings/github/ by a maintainer with admin rights on
the repository; it cannot be set up from the codebase. Releases published while
the switch is off are not archived retroactively.

Release checklist
-----------------

Most of this is enforced by ``tests/test_release_metadata.py``, which runs on
every commit rather than only at release time — drift is introduced between
releases and would otherwise only surface during one. It checks that:

* the version agrees across ``pyproject.toml`` (both the Poetry and commitizen
  blocks), ``src/oversampleqa/__init__.py``, ``CITATION.cff``, the BibTeX in
  ``README.md``, and the BibTeX here;
* ``date-released`` in ``CITATION.cff`` matches the dated ``CHANGELOG`` heading
  for that version;
* the current version has a DOI recorded in ``CITATION.cff``;
* the README badge points at the *concept* DOI, and the "cite this exact
  release" line points at the *current version* DOI;
* author, affiliation, ORCID and licence agree between ``CITATION.cff`` and
  ``.zenodo.json``;
* ``.zenodo.json`` declares no ``version`` field.

A failing test names the file and the expected value. Two things it cannot
check, which remain manual:

* Author list in ``AUTHORS.md``, which has no machine-readable version to
  compare against.
* Whether the recorded DOI is the *right* one — only that one exists. Copy it
  from the Zenodo record page.

The version DOI is minted only after the GitHub release is published, so
``test_current_version_has_a_recorded_doi`` fails between the version bump and
the archive. That is the intended sequence: bump and tag, publish to PyPI and
archive on Zenodo, then commit the DOI. It is also why both 0.4.0 and 0.5.0
shipped without it — nothing failed to remind anyone.

``.zenodo.json`` deliberately omits a ``version`` field: Zenodo takes the
version from the git tag, so there is nothing to keep in sync there.
