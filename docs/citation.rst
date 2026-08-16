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

Archiving a release on Zenodo
-----------------------------

The GitHub-Zenodo link is already enabled for this repository, so publishing a
release is all that is needed. Releases are cut by hand; nothing in CI tags or
publishes them.

1. Update ``version`` and ``date-released`` in ``CITATION.cff``, the version in
   ``pyproject.toml``, ``[tool.commitizen]``, and ``src/oversampleqa/__init__.py``,
   and promote the *Unreleased* changelog section. Commit.
2. Tag the release and push the tag::

      git tag -s vX.Y.Z -m "vX.Y.Z"
      git push origin vX.Y.Z

3. Publish a GitHub release for that tag. The webhook fires on release
   publication only, so pushing a tag alone archives nothing.
4. Zenodo creates the record and mints a fresh version DOI within a few minutes.
   The concept DOI stays the same. Confirm the metadata came from
   ``.zenodo.json`` by opening the new record.
5. Add the new version DOI to the ``identifiers`` block in ``CITATION.cff``. The
   README badge and BibTeX entry use the concept DOI and need no change.

Should the integration ever need re-enabling, it is done by hand at
https://zenodo.org/account/settings/github/ by a maintainer with admin rights on
the repository; it cannot be set up from the codebase. Releases published while
the switch is off are not archived retroactively.

Release checklist
-----------------

Before publishing a release that will be archived:

* ``version`` in ``CITATION.cff`` matches the release tag.
* ``date-released`` in ``CITATION.cff`` matches the release date.
* Author list and affiliations are current in ``CITATION.cff``, ``.zenodo.json``,
  and ``AUTHORS.md``.

``.zenodo.json`` deliberately omits a ``version`` field: Zenodo takes the
version from the git tag, so there is nothing to keep in sync there.
