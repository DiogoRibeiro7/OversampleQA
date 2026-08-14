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
     version = {0.2.0},
     year    = {2026},
     url     = {https://github.com/diogoribeiro7/OversampleQA}
   }

Once a release has been archived, add the DOI to the entry with
``doi = {10.5281/zenodo.XXXXXXX}``.

Which DOI to cite
-----------------

Zenodo mints two kinds of DOI:

Concept DOI
   Always resolves to the newest archived version. Zenodo labels it *Cite all
   versions* on the record page. Use it in the README badge and in
   ``CITATION.cff``, because it never goes stale.

Version DOI
   Points at one specific release. Use it in a paper, where the reader needs the
   exact code that produced the results.

Archiving a release on Zenodo
-----------------------------

The link between GitHub and Zenodo is enabled once, by hand, by a maintainer
with admin rights on the repository. It cannot be set up from the codebase.

1. Sign in at https://zenodo.org with the GitHub account that owns the
   repository and grant the Zenodo GitHub application access.
2. Open https://zenodo.org/account/settings/github/ and flip the switch next to
   ``diogoribeiro7/OversampleQA``. Only repositories where you have admin rights
   appear; use *Sync now* if the repository is missing from the list.
3. Publish a GitHub release. The webhook fires on release publication only, so
   pushing a tag alone does nothing, and releases published *before* the switch
   was enabled are not archived retroactively.
4. Zenodo creates the record and mints the DOI within a few minutes. Check the
   result at https://zenodo.org/account/settings/github/ and open the record to
   confirm the metadata came from ``.zenodo.json``.
5. Copy the concept DOI into the README badge, the BibTeX entry above, and the
   ``identifiers`` block in ``CITATION.cff`` (commented out until the first
   archive exists). Commit that change.

Release checklist
-----------------

Before publishing a release that will be archived:

* ``version`` in ``CITATION.cff`` matches the release tag.
* ``date-released`` in ``CITATION.cff`` matches the release date.
* Author list and affiliations are current in ``CITATION.cff``, ``.zenodo.json``,
  and ``AUTHORS.md``.

``.zenodo.json`` deliberately omits a ``version`` field: Zenodo takes the
version from the git tag, so there is nothing to keep in sync there.
