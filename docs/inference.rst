Statistical Inference
=====================

The hidden-majority error rate on its own is a bare number. This page covers the
machinery that makes it interpretable: a null distribution to compare it
against, tests for whether synthetic points are distributionally
indistinguishable from real ones, and intervals that account for the dependence
between synthetic points.

Why 0.13 means nothing on its own
---------------------------------

The error rate depends on dimensionality, minority density, ``hidden_ratio`` and
the choice of metric. Two datasets can produce the same rate for entirely
different reasons, and there is no threshold above which an oversampler is
"bad". It is a **relative** quantity, so it needs a reference.

Calibrating against a null
--------------------------

:func:`~oversampleqa.null_error_rate` supplies one. Instead of synthetic points,
it scores **real held-out minority points** through the identical pipeline.
Those points are drawn from the true minority distribution by construction, so
their error rate is what a perfect generator would achieve on this data. A
**ceiling** is computed the same way from majority-drawn points -- a generator
that learned the wrong distribution entirely.

.. code-block:: python

   from oversampleqa import validate_oversampling, null_error_rate
   from imblearn.over_sampling import SMOTE

   observed = validate_oversampling(X, y, minority_label=1,
                                    oversampler=SMOTE(random_state=0))

   calibration = null_error_rate(X, y, minority_label=1, observed=observed)
   print(calibration.interpret())

That converts a bare number into a statement:

.. code-block:: text

   0.000 is within the null interval [0.000, 0.067] --
   indistinguishable from an ideal generator on this data.

or, for a generator drawing from the wrong distribution:

.. code-block:: text

   0.950 is above the null interval [0.000, 0.067] (z=41.18) --
   worse than an ideal generator would achieve here.

or, when the observed rate falls *below* the interval:

.. code-block:: text

   0.010 is below the null interval [0.098, 0.184] (z=-2.71) --
   better than real held-out minority points score. Check memorisation before
   reading this as quality.

That third case is not a better result. Real minority points define the bar;
scoring below it usually means the synthetic points sit on top of the training
data, which :doc:`/fidelity` measures directly.

.. important::

   ``hidden_ratio`` and ``metric`` must match the run that produced
   ``observed``. The rate's scale depends on both, so calibrating against a null
   computed with different settings compares two different quantities.

How the split is built
----------------------

Calibration divides the minority into **three** disjoint sets, not two:

* one standing in for the sampler's training data;
* one **common reference**, the same set ``validate_oversampling`` scores its
  synthetic points against;
* one supplying the real null candidates, which stand in for synthetic points.

The third must be disjoint from the second, because scoring a point against a
set it belongs to measures nothing. Ceiling candidates are drawn from the
**visible** majority for the same reason: a point that is itself in the hidden
majority sits at distance zero from it and is counted as an error by
construction.

.. note::

   Before 0.6, the null scored held-out minority points against ``fit_minority``
   -- roughly 90% of the minority -- while the validator scored synthetic points
   against the held-out 10%. A denser reference means closer nearest neighbours
   and fewer errors, so the null sat around 0.033 where the same experiment
   gives 0.133. The bar was roughly four times too low, and ordinary samplers
   were therefore reported as significantly worse than ideal. Ceiling candidates
   were also drawn from the full majority, overlapping the hidden set in 64% of
   draws. Calibrations produced before 0.6 carry both defects.

Two-sample tests
----------------

Counting how often a point's nearest neighbour comes from the other sample *is*
the nearest-neighbour two-sample statistic of Schilling (1986) and Henze (1988).
Naming it brings a null distribution and a literature with it.

Applied to synthetic points against held-out real minority points, these test
the question a user actually has: **are the synthetic points distributionally
indistinguishable from real minority points?** A high p-value is evidence of
good synthesis.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Test
     - Statistic
     - Tail
   * - :func:`~oversampleqa.nn_two_sample_test`
     - k-nearest-neighbour pairs sharing a sample label
     - right
   * - :func:`~oversampleqa.mst_two_sample_test`
     - minimum-spanning-tree edges joining the samples
     - left
   * - :func:`~oversampleqa.cross_match_test`
     - cross-sample pairs in a greedy matching
     - left

All accept any metric from the registry, so ``hassanat`` composes with the
inferential layer. p-values come from permutation, so no asymptotic assumption
is needed; the nearest-neighbour test also reports the asymptotic normal
approximation alongside, so you can see when the two disagree.

.. note::

   The cross-match test uses a **greedy** matching rather than Rosenbaum's
   optimal non-bipartite matching, so the exact null distribution does not apply
   and the p-value is obtained by permutation. Treat it as an approximation to
   the published test.

Honest intervals
----------------

A binomial interval assumes independent Bernoulli trials. Synthetic points are
not independent: SMOTE places each point on a segment between a minority point
and one of its neighbours, so points sharing a parent lie in the same
neighbourhood and are scored the same way. The effective sample size is closer
to the number of **parents** than the number of synthetic points.

:func:`~oversampleqa.error_rate_interval` offers both:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - method
     - assumes
     - too narrow?
   * - ``wilson``
     - points are independent Bernoulli trials
     - almost always
   * - ``block_bootstrap``
     - points sharing a parent move together
     - rarely

The difference is not academic. On 200 synthetic points from 40 parents, where
children of a parent share their outcome:

.. code-block:: text

   rate            0.2500   (200 points from 40 parents)
   wilson          [0.1951, 0.3143]  width 0.1193
   block bootstrap [0.1250, 0.4000]  width 0.2750
   block / wilson  2.31x wider

The naive interval is **2.3x too narrow** here, which is the difference between
declaring a result and not.

Comparing methods across datasets
---------------------------------

Running a test per dataset and counting wins does not control error across the
family and ignores that datasets are blocks. The standard protocol is Friedman
followed by a Nemenyi post-hoc (Demšar 2006), in
:func:`~oversampleqa.inference.friedman_nemenyi`, with a critical-difference
diagram in :func:`~oversampleqa.plotting.plot_critical_difference`.

Pairwise p-values within a benchmark are corrected by Holm (family-wise error
rate, the default) or Benjamini-Hochberg (false discovery rate). With 8
oversamplers there are 28 comparisons, and uncorrected p-values manufacture
significance at that many looks.

What these p-values do not tell you
-----------------------------------

.. warning::

   **Failing to reject is not proof of equality.** A high p-value from any of
   these tests is weak evidence of similarity, never a demonstration of it.

   The power of every nearest-neighbour test **collapses as dimension grows**.
   On high-dimensional data a large p-value may reflect a lack of power rather
   than genuine similarity, and no amount of care in the test can recover it.
   Every result therefore reports ``n_synthetic`` and ``n_real``: read the
   p-value next to them, and treat a non-rejection on small samples in high
   dimension as uninformative rather than reassuring.

   **A significant Friedman test does not say which methods differ.** It says
   only that they are not all equivalent. The Nemenyi critical difference
   identifies pairs, and it is wide unless there are many datasets -- with 5
   methods over 5 datasets, mean ranks must differ by roughly 2.7 out of a
   possible 4. Failing to separate methods usually means too few datasets, not
   that the methods are equivalent.

   **The null calibration is conditional on this dataset.** It says how the
   observed rate compares to what an ideal generator achieves *here*, with
   *these* settings. It does not transfer to another dataset.

   **Corrected p-values are not effect sizes.** A comparison can be significant
   and negligible at once. Read the paired differences and their intervals,
   which are reported alongside.

References
----------

See :doc:`bibliography` for the full list: Schilling (1986), Henze (1988),
Friedman & Rafsky (1979), Rosenbaum (2005), Demšar (2006), and
Benjamini & Hochberg (1995).
