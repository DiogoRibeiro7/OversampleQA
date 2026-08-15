Fidelity and Diversity
======================

The hidden-majority error rate is one scalar, and it conflates two failures that
call for **opposite** remedies.

**Low fidelity** — synthetic points land in implausible regions: between
clusters, inside majority territory, off the data manifold. The fix is a more
conservative generator.

**Low diversity** — synthetic points are perfectly realistic but merely copy the
training minority, adding no information. The fix is a *less* conservative one.

A single number cannot tell you which you have, and pushing it in the wrong
direction makes things worse.

The case that makes the argument
--------------------------------

``RandomOverSampler`` duplicates real minority points. It is therefore maximally
realistic and completely uninformative. On a representative dataset:

.. code-block:: text

   SMOTE              err=0.204  precision=0.94 coverage=1.00  memorisation=0.401
   RandomOverSampler  err=0.242  precision=0.95 coverage=1.00  memorisation=0.000

The error rates are close. Precision and coverage are close. Only the
memorisation ratio separates them, and it does so unambiguously: 0.000 means
every synthetic point sits exactly on a training point.

What each metric measures
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Metric
     - Question
     - Axis
   * - ``precision``
     - Are synthetic points inside the real manifold?
     - fidelity
   * - ``recall``
     - Are real points inside the synthetic manifold?
     - diversity
   * - ``density``
     - *How many* real neighbourhoods contain each synthetic point?
     - fidelity, unsaturated
   * - ``coverage``
     - What fraction of real points have a synthetic neighbour?
     - diversity, robust
   * - ``memorisation_distance_ratio``
     - Is the generator closer to its training data than real points are to each other?
     - novelty
   * - ``boundary_violation``
     - Do synthetic points land among majority neighbours?
     - fidelity

**Density and coverage are the more reliable pair** (Naeem et al. 2020).
Precision saturates at 1 and a single real outlier with an enormous k-NN sphere
can certify every synthetic point at once; density counts spheres instead, so it
keeps resolving past the point where precision stops.

Reading the combinations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Pattern
     - Reading
   * - high precision, low recall
     - **Conservative.** Realistic but not diverse — the generator is hugging a
       sub-region of the minority.
   * - low precision, high recall
     - **Overreaching.** Diverse but implausible — it is spreading into space
       the real data does not occupy.
   * - memorisation ratio ≈ 0
     - **Copying.** Synthetic points are training points. The error rate cannot
       say anything about synthesis quality here.
   * - high boundary violation
     - **Generating into majority territory** — the specific failure this
       package exists to detect.
   * - good geometry, no utility gain
     - The classifier did not need the help. Geometry is necessary, not
       sufficient.

Choosing ``k``
--------------

.. warning::

   These metrics are **sensitive to** ``k``. Reporting one value and hiding that
   sensitivity would repeat the error rate's original sin — a number with no
   indication of what it depends on.

   Use :func:`~oversampleqa.fidelity.sweep_k`, which recomputes across several
   ``k`` and returns a frame. A metric that moves sharply with ``k`` is telling
   you the manifold estimate is unstable, not that the generator changed.

.. warning::

   **All manifold metrics degrade in high dimension.** Distances concentrate, so
   the k-NN spheres stop distinguishing anything, and they become unreliable
   well before they become obviously wrong. A warning is emitted when the
   feature count exceeds one tenth of the real sample size. Reduce dimension
   first, or read the numbers as indicative only.

Downstream utility
------------------

Geometry can look fine while the classifier gains nothing.
:func:`~oversampleqa.fidelity.downstream_utility` trains with and without
oversampling under identical cross-validation and reports the paired difference
with a bootstrap interval.

.. danger::

   **Oversampling must happen inside each training fold.** Resampling before
   splitting leaks synthetic points derived from validation-fold minority
   samples into training, and inflates the score — a SMOTE point interpolated
   from a validation point is nearly that point, so the model is scored on data
   it effectively trained on.

   This function uses :class:`imblearn.pipeline.Pipeline`, which resamples per
   fold. ``sklearn.pipeline.Pipeline`` does not handle samplers this way.
   ``tests/test_fidelity.py`` builds the leaky version deliberately and asserts
   it scores higher, so the correct construction is pinned by evidence.

Scoring defaults to ``average_precision`` (PR-AUC), never accuracy. On
imbalanced data accuracy is dominated by the majority class: predicting the
majority for everything scores well while being useless.

Worked example
--------------

.. code-block:: python

   from oversampleqa.fidelity import fidelity_report
   from imblearn.over_sampling import RandomOverSampler

   report = fidelity_report(X, y, minority_label=1,
                            oversampler=RandomOverSampler(random_state=0))

   print(report.error_rate)
   for note in report.interpret():
       print("-", note)

.. code-block:: text

   0.242
   - Memorisation: synthetic points sit on top of training points.
     The error rate cannot say anything about synthesis quality here.

References
----------

See :doc:`bibliography`: Sajjadi et al. (2018), Kynkäänniemi et al. (2019),
Naeem et al. (2020), Alaa et al. (2022), van Breugel et al. (2023).
