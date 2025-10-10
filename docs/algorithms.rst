Algorithms
==========

Probability Distance Metrics
----------------------------

Hellinger Distance
~~~~~~~~~~~~~~~~~~
The Hellinger distance between two discrete probability vectors :math:`p` and :math:`q` is defined as:

.. math::

   H(p, q) = \frac{1}{\sqrt{2}} \left\| \sqrt{p} - \sqrt{q} \right\|_2

This metric ranges between 0 and 1 and measures the similarity between probability distributions.

Jensen--Shannon Distance
~~~~~~~~~~~~~~~~~~~~~~~~
The Jensen--Shannon distance derives from the Jensen--Shannon divergence:

.. math::
   :nowrap:

   \begin{align}
   JSD(p, q) &= \frac{1}{2} KL(p \parallel m) + \frac{1}{2} KL(q \parallel m)\\
   m &= \frac{1}{2}(p + q)
   \end{align}

.. math::

   JSDist(p, q) = \sqrt{JSD(p, q)}

This symmetric measure is always finite and lies between 0 and 1.

Hassanat Distance
~~~~~~~~~~~~~~~~~
For real-valued vectors :math:`x` and :math:`y`, the Hassanat distance computes per-dimension contributions:

.. math::

   d_i = \begin{cases}
       0 & \text{if } \max(x_i, y_i) = 0\\
       1 - \frac{\min(x_i, y_i)}{\max(x_i, y_i)} & \text{otherwise}
   \end{cases}

The overall distance is the sum :math:`\sum_i d_i`, providing a scale-invariant similarity measure.

See :doc:`bibliography` for full references.
