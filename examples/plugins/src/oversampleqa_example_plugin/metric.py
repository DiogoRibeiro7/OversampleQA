"""A custom distance metric, as a plugin would define one."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class LorentzianDistance:
    """Lorentzian distance: ``sum(log(1 + |x - y|))``.

    An L1 distance with each coordinate difference passed through ``log1p``, so
    large differences are damped. Two points differing by 1000 in one feature
    are scored roughly twice as far apart as two differing by 30, rather than
    thirty times. That makes it markedly less sensitive to outliers than
    Manhattan distance, which is the usual reason to reach for it.

    It is a genuine metric, not merely a similarity score. Non-negativity,
    symmetry and identity are immediate. The triangle inequality holds because
    ``log`` is applied per coordinate to a quantity that already satisfies it::

        log(1 + |a - c|) <= log(1 + |a - b| + |b - c|)
                         <= log(1 + |a - b|) + log(1 + |b - c|)

    The second step is ``log(1 + s + t) <= log((1 + s)(1 + t))``, which holds
    because the product expands to ``1 + s + t + st`` and ``st >= 0``.

    This matters here. OversampleQA runs an axiom smoke check at registration
    and refuses a metric that fails it, because its own built-in Hassanat
    implementation shipped for the project's entire history scoring ``[-5]`` and
    ``[5]`` as distance zero. Note that the smoke check tests identity,
    symmetry, non-negativity and finiteness -- it does not test the triangle
    inequality, so passing it is necessary rather than sufficient. If you are
    writing a metric, prove that step as above rather than assuming it.
    """

    name = "lorentzian"

    #: The input this metric is defined on, read by entry-point discovery.
    #:
    #: ``"real"`` is the default and could be omitted; it is written out because
    #: this file is meant to be copied, and the knob is easier to find than to
    #: guess. The alternatives are ``"non_negative"``, ``"boolean"`` and
    #: ``"sample"``.
    #:
    #: It decides what input the axiom check uses. Get it wrong and a correct
    #: metric is checked where it is not defined: the host's own ``hellinger``
    #: raises on negative input, so declaring it ``"real"`` would see it
    #: rejected as violating axioms it satisfies. Lorentzian is defined on all
    #: of R^n, so ``"real"`` is right here.
    domain = "real"

    def __call__(
        self,
        x1: NDArray[np.floating],
        x2: NDArray[np.floating],
        **kwargs: Any,
    ) -> float:
        """Return the Lorentzian distance between two points.

        Args:
            x1: First point.
            x2: Second point.
            **kwargs: Ignored; present to satisfy the metric protocol.

        Returns:
            The distance, as a float.
        """
        a = np.asarray(x1, dtype=float)
        b = np.asarray(x2, dtype=float)
        return float(np.log1p(np.abs(a - b)).sum())
