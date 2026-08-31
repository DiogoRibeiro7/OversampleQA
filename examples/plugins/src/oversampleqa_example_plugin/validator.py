"""A custom validator, as a plugin would define one."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class MedianRatioValidator:
    """Scores a sampler by how far its synthetic points sit from real ones.

    The built-in estimand counts synthetic points nearer to held-out majority
    than to real minority. This one reports a different quantity: the median
    distance from each synthetic point to its nearest real minority point,
    divided by the median nearest-neighbour distance *within* the real minority.

    The denominator is the natural spacing of the real data, so the result is
    free of feature scale. A value near zero means the sampler is duplicating
    its training data; a value near one means it is placing points at roughly
    the spacing real points already have.

    It is included to show a validator that returns a number on a different
    scale from the built-in error rate, because the plugin protocol only
    promises a float -- it does not promise that two validators are comparable.
    """

    name = "median_ratio"

    def validate(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        minority_label: int,
        oversampler: Any,
        **kwargs: Any,
    ) -> float:
        """Run the sampler and score it.

        Args:
            X: Feature matrix.
            y: Labels.
            minority_label: Label of the minority class.
            oversampler: Object with ``fit_resample``.
            **kwargs: Ignored; present to satisfy the validator protocol.

        Returns:
            The scale-free median distance ratio, or ``nan`` when the sampler
            produced no synthetic points or the minority has fewer than two
            members. Returning ``0.0`` in those cases would be indistinguishable
            from a sampler that copied its input exactly.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        minority = X[y == minority_label]
        if len(minority) < 2:
            return float("nan")

        X_resampled, _ = oversampler.fit_resample(X, y)
        X_resampled = np.asarray(X_resampled, dtype=float)
        synthetic = X_resampled[len(X) :]
        if len(synthetic) == 0:
            return float("nan")

        to_real = np.linalg.norm(synthetic[:, None, :] - minority[None, :, :], axis=2)
        median_to_real = float(np.median(to_real.min(axis=1)))

        within = np.linalg.norm(minority[:, None, :] - minority[None, :, :], axis=2)
        np.fill_diagonal(within, np.inf)
        median_within = float(np.median(within.min(axis=1)))

        if median_within == 0.0:
            # Duplicate real points make the scale undefined rather than zero.
            return float("nan")
        return median_to_real / median_within
