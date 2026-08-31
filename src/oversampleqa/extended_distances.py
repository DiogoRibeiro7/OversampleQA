"""
Extended distance metrics for oversampleqa package.

This module adds comprehensive distance metrics beyond the basic ones,
with proper validation and testing strategies.
"""

import numpy as np
from numpy.typing import NDArray


def minkowski_distance(
    x1: NDArray[np.floating], x2: NDArray[np.floating], p: float = 3.0
) -> float:
    """Compute Minkowski distance between two vectors.

    Parameters
    ----------
    x1, x2 : np.ndarray
        Input vectors of same shape
    p : float, default=3.0
        Order of the norm (``p >= 1``). ``np.inf`` is accepted and gives the
        Chebyshev distance, which is the limit as p grows.

    Returns
    -------
    float
        Minkowski distance

    Raises
    ------
    ValueError
        If the shapes differ, or ``p < 1``.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    if p < 1:
        raise ValueError("p must be >= 1")

    diff = np.abs(x1 - x2)
    largest = float(diff.max()) if diff.size else 0.0
    if largest == 0.0:
        return 0.0

    if np.isinf(p):
        # The p -> infinity limit is the Chebyshev distance. The general
        # formula cannot produce it: every |d| > 1 raised to inf is inf, the
        # sum is inf, and inf ** (1 / inf) is inf ** 0, which is 1.0. So
        # `p=inf` returned 1.0 for any input at all, regardless of the data.
        return largest

    # The largest term is factored out before exponentiating. Computed
    # directly, `diff ** p` overflows at moderate p -- p=1000 gives inf, as it
    # does in scipy -- when the answer is simply the largest term. Scaling
    # every term into [0, 1] first makes the sum well behaved, and the result
    # is identical for the p values that never overflowed.
    scaled = diff / largest
    return float(largest * np.sum(scaled**p) ** (1 / p))


def chebyshev_distance(
    x1: NDArray[np.floating], x2: NDArray[np.floating]
) -> float:
    """Compute Chebyshev (L-infinity) distance between two vectors.

    This is the maximum absolute difference across all dimensions.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    return float(np.max(np.abs(x1 - x2)))


def mahalanobis_distance(
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    cov_inv: NDArray[np.floating] | None = None,
) -> float:
    """Compute Mahalanobis distance between two vectors.

    Parameters
    ----------
    x1, x2 : np.ndarray
        Input vectors
    cov_inv : np.ndarray
        Inverse covariance matrix. Required, and must be symmetric positive
        semi-definite -- that is what makes the result a distance. It is not
        validated as such on every call, because an eigenvalue check per pair
        would cost more than the distance itself; a negative squared distance
        is caught instead, which is how a non-PSD matrix usually shows up.

        Note the residual case: a matrix that is not PSD can still return 0
        for two distinct points, and no per-pair check can detect that. If you
        build ``cov_inv`` by any route other than inverting a sample
        covariance, check it once with ``np.linalg.eigvalsh``.

    Returns
    -------
    float
        Mahalanobis distance

    Raises
    ------
    ValueError
        If ``cov_inv`` is omitted, or if it yields a negative squared distance.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    diff = x1 - x2

    if cov_inv is None:
        # Silently returning Euclidean relabels one metric as another. It was
        # doing exactly that in the advanced benchmark's default metric list,
        # where every "mahalanobis" row was a byte-identical copy of the
        # "euclidean" row -- double-weighting euclidean in the rankings and
        # making the pairwise correction treat one comparison as two.
        raise ValueError(
            "mahalanobis requires cov_inv: Mahalanobis distance with an "
            "identity covariance is Euclidean distance, so defaulting to it "
            "would report one metric under another's name. Estimate the "
            "inverse from the reference data, e.g. "
            "cov_inv=np.linalg.pinv(np.cov(X, rowvar=False)), and pass it "
            "through metric_kwargs."
        )

    squared = float(np.dot(diff, np.dot(cov_inv, diff)))
    if squared < 0.0:
        # A genuine inverse covariance is positive semi-definite, so this
        # quadratic form cannot be negative. When it is, np.sqrt returns nan
        # with nothing but a bare "invalid value encountered in sqrt" to say
        # why -- a warning users routinely filter, pointing at a line inside
        # this library rather than at the matrix they passed.
        #
        # A near-singular inverse can produce a tiny negative through rounding
        # alone, which is noise rather than an error, so that is clamped.
        # Anything larger means cov_inv is not an inverse covariance.
        tolerance = 1e-12 * max(1.0, float(np.dot(diff, diff)))
        if squared < -tolerance:
            raise ValueError(
                "mahalanobis requires a positive semi-definite cov_inv: the "
                f"squared distance came out negative ({squared:.6g}), which "
                "np.sqrt reports as nan. Passing the covariance itself rather "
                "than its inverse, or inverting a covariance estimated from "
                "fewer samples than features, both produce a matrix that is "
                "not. Estimate it with "
                "cov_inv=np.linalg.pinv(np.cov(X, rowvar=False))."
            )
        squared = 0.0

    return float(np.sqrt(squared))


def canberra_distance(x1: NDArray[np.floating], x2: NDArray[np.floating]) -> float:
    """Compute Canberra distance between two vectors.

    Canberra distance is a weighted version of Manhattan distance,
    useful when dealing with features of different scales.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    numerator = np.abs(x1 - x2)
    denominator = np.abs(x1) + np.abs(x2)

    # Handle division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator == 0, 0.0, numerator / denominator)

    return float(np.sum(ratio))


def hamming_distance(x1: NDArray[np.generic], x2: NDArray[np.generic]) -> float:
    """Compute Hamming distance between two vectors.

    Counts the number of positions where elements differ.
    Useful for categorical or binary features.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    return float(np.sum(x1 != x2))


def _is_binary(values: NDArray[np.generic]) -> bool:
    """Whether an array holds only booleans or 0/1.

    Jaccard is defined on sets, and the implementations reach that by casting
    to bool. Any non-zero becomes True, so two vectors sharing a zero pattern
    compare as identical however different their values are.
    """
    if values.dtype == bool:
        return True
    return bool(np.all((values == 0) | (values == 1)))


def jaccard_distance(x1: NDArray[np.generic], x2: NDArray[np.generic]) -> float:
    """Compute Jaccard distance between two binary vectors.

    Jaccard distance = 1 - Jaccard similarity
    where Jaccard similarity = :math:`|intersection| / |union|`
    """
    x1_raw = np.asarray(x1)
    x2_raw = np.asarray(x2)
    x1_bool = x1_raw.astype(bool)
    x2_bool = x2_raw.astype(bool)
    if x1_bool.shape != x2_bool.shape:
        raise ValueError("Input vectors must have the same shape")
    if not _is_binary(x1_raw) or not _is_binary(x2_raw):
        # Without this, casting to bool made every non-zero identical:
        # d([1.0, 3.0], [7.0, 0.2]) was 0.0, two distinct points at distance
        # zero. `boolean` is the domain this metric declares in METRIC_DOMAINS.
        raise ValueError(
            "Jaccard distance requires binary inputs: values must be 0 or 1, "
            "or a boolean array. Casting other values to bool treats every "
            "non-zero as identical, so distinct points come out at distance "
            "zero. Binarise the features first, choosing the threshold "
            "deliberately."
        )

    intersection = np.sum(x1_bool & x2_bool)
    union = np.sum(x1_bool | x2_bool)

    if union == 0:
        return 0.0  # Both vectors are all zeros

    similarity: float = intersection / union
    return 1.0 - similarity


def braycurtis_distance(
    x1: NDArray[np.floating], x2: NDArray[np.floating]
) -> float:
    """Compute Bray-Curtis distance between two vectors.

    Often used in ecology and environmental science.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    if np.any(x1 < 0) or np.any(x2 < 0):
        raise ValueError("Bray-Curtis distance requires non-negative inputs")

    numerator = np.sum(np.abs(x1 - x2))
    denominator = np.sum(np.abs(x1 + x2))

    if denominator == 0:
        # Sound only because the inputs are non-negative: the sum of absolute
        # values is then zero exactly when both vectors are all-zero, and the
        # distance between them really is zero. Allow a negative through and
        # the terms cancel instead -- d([-1, 0], [1, 0]) came out as 0.0, two
        # distinct points at distance zero, which is the identity-of-
        # indiscernibles violation check_metric_axioms exists to catch.
        return 0.0

    ratio: float = numerator / denominator
    return ratio


def correlation_distance(
    x1: NDArray[np.floating], x2: NDArray[np.floating]
) -> float:
    """Compute correlation distance between two vectors.

    Correlation distance = 1 - Pearson correlation coefficient
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    if len(x1) < 2:
        # Correlation needs at least two components to have any variance.
        raise ValueError(
            "correlation distance is undefined for vectors of length < 2: "
            "there is no variance to correlate."
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        # A constant vector makes corrcoef divide by a zero standard deviation.
        # That is the case handled immediately below, so the warning is noise.
        corr_coef = np.corrcoef(x1, x2)[0, 1]

    if np.isnan(corr_coef):
        # A constant vector has zero variance, so the correlation is undefined.
        # This returned 0.0 -- "perfectly correlated" -- which made a constant
        # vector distance-zero from every other vector. METRIC_DOMAINS has
        # documented the case as undefined all along; the code disagreed.
        raise ValueError(
            "correlation distance is undefined when either vector is constant: "
            "zero variance leaves nothing to correlate. Drop constant features "
            "or rows, or use a metric defined on them such as 'euclidean'."
        )

    coefficient: float = corr_coef
    return float(np.clip(1.0 - coefficient, 0.0, 2.0))


def hellinger_distance(x1: NDArray[np.floating], x2: NDArray[np.floating]) -> float:
    """Compute the Hellinger distance between two probability vectors.

    The input vectors are normalized to sum to ``1`` and must contain
    non-negative values. The distance is bounded between ``0`` and ``1``.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    if np.any(x1 < 0) or np.any(x2 < 0):
        raise ValueError("Hellinger distance requires non-negative inputs")

    p = x1 / x1.sum() if x1.sum() != 0 else np.zeros_like(x1)
    q = x2 / x2.sum() if x2.sum() != 0 else np.zeros_like(x2)

    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))


def jensen_shannon_distance(
    x1: NDArray[np.floating], x2: NDArray[np.floating]
) -> float:
    """Compute the Jensen-Shannon distance between two probability vectors.

    The Jensen-Shannon distance is the square root of the
    Jensen-Shannon divergence and is symmetric and bounded between ``0`` and
    ``sqrt(log(2))`` when using natural logarithms.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")
    if np.any(x1 < 0) or np.any(x2 < 0):
        raise ValueError("Jensen-Shannon distance requires non-negative inputs")

    p = x1 / x1.sum() if x1.sum() != 0 else np.zeros_like(x1)
    q = x2 / x2.sum() if x2.sum() != 0 else np.zeros_like(x2)
    m = 0.5 * (p + q)

    def _kl_div(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a == 0, 1.0, a / b)
            log_term = np.log(ratio)
        return float(np.sum(np.where(a == 0, 0.0, a * log_term)))

    js_div = 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)
    return float(np.sqrt(js_div))


def energy_distance(x1: NDArray[np.floating], x2: NDArray[np.floating]) -> float:
    """Compute energy distance between two 1D or 2D vectors.

    The implementation follows the definition from energy statistics.

    .. warning::

       This is a **sample-based** metric, not a point metric. A 1-D input is
       reshaped to ``(len(x), 1)`` and treated as a *set of scalar
       observations*, not as one point in ``len(x)``-dimensional feature
       space. It therefore does not measure the same kind of quantity as
       ``euclidean`` or ``hassanat``, even though it is reachable through the
       same registry. Use it to compare two samples, not two points.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1.reshape(len(x1), -1)
    x2 = x2.reshape(len(x2), -1)

    diff_cross = np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=-1)
    term_a = diff_cross.mean()

    term_b: float
    if len(x1) > 1:
        diff_x1 = np.linalg.norm(x1[:, None, :] - x1[None, :, :], axis=-1)
        term_b = float(diff_x1[np.triu_indices(len(x1), 1)].mean())
    else:
        term_b = 0.0

    term_c: float
    if len(x2) > 1:
        diff_x2 = np.linalg.norm(x2[:, None, :] - x2[None, :, :], axis=-1)
        term_c = float(diff_x2[np.triu_indices(len(x2), 1)].mean())
    else:
        term_c = 0.0

    return float(2.0 * term_a - term_b - term_c)


def wasserstein_1d_distance(
    x1: NDArray[np.floating], x2: NDArray[np.floating]
) -> float:
    """Compute the 1D Wasserstein distance between two empirical distributions.

    .. warning::

       This is a **sample-based** metric, not a point metric. The input vector
       is flattened and treated as a *set of scalar observations* drawn from a
       distribution, not as one point in feature space. It therefore does not
       measure the same kind of quantity as ``euclidean`` or ``hassanat``,
       even though it is reachable through the same registry. Use it to
       compare two samples, not two points.

    Args:
        x1: Samples from distribution 1.
        x2: Samples from distribution 2.

    Returns:
        Wasserstein distance.
    """
    x1 = np.sort(np.asarray(x1, dtype=float).ravel())
    x2 = np.sort(np.asarray(x2, dtype=float).ravel())

    n = len(x1)
    m = len(x2)
    if n == 0 or m == 0:
        return 0.0

    i = j = 0
    cdf1 = cdf2 = 0.0
    last_x = min(x1[0], x2[0])
    dist = 0.0

    # W1 = integral of |F1(t) - F2(t)| dt. Between consecutive sorted points the
    # two CDFs are flat, so each interval [last_x, x) contributes
    # |F1 - F2| * (x - last_x) using the CDF values that hold *across* it --
    # that is, the values before the jump at x.
    #
    # The previous version advanced the CDF first and then added, crediting each
    # interval with the value from after its right-hand jump. On [0, 1] vs
    # [0, 3] that returned 0.5 where the true W1 is 1.0.
    while i < n and j < m:
        x = x1[i] if x1[i] <= x2[j] else x2[j]
        dist += abs(cdf1 - cdf2) * (x - last_x)
        last_x = x
        # Advance every tie at this position before moving on.
        while i < n and x1[i] == x:
            i += 1
            cdf1 = i / n
        while j < m and x2[j] == x:
            j += 1
            cdf2 = j / m

    while i < n:
        x = x1[i]
        dist += abs(cdf1 - cdf2) * (x - last_x)
        last_x = x
        i += 1
        cdf1 = i / n

    while j < m:
        x = x2[j]
        dist += abs(cdf1 - cdf2) * (x - last_x)
        last_x = x
        j += 1
        cdf2 = j / m

    return float(dist)
