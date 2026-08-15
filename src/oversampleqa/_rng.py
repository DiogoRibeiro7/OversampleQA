"""Random-number normalisation shared by every validator.

Seeding behaviour lives here so it cannot drift between the validators again.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RandomStateLike", "as_generator", "spawn_generators"]


RandomStateLike = int | np.random.Generator | np.random.SeedSequence | None
"""Anything accepted as a seed by the validators."""


def as_generator(random_state: RandomStateLike) -> np.random.Generator:
    """Normalise a seed-like value to a :class:`numpy.random.Generator`.

    Parameters
    ----------
    random_state : int, Generator, SeedSequence or None
        ``None`` draws from fresh entropy, meaning results are not
        reproducible. An ``int`` seeds a new generator. A ``Generator`` is
        returned unchanged, so callers can thread one generator through a whole
        pipeline and keep a single stream.

    Returns
    -------
    numpy.random.Generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.SeedSequence):
        return np.random.default_rng(random_state)
    return np.random.default_rng(random_state)


def spawn_generators(
    random_state: RandomStateLike, n: int
) -> list[np.random.Generator]:
    """Return ``n`` independent generators derived from ``random_state``.

    Uses :class:`numpy.random.SeedSequence` spawning, which guarantees the
    streams are statistically independent.

    Deriving repeat seeds as ``seed + i`` would **not** be safe: consecutive
    integer seeds produce correlated streams, so repeats built that way share
    structure and the resulting dispersion is understated.

    Parameters
    ----------
    random_state : int, Generator, SeedSequence or None
        Parent seed.
    n : int
        Number of child generators.

    Returns
    -------
    list of numpy.random.Generator
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")

    if isinstance(random_state, np.random.SeedSequence):
        parent = random_state
    elif isinstance(random_state, np.random.Generator):
        # Draw a fresh entropy value from the supplied generator so repeated
        # calls on the same generator do not replay the same children.
        parent = np.random.SeedSequence(int(random_state.integers(0, 2**63 - 1)))
    else:
        parent = np.random.SeedSequence(random_state)

    return [np.random.default_rng(child) for child in parent.spawn(n)]


def integer_seed(rng: np.random.Generator) -> int:
    """Draw an integer seed from ``rng``.

    For interoperability with scikit-learn estimators, which accept
    ``int``/``RandomState`` but not ``Generator``.
    """
    return int(rng.integers(0, 2**31 - 1))
