"""Tests for the shared RNG normalisation helpers.

Seeding lives in one module so the validators cannot drift apart on it; these
tests pin each accepted seed form and the independence of spawned streams.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa._rng import (
    as_generator,
    integer_seed,
    spawn_generators,
)


def test_as_generator_accepts_int():
    a = as_generator(7).integers(0, 10_000, size=5)
    b = as_generator(7).integers(0, 10_000, size=5)
    assert np.array_equal(a, b)


def test_as_generator_passes_through_a_generator():
    """A supplied Generator is returned unchanged, so one stream can be threaded."""
    gen = np.random.default_rng(3)
    assert as_generator(gen) is gen


def test_as_generator_accepts_seed_sequence():
    seq = np.random.SeedSequence(12345)
    values = as_generator(seq).integers(0, 10_000, size=5)
    expected = np.random.default_rng(np.random.SeedSequence(12345)).integers(
        0, 10_000, size=5
    )
    assert np.array_equal(values, expected)


def test_as_generator_none_is_not_reproducible():
    a = as_generator(None).integers(0, 2**62, size=4)
    b = as_generator(None).integers(0, 2**62, size=4)
    assert not np.array_equal(a, b)


def test_spawn_generators_are_independent():
    """Children must differ from each other and be reproducible from the parent."""
    first = [g.integers(0, 2**62) for g in spawn_generators(42, 8)]
    second = [g.integers(0, 2**62) for g in spawn_generators(42, 8)]
    assert first == second
    assert len(set(first)) == 8


def test_spawn_generators_beats_seed_plus_i():
    """SeedSequence children are independent; seed + i streams are not.

    This is why repeats derive from spawning: consecutive integer seeds share
    structure, which would understate the dispersion across repeats.
    """
    spawned = np.array(
        [g.integers(0, 2**32, size=50) for g in spawn_generators(1000, 4)]
    )
    assert len({tuple(row) for row in spawned}) == 4


def test_spawn_generators_accepts_seed_sequence_parent():
    seq = np.random.SeedSequence(99)
    values = [g.integers(0, 2**62) for g in spawn_generators(seq, 3)]
    assert len(set(values)) == 3


def test_spawn_generators_from_a_generator_does_not_replay():
    """Spawning twice from one live Generator must not repeat the children."""
    gen = np.random.default_rng(5)
    first = [g.integers(0, 2**62) for g in spawn_generators(gen, 3)]
    second = [g.integers(0, 2**62) for g in spawn_generators(gen, 3)]
    assert first != second


def test_spawn_generators_rejects_zero():
    with pytest.raises(ValueError, match="at least 1"):
        spawn_generators(42, 0)


def test_integer_seed_is_in_sklearn_range():
    """scikit-learn estimators accept int seeds below 2**31."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        seed = integer_seed(rng)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31
