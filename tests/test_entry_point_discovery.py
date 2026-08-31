"""Tests for entry-point plugin discovery.

Discovery is exercised against synthetic entry points rather than an installed
distribution, so the suite does not depend on ``examples/plugins`` having been
pip-installed first.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa.exceptions import PluginError
from oversampleqa.plugin_system import (
    METRIC_ENTRY_POINT_GROUP,
    VALIDATOR_ENTRY_POINT_GROUP,
    PluginManager,
)


class GoodMetric:
    def __call__(self, x1, x2, **kwargs):
        return float(np.abs(np.asarray(x1) - np.asarray(x2)).sum())


class BadMetric:
    """Violates the identity of indiscernibles, like the old hassanat did."""

    def __call__(self, x1, x2, **kwargs):
        return float(abs(np.abs(np.asarray(x1)).sum() - np.abs(np.asarray(x2)).sum()))


class GoodValidator:
    def validate(self, X, y, minority_label, oversampler, **kwargs):
        return 0.0


class NotAValidator:
    pass


class _FakeEntryPoint:
    """Stub with the only two members discovery touches: ``name`` and ``load``.

    Deliberately not a subclass of :class:`importlib.metadata.EntryPoint`, which
    stopped being a NamedTuple and cannot be constructed by subclasses portably.
    Standing in for the real thing here would test the stub, not the contract.
    """

    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


@pytest.fixture
def patch_entry_points(monkeypatch):
    def apply(metrics=(), validators=()):
        def fake(group):
            if group == METRIC_ENTRY_POINT_GROUP:
                return list(metrics)
            if group == VALIDATOR_ENTRY_POINT_GROUP:
                return list(validators)
            return []

        monkeypatch.setattr(
            "oversampleqa.plugin_system.metadata.entry_points",
            lambda *, group: fake(group),
        )

    return apply


def test_discovers_a_metric(patch_entry_points):
    patch_entry_points(metrics=[_FakeEntryPoint("good", GoodMetric)])
    manager = PluginManager()
    assert manager.discover_entry_points() == ["good"]
    assert manager.get_metric("good") is GoodMetric


def test_discovers_a_validator(patch_entry_points):
    patch_entry_points(validators=[_FakeEntryPoint("val", GoodValidator)])
    manager = PluginManager()
    assert manager.discover_entry_points() == ["val"]


def test_discovers_both_groups(patch_entry_points):
    patch_entry_points(
        metrics=[_FakeEntryPoint("good", GoodMetric)],
        validators=[_FakeEntryPoint("val", GoodValidator)],
    )
    assert set(PluginManager().discover_entry_points()) == {"good", "val"}


def test_no_entry_points_is_not_an_error(patch_entry_points):
    patch_entry_points()
    assert PluginManager().discover_entry_points() == []


def test_a_metric_failing_the_axioms_is_refused(patch_entry_points):
    """The check exists because the built-in hassanat would have failed it."""
    patch_entry_points(metrics=[_FakeEntryPoint("bad", BadMetric)])
    manager = PluginManager()
    with pytest.warns(RuntimeWarning, match="bad"):
        assert manager.discover_entry_points() == []


def test_one_bad_plugin_does_not_block_the_others(patch_entry_points):
    """Otherwise installing a broken plugin silently disables every other one."""
    patch_entry_points(
        metrics=[_FakeEntryPoint("bad", BadMetric), _FakeEntryPoint("good", GoodMetric)]
    )
    manager = PluginManager()
    with pytest.warns(RuntimeWarning):
        registered = manager.discover_entry_points()
    assert registered == ["good"]


def test_failure_warning_names_the_plugin_and_the_reason(patch_entry_points):
    patch_entry_points(metrics=[_FakeEntryPoint("bad", BadMetric)])
    with pytest.warns(RuntimeWarning) as caught:
        PluginManager().discover_entry_points()
    message = str(caught[0].message)
    assert "bad" in message
    assert "axiom" in message.lower()


def test_an_import_failure_is_reported_not_swallowed(patch_entry_points):
    """A plugin that fails to load must not look like one never installed."""
    patch_entry_points(
        metrics=[_FakeEntryPoint("broken", ImportError("no module named nope"))]
    )
    with pytest.warns(RuntimeWarning, match="no module named nope"):
        assert PluginManager().discover_entry_points() == []


def test_strict_raises_instead_of_warning(patch_entry_points):
    patch_entry_points(metrics=[_FakeEntryPoint("bad", BadMetric)])
    with pytest.raises(PluginError, match="bad"):
        PluginManager().discover_entry_points(strict=True)


def test_a_validator_without_validate_is_refused(patch_entry_points):
    patch_entry_points(validators=[_FakeEntryPoint("nope", NotAValidator)])
    with pytest.warns(RuntimeWarning, match="validate"):
        assert PluginManager().discover_entry_points() == []


def test_collision_with_a_builtin_metric_is_refused(patch_entry_points):
    patch_entry_points(metrics=[_FakeEntryPoint("euclidean", GoodMetric)])
    with pytest.warns(RuntimeWarning, match="built-in"):
        assert PluginManager().discover_entry_points() == []


def test_validator_collision_is_refused():
    """Load order is not something either plugin author controls."""
    manager = PluginManager()
    manager.register_validator("dup", GoodValidator)
    with pytest.raises(PluginError, match="already registered"):
        manager.register_validator("dup", GoodValidator)


def test_unregister_validator_round_trip():
    manager = PluginManager()
    manager.register_validator("tmp", GoodValidator)
    manager.unregister_validator("tmp")
    with pytest.raises(PluginError, match="no validator plugin"):
        manager.unregister_validator("tmp")


# --- a plugin whose metric is not defined on all of R^n ---


class NonNegativeMetric:
    """Correct, but only on non-negative input -- like the built-in hellinger.

    Declares its domain the only way an entry point allows: an attribute on the
    object it exports. The entry point carries a callable and nothing else.
    """

    domain = "non_negative"

    def __call__(self, x1, x2, **kwargs):
        a, b = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("requires non-negative inputs")
        return float(np.abs(np.sqrt(a) - np.sqrt(b)).sum())


class WrongDomainMetric(NonNegativeMetric):
    domain = "non-negative"  # hyphen: a plausible typo


def test_a_restricted_domain_metric_can_be_discovered(patch_entry_points):
    """It used to be rejected as violating the axioms it satisfies.

    Discovery registered every metric without a domain, so the axiom check ran
    on real-valued input. A metric defined only on non-negative input raised
    there, and the failure was reported as "does not satisfy the distance
    axioms" -- which was true of the check, not of the metric.
    """
    patch_entry_points(metrics=[_FakeEntryPoint("nonneg", NonNegativeMetric)])
    manager = PluginManager()

    registered = manager.discover_entry_points(strict=True)

    assert "nonneg" in registered


def test_an_undeclared_domain_still_defaults_to_real(patch_entry_points):
    """The common case must not need an attribute nobody knew to set."""
    patch_entry_points(metrics=[_FakeEntryPoint("good", GoodMetric)])
    manager = PluginManager()

    assert "good" in manager.discover_entry_points(strict=True)


def test_a_misspelled_domain_is_refused(patch_entry_points):
    """Silently treating it as `real` would check the metric where it is not
    defined, which is the failure this whole mechanism exists to avoid."""
    patch_entry_points(metrics=[_FakeEntryPoint("typo", WrongDomainMetric)])
    manager = PluginManager()

    with pytest.raises(PluginError, match="declares domain"):
        manager.discover_entry_points(strict=True)


def test_a_genuinely_broken_metric_is_still_refused(patch_entry_points):
    """Declaring a domain must not become a way to skip the axiom check."""
    patch_entry_points(metrics=[_FakeEntryPoint("bad", BadMetric)])
    manager = PluginManager()

    with pytest.raises(PluginError, match="axioms"):
        manager.discover_entry_points(strict=True)
