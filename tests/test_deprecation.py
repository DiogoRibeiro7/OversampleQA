"""Tests for the deprecation decorator.

The policy in docs/api_stability.md promises a warning that names the
replacement and the removal version, so those are asserted rather than merely
that some warning appeared.
"""

from __future__ import annotations

import os
import warnings

import pytest

from oversampleqa.deprecation import deprecated


@deprecated(removal_version="0.6.0", replacement="new_function")
def old_function(a: int, b: int = 2) -> int:
    """Original docstring."""
    return a + b


@deprecated(removal_version="0.7.0", replacement="NewClass")
class OldClass:
    """Original class docstring."""

    def __init__(self, value: int = 3) -> None:
        self.value = value

    def double(self) -> int:
        return self.value * 2


def _capture(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, caught


def test_warns_on_call():
    _, caught = _capture(old_function, 1)
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)


def test_message_names_the_removal_version():
    """A deprecation without a deadline is just a permanent warning."""
    _, caught = _capture(old_function, 1)
    assert "0.6.0" in str(caught[0].message)


def test_message_names_the_replacement():
    _, caught = _capture(old_function, 1)
    assert "new_function" in str(caught[0].message)


def test_return_value_is_untouched():
    result, _ = _capture(old_function, 1, b=5)
    assert result == 6


def test_signature_metadata_is_preserved():
    assert old_function.__name__ == "old_function"
    assert "Original docstring." in (old_function.__doc__ or "")


def test_docstring_gains_a_deprecation_note():
    """So it shows up in the rendered docs, not only at runtime."""
    assert ".. deprecated:: 0.6.0" in (old_function.__doc__ or "")


def test_stacklevel_points_at_the_caller():
    """Python's default filters key on the reported location.

    A warning that reports itself as originating inside oversampleqa is hidden
    from precisely the people who need to act on it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_function(1)
    # Compared case-insensitively on Windows, where the drive letter reaches
    # the warning as "c:" but ``__file__`` as "C:" (or the reverse, depending
    # on how the module was imported). The paths denote the same file; only
    # their spelling differs, so a plain == failed for every Windows
    # contributor while saying nothing about the stacklevel this asserts.
    # ``normcase`` is a no-op on POSIX, so the check stays exact there.
    assert os.path.normcase(caught[0].filename) == os.path.normcase(__file__)


def test_category_is_configurable():
    @deprecated(removal_version="0.9.0", category=FutureWarning)
    def changes_results() -> int:
        return 1

    _, caught = _capture(changes_results)
    assert issubclass(caught[0].category, FutureWarning)


def test_reason_is_appended():
    @deprecated(
        removal_version="0.9.0",
        reason="The estimand changed; there is no direct substitute.",
    )
    def gone() -> None:
        return None

    _, caught = _capture(gone)
    assert "estimand changed" in str(caught[0].message)


def test_message_without_a_replacement_omits_the_clause():
    @deprecated(removal_version="0.9.0")
    def gone() -> None:
        return None

    _, caught = _capture(gone)
    assert "Use " not in str(caught[0].message)


def test_class_warns_on_instantiation():
    _, caught = _capture(OldClass)
    assert len(caught) == 1
    assert "0.7.0" in str(caught[0].message)


def test_decorated_class_is_still_a_class():
    """Wrapping the class in a function would break all of this."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instance = OldClass(5)
    assert isinstance(instance, OldClass)
    assert instance.value == 5
    assert instance.double() == 10


def test_decorated_class_can_still_be_subclassed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        class Child(OldClass):
            pass

        child = Child(4)
    assert isinstance(child, OldClass)
    assert child.double() == 8


def test_removal_version_is_required():
    with pytest.raises(TypeError):
        deprecated(replacement="x")  # type: ignore[call-arg]


def test_nothing_in_the_package_is_currently_deprecated():
    """Guards the claim made in deprecation.py and docs/api_stability.md.

    If a deprecation is added, this test should be updated in the same commit
    that adds the changelog entry, so the two cannot drift.
    """
    import oversampleqa

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in oversampleqa.__all__:
            getattr(oversampleqa, name)
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []
