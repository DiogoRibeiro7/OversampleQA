"""The deprecation mechanism promised by :doc:`/api_stability`.

That document commits to a specific shape for every deprecated name: a warning
that names both the replacement and the release in which the name disappears,
and at least two minor releases of continued working behaviour. Written by hand
at each site, "names the removal version" is the part that gets forgotten, and a
warning that says only "this is deprecated" leaves the reader to guess how long
they have and what to move to.

Nothing in the package is deprecated at the time of writing. This exists so the
next deprecation matches the documented policy instead of inventing its own
wording.

Note that :func:`~oversampleqa.validator.warn_reference_bias` deliberately does
not use this. ``reference="train_minority"`` is not an old spelling of a current
name -- it computes a biased quantity, and its warning explains the bias, which
is a different message with a different category.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

__all__ = ["deprecated"]

F = TypeVar("F", bound=Callable[..., Any])


def _build_message(
    name: str,
    replacement: str | None,
    removal_version: str,
    reason: str | None,
) -> str:
    parts = [f"{name} is deprecated and will be removed in {removal_version}."]
    if replacement:
        parts.append(f"Use {replacement} instead.")
    if reason:
        parts.append(reason)
    return " ".join(parts)


def deprecated(
    *,
    removal_version: str,
    replacement: str | None = None,
    reason: str | None = None,
    category: type[Warning] = DeprecationWarning,
) -> Callable[[F], F]:
    """Mark a function, method or class as deprecated.

    The emitted warning names the replacement and the removal version, which is
    what :doc:`/api_stability` promises and what a caller needs in order to act.
    A note is appended to the docstring so the deprecation is visible in the
    rendered documentation as well as at runtime.

    The warning is raised with ``stacklevel`` pointing at the **caller**, not at
    this wrapper. This matters more than it looks: Python's default filters hide
    ``DeprecationWarning`` unless it originates in ``__main__``, and per-module
    filters key on the reported location. A warning that reports itself as
    coming from inside oversampleqa is invisible to exactly the people who need
    to see it.

    Args:
        removal_version: Release in which the name disappears, e.g. ``"0.6.0"``.
            Required -- a deprecation without a deadline is a permanent warning.
        replacement: What to use instead, if there is a direct successor.
        reason: Extra context appended to the message, for cases where the
            replacement is not a simple substitution.
        category: Warning class. Defaults to ``DeprecationWarning``. Use
            ``FutureWarning`` when the change alters results rather than
            spelling, since that one is shown to end users by default.

    Returns:
        A decorator that wraps the target, preserving its metadata.

    Example:
        >>> @deprecated(removal_version="0.6.0", replacement="new_name")
        ... def old_name() -> int:
        ...     return 1
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     old_name()
        ...     str(caught[0].message)
        1
        'old_name is deprecated and will be removed in 0.6.0. Use new_name instead.'
    """

    def decorate(target: F) -> F:
        message = _build_message(
            getattr(target, "__name__", str(target)),
            replacement,
            removal_version,
            reason,
        )
        note = f"\n\n.. deprecated:: {removal_version}\n   {message}\n"

        if isinstance(target, type):
            # Wrap __init__ so the warning fires at the instantiation site.
            # Wrapping the class in a function instead would break isinstance,
            # subclassing and the repr.
            #
            # Rebinding a dunder on a class object is exactly the kind of thing
            # a type checker is right to distrust in general, so the surgery is
            # done through an explicitly untyped alias rather than scattered
            # per-line suppressions.
            klass: Any = target
            original_init = klass.__init__

            @functools.wraps(original_init)
            def init_wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, category, stacklevel=2)
                original_init(self, *args, **kwargs)

            klass.__init__ = init_wrapper
            klass.__doc__ = (klass.__doc__ or "") + note
            return cast(F, klass)

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, category, stacklevel=2)
            return target(*args, **kwargs)

        wrapper.__doc__ = (target.__doc__ or "") + note
        return wrapper  # type: ignore[return-value]

    return decorate
