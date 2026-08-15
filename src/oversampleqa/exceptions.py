"""Exception hierarchy for OversampleQA.

Every error the package raises deliberately derives from
:class:`OversampleQAError`, so callers can catch the whole family without
catching unrelated failures.

These live in their own module rather than in ``types.py`` because the error
path had been raising ``ConfigurationError`` without it being defined anywhere
-- a ``NameError`` from inside an error handler, which masks the real failure
with a confusing one.
"""

from __future__ import annotations

__all__ = [
    "OversampleQAError",
    "ConfigurationError",
    "ValidationError",
    "UnsupportedSamplerError",
    "MetricError",
    "PluginError",
]


class OversampleQAError(Exception):
    """Base class for every error raised by OversampleQA."""


class ConfigurationError(OversampleQAError):
    """Configuration is invalid, missing, or internally inconsistent.

    Raised for bad parameter combinations and for lookups of things that were
    never registered.
    """


class ValidationError(OversampleQAError):
    """A validation run could not produce a meaningful result.

    Covers malformed input as well as data that cannot support the estimand --
    for example a minority class too small to hold anything out of.
    """


class UnsupportedSamplerError(ValidationError):
    """The oversampler cannot be validated by this package.

    Synthetic samples are identified positionally, which requires the sampler
    to append its output to the data it was given. Combined over/under-samplers
    such as ``SMOTEENN`` and ``SMOTETomek`` delete original rows instead, so
    their synthetic points cannot be identified this way.
    """


class MetricError(OversampleQAError):
    """A distance metric could not be resolved or computed."""


class PluginError(OversampleQAError):
    """A plugin failed to register, resolve, or execute."""
