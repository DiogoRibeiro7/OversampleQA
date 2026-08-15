"""A scikit-learn-shaped entry point.

Users of an ``imbalanced-learn``-adjacent tool expect estimator conventions:
constructor parameters, ``get_params``, ``fit``, ``score``, composability with
``cross_validate``. A free function with eleven keyword arguments does not
compose with any of that.

The free functions remain and are not deprecated -- the one-line quick start is
the package's best on-ramp. This is an additional surface, not a replacement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from ._rng import RandomStateLike
from .exceptions import ValidationError
from .reports import RunMetadata, ValidationReport
from .types import ReferenceSet, ValidationDetails

__all__ = ["OversamplingValidator", "validation_scorer"]


class OversamplingValidator(BaseEstimator):
    """Validate an oversampler, following the scikit-learn estimator contract.

    Lower scores are better: the score *is* the hidden-majority error rate, so
    ``score`` returns its negation, matching scikit-learn's "greater is better"
    convention for scorers.

    Parameters
    ----------
    oversampler : object
        An ``imbalanced-learn`` sampler.
    minority_label : int, optional
        Minority class. Inferred as the least frequent label when omitted.
    hidden_ratio : float, default=0.1
        Fraction held out.
    reference : {"hidden_minority", "train_minority"}, default="hidden_minority"
        Which minority set to compare against.
    metric : str, default="hassanat"
        Distance metric.
    metric_params : dict, optional
        Extra keyword arguments for the metric.
    n_repeats : int, default=1
        Independent hold-out splits.
    random_state : int, Generator, SeedSequence or None, default=42
        Seeds the split.

    Attributes
    ----------
    report_ : ValidationReport
        Set by :meth:`fit`.
    error_rate_ : float
        Set by :meth:`fit`.

    Notes
    -----
    The constructor stores its arguments unchanged and does no validation or
    computation, as scikit-learn requires -- ``get_params`` / ``set_params``
    round-trip, and ``clone`` works. All checking happens in :meth:`fit`.

    .. warning::

       **Cross-validation folds must be large enough to support the estimand.**
       Scoring runs a full validation on each test fold, which holds out
       ``hidden_ratio`` of *that fold's* minority. With ``cv=3`` on 136 minority
       points, a test fold has ~45 and a 10% hold-out leaves 4 — below
       ``min_hidden``, so validation raises.

       scikit-learn catches scorer exceptions and records ``nan``, so this
       surfaces as an all-``nan`` ``cv_results_`` with no explanation. Pass
       ``error_score="raise"`` to see the real message. Either use fewer folds,
       supply more minority data, or lower ``min_hidden`` deliberately.

    Examples
    --------
    Tuning a sampler against synthetic-sample quality becomes two lines::

        search = GridSearchCV(
            OversamplingValidator(SMOTE(random_state=0)),
            {"oversampler": [SMOTE(k_neighbors=k) for k in (3, 5, 9)]},
            scoring=validation_scorer,
        )
        search.fit(X, y)
    """

    def __init__(
        self,
        oversampler: Any,
        *,
        minority_label: int | None = None,
        hidden_ratio: float = 0.1,
        reference: ReferenceSet = "hidden_minority",
        metric: str = "hassanat",
        metric_params: dict[str, Any] | None = None,
        n_repeats: int = 1,
        random_state: RandomStateLike = 42,
    ) -> None:
        # Stored unchanged. No validation here: scikit-learn requires that
        # __init__ be a pure assignment so clone() and set_params() behave.
        self.oversampler = oversampler
        self.minority_label = minority_label
        self.hidden_ratio = hidden_ratio
        self.reference = reference
        self.metric = metric
        self.metric_params = metric_params
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _resolve_minority_label(self, y: NDArray[np.integer]) -> int:
        """Infer the minority label as the least frequent one."""
        if self.minority_label is not None:
            return int(self.minority_label)
        labels, counts = np.unique(y, return_counts=True)
        return int(labels[int(np.argmin(counts))])

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.integer]
    ) -> OversamplingValidator:
        """Run validation and store the report.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            self, so calls chain.

        Raises:
            ValidationError: If the inputs cannot support validation.
        """
        from .validator import validate_oversampling

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValidationError(f"X must be 2-D; got shape {X.shape}")
        if len(X) != len(y):
            raise ValidationError(f"X has {len(X)} rows but y has {len(y)}")

        minority_label = self._resolve_minority_label(y)

        details = validate_oversampling(
            X,
            y,
            minority_label,
            self.oversampler,
            hidden_ratio=self.hidden_ratio,
            metric=self.metric,
            metric_kwargs=self.metric_params,
            return_details=True,
            reference=self.reference,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        # return_details=True always yields ValidationDetails.
        if not isinstance(details, ValidationDetails):  # pragma: no cover
            raise ValidationError(
                "validate_oversampling(return_details=True) must return "
                "ValidationDetails"
            )
        self.error_rate_ = float(details.error_rate)
        self.report_ = ValidationReport(
            error_rate=self.error_rate_,
            metadata=RunMetadata.capture(
                X,
                y,
                self.oversampler,
                minority_label=minority_label,
                metric=self.metric,
                hidden_ratio=self.hidden_ratio,
                reference=self.reference,
                random_state=(
                    self.random_state
                    if isinstance(self.random_state, (int, type(None)))
                    else None
                ),
                n_repeats=self.n_repeats,
            ),
            details=details,
        )
        self.minority_label_ = minority_label
        return self

    def score(
        self,
        X: NDArray[np.floating] | None = None,
        y: NDArray[np.integer] | None = None,
    ) -> float:
        """Return the negated error rate, so greater is better.

        Scikit-learn's convention is that a higher score is better, but a
        higher error rate is worse. Returning the raw rate would make
        ``GridSearchCV`` select the *worst* sampler, so it is negated here.

        Args:
            X: Optional data to validate instead of the fitted run.
            y: Labels matching ``X``.

        Returns:
            Negated error rate.
        """
        if X is not None and y is not None:
            # deep=False: nested `oversampler__*` keys are not constructor
            # arguments, which is why sklearn's own clone() uses shallow params.
            fresh = self.__class__(**self.get_params(deep=False))
            return -float(fresh.fit(X, y).error_rate_)
        if not hasattr(self, "error_rate_"):
            raise ValidationError("call fit before score, or pass X and y")
        return -self.error_rate_


def validation_scorer(
    estimator: OversamplingValidator,
    X: NDArray[np.floating],
    y: NDArray[np.integer],
) -> float:
    """Scorer callable for ``cross_validate`` and ``GridSearchCV``.

    Follows the ``scorer(estimator, X, y)`` signature and the greater-is-better
    convention, so it can be passed directly as ``scoring=``.
    """
    return estimator.score(X, y)
