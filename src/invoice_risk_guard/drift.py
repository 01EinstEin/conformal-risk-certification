"""Distribution-shift alarm for extractor confidence scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log, sqrt

from .models import validate_unit_interval


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Two-sample confidence-score shift result."""

    reference_size: int
    recent_size: int
    empirical_distance: float
    critical_value: float
    alpha: float
    shifted: bool


def _validated_scores(scores: Sequence[float], name: str) -> tuple[float, ...]:
    if not scores:
        raise ValueError(f"{name} must contain at least one score")
    return tuple(validate_unit_interval(score, name) for score in scores)


def two_sample_ks_distance(
    reference_scores: Sequence[float],
    recent_scores: Sequence[float],
) -> float:
    """Return sup_x |F_reference(x) - F_recent(x)|."""
    reference = sorted(_validated_scores(reference_scores, "reference score"))
    recent = sorted(_validated_scores(recent_scores, "recent score"))
    reference_index = 0
    recent_index = 0
    distance = 0.0

    for value in sorted(set(reference) | set(recent)):
        while (
            reference_index < len(reference)
            and reference[reference_index] <= value
        ):
            reference_index += 1
        while recent_index < len(recent) and recent[recent_index] <= value:
            recent_index += 1

        reference_cdf = reference_index / len(reference)
        recent_cdf = recent_index / len(recent)
        distance = max(distance, abs(reference_cdf - recent_cdf))

    return distance


def detect_score_drift(
    reference_scores: Sequence[float],
    recent_scores: Sequence[float],
    *,
    alpha: float = 0.05,
) -> DriftReport:
    """Flag a score-distribution shift using a two-sample DKW bound.

    The bound is conservative and distribution-free under independent
    samples from the same reference distribution. It does not test whether
    confidence remains calibrated to correctness.
    """
    reference = _validated_scores(reference_scores, "reference score")
    recent = _validated_scores(recent_scores, "recent score")
    alpha_value = validate_unit_interval(alpha, "alpha")
    if alpha_value == 0.0 or alpha_value == 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    distance = two_sample_ks_distance(reference, recent)
    log_term = log(4.0 / alpha_value)
    critical_value = sqrt(log_term / (2.0 * len(reference))) + sqrt(
        log_term / (2.0 * len(recent))
    )

    return DriftReport(
        reference_size=len(reference),
        recent_size=len(recent),
        empirical_distance=distance,
        critical_value=critical_value,
        alpha=alpha_value,
        shifted=distance > critical_value,
    )