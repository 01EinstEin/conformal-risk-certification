"""Finite-sample risk calibration for field-level AI extraction scores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import exp, expm1, log, log1p, sqrt

from .models import (
    AuditRecord,
    CalibrationPolicy,
    Decision,
    ThresholdEvidence,
    validate_unit_interval,
)


# This grid is fixed before the audit outcomes are inspected.
DEFAULT_THRESHOLDS = tuple(index / 20 for index in range(21))


def _validate_open_interval(value: float, name: str) -> float:
    result = validate_unit_interval(value, name)
    if result == 0.0 or result == 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return result


def _logaddexp(left: float, right: float) -> float:
    larger = max(left, right)
    smaller = min(left, right)
    return larger + log1p(exp(smaller - larger))


def _binomial_cdf(errors: int, trials: int, probability: float) -> float:
    """Compute P[X <= errors] for X ~ Binomial(trials, probability)."""
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0 if errors < trials else 1.0

    log_p = log(probability)
    log_q = log1p(-probability)
    log_term = trials * log_q  # P[X = 0]
    log_total = log_term

    for successes in range(1, errors + 1):
        log_term += (
            log(trials - successes + 1)
            - log(successes)
            + log_p
            - log_q
        )
        log_total = _logaddexp(log_total, log_term)

    return min(1.0, exp(log_total))


def clopper_pearson_upper(errors: int, trials: int, alpha: float) -> float:
    """One-sided exact binomial upper confidence bound.

    The returned bound U solves P[Binomial(trials, U) <= errors] = alpha.
    It is conservative at the discrete boundary and uses no third-party
    numerical library.
    """
    if isinstance(errors, bool) or not isinstance(errors, int):
        raise TypeError("errors must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= errors <= trials:
        raise ValueError("errors must be between 0 and trials")
    alpha_value = _validate_open_interval(alpha, "alpha")

    if errors == trials:
        return 1.0
    if errors == 0:
        return -expm1(log(alpha_value) / trials)

    low = 0.0
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(errors, trials, midpoint) > alpha_value:
            low = midpoint
        else:
            high = midpoint
    return high


def calibrate_threshold(
    records: Sequence[AuditRecord],
    *,
    risk_limit: float,
    alpha: float = 0.05,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> CalibrationPolicy:
    """Choose maximum observed coverage among certified thresholds.

    `thresholds` must be fixed before inspecting the human-audit outcomes.
    Bonferroni correction controls the family-wise confidence error across
    this threshold grid. If no threshold passes, the result abstains on every
    future record.
    """
    if not records:
        raise ValueError("records must contain at least one audited prediction")

    risk_value = _validate_open_interval(risk_limit, "risk_limit")
    alpha_value = _validate_open_interval(alpha, "alpha")
    threshold_grid = tuple(
        validate_unit_interval(value, "threshold") for value in thresholds
    )
    if not threshold_grid:
        raise ValueError("thresholds must contain at least one candidate")
    if len(set(threshold_grid)) != len(threshold_grid):
        raise ValueError("thresholds must be unique")

    fields = {record.field for record in records}
    if len(fields) != 1:
        raise ValueError(
            "calibrate_threshold accepts one field at a time; "
            "use calibrate_by_field for multiple fields"
        )
    field = next(iter(fields))
    alpha_per_threshold = alpha_value / len(threshold_grid)
    evidence: list[ThresholdEvidence] = []

    for threshold in threshold_grid:
        accepted = [
            record for record in records if record.confidence >= threshold
        ]
        accepted_count = len(accepted)
        error_count = sum(not record.correct for record in accepted)
        coverage = accepted_count / len(records)
        observed_error_rate = (
            error_count / accepted_count if accepted_count else None
        )

        if accepted_count:
            upper_bound = clopper_pearson_upper(
                error_count, accepted_count, alpha_per_threshold
            )
        else:
            upper_bound = 1.0

        evidence.append(
            ThresholdEvidence(
                threshold=threshold,
                accepted_count=accepted_count,
                error_count=error_count,
                coverage=coverage,
                observed_error_rate=observed_error_rate,
                upper_error_bound=upper_bound,
                alpha_spent=alpha_per_threshold,
                certified=accepted_count > 0 and upper_bound <= risk_value,
            )
        )

    certified = [candidate for candidate in evidence if candidate.certified]
    if not certified:
        return CalibrationPolicy(
            field=field,
            risk_limit=risk_value,
            familywise_alpha=alpha_value,
            threshold=None,
            accepted_count=0,
            total_count=len(records),
            coverage=0.0,
            observed_error_rate=None,
            upper_error_bound=None,
            candidates=tuple(evidence),
        )

    selected = max(
        certified,
        key=lambda candidate: (candidate.coverage, -candidate.threshold),
    )
    return CalibrationPolicy(
        field=field,
        risk_limit=risk_value,
        familywise_alpha=alpha_value,
        threshold=selected.threshold,
        accepted_count=selected.accepted_count,
        total_count=len(records),
        coverage=selected.coverage,
        observed_error_rate=selected.observed_error_rate,
        upper_error_bound=selected.upper_error_bound,
        candidates=tuple(evidence),
    )


def calibrate_by_field(
    records: Sequence[AuditRecord],
    *,
    risk_limits: Mapping[str, float],
    alpha: float = 0.05,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, CalibrationPolicy]:
    """Calibrate each field with a family-wise alpha adjustment.

    Alpha is divided across both fields and candidate thresholds. Every
    observed field must have an explicit risk limit so a missing limit cannot
    silently inherit a permissive default.
    """
    if not records:
        raise ValueError("records must contain at least one audited prediction")
    alpha_value = _validate_open_interval(alpha, "alpha")
    threshold_grid = tuple(thresholds)
    if not threshold_grid:
        raise ValueError("thresholds must contain at least one candidate")

    grouped: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        grouped[record.field].append(record)

    fields = set(grouped)
    configured_fields = set(risk_limits)
    missing = fields - configured_fields
    unknown = configured_fields - fields
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing risk limits for {sorted(missing)}")
        if unknown:
            details.append(f"risk limits for absent fields {sorted(unknown)}")
        raise ValueError("; ".join(details))

    per_field_alpha = alpha_value / len(fields)
    return {
        field: calibrate_threshold(
            field_records,
            risk_limit=risk_limits[field],
            alpha=per_field_alpha,
            thresholds=threshold_grid,
        )
        for field, field_records in sorted(grouped.items())
    }


def decide(
    confidence: float,
    policy: CalibrationPolicy,
    *,
    drift_detected: bool = False,
) -> Decision:
    """Apply a certified policy; drift or missing certification means review."""
    score = validate_unit_interval(confidence, "confidence")
    if not isinstance(drift_detected, bool):
        raise TypeError("drift_detected must be a boolean")
    if drift_detected or policy.threshold is None:
        return Decision.HUMAN_REVIEW
    if score >= policy.threshold:
        return Decision.AUTO_ACCEPT
    return Decision.HUMAN_REVIEW