"""Demonstrate why confidence-score drift alone cannot certify future risk.

This deterministic synthetic experiment preserves the score-generating
distribution after deployment while changing the relationship between scores
and correctness. It is a counterexample to score-only drift monitoring, not
evidence about real invoice systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from invoice_risk_guard import (
    AuditRecord,
    CalibrationPolicy,
    calibrate_threshold,
    decide,
    detect_score_drift,
)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Aggregate measurements from repeated synthetic post-shift batches."""

    risk_limit: float
    selected_threshold: float
    calibration_coverage: float
    calibration_upper_error_bound: float
    score_alarm_rate: float
    risk_violation_rate: float
    accepted_error_rate: float
    replications: int
    batch_size: int


def _make_calibration_data(
    rng: Random, count: int
) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for _ in range(count):
        score = rng.betavariate(8, 2)
        correctness_probability = 0.65 + 0.35 * score
        records.append(
            AuditRecord(
                confidence=score,
                correct=rng.random() < correctness_probability,
                field="total",
            )
        )
    return records


def run_experiment(
    *,
    seed: int = 20261004,
    calibration_size: int = 10_000,
    batch_size: int = 1_000,
    replications: int = 100,
    risk_limit: float = 0.05,
    post_shift_error_probability: float = 0.25,
) -> ExperimentResult:
    """Calibrate once, then simulate conditional-risk drift with stable scores."""
    if calibration_size <= 0 or batch_size <= 0 or replications <= 0:
        raise ValueError("sample sizes and replications must be positive")
    if not 0.0 < post_shift_error_probability < 1.0:
        raise ValueError("post_shift_error_probability must be in (0, 1)")

    rng = Random(seed)
    calibration = _make_calibration_data(rng, calibration_size)
    policy: CalibrationPolicy = calibrate_threshold(
        calibration,
        risk_limit=risk_limit,
        alpha=0.05,
    )
    if policy.threshold is None or policy.upper_error_bound is None:
        raise RuntimeError("synthetic calibration did not certify a threshold")

    reference_scores = [record.confidence for record in calibration]
    alarm_count = 0
    violation_count = 0
    total_accepted = 0
    total_accepted_errors = 0

    for _ in range(replications):
        batch: list[AuditRecord] = []
        for _ in range(batch_size):
            score = rng.betavariate(8, 2)
            batch.append(
                AuditRecord(
                    confidence=score,
                    correct=rng.random() >= post_shift_error_probability,
                    field="total",
                )
            )
        report = detect_score_drift(
            reference_scores,
            [record.confidence for record in batch],
        )
        alarm_count += report.shifted

        accepted = [
            record
            for record in batch
            if decide(record.confidence, policy).value == "auto_accept"
        ]
        if not accepted:
            raise RuntimeError("synthetic batch contained no accepted fields")

        errors = sum(not record.correct for record in accepted)
        total_accepted += len(accepted)
        total_accepted_errors += errors
        violation_count += errors / len(accepted) > risk_limit

    return ExperimentResult(
        risk_limit=risk_limit,
        selected_threshold=policy.threshold,
        calibration_coverage=policy.coverage,
        calibration_upper_error_bound=policy.upper_error_bound,
        score_alarm_rate=alarm_count / replications,
        risk_violation_rate=violation_count / replications,
        accepted_error_rate=total_accepted_errors / total_accepted,
        replications=replications,
        batch_size=batch_size,
    )


def main() -> None:
    result = run_experiment()
    print("Synthetic silent conditional-drift experiment")
    print("  not real-invoice data and not a production-performance claim")
    print(f"  risk limit: {result.risk_limit:.1%}")
    print(f"  selected threshold: {result.selected_threshold:.2f}")
    print(f"  calibration coverage: {result.calibration_coverage:.1%}")
    print(
        "  calibration upper error bound: "
        f"{result.calibration_upper_error_bound:.2%}"
    )
    print(f"  batches with risk-limit violations: {result.risk_violation_rate:.1%}")
    print(f"  score-only alarm rate: {result.score_alarm_rate:.1%}")
    print(f"  aggregate accepted-field error rate: {result.accepted_error_rate:.1%}")
    print(
        "  interpretation: the score distribution is sampled from the same "
        "process, while correctness degrades independently of the score."
    )


if __name__ == "__main__":
    main()
