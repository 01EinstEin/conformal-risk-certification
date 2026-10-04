"""Deterministic synthetic demonstration; not real-world evidence."""

from __future__ import annotations

from random import Random

from invoice_risk_guard import (
    AuditRecord,
    calibrate_threshold,
    decide,
    detect_score_drift,
)


def make_synthetic_audits(rng: Random, count: int) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for _ in range(count):
        confidence = rng.betavariate(8, 2)
        # A synthetic extractor whose correctness improves with its score.
        correctness_probability = 0.65 + 0.35 * confidence
        records.append(
            AuditRecord(
                confidence=confidence,
                correct=rng.random() < correctness_probability,
                field="total",
            )
        )
    return records


def main() -> None:
    rng = Random(20261004)
    calibration = make_synthetic_audits(rng, 10_000)
    policy = calibrate_threshold(
        calibration,
        risk_limit=0.05,
        alpha=0.05,
    )

    print("Synthetic calibration (not a business result)")
    print(f"  audited fields: {len(calibration)}")
    print("  example risk budget: 5%")
    if policy.threshold is None:
        print("  certified threshold: none; route all fields to human review")
    else:
        print(f"  certified confidence threshold: {policy.threshold:.2f}")
        print(
            f"  calibration coverage: {policy.coverage:.1%} "
            f"({policy.accepted_count}/{policy.total_count})"
        )
        print(
            "  observed calibration error: "
            f"{policy.observed_error_rate:.2%}"
        )
        print(
            "  one-sided upper risk bound: "
            f"{policy.upper_error_bound:.2%}"
        )

        holdout = make_synthetic_audits(rng, 5_000)
        accepted = [
            record
            for record in holdout
            if decide(record.confidence, policy).value == "auto_accept"
        ]
        if accepted:
            holdout_errors = sum(not record.correct for record in accepted)
            print(
                "  independent synthetic holdout: "
                f"{holdout_errors}/{len(accepted)} errors "
                f"({holdout_errors / len(accepted):.2%}); "
                "descriptive only, not a certification"
            )

    reference_scores = [record.confidence for record in calibration]
    shifted_scores = [rng.betavariate(2, 8) for _ in range(500)]
    drift = detect_score_drift(reference_scores, shifted_scores)
    print(
        "  confidence-distribution drift alarm: "
        f"{drift.shifted} (distance={drift.empirical_distance:.3f}, "
        f"bound={drift.critical_value:.3f})"
    )


if __name__ == "__main__":
    main()