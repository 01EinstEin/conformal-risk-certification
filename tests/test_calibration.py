import math
import unittest

from invoice_risk_guard import (
    AuditRecord,
    Decision,
    calibrate_by_field,
    calibrate_threshold,
    clopper_pearson_upper,
    decide,
)


class ClopperPearsonTests(unittest.TestCase):
    def test_zero_errors_uses_exact_one_sided_bound(self) -> None:
        expected = -math.expm1(math.log(0.05) / 10)
        self.assertAlmostEqual(clopper_pearson_upper(0, 10, 0.05), expected)

    def test_all_errors_has_unit_upper_bound(self) -> None:
        self.assertEqual(clopper_pearson_upper(4, 4, 0.05), 1.0)

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clopper_pearson_upper(3, 2, 0.05)
        with self.assertRaises(ValueError):
            clopper_pearson_upper(0, 0, 0.05)


class CalibrationTests(unittest.TestCase):
    def test_selects_highest_coverage_certified_threshold(self) -> None:
        high_confidence = [
            AuditRecord(0.95, correct=index != 0, field="total")
            for index in range(80)
        ]
        low_confidence = [
            AuditRecord(0.40, correct=index >= 20, field="total")
            for index in range(80)
        ]

        policy = calibrate_threshold(
            high_confidence + low_confidence,
            risk_limit=0.15,
            alpha=0.05,
            thresholds=(0.0, 0.5, 0.9),
        )

        self.assertEqual(policy.threshold, 0.5)
        self.assertEqual(policy.accepted_count, 80)
        self.assertAlmostEqual(policy.coverage, 0.5)
        self.assertLessEqual(policy.upper_error_bound or 1.0, 0.15)

    def test_abstains_when_sample_cannot_certify_the_risk_limit(self) -> None:
        policy = calibrate_threshold(
            [AuditRecord(0.99, True, "total")],
            risk_limit=0.05,
            thresholds=(0.9,),
        )

        self.assertIsNone(policy.threshold)
        self.assertEqual(policy.coverage, 0.0)
        self.assertEqual(decide(0.999, policy), Decision.HUMAN_REVIEW)

    def test_drift_forces_human_review(self) -> None:
        records = [
            AuditRecord(0.95, True, "total") for _ in range(200)
        ]
        policy = calibrate_threshold(
            records,
            risk_limit=0.1,
            thresholds=(0.9,),
        )

        self.assertEqual(decide(0.99, policy), Decision.AUTO_ACCEPT)
        self.assertEqual(
            decide(0.99, policy, drift_detected=True),
            Decision.HUMAN_REVIEW,
        )

    def test_calibrates_each_field_with_explicit_risk_limits(self) -> None:
        records = [
            AuditRecord(0.9, True, "total"),
            AuditRecord(0.9, True, "due_date"),
        ]
        policies = calibrate_by_field(
            records,
            risk_limits={"total": 0.5, "due_date": 0.5},
            thresholds=(0.5,),
        )

        self.assertEqual(set(policies), {"total", "due_date"})
        self.assertAlmostEqual(policies["total"].familywise_alpha, 0.025)

    def test_missing_field_risk_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing risk limits"):
            calibrate_by_field(
                [AuditRecord(0.9, True, "total")],
                risk_limits={},
                thresholds=(0.5,),
            )

    def test_multiple_fields_cannot_be_calibrated_as_one_pool(self) -> None:
        records = [
            AuditRecord(0.9, True, "total"),
            AuditRecord(0.9, True, "due_date"),
        ]
        with self.assertRaisesRegex(ValueError, "one field at a time"):
            calibrate_threshold(records, risk_limit=0.5)


if __name__ == "__main__":
    unittest.main()