import unittest

from invoice_risk_guard import detect_score_drift, two_sample_ks_distance


class DriftTests(unittest.TestCase):
    def test_identical_samples_have_zero_distance_and_no_alarm(self) -> None:
        scores = [0.1, 0.2, 0.4, 0.7, 0.9]
        report = detect_score_drift(scores, scores)

        self.assertEqual(report.empirical_distance, 0.0)
        self.assertFalse(report.shifted)

    def test_separated_samples_trigger_alarm_with_enough_observations(self) -> None:
        reference = [0.1] * 200
        recent = [0.9] * 200
        report = detect_score_drift(reference, recent, alpha=0.05)

        self.assertEqual(report.empirical_distance, 1.0)
        self.assertTrue(report.shifted)

    def test_ks_distance_handles_ties(self) -> None:
        self.assertEqual(
            two_sample_ks_distance([0.2, 0.2, 0.8], [0.2, 0.8, 0.8]),
            1 / 3,
        )

    def test_small_samples_are_not_overconfident(self) -> None:
        report = detect_score_drift([0.1], [0.9])
        self.assertGreater(report.critical_value, 1.0)
        self.assertFalse(report.shifted)

    def test_empty_or_invalid_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect_score_drift([], [0.5])
        with self.assertRaises(ValueError):
            detect_score_drift([0.5], [1.1])


if __name__ == "__main__":
    unittest.main()