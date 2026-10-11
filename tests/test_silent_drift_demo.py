import unittest

from experiments.silent_drift_demo import run_experiment


class SilentDriftExperimentTests(unittest.TestCase):
    def test_score_alarm_misses_conditional_error_degradation(self) -> None:
        result = run_experiment()

        self.assertLess(result.calibration_upper_error_bound, result.risk_limit)
        self.assertGreater(result.calibration_coverage, 0.0)
        self.assertGreater(result.risk_violation_rate, 0.9)
        self.assertGreater(result.accepted_error_rate, result.risk_limit * 3)
        self.assertLess(result.score_alarm_rate, 0.2)


if __name__ == "__main__":
    unittest.main()
