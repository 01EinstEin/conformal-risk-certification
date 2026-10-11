import math
import unittest

from estimators import (
    AdaptiveConformalInference,
    IPWConformal,
    NaiveThresholding,
    prediction_set,
    split_conformal_quantile,
    time_uniform_hoeffding_upper,
    weighted_conformal_quantile,
)


class ConformalQuantileTests(unittest.TestCase):
    def test_split_quantile_uses_finite_sample_rank(self) -> None:
        self.assertAlmostEqual(split_conformal_quantile([0.1, 0.2], 0.5), 0.2)
        self.assertTrue(math.isinf(split_conformal_quantile([0.1, 0.2], 0.01)))

    def test_empty_calibration_is_conservative(self) -> None:
        self.assertEqual(split_conformal_quantile([], 0.05), float("inf"))

    def test_weighted_quantile_respects_added_test_weight(self) -> None:
        result = weighted_conformal_quantile(
            [0.1, 0.2, 0.3],
            [10.0, 10.0, 10.0],
            0.25,
            test_weight=1.0,
        )
        self.assertEqual(result, 0.3)

    def test_prediction_set_contains_classes_under_quantile(self) -> None:
        self.assertEqual(prediction_set([0.8, 0.2], 0.25), (0,))


class StrategyTests(unittest.TestCase):
    def test_naive_threshold_is_fixed_and_has_no_risk_bound_api(self) -> None:
        strategy = NaiveThresholding(0.7)
        self.assertTrue(strategy.accepts([0.7, 0.3]))
        self.assertFalse(strategy.accepts([0.69, 0.31]))

    def test_aci_reduces_alpha_after_miscoverage(self) -> None:
        strategy = AdaptiveConformalInference(
            [0.1] * 19,
            alpha=0.05,
            gamma=0.01,
        )
        prediction = strategy.predict_set([0.9, 0.1])
        missed = strategy.observe([0.9, 0.1], 1, prediction)
        self.assertTrue(missed)
        self.assertLess(strategy.alpha_t, 0.05)
        self.assertAlmostEqual(strategy.calibration_scores[-1], 0.9)

    def test_ipw_only_updates_risk_from_observed_audits(self) -> None:
        strategy = IPWConformal(
            [0.1] * 19,
            audit_probability=0.1,
        )
        strategy.observe(
            accepted=True,
            audit_loss=None,
            audited_nonconformity=None,
        )
        self.assertEqual(strategy.accepted_count, 1)
        self.assertEqual(strategy.weighted_accepted_errors, 0.0)
        strategy.observe(
            accepted=True,
            audit_loss=1,
            audited_nonconformity=0.9,
        )
        self.assertEqual(strategy.accepted_count, 2)
        self.assertAlmostEqual(strategy.weighted_accepted_errors, 10.0)
        self.assertEqual(strategy.audit_count, 1)
        self.assertEqual(strategy.calibration_weights[-1], 10.0)

    def test_ipw_rejects_unpaired_audit_values(self) -> None:
        strategy = IPWConformal([0.1] * 19)
        with self.assertRaises(ValueError):
            strategy.observe(
                accepted=True,
                audit_loss=1,
                audited_nonconformity=None,
            )

    def test_time_uniform_bound_is_null_without_accepted_examples(self) -> None:
        self.assertIsNone(time_uniform_hoeffding_upper(0.0, 0))

    def test_time_uniform_bound_is_valid_probability(self) -> None:
        bound = time_uniform_hoeffding_upper(
            2.0,
            10,
            delta=0.05,
            number_of_streams=5,
        )
        self.assertIsNotNone(bound)
        self.assertGreaterEqual(bound, 0.2)
        self.assertLessEqual(bound, 1.0)


if __name__ == "__main__":
    unittest.main()
