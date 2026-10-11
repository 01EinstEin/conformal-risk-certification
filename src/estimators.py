"""Selective-classification baselines for the German Credit benchmark.

The module intentionally separates prediction-set construction from risk
estimation. A conformal prediction-set guarantee is a marginal coverage
statement; it is not, by itself, a guarantee on error conditional on
automation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, isfinite, log, sqrt

import numpy as np


def _validate_open_probability(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return result


def _validate_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("probabilities must be a one-dimensional class vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("class probabilities must be finite values in [0, 1]")
    if not np.isclose(float(values.sum()), 1.0, atol=1e-6):
        raise ValueError("class probabilities must sum to one")
    return values


def split_conformal_quantile(
    calibration_scores: Sequence[float], alpha: float
) -> float:
    """Return the finite-sample split-conformal order statistic.

    An infinite threshold is returned when the calibration sample is too small
    to support the requested miscoverage level.
    """
    alpha_value = float(alpha)
    if not isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores must be a finite one-dimensional array")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("nonconformity scores must be in [0, 1]")
    if alpha_value == 0.0 or scores.size == 0:
        return float("inf")
    if alpha_value == 1.0:
        return float("-inf")

    rank = ceil((scores.size + 1) * (1.0 - alpha_value))
    if rank > scores.size:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def weighted_conformal_quantile(
    calibration_scores: Sequence[float],
    calibration_weights: Sequence[float],
    alpha: float,
    *,
    test_weight: float = 1.0,
) -> float:
    """Return a conservative weighted conformal quantile.

    The added test-point weight accounts for the unseen example. In the
    benchmark, audited stream examples receive inverse-propensity weight
    ``1 / q`` while the fully labelled, held-out calibration split has unit
    weights.
    """
    alpha_value = float(alpha)
    if not isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    scores = np.asarray(calibration_scores, dtype=float)
    weights = np.asarray(calibration_weights, dtype=float)
    if scores.ndim != 1 or weights.ndim != 1 or scores.size != weights.size:
        raise ValueError("calibration scores and weights must be aligned vectors")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("calibration scores must be finite values in [0, 1]")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("calibration weights must be finite and positive")
    if not isfinite(test_weight) or test_weight <= 0.0:
        raise ValueError("test_weight must be finite and positive")
    if alpha_value == 0.0 or scores.size == 0:
        return float("inf")
    if alpha_value == 1.0:
        return float("-inf")

    target_weight = (1.0 - alpha_value) * (float(weights.sum()) + test_weight)
    if target_weight > float(weights.sum()):
        return float("inf")

    order = np.argsort(scores, kind="stable")
    cumulative_weights = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative_weights, target_weight, side="left"))
    return float(scores[order[index]])


def prediction_set(
    probabilities: Sequence[float], quantile: float
) -> tuple[int, ...]:
    """Return class indices whose probability nonconformity is below quantile."""
    values = _validate_probabilities(probabilities)
    if np.isnan(quantile):
        raise ValueError("quantile must not be NaN")
    return tuple(
        int(index)
        for index, probability in enumerate(values)
        if 1.0 - float(probability) <= quantile
    )


def time_uniform_hoeffding_upper(
    error_sum: float,
    sample_size: int,
    *,
    delta: float = 0.05,
    range_width: float = 1.0,
    number_of_streams: int = 1,
) -> float | None:
    """Compute an anytime Hoeffding upper bound with alpha spending.

    Spending ``delta / (number_of_streams * n * (n + 1))`` at accepted-sample
    count ``n`` gives a simultaneous bound over all positive sample sizes and
    the specified independent streams. For IPW observations, ``range_width``
    is ``1 / q``.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    delta_value = _validate_open_probability(delta, "delta")
    if isinstance(number_of_streams, bool) or not isinstance(number_of_streams, int):
        raise TypeError("number_of_streams must be an integer")
    if number_of_streams <= 0:
        raise ValueError("number_of_streams must be positive")
    if not isfinite(range_width) or range_width <= 0.0:
        raise ValueError("range_width must be finite and positive")
    if not isfinite(error_sum) or error_sum < 0.0:
        raise ValueError("error_sum must be finite and nonnegative")
    if sample_size == 0:
        return None

    estimate = error_sum / sample_size
    delta_spent = delta_value / (
        number_of_streams * sample_size * (sample_size + 1)
    )
    radius = range_width * sqrt(log(1.0 / delta_spent) / (2.0 * sample_size))
    return min(1.0, estimate + radius)


class NaiveThresholding:
    """Fixed confidence cutoff with no feedback or statistical risk bound."""

    def __init__(self, threshold: float = 0.70) -> None:
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        self.threshold = float(threshold)

    def accepts(self, probabilities: Sequence[float]) -> bool:
        values = _validate_probabilities(probabilities)
        return float(values.max()) >= self.threshold


class AdaptiveConformalInference:
    """Adaptive Conformal Inference (ACI) with fully observed online labels."""

    def __init__(
        self,
        calibration_scores: Sequence[float],
        *,
        alpha: float = 0.05,
        gamma: float = 0.01,
    ) -> None:
        self.alpha_target = _validate_open_probability(alpha, "alpha")
        if not isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be finite and positive")
        self.gamma = float(gamma)
        self.alpha_t = self.alpha_target
        self.calibration_scores = self._validated_scores(calibration_scores)

    @staticmethod
    def _validated_scores(scores: Sequence[float]) -> list[float]:
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("calibration scores must be a finite vector")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("nonconformity scores must be in [0, 1]")
        return [float(value) for value in values]

    def predict_set(self, probabilities: Sequence[float]) -> tuple[int, ...]:
        quantile = split_conformal_quantile(
            self.calibration_scores, self.alpha_t
        )
        return prediction_set(probabilities, quantile)

    def observe(
        self,
        probabilities: Sequence[float],
        true_label: int,
        predicted_set: Sequence[int],
    ) -> bool:
        """Update ACI after observing the label; return the miscoverage flag."""
        values = _validate_probabilities(probabilities)
        if isinstance(true_label, bool) or not isinstance(true_label, int):
            raise TypeError("true_label must be an integer class index")
        if not 0 <= true_label < values.size:
            raise ValueError("true_label is outside the class vector")
        miss = true_label not in predicted_set
        self.alpha_t = min(
            1.0,
            max(0.0, self.alpha_t + self.gamma * (self.alpha_target - int(miss))),
        )
        self.calibration_scores.append(1.0 - float(values[true_label]))
        return miss


class IPWConformal:
    """Conformal sets and selective-risk monitoring under Bernoulli audits."""

    def __init__(
        self,
        calibration_scores: Sequence[float],
        *,
        audit_probability: float = 0.10,
        alpha: float = 0.05,
    ) -> None:
        self.audit_probability = _validate_open_probability(
            audit_probability, "audit_probability"
        )
        self.alpha = _validate_open_probability(alpha, "alpha")
        self.calibration_scores = AdaptiveConformalInference._validated_scores(
            calibration_scores
        )
        self.calibration_weights = [1.0] * len(self.calibration_scores)
        self.accepted_count = 0
        self.weighted_accepted_errors = 0.0
        self.audit_count = 0

    def predict_set(self, probabilities: Sequence[float]) -> tuple[int, ...]:
        quantile = weighted_conformal_quantile(
            self.calibration_scores,
            self.calibration_weights,
            self.alpha,
        )
        return prediction_set(probabilities, quantile)

    def observe(
        self,
        *,
        accepted: bool,
        audit_loss: int | None,
        audited_nonconformity: float | None,
    ) -> None:
        """Update from an audited label, without accepting hidden labels.

        Audits are drawn independently for each stream decision. When audited,
        the nonconformity score is inverse-propensity weighted for calibration;
        the accepted-only loss contributes ``loss / q`` to the IPW estimator.
        """
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be a boolean")
        if (audit_loss is None) != (audited_nonconformity is None):
            raise ValueError("audit loss and nonconformity must be observed together")
        if audit_loss is not None:
            if isinstance(audit_loss, bool):
                loss_value = int(audit_loss)
            elif isinstance(audit_loss, int) and audit_loss in (0, 1):
                loss_value = audit_loss
            else:
                raise ValueError("audit_loss must be 0, 1, or None")
            if (
                not isfinite(audited_nonconformity)
                or not 0.0 <= audited_nonconformity <= 1.0
            ):
                raise ValueError("audited_nonconformity must be in [0, 1]")
            self.audit_count += 1
            self.calibration_scores.append(float(audited_nonconformity))
            self.calibration_weights.append(1.0 / self.audit_probability)
            if accepted:
                self.weighted_accepted_errors += (
                    loss_value / self.audit_probability
                )
        if accepted:
            self.accepted_count += 1

    def risk_upper_bound(
        self,
        *,
        delta: float = 0.05,
        number_of_streams: int = 1,
    ) -> float | None:
        """Return a time-uniform Hoeffding UCB for accepted-only risk."""
        return time_uniform_hoeffding_upper(
            self.weighted_accepted_errors,
            self.accepted_count,
            delta=delta,
            range_width=1.0 / self.audit_probability,
            number_of_streams=number_of_streams,
        )
