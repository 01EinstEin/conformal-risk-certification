"""Typed records and policy results used by the research prototype."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


def validate_unit_interval(value: float, name: str) -> float:
    """Return a finite probability-like value in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number between 0 and 1")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return result


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One human-verified field prediction from an extractor."""

    confidence: float
    correct: bool
    field: str

    def __post_init__(self) -> None:
        validate_unit_interval(self.confidence, "confidence")
        if not isinstance(self.correct, bool):
            raise TypeError("correct must be a boolean human-audit outcome")
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("field must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ThresholdEvidence:
    """Risk evidence for one predeclared confidence threshold."""

    threshold: float
    accepted_count: int
    error_count: int
    coverage: float
    observed_error_rate: float | None
    upper_error_bound: float
    alpha_spent: float
    certified: bool


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    """The selected field-level threshold, or an abstain-all policy."""

    field: str
    risk_limit: float
    familywise_alpha: float
    threshold: float | None
    accepted_count: int
    total_count: int
    coverage: float
    observed_error_rate: float | None
    upper_error_bound: float | None
    candidates: tuple[ThresholdEvidence, ...]


class Decision(str, Enum):
    """Action to take for a new extracted field."""

    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"