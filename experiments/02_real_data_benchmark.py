"""Run the leakage-controlled German Credit selective-risk benchmark."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from estimators import (  # noqa: E402
    AdaptiveConformalInference,
    IPWConformal,
    NaiveThresholding,
    time_uniform_hoeffding_upper,
)


DATA_ID = 31
RANDOM_SEED = 20261011
N_FOLDS = 5
AUDIT_PROBABILITY = 0.10
CONFIDENCE_DELTA = 0.05
ACI_MIS_COVERAGE = 0.05
ACI_GAMMA = 0.01
NAIVE_THRESHOLD = 0.70

METHODS = (
    "Naive threshold",
    "ACI (full feedback)",
    "IPW-Conformal",
)
COLORS = {
    "Naive threshold": "#6B7280",
    "ACI (full feedback)": "#2563EB",
    "IPW-Conformal": "#D1495B",
}


def _make_classifier(features: pd.DataFrame, seed: int) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    transformations: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformations.append(
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformations.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            )
        )

    preprocessing = ColumnTransformer(transformations)
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _load_credit_data() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    dataset = fetch_openml(data_id=DATA_ID, as_frame=True)
    features = dataset.data
    labels = dataset.target.astype(str).str.lower()
    if features.shape != (1_000, 20):
        raise RuntimeError(
            f"Expected 1,000 rows and 20 features; received {features.shape}"
        )
    if set(labels.unique()) != {"bad", "good"}:
        raise RuntimeError(f"Unexpected target classes: {sorted(labels.unique())}")

    serialized = pd.concat(
        [features.reset_index(drop=True), labels.reset_index(drop=True).rename("class")],
        axis=1,
    ).to_csv(index=False).encode("utf-8")
    fingerprint = hashlib.sha256(serialized).hexdigest()
    metadata = {
        "openml_data_id": DATA_ID,
        "openml_name": dataset.details.get("name"),
        "openml_version": dataset.details.get("version"),
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "class_counts": {
            str(label): int(count)
            for label, count in labels.value_counts().sort_index().items()
        },
        "data_sha256": fingerprint,
    }
    return features, labels, metadata


def _calibration_nonconformity(
    classifier: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
) -> list[float]:
    probabilities = classifier.predict_proba(features)
    class_indices = {
        str(label).lower(): index
        for index, label in enumerate(classifier.classes_)
    }
    true_indices = np.asarray(
        [class_indices[label] for label in labels.astype(str).str.lower()],
        dtype=int,
    )
    return (
        1.0
        - probabilities[np.arange(len(true_indices)), true_indices]
    ).astype(float).tolist()


def _weighted_aggregate_bound(
    counts: list[int], bounds: list[float | None]
) -> float | None:
    total = sum(counts)
    if total == 0:
        return None
    if any(count > 0 and bound is None for count, bound in zip(counts, bounds)):
        raise RuntimeError("an active stream is missing its risk bound")
    return sum(
        count * float(bound)
        for count, bound in zip(counts, bounds)
        if count > 0 and bound is not None
    ) / total


def _summarize(
    frame: pd.DataFrame,
    method: str,
) -> dict[str, float | int | None]:
    last_rows = frame[
        (frame["method"] == method)
        & (frame["global_index"] == frame["global_index"].max())
    ]
    if len(last_rows) != 1:
        raise RuntimeError(f"Expected one final summary row for {method}")
    final = last_rows.iloc[0]
    accepted = int(final["accepted_count"])
    errors = int(final["accepted_error_count"])
    decisions = int(final["global_index"])
    bound = final["upper_risk_bound"]
    return {
        "decisions": decisions,
        "accepted": accepted,
        "coverage": accepted / decisions if decisions else 0.0,
        "oracle_risk": errors / accepted if accepted else None,
        "final_pooled_ucb": float(bound) if pd.notna(bound) else None,
        "audited": int(final["audit_count"]),
    }


def _plot_results(frame: pd.DataFrame, output_path: Path) -> None:
    summary = frame.sort_values(["method", "global_index"])
    figure, (risk_axis, coverage_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        layout="constrained",
    )

    for method in METHODS:
        method_data = summary[summary["method"] == method]
        color = COLORS[method]
        if method == "Naive threshold":
            risk_axis.plot(
                method_data["global_index"],
                method_data["oracle_risk"],
                color=color,
                linestyle="--",
                linewidth=2.0,
                label="Naive: oracle risk (no UCB)",
            )
        else:
            risk_axis.plot(
                method_data["global_index"],
                method_data["upper_risk_bound"],
                color=color,
                linestyle="-",
                linewidth=2.0,
                label=f"{method}: 95% upper bound",
            )
            risk_axis.plot(
                method_data["global_index"],
                method_data["oracle_risk"],
                color=color,
                linestyle=":",
                linewidth=1.8,
                label=f"{method}: oracle risk",
            )
        coverage_axis.plot(
            method_data["global_index"],
            method_data["automation_coverage"],
            color=color,
            linewidth=2.0,
            label=method,
        )

    for boundary in range(200, 1_000, 200):
        risk_axis.axvline(boundary, color="#9CA3AF", linewidth=0.8, alpha=0.45)
        coverage_axis.axvline(
            boundary, color="#9CA3AF", linewidth=0.8, alpha=0.45
        )
    risk_axis.set(
        title="Selective risk: confidence bounds and full-label evaluation",
        ylabel="Cumulative accepted-decision error rate",
        xlim=(1, 1_000),
        ylim=(0.0, 1.02),
    )
    risk_axis.grid(alpha=0.22)
    risk_axis.legend(loc="best", ncols=2, fontsize=9)
    risk_axis.text(
        0.01,
        0.02,
        "Naive thresholding has no statistical UCB; its line is evaluation-only.",
        transform=risk_axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )

    coverage_axis.set(
        title="Automation coverage",
        xlabel="Sequential decision index (five fold-isolated streams)",
        ylabel="Cumulative automated fraction",
        xlim=(1, 1_000),
        ylim=(0.0, 1.02),
    )
    coverage_axis.grid(alpha=0.22)
    coverage_axis.legend(loc="best")
    figure.suptitle(
        "German Credit Statlog: selective auditing at q = 0.10",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_benchmark() -> tuple[pd.DataFrame, dict[str, Any]]:
    features, labels, metadata = _load_credit_data()
    outer_splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    rows: list[dict[str, Any]] = []
    stream_size = len(features) // N_FOLDS
    fold_counts = {
        method: [0] * N_FOLDS
        for method in METHODS
    }
    fold_errors = {
        method: [0] * N_FOLDS
        for method in METHODS
    }
    fold_set_covered = {
        method: [0] * N_FOLDS
        for method in METHODS
    }
    fold_audits = [0] * N_FOLDS
    fold_ipw_estimators = [0.0] * N_FOLDS

    global_index = 0
    for fold, (pool_indices, test_indices) in enumerate(
        outer_splitter.split(features, labels)
    ):
        train_indices, calibration_indices = train_test_split(
            pool_indices,
            test_size=0.25,
            stratify=labels.iloc[pool_indices],
            random_state=RANDOM_SEED + fold,
        )
        classifier = _make_classifier(features, RANDOM_SEED + fold)
        classifier.fit(
            features.iloc[train_indices],
            labels.iloc[train_indices],
        )
        calibration_scores = _calibration_nonconformity(
            classifier,
            features.iloc[calibration_indices],
            labels.iloc[calibration_indices],
        )
        class_indices = {
            str(label).lower(): index
            for index, label in enumerate(classifier.classes_)
        }
        order_rng = np.random.default_rng(RANDOM_SEED + 10_000 + fold)
        ordered_test_indices = order_rng.permutation(test_indices)
        stream_features = features.iloc[ordered_test_indices]
        stream_labels = labels.iloc[ordered_test_indices].astype(str).str.lower()
        stream_probabilities = classifier.predict_proba(stream_features)
        stream_true_indices = np.asarray(
            [class_indices[label] for label in stream_labels],
            dtype=int,
        )
        audit_rng = np.random.default_rng(RANDOM_SEED + 20_000 + fold)

        naive = NaiveThresholding(NAIVE_THRESHOLD)
        aci = AdaptiveConformalInference(
            calibration_scores,
            alpha=ACI_MIS_COVERAGE,
            gamma=ACI_GAMMA,
        )
        ipw = IPWConformal(
            calibration_scores,
            audit_probability=AUDIT_PROBABILITY,
            alpha=ACI_MIS_COVERAGE,
        )
        fold_set_covered["Naive threshold"][fold] = 0

        for position, (probabilities, true_index) in enumerate(
            zip(stream_probabilities, stream_true_indices),
            start=1,
        ):
            global_index += 1
            prediction = int(np.argmax(probabilities))

            naive_accepted = naive.accepts(probabilities)
            naive_prediction_set = (
                (prediction,)
                if naive_accepted
                else tuple(range(len(probabilities)))
            )
            naive_loss = int(naive_accepted and prediction != true_index)
            fold_counts["Naive threshold"][fold] += int(naive_accepted)
            fold_errors["Naive threshold"][fold] += naive_loss
            fold_set_covered["Naive threshold"][fold] += int(
                true_index in naive_prediction_set
            )

            aci_set = aci.predict_set(probabilities)
            aci_accepted = len(aci_set) == 1
            aci_prediction = aci_set[0] if aci_accepted else prediction
            aci_loss = int(aci_accepted and aci_prediction != true_index)
            aci_miss = aci.observe(probabilities, int(true_index), aci_set)
            fold_counts["ACI (full feedback)"][fold] += int(aci_accepted)
            fold_errors["ACI (full feedback)"][fold] += aci_loss
            fold_set_covered["ACI (full feedback)"][fold] += int(not aci_miss)

            ipw_set = ipw.predict_set(probabilities)
            ipw_accepted = len(ipw_set) == 1
            ipw_prediction = ipw_set[0] if ipw_accepted else prediction
            ipw_loss = int(ipw_accepted and ipw_prediction != true_index)
            is_audited = bool(audit_rng.random() < AUDIT_PROBABILITY)
            audit_loss = ipw_loss if is_audited else None
            audited_nonconformity = (
                1.0 - float(probabilities[true_index]) if is_audited else None
            )
            ipw.observe(
                accepted=ipw_accepted,
                audit_loss=audit_loss,
                audited_nonconformity=audited_nonconformity,
            )
            fold_counts["IPW-Conformal"][fold] += int(ipw_accepted)
            fold_errors["IPW-Conformal"][fold] += ipw_loss
            fold_set_covered["IPW-Conformal"][fold] += int(
                true_index in ipw_set
            )
            fold_audits[fold] = ipw.audit_count
            fold_ipw_estimators[fold] = ipw.weighted_accepted_errors

            policy_states: dict[str, dict[str, Any]] = {
                "Naive threshold": {
                    "risk_estimate": None,
                    "upper_risk_bound": None,
                    "accepted": naive_accepted,
                    "audited": False,
                },
                "ACI (full feedback)": {
                    "risk_estimate": (
                        fold_errors["ACI (full feedback)"][fold]
                        / fold_counts["ACI (full feedback)"][fold]
                        if fold_counts["ACI (full feedback)"][fold]
                        else None
                    ),
                    "upper_risk_bound": time_uniform_hoeffding_upper(
                        fold_errors["ACI (full feedback)"][fold],
                        fold_counts["ACI (full feedback)"][fold],
                        delta=CONFIDENCE_DELTA,
                        number_of_streams=N_FOLDS,
                    ),
                    "accepted": aci_accepted,
                    "audited": False,
                },
                "IPW-Conformal": {
                    "risk_estimate": (
                        fold_ipw_estimators[fold] / fold_counts["IPW-Conformal"][fold]
                        if fold_counts["IPW-Conformal"][fold]
                        else None
                    ),
                    "upper_risk_bound": ipw.risk_upper_bound(
                        delta=CONFIDENCE_DELTA,
                        number_of_streams=N_FOLDS,
                    ),
                    "accepted": ipw_accepted,
                    "audited": is_audited,
                },
            }
            set_coverage_by_method = {
                "Naive threshold": fold_set_covered["Naive threshold"][fold],
                "ACI (full feedback)": fold_set_covered["ACI (full feedback)"][fold],
                "IPW-Conformal": fold_set_covered["IPW-Conformal"][fold],
            }

            for method in METHODS:
                accepted_so_far = sum(fold_counts[method])
                error_so_far = sum(fold_errors[method])
                total_stream_decisions = global_index
                active_fold_counts = fold_counts[method]
                if method == "Naive threshold":
                    pooled_bound = None
                elif method == "ACI (full feedback)":
                    fold_bounds = [
                        time_uniform_hoeffding_upper(
                            fold_errors[method][fold_index],
                            fold_counts[method][fold_index],
                            delta=CONFIDENCE_DELTA,
                            number_of_streams=N_FOLDS,
                        )
                        for fold_index in range(N_FOLDS)
                    ]
                    pooled_bound = _weighted_aggregate_bound(
                        active_fold_counts, fold_bounds
                    )
                elif method == "IPW-Conformal":
                    fold_bounds = [
                        time_uniform_hoeffding_upper(
                            fold_ipw_estimators[fold_index],
                            fold_counts[method][fold_index],
                            delta=CONFIDENCE_DELTA,
                            range_width=1.0 / AUDIT_PROBABILITY,
                            number_of_streams=N_FOLDS,
                        )
                        for fold_index in range(N_FOLDS)
                    ]
                    pooled_bound = _weighted_aggregate_bound(
                        active_fold_counts, fold_bounds
                    )
                else:
                    raise RuntimeError(f"Unknown method: {method}")
                risk_estimate = (
                    sum(
                        fold_ipw_estimators
                    )
                    / accepted_so_far
                    if method == "IPW-Conformal" and accepted_so_far
                    else (
                        error_so_far / accepted_so_far
                        if method == "ACI (full feedback)" and accepted_so_far
                        else None
                    )
                )
                set_covered_so_far = sum(fold_set_covered[method])
                audit_total = sum(fold_audits) if method == "IPW-Conformal" else 0
                oracle_risk = (
                    error_so_far / accepted_so_far if accepted_so_far else np.nan
                )
                rows.append(
                    {
                        "method": method,
                        "fold": fold + 1,
                        "position": position,
                        "stream_size": stream_size,
                        "global_index": global_index,
                        "accepted": policy_states[method]["accepted"],
                        "audited_this_decision": policy_states[method]["audited"],
                        "audit_count": audit_total,
                        "accepted_count": accepted_so_far,
                        "accepted_error_count": error_so_far,
                        "automation_coverage": accepted_so_far / total_stream_decisions,
                        "oracle_risk": oracle_risk,
                        "risk_estimate": risk_estimate,
                        "upper_risk_bound": pooled_bound,
                        "set_coverage": set_covered_so_far / total_stream_decisions,
                        "set_coverage_by_method_fold": (
                            set_coverage_by_method[method] / position
                        ),
                    }
                )

    result_frame = pd.DataFrame(rows)
    metadata.update(
        {
            "seed": RANDOM_SEED,
            "outer_folds": N_FOLDS,
            "stream_decisions_per_fold": stream_size,
            "total_out_of_fold_decisions": int(
                result_frame["global_index"].max()
            ),
            "training_rows_per_fold": 600,
            "calibration_rows_per_fold": 200,
            "audit_probability": AUDIT_PROBABILITY,
            "confidence_level": 1.0 - CONFIDENCE_DELTA,
            "aci_miscoverage_target": ACI_MIS_COVERAGE,
            "aci_step_size": ACI_GAMMA,
            "naive_confidence_threshold": NAIVE_THRESHOLD,
            "package_versions": {
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "matplotlib": version("matplotlib"),
                "scipy": version("scipy"),
            },
            "estimator_notes": (
                "Naive has no UCB. ACI UCB is diagnostic full-feedback "
                "Hoeffding; ACI marginal coverage is not selective-risk control. "
                "IPW UCB uses inverse-propensity losses and an anytime "
                "Hoeffding bound with range 1/q."
            ),
        }
    )
    return result_frame, metadata


def main() -> None:
    output_dir = PROJECT_ROOT / "generated-artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metadata = run_benchmark()
    plot_path = output_dir / "german_credit_risk_coverage.png"
    csv_path = output_dir / "german_credit_metrics.csv"
    metadata_path = output_dir / "german_credit_metadata.json"

    frame.to_csv(csv_path, index=False)
    _plot_results(frame, plot_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("German Credit Statlog selective-risk benchmark")
    print(
        f"  data: OpenML {metadata['openml_data_id']} "
        f"({metadata['openml_name']}, version {metadata['openml_version']})"
    )
    print(
        f"  decisions: {metadata['total_out_of_fold_decisions']} "
        f"across {N_FOLDS} fold-isolated streams"
    )
    for method in METHODS:
        summary = _summarize(frame, method)
        risk = (
            f"{summary['oracle_risk']:.3f}"
            if summary["oracle_risk"] is not None
            else "undefined (no automated decisions)"
        )
        bound = (
            f"{summary['final_pooled_ucb']:.3f}"
            if summary["final_pooled_ucb"] is not None
            else "not available"
        )
        print(
            f"  {method}: coverage={summary['coverage']:.3f}, "
            f"oracle selective risk={risk}, final pooled UCB={bound}"
        )
    print(
        f"  IPW audit fraction: "
        f"{_summarize(frame, 'IPW-Conformal')['audited'] / 1_000:.3f}"
    )
    print(f"  plot: {plot_path.relative_to(PROJECT_ROOT)}")
    print(f"  metrics: {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
