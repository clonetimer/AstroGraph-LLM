#!/usr/bin/env python3
"""Cross-validated utility audit for truth-free OPS-SAT shadow probes.

Only DEVELOPMENT_SELECTION groups are read.  Labels are joined after each
truth-free probe vector is computed and are used solely by this offline audit.
No LLM, formal evidence path, frozen detector, or external-validation group is
modified or consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.ops_sat_probe_utility import (
    OPS_SAT_PROBE_IDS,
    OPS_SAT_PROBE_LIBRARY_VERSION,
    measure_ops_sat_probe,
)
from astrograph.evaluation.replication_r22 import ops_r21_frozen_remediation_prediction


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def cross_validated_probabilities(
    matrix: np.ndarray, labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros(labels.size, dtype=float)
    priors = np.zeros(labels.size, dtype=float)
    for train_index, test_index in folds:
        train_labels = labels[train_index]
        prior = (float(np.sum(train_labels)) + 1.0) / (float(train_labels.size) + 2.0)
        priors[test_index] = prior
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=seed, max_iter=1000),
        )
        estimator.fit(matrix[train_index], train_labels)
        probabilities[test_index] = estimator.predict_proba(matrix[test_index])[:, 1]
    return probabilities, priors


def score_probabilities(labels: np.ndarray, probabilities: np.ndarray, priors: np.ndarray) -> dict:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    predicted = (clipped >= 0.5).astype(int)
    baseline_loss = float(log_loss(labels, priors, labels=[0, 1]))
    model_loss = float(log_loss(labels, clipped, labels=[0, 1]))
    false_positive = int(np.sum((predicted == 1) & (labels == 0)))
    true_negative = int(np.sum((predicted == 0) & (labels == 0)))
    return {
        "information_gain_bits_per_case": (baseline_loss - model_loss) / math.log(2.0),
        "log_loss": model_loss,
        "prior_log_loss": baseline_loss,
        "roc_auc": float(roc_auc_score(labels, clipped)),
        "accuracy_at_0_5": float(accuracy_score(labels, predicted)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predicted)),
        "precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, predicted, zero_division=0)),
        "healthy_false_alarm_count_at_0_5": false_positive,
        "healthy_case_count": false_positive + true_negative,
    }


def permutation_p_value(
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    observed_gain: float,
    *,
    permutations: int,
    seed: int,
) -> float | None:
    if permutations == 0:
        return None
    rng = np.random.default_rng(seed)
    at_least_observed = 0
    completed = 0
    for index in range(permutations):
        permuted = rng.permutation(labels)
        if any(np.unique(permuted[train]).size < 2 for train, _ in folds):
            continue
        probabilities, priors = cross_validated_probabilities(matrix, permuted, folds, seed=seed + index + 1)
        gain = score_probabilities(permuted, probabilities, priors)["information_gain_bits_per_case"]
        at_least_observed += int(gain >= observed_gain)
        completed += 1
    return (at_least_observed + 1.0) / (completed + 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 3 <= args.folds <= 10:
        parser.error("folds must be between 3 and 10")
    if not 0 <= args.permutations <= 2000:
        parser.error("permutations must be between 0 and 2000")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")

    dataset_path = args.data_root / "dataset.csv"
    segments_path = args.data_root / "segments.csv"
    if not dataset_path.is_file() or not segments_path.is_file():
        parser.error("data-root must contain dataset.csv and segments.csv")
    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config_path = ROOT / "configs/evaluation/r20_ops_sat.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    partitions = build_partitions(truth, config)
    partition_by = {row["group_id"]: row["partition"] for row in partitions["groups"]}
    selected = [group_id for group_id in sorted(raw, key=int) if partition_by[group_id] == "DEVELOPMENT_SELECTION"]
    labels = np.asarray([truth[group_id]["anomaly"] for group_id in selected], dtype=int)
    if min(np.bincount(labels)) < args.folds:
        raise ValueError("insufficient class coverage for requested folds")

    rows: list[dict] = []
    matrix_rows: list[list[float]] = []
    frozen_predictions: list[int] = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=selected) as admission:
        for group_id in selected:
            source_sha = _digest(raw[group_id])
            observations = [
                measure_ops_sat_probe(raw[group_id], probe_id, source_sha256=source_sha)
                for probe_id in OPS_SAT_PROBE_IDS
            ]
            matrix_rows.append([float(observation["value"]) for observation in observations])
            frozen = ops_r21_frozen_remediation_prediction(raw[group_id])
            frozen_predictions.append(int(bool(frozen["predicted_anomaly"])))
            rows.append({
                "group_id": group_id,
                "channel": raw[group_id][0]["channel"],
                "sample_count": len(raw[group_id]),
                "segment_sha256": source_sha,
                "observations": observations,
                "frozen_prediction": frozen,
                "offline_truth": "ANOMALOUS" if truth[group_id]["anomaly"] else "NOMINAL",
            })

    matrix = np.asarray(matrix_rows, dtype=float)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = [(train, test) for train, test in splitter.split(matrix, labels)]
    channels = np.asarray([row["channel"] for row in rows])
    channel_folds = [
        (train, test)
        for train, test in LeaveOneGroupOut().split(matrix, labels, groups=channels)
        if np.unique(labels[train]).size == 2
    ]
    covered_channel_indices = sorted(int(index) for _, test in channel_folds for index in test)
    if covered_channel_indices != list(range(labels.size)):
        raise ValueError("leave-one-channel-out folds do not cover every development case")
    candidates: dict[str, dict] = {}
    feature_sets = {probe_id: [index] for index, probe_id in enumerate(OPS_SAT_PROBE_IDS)}
    feature_sets["ALL_REGISTERED_PROBES"] = list(range(len(OPS_SAT_PROBE_IDS)))
    for candidate_index, (candidate_id, columns) in enumerate(feature_sets.items()):
        candidate_matrix = matrix[:, columns]
        probabilities, priors = cross_validated_probabilities(
            candidate_matrix, labels, folds, seed=args.seed
        )
        metrics = score_probabilities(labels, probabilities, priors)
        metrics["permutation_p_value_one_sided"] = permutation_p_value(
            candidate_matrix,
            labels,
            folds,
            metrics["information_gain_bits_per_case"],
            permutations=args.permutations,
            seed=args.seed + 1000 * (candidate_index + 1),
        )
        metrics["probe_ids"] = [OPS_SAT_PROBE_IDS[index] for index in columns]
        channel_probabilities, channel_priors = cross_validated_probabilities(
            candidate_matrix, labels, channel_folds, seed=args.seed
        )
        metrics["leave_one_channel_out"] = score_probabilities(
            labels, channel_probabilities, channel_priors
        )
        candidates[candidate_id] = metrics

    comparison_count = len(candidates)
    for metrics in candidates.values():
        raw_p = metrics["permutation_p_value_one_sided"]
        metrics["permutation_p_value_bonferroni"] = (
            None if raw_p is None else min(1.0, raw_p * comparison_count)
        )

    ranked = sorted(
        candidates,
        key=lambda key: (-candidates[key]["information_gain_bits_per_case"], key),
    )
    ranked_single = [candidate_id for candidate_id in ranked if candidate_id != "ALL_REGISTERED_PROBES"]
    best = candidates[ranked[0]]
    frozen_array = np.asarray(frozen_predictions, dtype=int)
    frozen_false_alarms = int(np.sum((frozen_array == 1) & (labels == 0)))
    report = {
        "schema_version": "astrograph-ops-sat-probe-utility-audit.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "probe_library_version": OPS_SAT_PROBE_LIBRARY_VERSION,
        "preflight_receipt": {
            "status": "PASS",
            "dataset_family": "OPS-SAT",
            "partition": "DEVELOPMENT_SELECTION",
            "purpose": "AUDIT",
            "selected_group_count": len(selected),
            "selected_group_ids_sha256": _digest(selected),
            "class_counts": {
                "NOMINAL": int(np.sum(labels == 0)),
                "ANOMALOUS": int(np.sum(labels == 1)),
            },
            "source_receipts": {
                "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
                "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
                "r20_ops_sat.json": {"sha256": _sha_file(config_path), "size": config_path.stat().st_size},
            },
            "external_validation_groups_consumed": False,
            "llm_invoked": False,
        },
        "dataset_admission": admission,
        "cross_validation": {
            "kind": "STRATIFIED_K_FOLD_OUT_OF_FOLD",
            "fold_count": args.folds,
            "shuffle": True,
            "seed": args.seed,
            "probability_model": "STANDARD_SCALER_PLUS_LIBLINEAR_LOGISTIC_REGRESSION",
            "decision_threshold": 0.5,
            "information_gain_baseline": "LAPLACE_SMOOTHED_TRAIN_FOLD_CLASS_PRIOR",
            "permutation_count": args.permutations,
        },
        "candidate_metrics": candidates,
        "ranking_by_information_gain": ranked,
        "best_candidate_id": ranked[0],
        "best_single_probe_id": ranked_single[0],
        "best_candidate_positive_information_gain": (
            best["information_gain_bits_per_case"] > 0.0
        ),
        "promotion_recommendation": (
            "ELIGIBLE_FOR_FURTHER_DEVELOPMENT_ONLY"
            if best["information_gain_bits_per_case"] > 0.0
            and best["permutation_p_value_bonferroni"] is not None
            and best["permutation_p_value_bonferroni"] <= 0.05
            and best["leave_one_channel_out"]["information_gain_bits_per_case"] > 0.0
            else "NO_RELIABLE_UTILITY_SIGNAL"
        ),
        "single_probe_action_recommendation": (
            "NO_SINGLE_PROBE_PASSES_MULTIPLE_COMPARISON_GATE"
            if not any(
                candidates[candidate_id]["information_gain_bits_per_case"] > 0.0
                and candidates[candidate_id]["permutation_p_value_bonferroni"] is not None
                and candidates[candidate_id]["permutation_p_value_bonferroni"] <= 0.05
                and candidates[candidate_id]["leave_one_channel_out"]["information_gain_bits_per_case"] > 0.0
                for candidate_id in ranked_single
            )
            else "ONE_OR_MORE_SINGLE_PROBES_ELIGIBLE_FOR_FURTHER_DEVELOPMENT"
        ),
        "frozen_detector_reference": {
            "case_count": len(selected),
            "accuracy": float(accuracy_score(labels, frozen_array)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, frozen_array)),
            "f1": float(f1_score(labels, frozen_array, zero_division=0)),
            "healthy_false_alarm_count": frozen_false_alarms,
            "healthy_case_count": int(np.sum(labels == 0)),
            "comparison_boundary": "Deterministic predictions only; no calibrated probabilities or information-gain score.",
        },
        "cases": rows,
        "claim_boundary": (
            "Development-selection, internally cross-validated probe utility audit on a previously seen dataset. "
            "Probe ranking used offline labels and is not an independent validation, LLM result, root-cause result, "
            "formal EvidenceFact, production qualification, safety claim, or flight claim."
        ),
    }
    report["report_sha256"] = _digest(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "best_candidate_id": report["best_candidate_id"],
        "candidate_metrics": candidates,
        "frozen_detector_reference": report["frozen_detector_reference"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
