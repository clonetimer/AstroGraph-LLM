#!/usr/bin/env python3
"""A11 nested-CV deterministic comparator for the A10 OPS-SAT bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.ops_sat_probe_utility import OPS_SAT_PROBE_IDS, measure_ops_sat_probe_bundle
from scripts.experiments.run_ops_sat_llm_active_shadow import _digest, _sha_file
from scripts.experiments.run_ops_sat_probe_utility_audit import (
    cross_validated_probabilities,
    score_probabilities,
)


def select_f05_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    """Select a precision-weighted threshold; ties prefer fewer FP then a higher threshold."""
    candidates = sorted({0.0, 0.5, math.nextafter(1.0, 2.0), *map(float, probabilities)})
    best_key: tuple[float, int, float] | None = None
    best_threshold = 0.5
    best_metrics: dict = {}
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        true_positive = int(np.sum((predicted == 1) & (labels == 1)))
        false_positive = int(np.sum((predicted == 1) & (labels == 0)))
        false_negative = int(np.sum((predicted == 0) & (labels == 1)))
        beta2 = 0.25
        denominator = (1.0 + beta2) * true_positive + beta2 * false_negative + false_positive
        f05 = 0.0 if denominator == 0 else (1.0 + beta2) * true_positive / denominator
        key = (f05, -false_positive, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = {
                "f0_5": f05,
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
            }
    return float(best_threshold), best_metrics


def nested_oof_policy(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 42,
    feature_names: tuple[str, ...] | None = None,
    feature_sets: dict[str, list[int]] | None = None,
) -> dict:
    """Select probe subset and decision threshold inside each outer training fold."""
    names = feature_names or OPS_SAT_PROBE_IDS
    if matrix.shape != (labels.size, len(names)):
        raise ValueError("OPS bundle matrix shape does not match the declared feature names")
    if feature_sets is None:
        candidate_sets = {probe_id: [index] for index, probe_id in enumerate(OPS_SAT_PROBE_IDS)}
        candidate_sets["ALL_REGISTERED_PROBES"] = list(range(len(OPS_SAT_PROBE_IDS)))
    else:
        candidate_sets = feature_sets
    if not candidate_sets or any(
        not columns or min(columns) < 0 or max(columns) >= len(names)
        for columns in candidate_sets.values()
    ):
        raise ValueError("candidate feature sets must contain valid declared columns")
    probabilities = np.zeros(labels.size, dtype=float)
    thresholds = np.zeros(labels.size, dtype=float)
    selected_ids = np.empty(labels.size, dtype=object)
    fold_receipts: list[dict] = []
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold_index, (train_index, test_index) in enumerate(outer.split(matrix, labels)):
        train_labels = labels[train_index]
        inner_splitter = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + 100 + fold_index,
        )
        inner = [(train, test) for train, test in inner_splitter.split(matrix[train_index], train_labels)]
        candidate_receipts: dict[str, dict] = {}
        for candidate_id, columns in candidate_sets.items():
            inner_probabilities, inner_priors = cross_validated_probabilities(
                matrix[train_index][:, columns], train_labels, inner, seed=seed + fold_index
            )
            metrics = score_probabilities(train_labels, inner_probabilities, inner_priors)
            candidate_receipts[candidate_id] = {
                "columns": columns,
                "information_gain_bits_per_case": metrics["information_gain_bits_per_case"],
                "inner_probabilities": inner_probabilities,
            }
        selected = min(
            candidate_receipts,
            key=lambda candidate_id: (
                -candidate_receipts[candidate_id]["information_gain_bits_per_case"],
                len(candidate_receipts[candidate_id]["columns"]),
                candidate_id,
            ),
        )
        receipt = candidate_receipts[selected]
        threshold, threshold_metrics = select_f05_threshold(train_labels, receipt["inner_probabilities"])
        columns = receipt["columns"]
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=seed + fold_index, max_iter=1000),
        )
        estimator.fit(matrix[train_index][:, columns], train_labels)
        probabilities[test_index] = estimator.predict_proba(matrix[test_index][:, columns])[:, 1]
        thresholds[test_index] = threshold
        selected_ids[test_index] = selected
        fold_receipts.append({
            "fold_index": fold_index,
            "train_case_count": int(train_index.size),
            "test_case_count": int(test_index.size),
            "selected_candidate_id": selected,
            "selected_feature_ids": [names[index] for index in columns],
            "selected_probe_ids": [names[index] for index in columns if names[index] in OPS_SAT_PROBE_IDS],
            "inner_information_gain_bits_per_case": receipt["information_gain_bits_per_case"],
            "selected_threshold": threshold,
            "inner_threshold_metrics": threshold_metrics,
            "test_group_indices_sha256": hashlib.sha256(
                json.dumps(test_index.tolist(), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
    return {
        "probabilities": probabilities,
        "thresholds": thresholds,
        "predictions": (probabilities >= thresholds).astype(int),
        "selected_candidate_ids": selected_ids,
        "fold_receipts": fold_receipts,
    }


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    false_positive = int(np.sum((predictions == 1) & (labels == 0)))
    balanced = (
        float(balanced_accuracy_score(labels, predictions))
        if np.unique(labels).size == 2 else None
    )
    return {
        "case_count": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": balanced,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "healthy_false_alarm_count": false_positive,
        "healthy_case_count": int(np.sum(labels == 0)),
        "abstain_or_ambiguous_count": 0,
    }


def exact_mcnemar_two_sided(a11_only_correct: int, a10_only_correct: int) -> float:
    discordant = a11_only_correct + a10_only_correct
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(a11_only_correct, a10_only_correct) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--utility-report", type=Path, required=True)
    parser.add_argument("--a10-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel-conditioned", action="store_true")
    parser.add_argument("--global-policy-report", type=Path)
    args = parser.parse_args()
    if not 3 <= args.outer_folds <= 10 or not 3 <= args.inner_folds <= 10:
        parser.error("outer-folds and inner-folds must be between 3 and 10")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    if args.channel_conditioned != (args.global_policy_report is not None):
        parser.error("--global-policy-report is required exactly when --channel-conditioned is enabled")

    utility = json.loads(args.utility_report.read_text(encoding="utf-8"))
    a10 = json.loads(args.a10_report.read_text(encoding="utf-8"))
    global_policy = (
        json.loads(args.global_policy_report.read_text(encoding="utf-8"))
        if args.global_policy_report is not None else None
    )
    if utility.get("schema_version") != "astrograph-ops-sat-probe-utility-audit.v1":
        parser.error("unsupported utility report schema")
    if a10.get("schema_version") != "astrograph-ops-sat-compact-bundle-comparator.v1":
        parser.error("unsupported A10 report schema")
    for document, name in ((utility, "utility"), (a10, "A10")):
        content = dict(document)
        claimed = content.pop("report_sha256", None)
        if not isinstance(claimed, str) or _digest(content) != claimed:
            raise ValueError(f"{name} report content digest mismatch")
    if global_policy is not None:
        if global_policy.get("schema_version") != "astrograph-ops-sat-deterministic-bundle-comparator.v1":
            parser.error("unsupported global policy report schema")
        content = dict(global_policy)
        claimed = content.pop("report_sha256", None)
        if not isinstance(claimed, str) or _digest(content) != claimed:
            raise ValueError("global policy report content digest mismatch")

    dataset_path = args.data_root / "dataset.csv"
    segments_path = args.data_root / "segments.csv"
    receipts = {
        "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
        "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
    }
    utility_receipts = utility["preflight_receipt"]["source_receipts"]
    a10_receipts = a10["preflight_receipt"]["source_receipts"]
    if any(receipts[name] != utility_receipts[name] or receipts[name] != a10_receipts[name] for name in receipts):
        raise ValueError("A11 source identity differs from its bound reports")

    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partition_by = {row["group_id"]: row["partition"] for row in build_partitions(truth, config)["groups"]}
    development_ids = [group_id for group_id in sorted(raw, key=int) if partition_by[group_id] == "DEVELOPMENT_SELECTION"]
    selected_ids = list(a10["preflight_receipt"]["selected_group_ids"])
    if not set(selected_ids) <= set(development_ids):
        raise ValueError("A10 cases are outside A11 development partition")

    rows: list[dict] = []
    matrix_rows: list[list[float]] = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=development_ids) as admission:
        for group_id in development_ids:
            source_sha = _digest(raw[group_id])
            bundle = measure_ops_sat_probe_bundle(raw[group_id], source_sha256=source_sha)
            matrix_rows.append([float(component["value"]) for component in bundle["components"]])
            rows.append({
                "group_id": group_id,
                "channel": raw[group_id][0]["channel"],
                "segment_sha256": source_sha,
                "bundle_observation_id": bundle["observation_id"],
                "offline_truth": "ANOMALOUS" if truth[group_id]["anomaly"] else "NOMINAL",
            })
    labels = np.asarray([truth[group_id]["anomaly"] for group_id in development_ids], dtype=int)
    matrix = np.asarray(matrix_rows, dtype=float)
    feature_names: tuple[str, ...] = OPS_SAT_PROBE_IDS
    candidate_sets: dict[str, list[int]] | None = None
    if args.channel_conditioned:
        channel_ids = sorted({row["channel"] for row in rows})
        one_hot = np.asarray([
            [float(row["channel"] == channel_id) for channel_id in channel_ids]
            for row in rows
        ], dtype=float)
        matrix = np.column_stack([matrix, one_hot])
        feature_names = (*OPS_SAT_PROBE_IDS, *(f"channel::{channel_id}" for channel_id in channel_ids))
        channel_columns = list(range(len(OPS_SAT_PROBE_IDS), len(feature_names)))
        candidate_sets = {
            probe_id: [index] for index, probe_id in enumerate(OPS_SAT_PROBE_IDS)
        }
        candidate_sets["ALL_REGISTERED_PROBES"] = list(range(len(OPS_SAT_PROBE_IDS)))
        candidate_sets["CHANNEL_INTERCEPTS_ONLY"] = channel_columns
        for index, probe_id in enumerate(OPS_SAT_PROBE_IDS):
            candidate_sets[f"CHANNEL_PLUS_{probe_id}"] = [index, *channel_columns]
        candidate_sets["CHANNEL_PLUS_ALL_REGISTERED_PROBES"] = [
            *range(len(OPS_SAT_PROBE_IDS)), *channel_columns,
        ]
    policy = nested_oof_policy(
        matrix, labels,
        outer_folds=args.outer_folds, inner_folds=args.inner_folds, seed=args.seed,
        feature_names=feature_names, feature_sets=candidate_sets,
    )
    for index, row in enumerate(rows):
        predicted = int(policy["predictions"][index])
        row.update({
            "out_of_fold_probability": float(policy["probabilities"][index]),
            "fold_selected_threshold": float(policy["thresholds"][index]),
            "fold_selected_candidate_id": str(policy["selected_candidate_ids"][index]),
            "validated_candidate_states": ["ANOMALOUS" if predicted else "NOMINAL"],
        })

    selected_rows = [row for row in rows if row["group_id"] in set(selected_ids)]
    selected_rows.sort(key=lambda row: selected_ids.index(row["group_id"]))
    a10_by_group = {row["group_id"]: row for row in a10["cases"]}
    if any(
        row["bundle_observation_id"] != a10_by_group[row["group_id"]]["arm"]["observation"]["observation_id"]
        for row in selected_rows
    ):
        raise ValueError("A11 bundle identity differs from A10")

    full_metrics = _classification_metrics(labels, policy["predictions"])
    selected_labels = np.asarray([truth[row["group_id"]]["anomaly"] for row in selected_rows], dtype=int)
    selected_predictions = np.asarray([
        int(row["validated_candidate_states"] == ["ANOMALOUS"]) for row in selected_rows
    ], dtype=int)
    selected_metrics = _classification_metrics(selected_labels, selected_predictions)
    a10_summary = a10["a10_compact_bundle"]
    a11_only_correct = 0
    a10_only_correct = 0
    for row, label, prediction in zip(selected_rows, selected_labels, selected_predictions):
        a11_correct = int(prediction) == int(label)
        a10_row = a10_by_group[row["group_id"]]["arm"]
        a10_correct = (
            len(a10_row["validated_candidate_states"]) == 1
            and not a10_row["revision"]["abstain_recommended"]
            and a10_row["validated_candidate_states"][0]
            == ("ANOMALOUS" if int(label) else "NOMINAL")
        )
        a11_only_correct += int(a11_correct and not a10_correct)
        a10_only_correct += int(a10_correct and not a11_correct)
    channel_metrics: dict[str, dict] = {}
    channels = sorted({row["channel"] for row in rows})
    for channel in channels:
        indices = np.asarray([index for index, row in enumerate(rows) if row["channel"] == channel], dtype=int)
        channel_metrics[channel] = {
            **_classification_metrics(labels[indices], policy["predictions"][indices]),
            "nominal_case_count": int(np.sum(labels[indices] == 0)),
            "anomalous_case_count": int(np.sum(labels[indices] == 1)),
        }
    frozen = utility["frozen_detector_reference"]
    fold_frequency = Counter(receipt["selected_candidate_id"] for receipt in policy["fold_receipts"])
    case_frequency = Counter(map(str, policy["selected_candidate_ids"]))
    report = {
        "schema_version": (
            "astrograph-ops-sat-channel-conditioned-bundle-comparator.v1"
            if args.channel_conditioned else
            "astrograph-ops-sat-deterministic-bundle-comparator.v1"
        ),
        "comparator_id": "A12_CHANNEL_CONDITIONED_BUNDLE" if args.channel_conditioned else "A11_GLOBAL_BUNDLE",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "preflight_receipt": {
            "status": "PASS",
            "source_receipts": receipts,
            "utility_report_file_sha256": _sha_file(args.utility_report),
            "utility_report_content_sha256": utility["report_sha256"],
            "a10_report_file_sha256": _sha_file(args.a10_report),
            "a10_report_content_sha256": a10["report_sha256"],
            **({
                "global_policy_report_file_sha256": _sha_file(args.global_policy_report),
                "global_policy_report_content_sha256": global_policy["report_sha256"],
            } if global_policy is not None and args.global_policy_report is not None else {}),
            "development_group_count": len(development_ids),
            "paired_a10_group_ids": selected_ids,
            "external_validation_groups_consumed": False,
            "llm_invoked": False,
            "channel_conditioning_enabled": args.channel_conditioned,
        },
        "dataset_admission": admission,
        "policy": {
            "kind": "NESTED_STRATIFIED_CV_PROBE_SELECTION_AND_F0_5_THRESHOLD",
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": args.seed,
            "channel_conditioning_enabled": args.channel_conditioned,
            "declared_feature_ids": list(feature_names),
            "feature_selection_objective": "INNER_OOF_INFORMATION_GAIN_BITS_PER_CASE",
            "threshold_objective": "INNER_OOF_F0_5_TIE_LOWER_FP_THEN_HIGHER_THRESHOLD",
            "fold_receipts": policy["fold_receipts"],
            "fold_selected_candidate_frequency": dict(fold_frequency),
            "case_selected_candidate_frequency": dict(case_frequency),
        },
        "all_development_metrics": full_metrics,
        "per_channel_metrics": channel_metrics,
        "paired_a10_metrics": selected_metrics,
        "paired_a10_cases": selected_rows,
        "comparison_to_a10": {
            "accuracy_delta_policy_minus_a10": selected_metrics["accuracy"] - a10_summary["accuracy_with_abstention_as_incorrect"],
            "healthy_false_alarm_delta_policy_minus_a10": selected_metrics["healthy_false_alarm_count"] - a10_summary["healthy_false_alarm_count"],
            "abstain_or_ambiguous_delta_policy_minus_a10": -a10_summary["abstain_or_ambiguous_count"],
            "same_bundle_observation_ids": True,
            "same_tool_action_count": True,
            "llm_call_delta_policy_minus_a10": -a10_summary["llm_call_count"],
            "policy_only_correct_count": a11_only_correct,
            "a10_only_correct_count": a10_only_correct,
            "paired_exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(
                a11_only_correct, a10_only_correct
            ),
            "comparison_boundary": "Paired mechanism comparison; A10 and deterministic-policy inference and compute costs differ.",
        },
        **({
            "comparison_to_global_a11": {
                "accuracy_delta_a12_minus_a11": full_metrics["accuracy"] - global_policy["all_development_metrics"]["accuracy"],
                "balanced_accuracy_delta_a12_minus_a11": full_metrics["balanced_accuracy"] - global_policy["all_development_metrics"]["balanced_accuracy"],
                "precision_delta_a12_minus_a11": full_metrics["precision"] - global_policy["all_development_metrics"]["precision"],
                "recall_delta_a12_minus_a11": full_metrics["recall"] - global_policy["all_development_metrics"]["recall"],
                "f1_delta_a12_minus_a11": full_metrics["f1"] - global_policy["all_development_metrics"]["f1"],
                "healthy_false_alarm_delta_a12_minus_a11": full_metrics["healthy_false_alarm_count"] - global_policy["all_development_metrics"]["healthy_false_alarm_count"],
                "verdict": "CHANNEL_CONDITIONING_REJECTED" if (
                    full_metrics["balanced_accuracy"] <= global_policy["all_development_metrics"]["balanced_accuracy"]
                    or full_metrics["healthy_false_alarm_count"] > global_policy["all_development_metrics"]["healthy_false_alarm_count"]
                ) else "CHANNEL_CONDITIONING_ELIGIBLE_FOR_FURTHER_DEVELOPMENT",
            }
        } if global_policy is not None else {}),
        "comparison_to_frozen_detector": {
            "accuracy_delta_policy_minus_frozen": full_metrics["accuracy"] - frozen["accuracy"],
            "balanced_accuracy_delta_policy_minus_frozen": (
                full_metrics["balanced_accuracy"] - frozen["balanced_accuracy"]
            ),
            "f1_delta_policy_minus_frozen": full_metrics["f1"] - frozen["f1"],
            "healthy_false_alarm_delta_policy_minus_frozen": (
                full_metrics["healthy_false_alarm_count"] - frozen["healthy_false_alarm_count"]
            ),
            "comparison_boundary": (
                "Same development cases but different operating points; accuracy gain is dominated by fewer healthy "
                "false alarms and does not imply higher anomaly sensitivity."
            ),
        },
        "claim_boundary": (
            "Development-only nested cross-validation on a previously seen dataset. Feature and threshold choices are "
            "made inside outer training folds, but this is not external validation, root-cause diagnosis, formal "
            "evidence, production qualification, or a flight claim."
        ),
    }
    report["report_sha256"] = _digest(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "all_development_metrics": full_metrics,
        "paired_a10_metrics": selected_metrics,
        "comparison_to_a10": report["comparison_to_a10"],
        **({"comparison_to_global_a11": report["comparison_to_global_a11"]} if global_policy is not None else {}),
        "fold_selected_candidate_frequency": report["policy"]["fold_selected_candidate_frequency"],
        "comparison_to_frozen_detector": report["comparison_to_frozen_detector"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
