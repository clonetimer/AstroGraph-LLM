#!/usr/bin/env python3
"""A13 audit of newly inspected official OPS-SAT derived-feature observations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import FEATURE_FIELDS, build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.ops_sat_probe_utility import (
    OPS_SAT_LENGTH_FREE_FEATURE_IDS,
    OPS_SAT_OFFICIAL_FEATURE_BUNDLE_VERSION,
    measure_ops_sat_official_feature_bundle,
)
from scripts.experiments.run_ops_sat_deterministic_bundle_comparator import (
    _classification_metrics,
    nested_oof_policy,
)
from scripts.experiments.run_ops_sat_llm_active_shadow import _digest, _sha_file


def _policy(
    matrix: np.ndarray,
    labels: np.ndarray,
    names: tuple[str, ...],
    all_candidate_id: str,
    *,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict:
    feature_sets = {name: [index] for index, name in enumerate(names)}
    feature_sets[all_candidate_id] = list(range(len(names)))
    return nested_oof_policy(
        matrix,
        labels,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        feature_names=names,
        feature_sets=feature_sets,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-policy-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 3 <= args.outer_folds <= 10 or not 3 <= args.inner_folds <= 10:
        parser.error("outer-folds and inner-folds must be between 3 and 10")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")

    raw_policy = json.loads(args.raw_policy_report.read_text(encoding="utf-8"))
    if raw_policy.get("schema_version") != "astrograph-ops-sat-deterministic-bundle-comparator.v1":
        parser.error("unsupported raw-policy report schema")
    content = dict(raw_policy)
    claimed = content.pop("report_sha256", None)
    if not isinstance(claimed, str) or _digest(content) != claimed:
        raise ValueError("raw-policy report content digest mismatch")

    dataset_path = args.data_root / "dataset.csv"
    segments_path = args.data_root / "segments.csv"
    receipts = {
        "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
        "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
    }
    raw_receipts = raw_policy["preflight_receipt"]["source_receipts"]
    if any(receipts[name] != raw_receipts[name] for name in receipts):
        raise ValueError("A13 source identity differs from A11")

    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partition_by = {row["group_id"]: row["partition"] for row in build_partitions(truth, config)["groups"]}
    development_ids = [group_id for group_id in sorted(feature, key=int) if partition_by[group_id] == "DEVELOPMENT_SELECTION"]
    labels = np.asarray([truth[group_id]["anomaly"] for group_id in development_ids], dtype=int)

    observations: list[dict] = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=development_ids) as admission:
        for group_id in development_ids:
            observation = measure_ops_sat_official_feature_bundle(
                feature[group_id], source_sha256=receipts["dataset.csv"]["sha256"]
            )
            observations.append(observation)

    full_names = tuple(FEATURE_FIELDS)
    length_free_names = tuple(OPS_SAT_LENGTH_FREE_FEATURE_IDS)
    full_matrix = np.asarray([
        [float(observation["values"][name]) for name in full_names] for observation in observations
    ])
    length_free_matrix = np.asarray([
        [float(observation["values"][name]) for name in length_free_names] for observation in observations
    ])
    full = _policy(
        full_matrix, labels, full_names, "ALL_OFFICIAL_FEATURES",
        outer_folds=args.outer_folds, inner_folds=args.inner_folds, seed=args.seed,
    )
    length_free = _policy(
        length_free_matrix, labels, length_free_names, "ALL_LENGTH_FREE_FEATURES",
        outer_folds=args.outer_folds, inner_folds=args.inner_folds, seed=args.seed,
    )
    full_metrics = _classification_metrics(labels, full["predictions"])
    length_free_metrics = _classification_metrics(labels, length_free["predictions"])
    raw_metrics = raw_policy["all_development_metrics"]

    paired_ids = list(raw_policy["preflight_receipt"]["paired_a10_group_ids"])
    index_by_id = {group_id: index for index, group_id in enumerate(development_ids)}
    paired_indices = np.asarray([index_by_id[group_id] for group_id in paired_ids], dtype=int)
    paired_metrics = _classification_metrics(labels[paired_indices], length_free["predictions"][paired_indices])
    cases = []
    for index, (group_id, observation) in enumerate(zip(development_ids, observations)):
        cases.append({
            "group_id": group_id,
            "channel": feature[group_id]["channel"],
            "observation_id": observation["observation_id"],
            "row_sha256": observation["row_sha256"],
            "offline_truth": "ANOMALOUS" if labels[index] else "NOMINAL",
            "full_policy": {
                "out_of_fold_probability": float(full["probabilities"][index]),
                "fold_selected_threshold": float(full["thresholds"][index]),
                "fold_selected_candidate_id": str(full["selected_candidate_ids"][index]),
                "predicted_state": "ANOMALOUS" if full["predictions"][index] else "NOMINAL",
            },
            "length_free_policy": {
                "out_of_fold_probability": float(length_free["probabilities"][index]),
                "fold_selected_threshold": float(length_free["thresholds"][index]),
                "fold_selected_candidate_id": str(length_free["selected_candidate_ids"][index]),
                "predicted_state": "ANOMALOUS" if length_free["predictions"][index] else "NOMINAL",
            },
        })

    eligible = (
        length_free_metrics["balanced_accuracy"] > raw_metrics["balanced_accuracy"]
        and length_free_metrics["recall"] > raw_metrics["recall"]
        and length_free_metrics["healthy_false_alarm_count"] <= 10
    )
    report = {
        "schema_version": "astrograph-ops-sat-official-feature-observation-audit.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "observation_bundle_version": OPS_SAT_OFFICIAL_FEATURE_BUNDLE_VERSION,
        "preflight_receipt": {
            "status": "PASS",
            "source_receipts": receipts,
            "raw_policy_report_file_sha256": _sha_file(args.raw_policy_report),
            "raw_policy_report_content_sha256": raw_policy["report_sha256"],
            "development_group_count": len(development_ids),
            "external_validation_groups_consumed": False,
            "llm_invoked": False,
            "observation_kind": "NEWLY_INSPECTED_OFFICIAL_DERIVED_FEATURES_NOT_NEW_RAW_SAMPLES",
        },
        "dataset_admission": admission,
        "excluded_process_sensitive_feature_ids": sorted(set(full_names) - set(length_free_names)),
        "full_feature_policy": {
            "metrics": full_metrics,
            "fold_receipts": full["fold_receipts"],
            "fold_selected_candidate_frequency": dict(Counter(
                receipt["selected_candidate_id"] for receipt in full["fold_receipts"]
            )),
            "interpretation": "SENSITIVITY_ONLY_INCLUDES_LENGTH_AND_DURATION_RELATED_FEATURES",
        },
        "length_free_policy": {
            "feature_ids": list(length_free_names),
            "metrics": length_free_metrics,
            "paired_a10_group_metrics": paired_metrics,
            "fold_receipts": length_free["fold_receipts"],
            "fold_selected_candidate_frequency": dict(Counter(
                receipt["selected_candidate_id"] for receipt in length_free["fold_receipts"]
            )),
            "promotion_recommendation": (
                "ELIGIBLE_FOR_LLM_SHADOW_EXPERIMENT_ONLY"
                if eligible else "NO_RELIABLE_REPRESENTATION_GAIN"
            ),
        },
        "comparison_to_raw_probe_a11": {
            "balanced_accuracy_delta": length_free_metrics["balanced_accuracy"] - raw_metrics["balanced_accuracy"],
            "recall_delta": length_free_metrics["recall"] - raw_metrics["recall"],
            "f1_delta": length_free_metrics["f1"] - raw_metrics["f1"],
            "healthy_false_alarm_delta": (
                length_free_metrics["healthy_false_alarm_count"] - raw_metrics["healthy_false_alarm_count"]
            ),
            "comparison_boundary": (
                "Same development groups and nested protocol, but A13 reads a different derived observation source."
            ),
        },
        "cases": cases,
        "claim_boundary": (
            "Development-only retrospective full-segment feature audit. The feature export is an admitted second "
            "source and may encode dataset construction effects; no external validation, early detection, root-cause "
            "diagnosis, formal evidence, production qualification, or flight claim."
        ),
    }
    report["report_sha256"] = _digest(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "full_feature_metrics": full_metrics,
        "length_free_metrics": length_free_metrics,
        "paired_a10_group_metrics": paired_metrics,
        "comparison_to_raw_probe_a11": report["comparison_to_raw_probe_a11"],
        "promotion_recommendation": report["length_free_policy"]["promotion_recommendation"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
