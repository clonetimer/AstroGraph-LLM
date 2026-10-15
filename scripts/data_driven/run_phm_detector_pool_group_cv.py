#!/usr/bin/env python3
"""Five-fold PHM Test audit for a Validation-selected detector family pool."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_phm_causal_windows as phm
from scripts.data_driven.run_hierarchical_group_cv import rotating_split
from astrograph.data_driven.alarm_policy import AlarmPolicy, alarm_event_metrics, apply_alarm_policy
from astrograph.data_driven.phm_v60 import audit_split, load_records


CANDIDATES = {
    "residual_legacy": ("residual_tcn", "legacy_summary", "detection", 0.0, None, 1.0, 0.0, 0.2),
    "multiview_legacy_logistic": (
        "causal_multiview_detector_tcn", "legacy_summary", "detection", 0.0, None, 1.0, 0.0, 0.2
    ),
    "multiview_aux_0_1": (
        "causal_multiview_detector_tcn", "legacy_summary", "detection_aux", 0.1, None, 1.0, 0.0, 0.2
    ),
    "multiview_aux_fixed_0_6": (
        "causal_multiview_detector_tcn", "legacy_summary", "detection_aux", 0.1, 0.6, 1.0, 0.0, 0.2
    ),
    "multiview_aux_nominal_2": (
        "causal_multiview_detector_tcn", "legacy_summary", "detection_aux", 0.1, None, 2.0, 0.0, 0.2
    ),
    "multiview_aux_nominal_4": (
        "causal_multiview_detector_tcn", "legacy_summary", "detection_aux", 0.1, None, 4.0, 0.0, 0.2
    ),
    "multiview_aux_nominal_4_cvar": (
        "causal_multiview_detector_tcn", "legacy_summary", "detection_aux", 0.1, None, 4.0, 0.25, 0.2
    ),
}
DEFAULT_CANDIDATES = tuple(name for name in CANDIDATES if "nominal_" not in name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=phm.DEFAULT_DATASET)
    parser.add_argument(
        "--candidates", nargs="+", choices=sorted(CANDIDATES), default=list(DEFAULT_CANDIDATES)
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--channel-limit", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label-contract", choices=["event_active", "manifestation_aware"], default="manifestation_aware")
    parser.add_argument("--training-sampling", choices=["window_shuffle", "trajectory_uniform"], default="trajectory_uniform")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = json.loads(phm.HIERARCHY.read_text(encoding="utf-8"))
    source = load_records(args.dataset_root)
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    folds = []
    test_scores = []
    for test_fold in range(5):
        records = [replace(row, split=rotating_split(row.sample_index, test_fold)) for row in source]
        split_audit = audit_split(records)
        if split_audit["status"] != "PASS":
            raise RuntimeError(split_audit)
        selected, median = phm.select_train_channels(
            records, class_names, args.channel_limit, label_contract=args.label_contract
        )
        windows = phm.scale_windows(phm.build_windows(
            records, selected, class_names, component_names, hierarchy["fault_to_component"],
            label_contract=args.label_contract,
        ), median)
        seed = args.seed + test_fold
        trained = []
        for candidate in args.candidates:
            (
                model_name, feature_profile, task_scope, auxiliary_weight,
                fixed_threshold, nominal_weight, cvar_weight, cvar_fraction,
            ) = CANDIDATES[candidate]
            print(f"[detector-pool] fold={test_fold} candidate={candidate}", flush=True)
            training, runtime = phm.train_one(
                model_name, windows["train"], windows["validation"], windows["test"],
                class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
                selected_channels=selected, device=args.device, batch_size=args.batch_size,
                minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
                evaluate_test_split=False, label_contract=args.label_contract,
                training_sampling=args.training_sampling, task_scope=task_scope,
                detector_feature_profile=feature_profile, return_runtime=True,
                diagnosis_auxiliary_weight=auxiliary_weight,
                fixed_detection_threshold=fixed_threshold,
                nominal_trajectory_weight=nominal_weight,
                detection_cvar_weight=cvar_weight,
                detection_cvar_fraction=cvar_fraction,
            )
            trained.append((candidate, feature_profile, training, runtime))
        chosen = max(
            trained,
            key=lambda item: (
                float(item[2]["best_validation"]["detection_macro_f1"]),
                -args.candidates.index(item[0]),
            ),
        )
        candidate, feature_profile, training, runtime = chosen
        selection = training["best_validation"]
        deep_score = phm.infer(runtime["model"], windows["test"], device=args.device)[0][:, 1]
        logistic_score = phm.logistic_detector(
            windows["train"], windows["test"], seed, feature_profile=feature_profile
        )
        score = logistic_score if selection["detector"] == "logistic" else deep_score
        policy = AlarmPolicy(
            on_threshold=float(selection["alarm_on_threshold"]),
            off_threshold=float(selection["alarm_off_threshold"]),
            minimum_consecutive=int(selection["alarm_minimum_consecutive"]),
        )
        predicted = np.asarray(apply_alarm_policy(
            score.tolist(), windows["test"].sample_ids,
            windows["test"].end_times_s.tolist(), policy,
        ))
        detection_f1 = phm.macro_f1(windows["test"].detection, predicted, [0, 1])
        injection_f1 = phm.macro_f1(windows["test"].injection_detection, predicted, [0, 1])
        test_scores.append(detection_f1)
        folds.append({
            "test_fold": test_fold,
            "validation_fold": (test_fold + 1) % 5,
            "train_folds": sorted(set(range(5)) - {test_fold, (test_fold + 1) % 5}),
            "split_audit": split_audit,
            "candidate_validation_scores": {
                item[0]: item[2]["best_validation"]["detection_macro_f1"] for item in trained
            },
            "selected_candidate": candidate,
            "selected_detector": selection["detector"],
            "selected_threshold": selection["detection_threshold"],
            "selected_training": training,
            "test": {
                "detection_macro_f1": detection_f1,
                "injection_awareness_macro_f1": injection_f1,
                "delay": phm.delays(windows["test"], score, predicted=predicted),
                "alarm_event_metrics": alarm_event_metrics(
                    windows["test"].detection.tolist(), predicted.tolist(),
                    windows["test"].sample_ids, windows["test"].end_times_s.tolist(),
                    windows["test"].onset_times_s.tolist(),
                ),
            },
        })
    report = {
        "schema_version": "astrograph-phm-detector-pool-group-cv.v1",
        "protocol": {
            "candidate_pool": args.candidates,
            "selection": "model family, checkpoint, detector source and threshold selected on Validation; Test evaluated once",
            "label_contract": args.label_contract,
            "training_sampling": args.training_sampling,
            "group_key": "sample_index shared across subsystems and phases",
        },
        "folds": folds,
        "aggregate": {
            "fold_count": len(test_scores),
            "detection_macro_f1_mean": statistics.fmean(test_scores),
            "detection_macro_f1_population_std": statistics.pstdev(test_scores),
            "detection_macro_f1_minimum": min(test_scores),
            "detection_macro_f1_maximum": max(test_scores),
            "detection_macro_f1_values": test_scores,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
