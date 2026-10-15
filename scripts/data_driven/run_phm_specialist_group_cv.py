#!/usr/bin/env python3
"""Five-fold PHM Test audit for frozen heterogeneous task specialists."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_phm_causal_windows as phm
from scripts.data_driven.run_hierarchical_group_cv import rotating_split
from astrograph.data_driven.phm_v60 import audit_split, load_records
from astrograph.data_driven.sota_backbones import SOTA_MODEL_FAMILIES


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "detection_macro_f1": ("detection", "macro_f1"),
        "localization_macro_f1": ("localization", "macro_f1"),
        "classification_macro_f1": ("classification_predicted_component", "macro_f1"),
        "end_to_end_macro_f1": ("end_to_end_18_class", "macro_f1"),
    }
    result = {"fold_count": len(rows)}
    for name, path in paths.items():
        values = [float(row[path[0]][path[1]]) for row in rows]
        result[name] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=phm.DEFAULT_DATASET)
    parser.add_argument(
        "--detector-model", default="residual_tcn",
        choices=["residual_tcn", "causal_multiview_detector_tcn", *SOTA_MODEL_FAMILIES],
    )
    parser.add_argument(
        "--detector-feature-profile", default="legacy_summary",
        choices=["legacy_summary", "change_summary"],
    )
    parser.add_argument(
        "--detector-task-scope", default="detection",
        choices=["detection", "detection_aux"],
    )
    parser.add_argument("--detector-diagnosis-auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--detector-nominal-trajectory-weight", type=float, default=1.0)
    parser.add_argument("--detector-cvar-weight", type=float, default=0.0)
    parser.add_argument("--detector-cvar-fraction", type=float, default=0.2)
    parser.add_argument(
        "--diagnosis-model", default="hierarchical_mechanism_gat_tcn",
        choices=["residual_tcn", "hierarchical_mechanism_gat_tcn", *SOTA_MODEL_FAMILIES],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--diagnosis-epochs", type=int, default=None)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--channel-limit", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--label-contract", choices=["event_active", "manifestation_aware"],
        default="manifestation_aware",
    )
    parser.add_argument(
        "--training-sampling", choices=["window_shuffle", "trajectory_uniform"],
        default="trajectory_uniform",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = json.loads(phm.HIERARCHY.read_text(encoding="utf-8"))
    source = load_records(args.dataset_root)
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    folds = []
    results = []
    for test_fold in range(5):
        records = [
            replace(row, split=rotating_split(row.sample_index, test_fold)) for row in source
        ]
        split_audit = audit_split(records)
        if split_audit["status"] != "PASS":
            raise RuntimeError(split_audit)
        selected, median = phm.select_train_channels(
            records, class_names, args.channel_limit, label_contract=args.label_contract
        )
        windows = phm.scale_windows(
            phm.build_windows(
                records, selected, class_names, component_names,
                hierarchy["fault_to_component"], label_contract=args.label_contract,
            ),
            median,
        )
        seed = args.seed + test_fold
        print(f"[specialist-cv] fold={test_fold} detector={args.detector_model}", flush=True)
        detector_report, detector_runtime = phm.train_one(
            args.detector_model,
            windows["train"], windows["validation"], windows["test"],
            class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
            selected_channels=selected, device=args.device, batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
            evaluate_test_split=False, label_contract=args.label_contract,
            training_sampling=args.training_sampling, task_scope=args.detector_task_scope,
            diagnosis_auxiliary_weight=args.detector_diagnosis_auxiliary_weight,
            nominal_trajectory_weight=args.detector_nominal_trajectory_weight,
            detection_cvar_weight=args.detector_cvar_weight,
            detection_cvar_fraction=args.detector_cvar_fraction,
            return_runtime=True,
            detector_feature_profile=args.detector_feature_profile,
        )
        print(f"[specialist-cv] fold={test_fold} diagnosis={args.diagnosis_model}", flush=True)
        diagnosis_report, diagnosis_runtime = phm.train_one(
            args.diagnosis_model,
            windows["train"], windows["validation"], windows["test"],
            class_names, component_names, hierarchy["fault_to_component"], seed,
            args.diagnosis_epochs or args.epochs,
            selected_channels=selected, device=args.device, batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
            evaluate_test_split=False, label_contract=args.label_contract,
            training_sampling=args.training_sampling, task_scope="diagnosis", return_runtime=True,
        )
        test_logistic = phm.logistic_detector(
            windows["train"], windows["test"], seed,
            feature_profile=args.detector_feature_profile,
        )
        test_result = phm.evaluate_test(
            diagnosis_runtime["model"], windows["test"], test_logistic,
            diagnosis_report["best_validation"], class_names, component_names,
            hierarchy["fault_to_component"], device=args.device,
            detection_model=detector_runtime["model"],
            detection_selection=detector_report["best_validation"],
        )
        results.append(test_result)
        folds.append({
            "test_fold": test_fold,
            "validation_fold": (test_fold + 1) % 5,
            "train_folds": sorted(set(range(5)) - {test_fold, (test_fold + 1) % 5}),
            "split_audit": split_audit,
            "selected_channels": selected,
            "detector_training": detector_report,
            "diagnosis_training": diagnosis_report,
            "test": test_result,
        })
    report = {
        "schema_version": "astrograph-phm-specialist-group-cv.v1",
        "protocol": {
            "detector_model": args.detector_model,
            "detector_selection_candidates": ["deep", "logistic"],
            "detector_feature_profile": args.detector_feature_profile,
            "detector_task_scope": args.detector_task_scope,
            "detector_diagnosis_auxiliary_weight": args.detector_diagnosis_auxiliary_weight,
            "detector_nominal_trajectory_weight": args.detector_nominal_trajectory_weight,
            "detector_cvar_weight": args.detector_cvar_weight,
            "detector_cvar_fraction": args.detector_cvar_fraction,
            "diagnosis_model": args.diagnosis_model,
            "label_contract": args.label_contract,
            "training_sampling": args.training_sampling,
            "selection": "each specialist selected on Validation; mixed Test evaluated once",
            "group_key": "sample_index shared across subsystems and phases",
        },
        "folds": folds,
        "aggregate": aggregate(results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
