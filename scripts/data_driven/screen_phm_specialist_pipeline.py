#!/usr/bin/env python3
"""Validation-only screen for explicitly decoupled PHM task specialists."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_phm_causal_windows as phm
from astrograph.data_driven.phm_v60 import audit_split, load_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=phm.DEFAULT_DATASET)
    parser.add_argument("--detector-model", default="residual_tcn", choices=[
        "residual_tcn", "hierarchy_event_tcn", "pooled_task_adapter_tcn",
    ])
    parser.add_argument("--diagnosis-models", nargs="+", default=[
        "residual_tcn", "hierarchical_mechanism_gat_tcn",
    ], choices=["residual_tcn", "hierarchical_mechanism_gat_tcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260826, 20260827, 20260828])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--channel-limit", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--label-contract", choices=["event_active", "manifestation_aware"],
        default="manifestation_aware",
    )
    parser.add_argument(
        "--training-sampling", choices=["window_shuffle", "trajectory_uniform"],
        default="trajectory_uniform",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = json.loads(phm.HIERARCHY.read_text(encoding="utf-8"))
    source = load_records(args.dataset_root)
    split_audit = audit_split(source)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    records = [row for row in source if row.split != "test"]
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    selected, median = phm.select_train_channels(
        records, class_names, args.channel_limit, label_contract=args.label_contract
    )
    unscaled = phm.build_windows(
        records, selected, class_names, component_names, hierarchy["fault_to_component"],
        label_contract=args.label_contract,
    )
    windows = phm.scale_windows(
        {name: unscaled[name] for name in ("train", "validation")}, median
    )

    runs = []
    for seed in args.seeds:
        detector = phm.train_one(
            args.detector_model,
            windows["train"], windows["validation"], windows["validation"],
            class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
            selected_channels=selected, device=args.device, batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
            evaluate_test_split=False, label_contract=args.label_contract,
            training_sampling=args.training_sampling, task_scope="detection",
        )
        for diagnosis_model in args.diagnosis_models:
            diagnosis = phm.train_one(
                diagnosis_model,
                windows["train"], windows["validation"], windows["validation"],
                class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
                selected_channels=selected, device=args.device, batch_size=args.batch_size,
                minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
                evaluate_test_split=False, label_contract=args.label_contract,
                training_sampling=args.training_sampling, task_scope="diagnosis",
            )
            runs.append({
                "seed": seed,
                "detector_model": args.detector_model,
                "diagnosis_model": diagnosis_model,
                "detection_macro_f1": detector["best_validation"]["detection_macro_f1"],
                "localization_macro_f1": diagnosis["best_validation"]["localization_macro_f1"],
                "classification_macro_f1": diagnosis["best_validation"]["classification_macro_f1"],
                "detector": detector,
                "diagnosis": diagnosis,
            })
    aggregate = {}
    for model in args.diagnosis_models:
        values = [row for row in runs if row["diagnosis_model"] == model]
        aggregate[model] = {
            metric: {
                "mean": statistics.fmean(float(row[metric]) for row in values),
                "values": [float(row[metric]) for row in values],
            }
            for metric in (
                "detection_macro_f1", "localization_macro_f1", "classification_macro_f1"
            )
        }
    report = {
        "schema_version": "astrograph-phm-specialist-validation-screen.v1",
        "protocol": {
            "scope": "validation_only; test records excluded before telemetry loading",
            "architecture": "independent detection encoder plus diagnosis encoder",
            "detector_model": args.detector_model,
            "detector_selection_candidates": ["deep", "logistic"],
            "diagnosis_models": args.diagnosis_models,
            "label_contract": args.label_contract,
            "training_sampling": args.training_sampling,
            "seeds": args.seeds,
        },
        "split_audit": split_audit,
        "selected_channels": selected,
        "runs": runs,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
