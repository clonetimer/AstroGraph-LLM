#!/usr/bin/env python3
"""Validation-only, group-isolated screening for three-task deep candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_adcs_deep_encoders as adcs_deep
from scripts.data_driven import compare_phm_causal_windows as phm_deep
from scripts.integration import run_adcs_v522_regression as adcs
from astrograph.data_driven import phm_v60

METRICS = (
    "detection_macro_f1",
    "localization_macro_f1",
    "classification_macro_f1",
    "selection_score",
)


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(str(run["family"]), []).append(run)
    result: dict[str, Any] = {}
    for name, values in by_model.items():
        result[name] = {
            metric: {
                "mean": statistics.fmean(float(row["best_validation"][metric]) for row in values),
                "values": [float(row["best_validation"][metric]) for row in values],
            }
            for metric in METRICS
        }
        result[name]["seeds"] = [int(row["seed"]) for row in values]
    return result


def run_adcs(args: argparse.Namespace) -> dict[str, Any]:
    source = adcs.load_trajectories(args.adcs_dataset)
    audit = adcs.audit_split(source)
    rows = [row for row in source if row.split != "test"]
    class_names = sorted({row.diagnostic_class for row in source if row.diagnostic_class != "nominal"})
    component_names = list(adcs.COMPONENT_NAMES)
    windows = adcs.build_windows(
        rows,
        {name: index for index, name in enumerate(class_names)},
        {name: index for index, name in enumerate(component_names)},
    )
    impute, mean, scale = adcs.fit_train_scaler(windows["train"].x)
    scaled = {
        name: adcs.scale_windows(windows[name], impute, mean, scale)
        for name in ("train", "validation")
    }
    runs = [
        adcs_deep.train_model(
            model, scaled["train"], scaled["validation"], scaled["validation"],
            class_names, component_names, epochs=args.epochs, seed=seed,
            device=args.device, batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs,
            early_stop_patience=args.early_stop_patience,
            evaluate_test_split=False,
            gradient_audit=args.gradient_audit,
        )
        for seed in args.seeds for model in args.models
    ]
    return {
        "dataset": "whole_spacecraft_adcs_paired_dataset_v5_2_2",
        "split_audit": audit,
        "test_file_access": "none; test trajectories were excluded before telemetry/label loading",
        "runs": runs,
        "aggregate": summarize(runs),
    }


def run_phm(args: argparse.Namespace) -> dict[str, Any]:
    hierarchy = json.loads(phm_deep.HIERARCHY.read_text(encoding="utf-8"))
    source = phm_v60.load_records(args.phm_dataset)
    audit = phm_v60.audit_split(source)
    records = [row for row in source if row.split != "test"]
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    selected, median = phm_deep.select_train_channels(
        records, class_names, args.channel_limit, label_contract=args.phm_label_contract
    )
    unscaled = phm_deep.build_windows(
        records, selected, class_names, component_names, hierarchy["fault_to_component"],
        label_contract=args.phm_label_contract,
    )
    windows = phm_deep.scale_windows(
        {name: unscaled[name] for name in ("train", "validation")}, median
    )
    runs = [
        phm_deep.train_one(
            model, windows["train"], windows["validation"], windows["validation"],
            class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
            selected_channels=selected, device=args.device, batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs,
            early_stop_patience=args.early_stop_patience,
            evaluate_test_split=False,
            gradient_audit=args.gradient_audit,
            label_contract=args.phm_label_contract,
            training_sampling=args.phm_training_sampling,
        )
        for seed in args.seeds for model in args.models
    ]
    return {
        "dataset": "whole_spacecraft_phm_dataset_v6_0",
        "split_audit": audit,
        "selected_channels": selected,
        "label_contract": args.phm_label_contract,
        "training_sampling": args.phm_training_sampling,
        "test_file_access": "none; test records were excluded before telemetry/label loading",
        "runs": runs,
        "aggregate": summarize(runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adcs", "phm"], required=True)
    parser.add_argument("--adcs-dataset", type=Path, default=adcs.DEFAULT_DATASET)
    parser.add_argument("--phm-dataset", type=Path, default=phm_deep.DEFAULT_DATASET)
    parser.add_argument("--models", nargs="+", choices=["residual_tcn", "hierarchy_event_tcn", "pooled_task_adapter_tcn", "late_fork_tcn", "tied_late_fork_tcn", "capacity_matched_shared_tcn", "mechanism_gated_gat_tcn", "hierarchical_mechanism_gat_tcn"], default=["residual_tcn", "hierarchy_event_tcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260814, 20260815, 20260816])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--channel-limit", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-audit", action="store_true")
    parser.add_argument(
        "--phm-label-contract", choices=["event_active", "manifestation_aware"],
        default="event_active",
    )
    parser.add_argument(
        "--phm-training-sampling", choices=["window_shuffle", "trajectory_uniform"],
        default="window_shuffle",
    )
    args = parser.parse_args()
    report = {
        "schema_version": "astrograph-three-task-validation-screen.v1",
        "protocol": {
            "scope": "validation_only",
            "models": args.models,
            "seeds": args.seeds,
            "maximum_epochs": args.epochs,
            "minimum_epochs": args.minimum_epochs,
            "early_stop_patience": args.early_stop_patience,
            "batch_size": args.batch_size,
            "device": args.device,
            "claim_boundary": "Screening only; no test score and no production promotion.",
        },
        args.dataset: run_adcs(args) if args.dataset == "adcs" else run_phm(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
