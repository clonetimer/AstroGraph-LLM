#!/usr/bin/env python3
"""Five-fold group CV for flat and subsystem-aware causal TCN candidates.

Each run uses one modulo-5 group as test, the next group as validation and the
remaining three as train.  Preprocessing, channel selection, checkpoint and
fusion selection are repeated inside every fold; no test fold participates in
selection.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_adcs_deep_encoders as adcs_deep
from scripts.data_driven import compare_phm_causal_windows as phm_deep
from scripts.integration import run_adcs_v522_regression as adcs
from astrograph.data_driven import phm_v60
from astrograph.data_driven.mechanism_features import (
    compose_columns, load_feature_bank, profile_columns, require_available,
)
from astrograph.data_driven.sota_backbones import SOTA_MODEL_FAMILIES

MECHANISM_FEATURE_BANK = ROOT / "configs/data_driven/mechanism_feature_banks.json"


def rotating_split(sample_index: int, test_fold: int) -> str:
    fold = sample_index % 5
    if fold == test_fold:
        return "test"
    if fold == (test_fold + 1) % 5:
        return "validation"
    return "train"


def aggregate(rows: list[dict[str, Any]], e2e_key: str) -> dict[str, Any]:
    paths = {
        "detection_macro_f1": ("detection", "macro_f1"),
        "localization_macro_f1": ("localization", "macro_f1"),
        "classification_macro_f1": ("classification_predicted_component", "macro_f1"),
        "end_to_end_macro_f1": (e2e_key, "macro_f1"),
    }
    result: dict[str, Any] = {"fold_count": len(rows)}
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


def run_adcs(args: argparse.Namespace) -> dict[str, Any]:
    source = adcs.load_trajectories(args.adcs_dataset)
    class_names = sorted({row.diagnostic_class for row in source if row.diagnostic_class != "nominal"})
    component_names = list(adcs.COMPONENT_NAMES)
    by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in args.models}
    folds = []
    bank = load_feature_bank(MECHANISM_FEATURE_BANK)
    feature_columns = compose_columns(
        adcs.FEATURE_COLUMNS, profile_columns(bank, "adcs_v5_2_2", args.adcs_feature_profile)
    )
    require_available(
        feature_columns, list(pd.read_csv(source[0].telemetry_path, nrows=0).columns)
    )
    for test_fold in range(5):
        rows = [replace(row, split=rotating_split(row.sample_index, test_fold)) for row in source]
        audit = adcs.audit_split(rows)
        if audit["status"] != "PASS":
            raise RuntimeError({"dataset": "adcs", "fold": test_fold, "audit": audit})
        windows = adcs.build_windows(
            rows,
            {name: index for index, name in enumerate(class_names)},
            {name: index for index, name in enumerate(component_names)},
            feature_columns=feature_columns,
        )
        preprocessor = adcs.fit_train_preprocessor(
            windows["train"].x, method=args.preprocessing
        )
        scaled = {
            name: adcs.preprocess_windows(value, preprocessor)
            for name, value in windows.items()
        }
        seed = args.seed + test_fold
        fold_models = []
        for name in args.models:
            print(f"[cv][adcs] test_fold={test_fold} model={name}", flush=True)
            result = adcs_deep.train_model(
                name, scaled["train"], scaled["validation"], scaled["test"],
                class_names, component_names, epochs=args.adcs_epochs, seed=seed,
                device=args.device, batch_size=args.batch_size,
                minimum_epochs=args.minimum_epochs,
                early_stop_patience=args.early_stop_patience,
                selected_channels=feature_columns,
                alarm_policy_mode=args.alarm_policy,
            )
            by_model[name].append(result)
            fold_models.append(result)
        folds.append({
            "test_fold": test_fold,
            "validation_fold": (test_fold + 1) % 5,
            "train_folds": sorted(set(range(5)) - {test_fold, (test_fold + 1) % 5}),
            "split_audit": audit,
            "models": fold_models,
        })
    return {
        "dataset": "whole_spacecraft_adcs_paired_dataset_v5_2_2",
        "group_key": "trajectory_pair_id; assignment by sample_index % 5",
        "feature_profile": args.adcs_feature_profile,
        "feature_columns": feature_columns,
        "folds": folds,
        "aggregate": {name: aggregate(rows, "end_to_end_15_class") for name, rows in by_model.items()},
    }


def run_phm(args: argparse.Namespace) -> dict[str, Any]:
    hierarchy = json.loads(phm_deep.HIERARCHY.read_text(encoding="utf-8"))
    source = phm_v60.load_records(args.phm_dataset)
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in args.models}
    folds = []
    bank = load_feature_bank(MECHANISM_FEATURE_BANK)
    required_channels = profile_columns(bank, "phm_v6_0", args.phm_feature_profile)
    for test_fold in range(5):
        records = [replace(row, split=rotating_split(row.sample_index, test_fold)) for row in source]
        audit = phm_v60.audit_split(records)
        if audit["status"] != "PASS":
            raise RuntimeError({"dataset": "phm", "fold": test_fold, "audit": audit})
        selected, median = phm_deep.select_train_channels(
            records, class_names, args.channel_limit, required_channels=required_channels,
            label_contract=args.phm_label_contract,
        )
        windows = phm_deep.scale_windows(
            phm_deep.build_windows(
                records, selected, class_names, component_names, hierarchy["fault_to_component"],
                label_contract=args.phm_label_contract,
                training_sampling=args.phm_training_sampling,
            ),
            median, method=args.preprocessing,
        )
        seed = args.seed + test_fold
        fold_models = []
        for name in args.models:
            print(f"[cv][phm] test_fold={test_fold} model={name}", flush=True)
            result = phm_deep.train_one(
                name, windows["train"], windows["validation"], windows["test"],
                class_names, component_names, hierarchy["fault_to_component"], seed,
                args.phm_epochs, selected_channels=selected,
                device=args.device, batch_size=args.batch_size,
                minimum_epochs=args.minimum_epochs,
                early_stop_patience=args.early_stop_patience,
                alarm_policy_mode=args.alarm_policy,
                label_contract=args.phm_label_contract,
            )
            by_model[name].append(result)
            fold_models.append(result)
        folds.append({
            "test_fold": test_fold,
            "validation_fold": (test_fold + 1) % 5,
            "train_folds": sorted(set(range(5)) - {test_fold, (test_fold + 1) % 5}),
            "split_audit": audit,
            "selected_channels": selected,
            "models": fold_models,
        })
    return {
        "dataset": "whole_spacecraft_phm_dataset_v6_0",
        "group_key": "sample_index shared across subsystems and phases; assignment by sample_index % 5",
        "feature_profile": args.phm_feature_profile,
        "required_mechanism_channels": required_channels,
        "label_contract": args.phm_label_contract,
        "training_sampling": args.phm_training_sampling,
        "folds": folds,
        "aggregate": {name: aggregate(rows, "end_to_end_18_class") for name, rows in by_model.items()},
    }


def render(report: dict[str, Any]) -> str:
    rows = []
    for key in ("adcs", "phm"):
        if key not in report:
            continue
        dataset = report[key]
        for model, values in dataset["aggregate"].items():
            def value(name: str) -> str:
                item = values[name]
                return f"{item['mean']:.4f} ± {item['population_std']:.4f}"
            rows.append(
                f"| {dataset['dataset']} | {model} | {value('detection_macro_f1')} | "
                f"{value('localization_macro_f1')} | {value('classification_macro_f1')} | "
                f"{value('end_to_end_macro_f1')} |"
            )
    return """# 卫星部件/分系统感知模型五折 Group CV

每轮 test fold 为 `sample_index % 5 == k`，validation 为下一 fold，其余三折训练。所有预处理、PHM 通道选择、checkpoint、检测器来源和层级融合仅在当轮 Train/Validation 完成。

| 数据集 | 模型 | Detection Macro-F1 | Localization Macro-F1 | Classification Macro-F1 | E2E Macro-F1 |
|---|---|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcs-dataset", type=Path, default=adcs.DEFAULT_DATASET)
    parser.add_argument("--phm-dataset", type=Path, default=phm_deep.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/integration/out/hierarchical_group_cv_20260814")
    parser.add_argument("--models", nargs="+", choices=["residual_tcn", "subsystem_tcn", "hierarchy_event_tcn", "pooled_task_adapter_tcn", "late_fork_tcn", "tied_late_fork_tcn", "capacity_matched_shared_tcn", "mechanism_gated_gat_tcn", "hierarchical_mechanism_gat_tcn", *SOTA_MODEL_FAMILIES], default=["inception_time", "timesnet", "modern_tcn", "residual_tcn"])
    parser.add_argument("--datasets", nargs="+", choices=["adcs", "phm"], default=["adcs", "phm"])
    parser.add_argument("--adcs-epochs", type=int, default=8)
    parser.add_argument("--phm-epochs", type=int, default=8)
    parser.add_argument("--channel-limit", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--minimum-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--preprocessing", choices=["standard", "robust_clip"], default="standard")
    parser.add_argument("--alarm-policy", choices=["point", "hysteresis"], default="point")
    parser.add_argument(
        "--phm-label-contract", choices=["event_active", "manifestation_aware"],
        default="event_active",
    )
    parser.add_argument(
        "--phm-training-sampling", choices=["window_shuffle", "trajectory_uniform"],
        default="window_shuffle",
    )
    parser.add_argument(
        "--adcs-feature-profile",
        choices=["base", "innovation", "rw_execution", "rw_current_pair", "full"],
        default="base",
    )
    parser.add_argument(
        "--phm-feature-profile", choices=["selected_only", "mechanism_required"],
        default="selected_only",
    )
    args = parser.parse_args()
    report = {
        "schema_version": "astrograph-hierarchical-group-cv.v1",
        "protocol": {
            "folds": 5,
            "validation_fold": "(test_fold + 1) % 5",
            "selection": "train/validation only; test once per fold",
            "model_families": args.models,
            "device": args.device,
            "preprocessing": args.preprocessing,
            "adcs_feature_profile": args.adcs_feature_profile,
            "phm_feature_profile": args.phm_feature_profile,
            "alarm_policy": args.alarm_policy,
            "phm_label_contract": args.phm_label_contract,
            "phm_training_sampling": args.phm_training_sampling,
            "claim_boundary": "E0 synthetic group-CV; subsystem topology is an architecture prior, not causal proof or flight evidence.",
        },
    }
    if "adcs" in args.datasets:
        report["adcs"] = run_adcs(args)
    if "phm" in args.datasets:
        report["phm"] = run_phm(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(render(report), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
