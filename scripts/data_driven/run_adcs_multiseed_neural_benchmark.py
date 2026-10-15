#!/usr/bin/env python3
"""Long-epoch, multi-seed ADCS neural benchmark with nominal envelope control."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.mechanism_features import (  # noqa: E402
    compose_columns, load_feature_bank, profile_columns, require_available,
)
from astrograph.data_driven.sota_backbones import SOTA_MODEL_FAMILIES  # noqa: E402
from astrograph.evaluation.data_exposure import dataset_use  # noqa: E402
from scripts.data_driven import compare_adcs_deep_encoders as deep  # noqa: E402
from scripts.integration import run_adcs_v522_regression as adcs  # noqa: E402


DEFAULT_DATASET = Path("/mnt/PRO6000_disk/Timer/sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2")
DEFAULT_DEVELOPMENT_NORMAL = Path("/mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_development_seed20260923")
FEATURE_BANK = ROOT / "configs/data_driven/mechanism_feature_banks.json"
DEFAULT_MODELS = (
    "residual_tcn", "gru", "inception_time", "modern_tcn",
    "progressive_hierarchical_gat_tcn", "gru_inception_specialist",
)


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "detection_accuracy": ("detection", "accuracy"),
        "detection_macro_precision": ("detection", "macro_precision"),
        "detection_macro_recall": ("detection", "macro_recall"),
        "detection_macro_f1": ("detection", "macro_f1"),
        "localization_accuracy": ("localization", "accuracy"),
        "localization_macro_precision": ("localization", "macro_precision"),
        "localization_macro_recall": ("localization", "macro_recall"),
        "localization_macro_f1": ("localization", "macro_f1"),
        "classification_accuracy": ("classification_predicted_component", "accuracy"),
        "classification_macro_precision": (
            "classification_predicted_component", "macro_precision",
        ),
        "classification_macro_recall": ("classification_predicted_component", "macro_recall"),
        "classification_macro_f1": ("classification_predicted_component", "macro_f1"),
        "end_to_end_macro_f1": ("end_to_end_15_class", "macro_f1"),
    }
    result: dict[str, Any] = {"seed_count": len(runs)}
    for name, path in paths.items():
        values = [float(run[path[0]][path[1]]) for run in runs]
        result[name] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    for name in (
        "fault_event_recall", "nominal_trajectory_false_alarm_rate",
        "false_alarm_episodes_per_hour",
    ):
        values = [float(run["best_validation"][name]) for run in runs]
        result[f"validation_{name}"] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    result["all_seeds_meet_validation_false_alarm_constraint"] = all(
        bool(run["best_validation"]["meets_nominal_false_alarm_constraint"])
        for run in runs
    )
    if all("independent_nominal_envelope" in run for run in runs):
        values = [
            float(run["independent_nominal_envelope"]["trajectory_false_alarm_rate"])
            for run in runs
        ]
        result["confirmation_nominal_trajectory_false_alarm_rate"] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
        result["all_seeds_zero_confirmation_false_alarms"] = all(value == 0.0 for value in values)
    return result


def render(report: dict[str, Any]) -> str:
    rows = []
    for family, metrics_ in report["aggregate"].items():
        def cell(name: str) -> str:
            value = metrics_[name]
            return f"{value['mean']:.4f} ± {value['population_std']:.4f}"
        rows.append(
            f"| {family} | {cell('detection_accuracy')} | "
            f"{cell('detection_macro_recall')} | {cell('detection_macro_f1')} | "
            f"{cell('localization_accuracy')} | {cell('localization_macro_recall')} | "
            f"{cell('localization_macro_f1')} | {cell('classification_accuracy')} | "
            f"{cell('classification_macro_recall')} | {cell('classification_macro_f1')} | "
            f"{cell('end_to_end_macro_f1')} |"
        )
    confirmation = ""
    if report.get("external_nominal_dataset"):
        confirmation_rows = []
        for family, metrics_ in report["aggregate"].items():
            value = metrics_["confirmation_nominal_trajectory_false_alarm_rate"]
            confirmation_rows.append(
                f"- {family}: {value['mean']:.4f} ± {value['population_std']:.4f}; "
                f"maximum={value['maximum']:.4f}"
            )
        confirmation = (
            f"\n独立正常工况确认数据：`{report['external_nominal_dataset']}`。"
            "逐随机种子误警结果见 `report.json`。\n\n"
            + "\n".join(confirmation_rows) + "\n"
        )
    return """# ADCS 长周期多随机种子神经网络对比

每个模型使用相同数据、完整机理特征、30 epoch 以上预算和相同训练随机种子集合。正常工况开发包只进入训练/验证；测试集不参与 checkpoint、融合或告警策略选择。

| 模型 | Det Acc | Det Recall | Det F1 | Loc Acc | Loc Recall | Loc F1 | Class Acc | Class Recall | Class F1 | E2E F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n" + confirmation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--development-nominal-root", type=Path, default=DEFAULT_DEVELOPMENT_NORMAL)
    parser.add_argument("--external-nominal-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "run_artifacts/post51-adcs-multiseed-neural-benchmark",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=[
            "residual_tcn", "gru", "transformer", "subsystem_tcn",
            "mechanism_gated_gat_tcn", "progressive_hierarchical_gat_tcn",
            "gru_inception_specialist",
            *SOTA_MODEL_FAMILIES,
        ],
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260930, 20260931, 20260932])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preprocessing", choices=["standard", "robust_clip"], default="standard")
    parser.add_argument("--alarm-policy", choices=["point", "hysteresis"], default="hysteresis")
    args = parser.parse_args(argv)
    if args.epochs < 30:
        parser.error("formal neural benchmark requires at least 30 epochs")
    if args.minimum_epochs < 20 or args.minimum_epochs > args.epochs:
        parser.error("minimum epochs must be within [20, epochs]")
    if len(set(args.seeds)) < 3:
        parser.error("formal neural benchmark requires at least three distinct seeds")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trajectories = adcs.load_trajectories(args.dataset_root)
    class_names = sorted({row.diagnostic_class for row in trajectories if row.diagnostic_class != "nominal"})
    component_names = list(adcs.COMPONENT_NAMES)
    bank = load_feature_bank(FEATURE_BANK)
    feature_columns = compose_columns(
        adcs.FEATURE_COLUMNS, profile_columns(bank, "adcs_v5_2_2", "full")
    )
    require_available(
        feature_columns, list(pd.read_csv(trajectories[0].telemetry_path, nrows=0).columns)
    )
    indexes = {name: index for index, name in enumerate(class_names)}
    component_indexes = {name: index for index, name in enumerate(component_names)}
    windows = adcs.build_windows(
        trajectories, indexes, component_indexes, feature_columns=feature_columns,
    )
    normal_trajectories = adcs.load_trajectories(args.development_nominal_root)
    if any(row.diagnostic_class != "nominal" for row in normal_trajectories):
        raise ValueError("development normal dataset contains a fault trajectory")
    normal = {
        name: deep.namespace_windows(part, "development-normal")
        for name, part in adcs.build_windows(
            normal_trajectories, indexes, component_indexes, feature_columns=feature_columns,
        ).items()
    }
    windows["train"] = deep.concatenate_windows([windows["train"], normal["train"]])
    windows["validation"] = deep.concatenate_windows([
        windows["validation"], normal["validation"], normal["test"],
    ])
    train_groups = sorted(set(windows["train"].trajectory_ids))
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=train_groups):
        preprocessor = adcs.fit_train_preprocessor(
            windows["train"].x, method=args.preprocessing,
        )
    scaled = {
        name: adcs.preprocess_windows(value, preprocessor) for name, value in windows.items()
    }
    external_nominal = None
    if args.external_nominal_root is not None:
        external_trajectories = adcs.load_trajectories(args.external_nominal_root)
        if any(row.diagnostic_class != "nominal" for row in external_trajectories):
            raise ValueError("external nominal dataset contains a fault trajectory")
        external_parts = adcs.build_windows(
            external_trajectories, indexes, component_indexes, feature_columns=feature_columns,
        )
        external_nominal = adcs.preprocess_windows(
            deep.namespace_windows(
                deep.concatenate_windows(list(external_parts.values())),
                "independent-confirmation",
            ),
            preprocessor,
        )
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in args.models}
    for family in args.models:
        for seed in args.seeds:
            print(f"[benchmark] model={family} seed={seed}", flush=True)
            runs[family].append(deep.train_model(
                family, scaled["train"], scaled["validation"], scaled["test"],
                class_names, component_names, epochs=args.epochs, seed=seed,
                device=args.device, batch_size=args.batch_size,
                minimum_epochs=args.minimum_epochs,
                early_stop_patience=args.early_stop_patience,
                selected_channels=feature_columns,
                alarm_policy_mode=args.alarm_policy,
                external_nominal=external_nominal,
            ))
    report = {
        "schema_version": "astrograph-adcs-multiseed-neural-benchmark.v1",
        "protocol": {
            "epochs": args.epochs,
            "minimum_epochs": args.minimum_epochs,
            "early_stop_patience": args.early_stop_patience,
            "seeds": args.seeds,
            "models": args.models,
            "feature_profile": "full",
            "device": args.device,
            "test_use": "one evaluation after train/validation selection per model and seed",
            "claim_boundary": "Basilisk synthetic DEVELOPMENT evidence; random initialization robustness only.",
        },
        "dataset": str(args.dataset_root.resolve()),
        "development_nominal_dataset": str(args.development_nominal_root.resolve()),
        "external_nominal_dataset": (
            str(args.external_nominal_root.resolve())
            if args.external_nominal_root is not None else None
        ),
        "feature_columns": feature_columns,
        "window_counts": {name: len(value.x) for name, value in scaled.items()},
        "runs": runs,
        "aggregate": {family: aggregate(values) for family, values in runs.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render(report), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
