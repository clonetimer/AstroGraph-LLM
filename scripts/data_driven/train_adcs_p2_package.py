#!/usr/bin/env python3
"""Train and freeze the ADCS P2 deployable package on the fixed group split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from astrograph.data_driven import (
    DataDrivenModelManifest, DataDrivenTaskType, ModelCapability,
    ModelOutputSemantics, ModelRuntime,
)
from astrograph.integrations.adcs_three_task_torch import (
    ADCSThreeTaskTorchProvider, declared_channel_order_contract,
    save_adcs_three_task_torch_package,
)
from astrograph.assurance import ArtifactRoot
from astrograph.data_driven.mechanism_features import (
    compose_columns, load_feature_bank, profile_columns, require_available,
)
from scripts.data_driven import compare_adcs_deep_encoders as deep
from scripts.integration import run_adcs_v522_regression as adcs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=adcs.DEFAULT_DATASET)
    parser.add_argument("--package", type=Path,
                        default=ROOT / "models/adcs_v5_2_2/p2_innovation.pt")
    parser.add_argument("--receipt", type=Path,
                        default=ROOT / "models/adcs_v5_2_2/p2_innovation.receipt.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    rows = adcs.load_trajectories(args.dataset_root)
    audit = adcs.audit_split(rows)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    hierarchy = json.loads(adcs.DIAGNOSIS_HIERARCHY.read_text(encoding="utf-8"))
    class_map = json.loads(adcs.FAULT_CLASS_MAP.read_text(encoding="utf-8"))["mappings"]
    universe = json.loads(adcs.CANDIDATE_UNIVERSE.read_text(encoding="utf-8"))
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = list(hierarchy["components"])
    bank = load_feature_bank(deep.MECHANISM_FEATURE_BANK)
    channels = compose_columns(
        adcs.FEATURE_COLUMNS, profile_columns(bank, "adcs_v5_2_2", "innovation")
    )
    require_available(channels, list(pd.read_csv(rows[0].telemetry_path, nrows=0).columns))
    windows = adcs.build_windows(
        rows, {name: index for index, name in enumerate(class_names)},
        {name: index for index, name in enumerate(component_names)}, feature_columns=channels,
    )
    preprocessor = adcs.fit_train_preprocessor(windows["train"].x, method="robust_clip")
    scaled = {name: adcs.preprocess_windows(value, preprocessor) for name, value in windows.items()}
    captured = {}

    def capture(model, logistic, selection):
        captured.update(model=model, logistic=logistic, selection=dict(selection))

    report = deep.train_model(
        "residual_tcn", scaled["train"], scaled["validation"], scaled["test"],
        class_names, component_names, epochs=args.epochs, seed=args.seed,
        device=args.device, batch_size=args.batch_size, minimum_epochs=args.minimum_epochs,
        early_stop_patience=args.early_stop_patience, selected_channels=channels,
        alarm_policy_mode="point", artifact_callback=capture,
    )
    selection = captured["selection"]
    logistic = captured["logistic"]
    manifest = DataDrivenModelManifest(
        model_id="adcs-p2-innovation-residual-tcn", model_version="0.7.1.post24-candidate.2",
        display_name="ADCS P2 innovation Residual TCN",
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION, runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS,
                      ModelCapability.TEMPORAL_CONTEXT],
        supported_fault_ids=sorted(universe["runtime_fault_ids"]),
        feature_schema_ids=["adcs-v5.2.2-p2-innovation-window-v2"],
        input_contract={
            "window_length": adcs.WINDOW,
            "feature_order": channels,
            "tensor_channel_order": channels,
            "channel_order_contract_schema": "astrograph-channel-order-contract.v1",
        },
    )
    provider = ADCSThreeTaskTorchProvider(
        manifest=manifest, model=captured["model"], preprocessor=preprocessor,
        selected_channels=tuple(channels), class_names=tuple(class_names),
        component_names=tuple(component_names), fault_to_component=hierarchy["fault_to_component"],
        class_map=class_map, runtime_fault_ids=frozenset(universe["runtime_fault_ids"]),
        detection_selection={
            "detector_deep_blend_alpha": selection["detector_deep_blend_alpha"],
            "detection_threshold": selection["detection_threshold"],
            "alarm_policy_mode": "point",
        },
        logistic_coefficient=logistic.coef_[0], logistic_intercept=float(logistic.intercept_[0]),
        diagnosis_selection={"hierarchy_blend_alpha": selection["hierarchy_blend_alpha"]},
        channel_order_contract=declared_channel_order_contract(
            channels, provenance="DECLARED_ORDER_ENFORCED_DURING_TRAINING"
        ),
    )
    digest = save_adcs_three_task_torch_package(provider, args.package)
    try:
        package_path = str(args.package.relative_to(ROOT))
    except ValueError:
        package_path = str(args.package)
    receipt = {
        "schema_version": "astrograph-adcs-p2-package-training-receipt.v2",
        "package_path": package_path, "package_sha256": digest,
        "dataset_root": str(args.dataset_root), "split_audit": audit,
        "training": {"seed": args.seed, "device": args.device, "epochs": args.epochs,
                     "minimum_epochs": args.minimum_epochs,
                     "early_stop_patience": args.early_stop_patience,
                     "batch_size": args.batch_size},
        "preprocessing": preprocessor.receipt(), "feature_profile": "innovation",
        "tensor_channel_order": list(channels),
        "channel_order_contract": dict(provider.channel_order_contract or {}),
        "metrics": report,
        "claim_boundary": "Fixed-split E0 engineering package; five-fold qualification is separate.",
    }
    ArtifactRoot(args.receipt.parent, source_root=ROOT).write_json(args.receipt.name, receipt)
    print(json.dumps({"package": str(args.package), "sha256": digest,
                      "receipt": str(args.receipt)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
