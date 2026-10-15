#!/usr/bin/env python3
"""Train the selected PHM specialists and exercise the registered full workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_phm_causal_windows as phm
from scripts.integration.run_phm_v60_regression import run_chain
from astrograph.data_driven import (
    DataDrivenModelManifest, DataDrivenTaskType, ModelCapability,
    ModelInferenceRequest, ModelOutputSemantics, ModelRuntime, ObservationScope,
)
from astrograph.assurance import ArtifactRoot, canonical_sha256
from astrograph.data_driven.phm_v60 import audit_split, load_records
from astrograph.integrations.phm_three_task_torch import (
    PHMThreeTaskTorchProvider, build_registry_from_phm_three_task_provider,
    load_phm_three_task_torch_package, save_phm_three_task_torch_package,
)
from astrograph.data_driven.mechanism_graph import load_hierarchical_mechanism_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--diagnosis-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    artifacts = ArtifactRoot(args.output_dir, source_root=ROOT)
    output_root = artifacts.ensure()

    hierarchy = json.loads(phm.HIERARCHY.read_text(encoding="utf-8"))
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    records = load_records(args.dataset_root)
    split_audit = audit_split(records)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    selected_channels, fallback_median = phm.select_train_channels(
        records, class_names, 96, label_contract="manifestation_aware"
    )
    raw_windows = phm.build_windows(
        records, selected_channels, class_names, component_names,
        hierarchy["fault_to_component"], label_contract="manifestation_aware",
    )
    windows, preprocessor = phm.scale_windows(
        raw_windows, fallback_median, return_state=True
    )
    common = dict(
        selected_channels=selected_channels, device=args.device, batch_size=args.batch_size,
        minimum_epochs=20, early_stop_patience=15, evaluate_test_split=False,
        label_contract="manifestation_aware", training_sampling="trajectory_uniform",
        return_runtime=True,
    )
    detector_report, detector_runtime = phm.train_one(
        "causal_multiview_detector_tcn", windows["train"], windows["validation"], windows["test"],
        class_names, component_names, hierarchy["fault_to_component"], 20260826, args.epochs,
        task_scope="detection_aux", diagnosis_auxiliary_weight=0.1,
        detector_feature_profile="legacy_summary", **common,
    )
    diagnosis_report, diagnosis_runtime = phm.train_one(
        "hierarchical_mechanism_gat_tcn", windows["train"], windows["validation"], windows["test"],
        class_names, component_names, hierarchy["fault_to_component"], 20260826,
        args.diagnosis_epochs, task_scope="diagnosis", **common,
    )
    test_logistic = phm.logistic_detector(windows["train"], windows["test"], 20260826)
    test_result = phm.evaluate_test(
        diagnosis_runtime["model"], windows["test"], test_logistic,
        diagnosis_report["best_validation"], class_names, component_names,
        hierarchy["fault_to_component"], device=args.device,
        detection_model=detector_runtime["model"],
        detection_selection=detector_report["best_validation"],
    )

    config_root = ROOT / "configs/datasets/phm_v6_0"
    class_map = json.loads((config_root / "fault_class_map.json").read_text(encoding="utf-8"))["mappings"]
    universe = json.loads((config_root / "candidate_universe.json").read_text(encoding="utf-8"))
    runtime_fault_ids = frozenset(universe["runtime_fault_ids"])
    manifest = DataDrivenModelManifest(
        model_id="phm-v60-causal-multiview-hierarchical-specialists",
        model_version="2026.08.26-candidate",
        display_name="PHM v6 causal detector plus hierarchical diagnoser",
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION,
        runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS,
                      ModelCapability.OOD_SCORE, ModelCapability.TEMPORAL_CONTEXT],
        supported_fault_ids=sorted(runtime_fault_ids),
        feature_schema_ids=["phm-v60-causal-window-v1"],
        input_contract={"window_length": phm.WINDOW, "feature_order": selected_channels},
        training_dataset_ids=["whole_spacecraft_phm_dataset_v6_0"],
        claim_boundary=(
            "E0 synthetic grouped causal-window research candidate; not calibrated, production, "
            "safety, flight, or formal diagnosis evidence."
        ),
        metadata={"task_models": {
            "detection": "causal_multiview_detector_tcn+detection_aux_0.1",
            "localization_classification": "hierarchical_mechanism_gat_tcn",
        }},
    )
    provider = PHMThreeTaskTorchProvider(
        manifest=manifest,
        detector_model=detector_runtime["model"], diagnosis_model=diagnosis_runtime["model"],
        preprocessor=preprocessor, selected_channels=tuple(selected_channels),
        class_names=tuple(class_names), component_names=tuple(component_names),
        fault_to_component=hierarchy["fault_to_component"], class_map=class_map,
        runtime_fault_ids=runtime_fault_ids,
        detection_selection=detector_report["best_validation"],
        diagnosis_selection=diagnosis_report["best_validation"],
    )
    hierarchical_contract = load_hierarchical_mechanism_graph(
        phm.HIERARCHICAL_GRAPH_CONFIG, "phm_v6_0", selected_channels, component_names
    )
    package_path = artifacts.resolve("models/phm_three_task_specialists.pt")
    package_sha256 = save_phm_three_task_torch_package(
        provider, package_path, hierarchical_contract=hierarchical_contract
    )
    reloaded_provider = load_phm_three_task_torch_package(
        package_path, expected_sha256=package_sha256, device=args.device
    )
    registry = build_registry_from_phm_three_task_provider(reloaded_provider)

    sample = next(
        row for row in records
        if row.split == "test" and row.condition_id == "thermal_heater_stuck_on"
    )
    frame = pd.read_csv(sample.telemetry_path, usecols=["time_s", *selected_channels], low_memory=False)
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    target_time = float(sample.fault_manifestation_time_s) + 30.0
    endpoint = int(np.searchsorted(times, target_time, side="left"))
    endpoint = max(phm.WINDOW - 1, min(endpoint, len(frame) - 1))
    raw_window = frame[selected_channels].iloc[endpoint - phm.WINDOW + 1:endpoint + 1]
    values = raw_window.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).tolist()
    payload = {"channel_order": selected_channels, "telemetry_window": values}
    request = ModelInferenceRequest(
        episode_id="preflight-phm-neural", model_id=manifest.model_id,
        model_version=manifest.model_version, feature_schema_id="phm-v60-causal-window-v1",
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=sorted(runtime_fault_ids), payload=payload,
        observation_scope=ObservationScope(
            scope_id=f"phm-v60-specialist-window-{endpoint}",
            source_id="phm-v60-telemetry",
            window_start_s=float(times[endpoint - phm.WINDOW + 1]),
            window_end_s=float(times[endpoint]),
            trajectory_group_id=str(getattr(sample, "monte_carlo_group_id", sample.sample_id)),
        ),
    )
    chains = {}
    for reasoner in (["rule"] if args.skip_llm else ["rule", "llm"]):
        try:
            chains[reasoner] = run_chain(
                dataset_root=args.dataset_root, sample_id=sample.sample_id, prediction=None,
                output_dir=output_root / "chains" / reasoner, reasoner=reasoner,
                vllm_url=args.vllm_url, vllm_model=args.vllm_model,
                model_registry=registry, inference_request=request,
            )
        except Exception as exc:  # preserve a failed external-service regression as evidence
            chains[reasoner] = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}

    report = {
        "schema_version": "astrograph-phm-neural-specialist-chain.v1",
        "split_audit": split_audit,
        "selected_channels": selected_channels,
        "preprocessor": preprocessor.receipt(),
        "detector_training": detector_report,
        "diagnosis_training": diagnosis_report,
        "test": test_result,
        "manifest": manifest.model_dump(mode="json"),
        "model_package": {"path": str(package_path), "sha256": package_sha256,
                          "load_mode": "torch.weights_only+explicit architecture"},
        "runtime_sample": {"sample_id": sample.sample_id, "window_end_time_s": float(times[endpoint])},
        "chains": chains,
        "claim_boundary": manifest.claim_boundary,
    }
    path = artifacts.write_json("report.json", report)
    print(path)
    return 0 if all(value.get("status") == "SUCCEEDED" for value in chains.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
