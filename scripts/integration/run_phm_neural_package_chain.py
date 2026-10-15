#!/usr/bin/env python3
"""Reload a frozen PHM neural package and run it through AstroGraph's workflow."""
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

from scripts.integration.run_phm_v60_regression import run_chain
from astrograph.assurance import ArtifactRoot, canonical_sha256
from astrograph.data_driven import ModelInferenceRequest, ObservationScope
from astrograph.data_driven.phm_v60 import load_records
from astrograph.integrations.phm_three_task_torch import (
    build_registry_from_phm_three_task_provider, load_phm_three_task_torch_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    artifacts = ArtifactRoot(args.output_dir, source_root=ROOT)
    output_root = artifacts.ensure()

    provider = load_phm_three_task_torch_package(
        args.package, expected_sha256=args.package_sha256, device=args.device
    )
    registry = build_registry_from_phm_three_task_provider(provider)
    records = load_records(args.dataset_root)
    sample = next(
        row for row in records
        if row.split == "test" and row.condition_id == "thermal_heater_stuck_on"
    )
    channels = list(provider.selected_channels)
    frame = pd.read_csv(sample.telemetry_path, usecols=["time_s", *channels], low_memory=False)
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    window_length = int(provider.manifest.input_contract["window_length"])
    endpoint = int(np.searchsorted(times, float(sample.fault_manifestation_time_s) + 30.0))
    endpoint = max(window_length - 1, min(endpoint, len(frame) - 1))
    raw = frame[channels].iloc[endpoint - window_length + 1:endpoint + 1]
    values = raw.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).tolist()
    payload = {"channel_order": channels, "telemetry_window": values}
    request = ModelInferenceRequest(
        episode_id="preflight-phm-neural", model_id=provider.manifest.model_id,
        model_version=provider.manifest.model_version,
        feature_schema_id="phm-v60-causal-window-v1",
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=sorted(provider.runtime_fault_ids),
        payload=payload,
        observation_scope=ObservationScope(
            scope_id=f"phm-v60-window-{endpoint}",
            source_id="phm-v60-telemetry",
            window_start_s=float(times[endpoint - window_length + 1]),
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
        except Exception as exc:
            chains[reasoner] = {
                "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)
            }
    report = {
        "schema_version": "astrograph-phm-neural-package-chain.v1",
        "model_package": {"path": str(args.package), "sha256": args.package_sha256},
        "manifest": provider.manifest.model_dump(mode="json"),
        "runtime_sample": {"sample_id": sample.sample_id, "window_end_time_s": float(times[endpoint])},
        "detection_selection": dict(provider.detection_selection),
        "diagnosis_selection": dict(provider.diagnosis_selection),
        "chains": chains,
    }
    path = artifacts.write_json("report.json", report)
    print(path)
    return 0 if all(value.get("status") == "SUCCEEDED" for value in chains.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
