#!/usr/bin/env python3
"""Reload the frozen ADCS P2 package and execute AstroGraph rule/LLM chains."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from astrograph.assurance import ArtifactRoot, canonical_sha256
from astrograph.data_driven import ModelInferenceRequest, ObservationScope
from astrograph.integrations.adcs_three_task_torch import (
    build_registry_from_adcs_three_task_provider, load_adcs_three_task_torch_package,
)
from astrograph.retrieval import (
    ExperienceKnowledgeCandidateInbox,
    ExperienceKnowledgeUpdateCandidate,
)
from scripts.integration import run_adcs_v522_regression as adcs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--dor-executable", type=Path, default=adcs.DEFAULT_DOR_EXECUTABLE)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--experience-ledger", type=Path)
    parser.add_argument("--memory-store", type=Path)
    parser.add_argument("--kg-candidate-inbox", type=Path)
    args = parser.parse_args()
    artifacts = ArtifactRoot(args.output_dir, source_root=ROOT)
    output_root = artifacts.ensure()

    provider = load_adcs_three_task_torch_package(
        args.package, expected_sha256=args.package_sha256, device=args.device
    )
    registry = build_registry_from_adcs_three_task_provider(provider)
    rows = adcs.load_trajectories(args.dataset_root)
    sample = next(
        row for row in rows if row.split == "test" and row.diagnostic_class == "adcs_rw_jamming"
    )
    labels = pd.read_csv(sample.labels_path)
    active_field = "label.fault_active" if "label.fault_active" in labels else "event_active"
    active = pd.to_numeric(labels[active_field], errors="coerce").fillna(0).to_numpy() > 0.5
    onset_index = int(np.flatnonzero(active)[0])
    channels = list(provider.selected_channels)
    frame = pd.read_csv(sample.telemetry_path, usecols=["time_s", *channels], low_memory=False)
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    length = int(provider.manifest.input_contract["window_length"])
    endpoint = min(len(frame) - 1, max(length - 1, onset_index + 30))
    values = frame[channels].iloc[endpoint - length + 1:endpoint + 1].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32).tolist()
    payload = {"channel_order": channels, "telemetry_window": values}
    request = ModelInferenceRequest(
        episode_id="preflight-adcs-p2", model_id=provider.manifest.model_id,
        model_version=provider.manifest.model_version,
        feature_schema_id=provider.manifest.feature_schema_ids[0],
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=sorted(provider.runtime_fault_ids),
        payload=payload,
        observation_scope=ObservationScope(
            scope_id=f"adcs-p2-window-{endpoint}",
            source_id="adcs-v522-telemetry",
            window_start_s=float(times[endpoint - length + 1]),
            window_end_s=float(times[endpoint]),
            trajectory_group_id=str(getattr(sample, "trajectory_pair_id", sample.sample_id)),
        ),
    )
    prediction = registry.predict(request)
    chains = {}
    for reasoner in (["rule"] if args.skip_llm else ["rule", "llm"]):
        try:
            chains[reasoner] = adcs.run_chain(
                dataset_root=args.dataset_root, sample=sample, prediction=prediction,
                output_dir=output_root / "chains", reasoner=reasoner,
                vllm_url=args.vllm_url, vllm_model=args.vllm_model,
                dor_executable=args.dor_executable,
                experience_ledger_path=args.experience_ledger,
                memory_store_path=args.memory_store,
            )
        except Exception as exc:
            chains[reasoner] = {
                "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)
            }
    staged_candidate_ids: list[str] = []
    if args.kg_candidate_inbox is not None:
        inbox = ExperienceKnowledgeCandidateInbox(args.kg_candidate_inbox)
        llm_payload = (chains.get("llm") or {}).get("output_payload") or {}
        structured = llm_payload.get("structured_diagnosis_output") or {}
        knowledge = structured.get("knowledge_and_retrieval_summary") or {}
        for raw in knowledge.get("knowledge_update_candidates", []):
            candidate = ExperienceKnowledgeUpdateCandidate.model_validate(raw)
            if inbox.stage(candidate):
                staged_candidate_ids.append(candidate.candidate_id)
    report = {
        "schema_version": "astrograph-adcs-p2-package-chain.v1",
        "model_package": {"path": str(args.package), "sha256": args.package_sha256},
        "manifest": provider.manifest.model_dump(mode="json"),
        "runtime_sample": {"sample_id": sample.sample_id,
                           "diagnostic_class": sample.diagnostic_class,
                           "window_end_time_s": float(times[endpoint])},
        "prediction": prediction.model_dump(mode="json"),
        "chains": chains,
        "experience_reuse": {
            "ledger_path": str(args.experience_ledger.resolve()) if args.experience_ledger else None,
            "memory_store_path": str(args.memory_store.resolve()) if args.memory_store else None,
            "kg_candidate_inbox_path": (
                str(args.kg_candidate_inbox.resolve()) if args.kg_candidate_inbox else None
            ),
            "staged_candidate_ids": staged_candidate_ids,
            "active_graph_modified": False,
        },
        "qualification_boundary": "Candidate chain evidence only; metric gates remain independent.",
    }
    path = artifacts.write_json("report.json", report)
    shutil.rmtree(artifacts.resolve("chains"), ignore_errors=True)
    print(path)
    return 0 if all(value.get("status") == "SUCCEEDED" for value in chains.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
