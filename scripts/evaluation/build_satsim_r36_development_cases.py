#!/usr/bin/env python3
"""Build an R36 DEVELOPMENT case artifact from the uploaded synthetic PHM/SatSim corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from astrograph.data_driven.phm_v60 import (
    audit_split,
    build_dataset,
    load_records,
    load_three_stage_promoted_bundle,
)

from astrograph.evaluation.data_exposure import dataset_use, guard_prediction_generation

ROOT = Path(__file__).resolve().parents[2]
HIERARCHY = ROOT / "configs" / "datasets" / "phm_v6_0" / "diagnosis_hierarchy.json"
DEFAULT_MODEL = ROOT / "models" / "phm_v6_0" / "promoted_extra_trees_v1"


def duration_s(path: Path) -> float:
    frame = pd.read_csv(path, usecols=["time_s"])
    values = pd.to_numeric(frame["time_s"], errors="raise")
    duration = float(values.iloc[-1] - values.iloc[0])
    if duration <= 0:
        raise ValueError(f"non-positive duration: {path}")
    return duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = json.loads(HIERARCHY.read_text(encoding="utf-8"))
    fault_to_component = hierarchy["fault_to_component"]
    records = load_records(args.dataset_root)
    split = audit_split(records)
    if split["status"] != "PASS":
        raise ValueError("PHM/SatSim frozen split audit failed")

    dataset = build_dataset(args.dataset_root, fault_to_component, progress=True)
    local_sample_ids = sorted(record.sample_id for record in records)
    local_sample_id_sha256 = hashlib.sha256(
        json.dumps(local_sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # PHM sample IDs are dataset-local numeric identifiers and collide by name with
    # unrelated real-data R21 group IDs.  Do not misrepresent them as globally
    # registered provenance identities; the promoted bundle separately verifies the
    # sample_index.csv content hash before prediction.
    with dataset_use(
        "SYNTHETIC_PROTOCOL", "DEVELOPMENT", "OPTIMIZE"
    ) as development_provenance:
        guard_prediction_generation()
        result = load_three_stage_promoted_bundle(
            dataset, args.model_dir, dataset_root=args.dataset_root
        )
    predictions = {item["sample_id"]: item for item in result["test_predictions"]}
    threshold = float(result["selection"]["detection"]["threshold"])

    cases = []
    for record in [item for item in records if item.split == "test"]:
        pred = predictions[record.sample_id]
        true_fault = record.condition_id != "nominal"
        predicted_fault = float(pred["detection_score"]) >= threshold
        onset = None
        if true_fault:
            # The corpus is retrospective trace-level screening with known activation.
            # Keep truth timing for provenance but do not fabricate a detection time.
            onset_candidate = record.fault_manifestation_time_s
            if math.isnan(onset_candidate):
                onset_candidate = record.event_start_s
            if math.isnan(onset_candidate):
                onset_candidate = record.activation_time_s
            onset = float(onset_candidate)
        cases.append({
            "case_id": record.sample_id,
            "true_detected": true_fault,
            "predicted_detected": predicted_fault,
            "duration_s": duration_s(record.telemetry_path),
            "true_onset_s": onset,
            "detection_time_s": None,
            "true_component_id": fault_to_component[record.condition_id] if true_fault else None,
            "predicted_component_id": pred["predicted_component"] if predicted_fault else None,
            "true_fault_mode_id": record.condition_id if true_fault else None,
            "predicted_fault_mode_id": pred["predicted_class"] if predicted_fault else None,
            "true_open_set": False,
            "abstained": bool(true_fault and not predicted_fault),
        })

    payload = {
        "schema_version": "astrograph-r36-diagnostic-cases.v1",
        "dataset_id": "SATSIM_PHM_V6_0_USER_SUPPLIED",
        "protocol_id": "R36_SATSIM_DEV_PROMOTED_EXTRATREES_V1",
        "cohort_id": "PHM_V6_0_FROZEN_TEST_SAMPLE_INDEX_4_9",
        "evidence_class": "DEVELOPMENT",
        "model_family": result["strategy"],
        "detection_threshold": threshold,
        "case_unit": "trace",
        "timing_scope": "retrospective known-activation trace screening; detection_time_s intentionally unavailable",
        "split_audit": split,
        "development_provenance": {
            **{key: value for key, value in development_provenance.items() if key != "group_ids"},
            "registered_group_count": 0,
            "local_sample_count": len(local_sample_ids),
            "local_sample_id_sha256": local_sample_id_sha256,
            "group_identity_boundary": (
                "PHM sample IDs are dataset-local and are not asserted as R21 global group IDs; "
                "dataset identity is verified by the promoted bundle sample_index.csv hash."
            ),
        },
        "source_result": {key: value for key, value in result.items() if key not in {"models", "test_predictions"}},
        "cases": cases,
        "claim_boundary": (
            "Synthetic PHM/SatSim development evaluation with a frozen promoted retrospective model. "
            "It is not unknown-onset online detection, flight evidence, or qualification."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({
        "out": str(args.out),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(cases),
        "fault_case_count": sum(item["true_detected"] for item in cases),
        "normal_case_count": sum(not item["true_detected"] for item in cases),
        "strategy": result["strategy"],
        "detection_threshold": threshold,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
