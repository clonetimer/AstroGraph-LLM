#!/usr/bin/env python3
"""Select (but never freeze) PHM v6.0 soft semantic calibration on groups 3/8."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.phm_v60 import load_records
from astrograph.evidence import (
    ConfiguredRuntimeEvidenceBuilder,
    SoftCalibrationEpisode,
    select_development_soft_calibration,
)
from astrograph.simulation import SimulationRunCompleted


DEFAULT_DATASET = Path(
    os.environ.get(
        "ASTROGRAPH_PHM_V60_DATA_ROOT",
        str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0"),
    )
)
DEFAULT_OUTPUT = ROOT / "research/evidence/phm_v6_0_soft_calibration_r001"
EVIDENCE_PACKAGE = ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json"
HARD_CALIBRATION_RECEIPT = ROOT / "configs/datasets/phm_v6_0/evidence_calibration_receipt.json"
SELECTED_DEFINITION_IDS = (
    "ED-PHM-V60-BATTERY-CAPACITY-ENERGY-V1",
    "ED-PHM-V60-BATTERY-CAPACITY-SOC-V1",
    "ED-PHM-V60-SOLAR-STRING-POWER-V1",
    "ED-PHM-V60-DOWNLINK-SERVICE-ACTIVE-V1",
    "ED-PHM-V60-DOWNLINK-SERVICE-COMMAND-V1",
    "ED-PHM-V60-RADIATOR-REJECTION-POWER-V1",
    "ED-PHM-V60-RADIATOR-REJECTION-TEMP-V1",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    records = load_records(dataset_root)
    receipt_payload = json.loads(HARD_CALIBRATION_RECEIPT.read_text(encoding="utf-8"))
    condition_to_fault = {
        value["dataset_condition_id"]: fault_id
        for fault_id, value in receipt_payload["faults"].items()
    }
    development = [
        item
        for item in records
        if item.sample_index in {3, 8} and item.condition_id != "nominal"
    ]
    nominal_by_key = {
        (item.subsystem, item.sample_index): item
        for item in records
        if item.condition_id == "nominal" and item.sample_index in {3, 8}
    }
    nominal_frames = {
        key: pd.read_csv(item.telemetry_path, low_memory=False)
        for key, item in nominal_by_key.items()
    }
    builder = ConfiguredRuntimeEvidenceBuilder(EVIDENCE_PACKAGE)
    episodes: list[SoftCalibrationEpisode] = []
    for index, record in enumerate(development, start=1):
        key = (record.subsystem, record.sample_index)
        event = SimulationRunCompleted(
            event_id=f"phm-soft-cal-{record.sample_id}",
            run_id=f"phm-soft-cal-{record.sample_id}",
            run_dir=str(record.telemetry_path.parent),
            nominal_run_dir=str(nominal_by_key[key].telemetry_path.parent),
            emitted_at="2026-08-31T00:00:00Z",
            metadata={"offline_label_separated_calibration": True},
        )
        source = pd.read_csv(record.telemetry_path, low_memory=False)
        bundle, _ = builder.build_from_frames(
            event,
            source=source,
            nominal=nominal_frames[key],
        )
        episodes.append(
            SoftCalibrationEpisode(
                group_id=record.sample_index,
                true_fault_id=condition_to_fault.get(
                    record.condition_id, f"DATASET_CONDITION::{record.condition_id}"
                ),
                bundle=bundle,
            )
        )
        if index % 20 == 0:
            print(f"built {index}/{len(development)} development episodes", flush=True)

    source_hashes = dict(receipt_payload["source_hashes"])
    source_hashes["configs/datasets/phm_v6_0/evidence_calibration_receipt.json"] = (
        _sha256(HARD_CALIBRATION_RECEIPT)
    )
    policy, receipt = select_development_soft_calibration(
        episodes,
        dataset_id=receipt_payload["dataset_id"],
        runtime_evidence_package_sha256=_sha256(EVIDENCE_PACKAGE),
        source_hashes=source_hashes,
        selected_evidence_definition_ids=SELECTED_DEFINITION_IDS,
        policy_id="phm.v6_0.soft-semantic-development-r001",
        policy_version="0.1.0",
        receipt_id="phm-v6.0-soft-semantic-selection-20260831-r001",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "development_selected_policy.json"
    receipt_path = args.output_dir / "selection_receipt.json"
    policy_path.write_text(
        json.dumps(policy.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(receipt.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policy": str(policy_path),
                "policy_sha256": policy.policy_sha256,
                "receipt": str(receipt_path),
                "receipt_sha256": receipt.receipt_sha256,
                "decision": receipt.decision,
                "freeze_blockers": receipt.freeze_blockers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
