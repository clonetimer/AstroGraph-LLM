#!/usr/bin/env python3
"""Audit current ADCS metric semantics and independent-confirmation readiness."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from astrograph.data_driven.adcs_metric_ledger import load_ledger, verify_ledger_sources

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "governance/implementation/post48_dev23/ADCS_METRIC_PROTOCOL_LEDGER.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_current_status() -> dict:
    ledger = load_ledger(LEDGER)
    verify_ledger_sources(ROOT, ledger)
    replay = next(r for r in ledger.records if r.evidence_role == "USED_GROUP_RETROSPECTIVE_REPLAY")
    current_model = ROOT / "models/adcs_v5_2_2/p2_innovation_v2.pt"
    current_sha = sha256(current_model)
    if current_sha != ledger.current_mission_model_sha256:
        raise ValueError("current Mission model file hash differs from ledger binding")
    m = replay.metrics
    three_task_gate = {
        "detection_executable_ge_0p95": bool(m["detection_macro_f1_executable"] >= 0.95),
        "localization_system_ge_0p95": bool(m["localization_system_derived_macro_f1"] >= 0.95),
        "classification_system_ge_0p95": bool(m["classification_system_macro_f1"] >= 0.95),
    }
    coverage = float(replay.denominators["execution_coverage"])
    return {
        "schema_version": "astrograph-adcs-current-status.v1",
        "version": "0.7.1.post48.dev23",
        "current_model_path": "models/adcs_v5_2_2/p2_innovation_v2.pt",
        "current_model_sha256": current_sha,
        "metric_record_id": replay.record_id,
        "protocol_id": replay.protocol_id,
        "cohort": replay.cohort,
        "three_task_metrics": {
            "detection_macro_f1_executable": m["detection_macro_f1_executable"],
            "localization_direct_head_macro_f1": m["localization_direct_head_macro_f1"],
            "localization_system_derived_macro_f1": m["localization_system_derived_macro_f1"],
            "classification_raw_head_macro_f1": m["classification_raw_head_macro_f1"],
            "classification_system_macro_f1": m["classification_system_macro_f1"],
        },
        "all_window_detection_macro_f1": m["detection_macro_f1_all_windows"],
        "execution": {
            "windows_all": replay.denominators["windows_all"],
            "windows_executable": replay.denominators["windows_executable"],
            "windows_unavailable": replay.denominators["windows_unavailable"],
            "execution_coverage": coverage,
        },
        "three_task_gate": three_task_gate,
        "three_task_gate_passed": all(three_task_gate.values()),
        "full_window_coverage_complete": coverage == 1.0,
        "independent_confirmation": False,
        "status": "THREE_TASK_USED_GROUP_REPLAY_METRICS_PASS_WITH_COVERAGE_GAP",
        "claim_boundary": "Three-task thresholds are met on the previously used sample_04/sample_09 replay cohort. This is current regression evidence, not new independent confirmation. Detection must also report the all-window metric because 45/546 windows are input-unavailable.",
    }


def build_independent_status(dataset_zip: Path) -> dict:
    dataset_zip = dataset_zip.resolve()
    with zipfile.ZipFile(dataset_zip) as archive:
        names = archive.namelist()
        sample_ids = sorted({m.group(1) for name in names if (m := re.search(r"sample_(\d{2})", name))})
        split_member = next((n for n in names if n.endswith("/audit/evaluation_splits.json")), None)
        if split_member is None:
            raise ValueError("dataset zip lacks audit/evaluation_splits.json")
        split_audit = json.loads(archive.read(split_member))
    historically_consumed = [f"{i:02d}" for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
    unused = sorted(set(sample_ids) - set(historically_consumed))
    protocol = split_audit["split_protocol"]
    return {
        "schema_version": "astrograph-adcs-independent-confirmation-status.v1",
        "version": "0.7.1.post48.dev23",
        "dataset_zip_sha256": sha256(dataset_zip),
        "dataset_name": "whole_spacecraft_adcs_paired_dataset_v5_2_2",
        "dataset_sample_ids": sample_ids,
        "historically_consumed_sample_ids": historically_consumed,
        "unused_sample_ids": unused,
        "primary_group_key": protocol["primary_group_key"],
        "same_initial_state_never_crosses_train_test": protocol["same_initial_state_never_crosses_train_test"],
        "new_independent_group_available_in_uploaded_dataset": bool(unused),
        "status": "READY" if unused else "NEW_GROUP_REQUIRED",
        "independent_confirmation_performed": False,
        "required_confirmation_contract": {
            "new_group_identity": "trajectory_pair_id values absent from all historical train/validation/test/replay artifacts",
            "minimum_data_scope": "new Monte-Carlo initial-state groups covering the supported ADCS fault/mission scope intended for the claim",
            "model_freeze": "p2_innovation_v2.pt SHA-256 39062457c1ed6d68c1478afa1d125ba54b80a6e07936de0a4078dca865852732",
            "prediction_before_truth": True,
            "labels_withheld_until_prediction_freeze": True,
            "threshold_tuning_on_confirmation_data": False,
            "all_windows_retained": True,
            "report_required": [
                "detection_macro_f1_executable",
                "detection_macro_f1_all_windows",
                "execution_coverage",
                "localization_direct_head_macro_f1",
                "localization_system_derived_macro_f1",
                "classification_raw_head_macro_f1",
                "classification_system_macro_f1",
            ],
        },
        "claim_boundary": "The uploaded ADCS.zip contains sample_00 through sample_09 only; all have already participated in historical train/validation/test or repeated regression. Re-splitting them cannot be relabeled as new independent confirmation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "governance/implementation/post48_dev23")
    args = parser.parse_args()
    current = build_current_status()
    independent = build_independent_status(args.dataset_zip)
    _save(args.output_dir / "ADCS_CURRENT_V2_STATUS.json", current)
    _save(args.output_dir / "ADCS_INDEPENDENT_CONFIRMATION_STATUS.json", independent)
    print(json.dumps({"current": current["status"], "independent": independent["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
