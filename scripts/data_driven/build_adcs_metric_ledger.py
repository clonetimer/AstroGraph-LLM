#!/usr/bin/env python3
"""Build the current ADCS metric/protocol ledger from frozen project evidence.

The output is descriptive governance evidence.  It does not train, recalibrate,
modify, admit, or promote any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.data_driven.adcs_metric_ledger import ADCSMetricProtocolLedger, file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def build() -> ADCSMetricProtocolLedger:
    p2_path = "governance/evidence/research/data_driven_runs/data_driven_mechanism_p2_20260818/report.json"
    prod_path = "governance/evidence/research/data_driven_runs/adcs_p2_production_group_cv_20260827/report.json"
    replay_path = "governance/implementation/post48_dev3/receipts/METRICS_SUMMARY.json"
    head_path = "governance/implementation/post48_dev3/used_test_final/report.json"
    binding_path = "configs/agent/mission.default.json"

    p2 = _load(p2_path)
    prod = _load(prod_path)
    replay = _load(replay_path)
    heads = _load(head_path)
    binding = _load(binding_path)
    current_sha = binding["approved_model_sha256"][0]
    replay_branch = replay["branches"]["sensor_agent"]
    raw_head = heads["model_heads"]["classification_raw_head"]
    direct_localization = heads["model_heads"]["localization_direct_head"]
    succeeded = int(heads["execution_status_counts"]["SUCCEEDED"])
    unavailable = int(heads["execution_status_counts"]["MODEL_UNAVAILABLE"])

    records = [
        {
            "record_id": "adcs-p2-innovation-5fold-20260818",
            "protocol_id": "ADCS.P2.GROUP_CV.5FOLD.INNOVATION",
            "evidence_role": "RESEARCH_GROUP_CV_BASELINE",
            "evidence_date": "2026-08-18",
            "cohort": "five-fold trajectory_pair_id group CV; test fold used once per fold after train/validation selection",
            "reuse_status": "TRAIN_TEST_GROUP_CV",
            "model_sha256": None,
            "metrics": {
                **{k: float(v) for k, v in p2["adcs_group_cv"]["candidate_absolute_mean"].items()},
            },
            "denominators": {"folds": 5},
            "source_path": p2_path,
            "source_sha256": file_sha256(ROOT / p2_path),
            "independent_confirmation": False,
            "current_operational_replay_evidence": False,
            "claim_boundary": "E0 synthetic five-fold research baseline. Useful for cross-group generalization evidence within this dataset; not the current Mission replay score and not flight/cross-satellite confirmation.",
        },
        {
            "record_id": "adcs-production-qualification-20260827",
            "protocol_id": "ADCS.P2.PRODUCTION_QUALIFICATION.5FOLD.20260827",
            "evidence_role": "HISTORICAL_PRODUCTION_QUALIFICATION",
            "evidence_date": prod["date"],
            "cohort": "five-fold trajectory_pair_id qualification for the frozen post24 P2 candidate",
            "reuse_status": "HISTORICAL_FROZEN_QUALIFICATION",
            "model_sha256": "8facdc8ef3968d4ad37ffdb650b85c93a4aeafa03eb868d9b28b90e8e446b8a7",
            "metrics": {
                "detection_macro_f1": float(prod["metrics"]["detection_macro_f1"]["mean"]),
                "localization_macro_f1": float(prod["metrics"]["localization_macro_f1"]["mean"]),
                "classification_macro_f1": float(prod["metrics"]["classification_macro_f1"]["mean"]),
                "end_to_end_macro_f1": float(prod["metrics"]["e2e_macro_f1"]["mean"]),
            },
            "denominators": {"folds": int(prod["protocol"]["folds"])},
            "source_path": prod_path,
            "source_sha256": file_sha256(ROOT / prod_path),
            "independent_confirmation": False,
            "current_operational_replay_evidence": False,
            "claim_boundary": "Historical qualification decision for the 2026-08-27 frozen candidate. It must not be reported as the current post48 Mission replay performance.",
        },
        {
            "record_id": "adcs-used-groups-normal-replay-post48-dev3",
            "protocol_id": "ADCS.POST48.USED_TEST_GROUPS.NORMAL_REPLAY",
            "evidence_role": "USED_GROUP_RETROSPECTIVE_REPLAY",
            "evidence_date": "2026-09-13",
            "cohort": replay["cohort"],
            "reuse_status": "PREVIOUSLY_USED_TEST_GROUPS",
            "model_sha256": replay["weight_sha256"],
            "metrics": {
                "detection_macro_f1_executable": float(replay_branch["detection_f1_executable"]),
                "detection_macro_f1_all_windows": float(replay_branch["detection_f1_all"]),
                "localization_direct_head_macro_f1": float(direct_localization["macro_f1"]),
                "localization_system_derived_macro_f1": float(replay_branch["system_derived_localization_f1"]),
                "classification_raw_head_macro_f1": float(raw_head["macro_f1"]),
                "classification_system_macro_f1": float(replay_branch["classification_f1"]),
            },
            "denominators": {
                "windows_all": int(replay["windows"]),
                "windows_executable": succeeded,
                "windows_unavailable": unavailable,
                "active_fault_windows": int(replay["active_windows"]),
                "execution_coverage": succeeded / int(replay["windows"]),
            },
            "source_path": replay_path,
            "source_sha256": file_sha256(ROOT / replay_path),
            "independent_confirmation": False,
            "current_operational_replay_evidence": True,
            "claim_boundary": replay["claim_boundary"] + " This cohort has been repeatedly used in post48 development and is not new independent confirmation.",
        },
        {
            "record_id": "adcs-current-mission-binding-post48-dev23",
            "protocol_id": "ADCS.MISSION.CURRENT_MODEL_BINDING",
            "evidence_role": "CURRENT_MISSION_BINDING",
            "evidence_date": "2026-09-15",
            "cohort": "configuration binding only; no accuracy denominator",
            "reuse_status": "CONFIGURATION_BINDING_ONLY",
            "model_sha256": current_sha,
            "metrics": {},
            "denominators": {},
            "source_path": binding_path,
            "source_sha256": file_sha256(ROOT / binding_path),
            "independent_confirmation": False,
            "current_operational_replay_evidence": False,
            "claim_boundary": "Records which model hash the default Mission configuration approves. A binding is not itself a performance result.",
        },
    ]
    payload = {
        "version": "0.7.1.post48.dev23",
        "current_mission_model_sha256": current_sha,
        "records": records,
        "reporting_rules": [
            "Never report an ADCS metric without protocol_id and cohort/reuse status.",
            "Always report detection on both executable windows and all windows when unavailable inputs exist.",
            "Keep direct localization-head metrics separate from system-derived localization.",
            "Keep raw classification-head metrics separate from final system classification.",
            "Previously used sample_04/sample_09 replay is retrospective regression evidence, not new independent confirmation.",
            "Historical production qualification metrics must not be substituted for current Mission normal-mode replay metrics.",
        ],
        "correction_note": "post48.dev23 corrects a development-planning error in which the 2026-08-27 historical production qualification metrics were temporarily treated as the current ADCS three-task result. No production model/KG/memory state is changed by this correction.",
    }
    return ADCSMetricProtocolLedger.model_validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "governance/implementation/post48_dev23/ADCS_METRIC_PROTOCOL_LEDGER.json")
    args = parser.parse_args()
    ledger = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
