#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astrograph.diagnosis.unified import UnifiedDiagnosisService  # noqa: E402


EXPECTED_LOCATION = {
    "eps_battery_capacity_loss": ("EPS", "Battery"),
    "adcs_rw_bearing_seizure": ("ADCS", "ReactionWheel"),
    "thermal_heater_stuck_on": ("THERMAL", "Heater"),
    "payload_instrument_off": ("PAYLOAD", "Instrument"),
    "payload_instrument_degraded_rate": ("PAYLOAD", "Instrument"),
    "comm_data_downlink_link_loss": ("COMM_DATA", "Transmitter"),
}


def _strip_case_id(value):
    if isinstance(value, dict):
        return {key: _strip_case_id(item) for key, item in value.items() if key != "case_id"}
    if isinstance(value, list):
        return [_strip_case_id(item) for item in value]
    return value


def _macro_f1(truth: list[str], predicted: list[str | None]) -> float:
    scores = []
    for label in sorted(set(truth)):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(campaign_root: Path) -> dict:
    campaign = json.loads((campaign_root / "campaign_manifest.json").read_text(encoding="utf-8"))
    service = UnifiedDiagnosisService()
    rows: list[dict] = []
    replay_mismatches = 0
    anonymized_mismatches = 0
    for index, item in enumerate(campaign.get("case_results", []), start=1):
        case_id = item["case_id"]
        truth_effect = item["scenario_id"]
        case_root = campaign_root / "cases" / case_id
        pair_truth = json.loads((case_root / "pair_validation.json").read_text(encoding="utf-8"))
        effect_manifest = json.loads((case_root / "effect" / "manifest.json").read_text(encoding="utf-8"))
        result = service.diagnose_paths(
            case_id=f"opaque-{index:03d}",
            telemetry_path=case_root / "effect" / "trace.csv",
            nominal_path=case_root / "nominal" / "trace.csv",
            backend="composite",
        )
        replay = service.diagnose_paths(
            case_id=f"opaque-{index:03d}",
            telemetry_path=case_root / "effect" / "trace.csv",
            nominal_path=case_root / "nominal" / "trace.csv",
            backend="composite",
        )
        anonymized = service.diagnose_paths(
            case_id=f"permuted-{1000-index}",
            telemetry_path=case_root / "effect" / "trace.csv",
            nominal_path=case_root / "nominal" / "trace.csv",
            backend="composite",
        )
        if result.model_dump(mode="json") != replay.model_dump(mode="json"):
            replay_mismatches += 1
        if _strip_case_id(result.model_dump(mode="json")) != _strip_case_id(anonymized.model_dump(mode="json")):
            anonymized_mismatches += 1
        expected_subsystem, expected_component = EXPECTED_LOCATION[truth_effect]
        predicted = result.primary_effect_id
        exact = result.status.value == "DIAGNOSED" and predicted == truth_effect
        false_diagnosis = result.status.value == "DIAGNOSED" and predicted != truth_effect
        config = pair_truth.get("fault_configuration", {})
        expected_onset = float(config.get("onset_time_s", 0.0))
        expected_severity = float(config.get("magnitude", 1.0))
        onset_error = (
            abs(float(result.estimated_onset_time_s) - expected_onset)
            if result.estimated_onset_time_s is not None else None
        )
        severity_error = (
            abs(float(result.severity_score) - expected_severity)
            if result.severity_score is not None else None
        )
        rows.append({
            "case_id": case_id,
            "split": effect_manifest.get("split", "unknown"),
            "truth_effect_id": truth_effect,
            "predicted_effect_id": predicted or "",
            "diagnosis_status": result.status.value,
            "exact": exact,
            "false_diagnosis": false_diagnosis,
            "truth_subsystem": expected_subsystem,
            "predicted_subsystem": result.primary_subsystem or "",
            "subsystem_exact": result.primary_subsystem == expected_subsystem,
            "truth_component": expected_component,
            "predicted_component": result.primary_component or "",
            "component_exact": result.primary_component == expected_component,
            "truth_onset_time_s": expected_onset,
            "estimated_onset_time_s": result.estimated_onset_time_s,
            "onset_absolute_error_s": onset_error,
            "truth_severity": expected_severity,
            "estimated_severity_score": result.severity_score,
            "severity_absolute_error": severity_error,
            "resolution_basis": result.resolution_basis,
            "label_or_truth_used": bool(result.labels_used or result.truth_used),
        })
    truth = [row["truth_effect_id"] for row in rows]
    predicted = [row["predicted_effect_id"] or None for row in rows]
    onset_errors = [float(row["onset_absolute_error_s"]) for row in rows if row["onset_absolute_error_s"] is not None]
    severity_errors = [float(row["severity_absolute_error"]) for row in rows if row["severity_absolute_error"] is not None]
    holdout = [row for row in rows if row["split"] == "holdout"]
    per_effect = {}
    for effect_id in sorted(set(truth)):
        selected = [row for row in rows if row["truth_effect_id"] == effect_id]
        per_effect[effect_id] = {
            "case_count": len(selected),
            "exact_count": sum(bool(row["exact"]) for row in selected),
            "false_diagnosis_count": sum(bool(row["false_diagnosis"]) for row in selected),
        }
    return {
        "schema_version": "astrograph-trusted-single-fault-evaluation.v1",
        "status": "PASS" if rows else "FAIL",
        "campaign_id": campaign.get("campaign_id"),
        "campaign_status": campaign.get("status"),
        "case_count": len(rows),
        "run_count": campaign.get("total_run_count", 0),
        "acceptance_valid_count": campaign.get("acceptance_valid_count", 0),
        "acceptance_valid_pair_count": campaign.get("acceptance_valid_pair_count", 0),
        "public_trace_leakage_count": campaign.get("public_trace_leakage_count", 0),
        "exact_diagnosis_count": sum(bool(row["exact"]) for row in rows),
        "exact_diagnosis_rate": sum(bool(row["exact"]) for row in rows) / len(rows) if rows else 0.0,
        "false_diagnosis_count": sum(bool(row["false_diagnosis"]) for row in rows),
        "safe_non_conclusion_count": sum(row["diagnosis_status"] != "DIAGNOSED" for row in rows),
        "macro_f1": _macro_f1(truth, predicted),
        "holdout_case_count": len(holdout),
        "holdout_exact_count": sum(bool(row["exact"]) for row in holdout),
        "subsystem_exact_count": sum(bool(row["subsystem_exact"]) for row in rows),
        "component_exact_count": sum(bool(row["component_exact"]) for row in rows),
        "onset_mae_s": sum(onset_errors) / len(onset_errors) if onset_errors else None,
        "onset_max_error_s": max(onset_errors) if onset_errors else None,
        "severity_mae": sum(severity_errors) / len(severity_errors) if severity_errors else None,
        "severity_max_error": max(severity_errors) if severity_errors else None,
        "label_or_truth_use_count": sum(bool(row["label_or_truth_used"]) for row in rows),
        "deterministic_replay_mismatch_count": replay_mismatches,
        "case_id_permutation_mismatch_count": anonymized_mismatches,
        "status_counts": dict(sorted(Counter(row["diagnosis_status"] for row in rows).items())),
        "per_effect": per_effect,
        "cases": rows,
        "claim_boundary": (
            "Six-mode paired laboratory single-fault evaluation on the frozen trusted Basilisk campaign; "
            "not broad fault coverage, open-set calibration, flight qualification, or safety approval."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path)
    args = parser.parse_args()
    result = evaluate(args.campaign_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.case_csv:
        args.case_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = result["cases"]
        with args.case_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case_id"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: result[key] for key in (
        "status", "case_count", "exact_diagnosis_count", "false_diagnosis_count",
        "holdout_exact_count", "subsystem_exact_count", "component_exact_count",
        "onset_mae_s", "severity_mae"
    )}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
