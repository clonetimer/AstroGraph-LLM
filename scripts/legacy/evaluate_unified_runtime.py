#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.diagnosis import UnifiedDiagnosisService, UnifiedBatchManifest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the unified diagnosis runtime on a trusted paired campaign")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trusted-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trusted_audit = json.loads(args.trusted_audit.read_text(encoding="utf-8"))
    if trusted_audit.get("status") != "PASS":
        raise SystemExit("trusted audit must be PASS")
    cases_root = args.dataset_root / "cases"
    service = UnifiedDiagnosisService()
    rows = []
    cli_api_mismatches = []
    replay_mismatches = []
    batch_cases = []
    api_payloads = {}
    with tempfile.TemporaryDirectory(prefix="astrograph-unified-eval-") as temp_dir:
        temp = Path(temp_dir)
        for case_root in sorted(path for path in cases_root.iterdir() if path.is_dir()):
            case_id = case_root.name
            truth = json.loads((case_root / "pair_validation.json").read_text(encoding="utf-8"))["scenario_id"]
            nominal = case_root / "nominal" / "trace.csv"
            effect = case_root / "effect" / "trace.csv"
            first = service.diagnose_paths(case_id=case_id, nominal_path=nominal, telemetry_path=effect, backend="auto")
            second = service.diagnose_paths(case_id=case_id, nominal_path=nominal, telemetry_path=effect, backend="auto")
            first_payload = first.model_dump(mode="json")
            api_payloads[case_id] = first_payload
            if first_payload != second.model_dump(mode="json"):
                replay_mismatches.append(case_id)
            cli_output = temp / f"{case_id}.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            completed = subprocess.run(
                [sys.executable, "-m", "astrograph.cli", "diagnose", "--case", case_id,
                 "--nominal", str(nominal), "--telemetry", str(effect), "--backend", "auto",
                 "--output-json", str(cli_output)],
                cwd=ROOT, env=env, text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0 or not cli_output.exists() or json.loads(cli_output.read_text()) != first_payload:
                cli_api_mismatches.append({"case_id": case_id, "returncode": completed.returncode,
                                           "stderr": completed.stderr[-500:]})
            correct = first.primary_effect_id == truth and first.status.value == "DIAGNOSED"
            false_diagnosis = first.status.value == "DIAGNOSED" and first.primary_effect_id != truth
            safe_non_conclusion = first.status.value in {"CONFLICTING_EVIDENCE", "AMBIGUOUS", "UNKNOWN_FAULT", "INSUFFICIENT_DATA"} and first.primary_effect_id is None
            rows.append({
                "case_id": case_id,
                "truth_effect_id": truth,
                "status": first.status.value,
                "primary_effect_id": first.primary_effect_id,
                "resolution_basis": first.resolution_basis,
                "exact_diagnosis": correct,
                "false_diagnosis": false_diagnosis,
                "safe_non_conclusion": safe_non_conclusion,
                "labels_used": first.labels_used,
                "truth_used": first.truth_used,
                "engine_runs": [
                    {"engine_id": run.engine_id, "execution_status": run.execution_status.value,
                     "status": run.status, "detected_effect_ids": run.detected_effect_ids}
                    for run in first.engine_runs
                ],
            })
            batch_cases.append({"case_id": case_id, "nominal": str(nominal), "telemetry": str(effect), "backend": "auto"})
        manifest = UnifiedBatchManifest.model_validate({
            "schema_version": "astrograph-unified-batch-manifest.v1", "cases": batch_cases
        })
        batch = service.diagnose_batch(manifest)
        batch_mismatches = [item.case_id for item in batch.results if item.model_dump(mode="json") != api_payloads[item.case_id]]

    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    payload = {
        "schema_version": "astrograph-unified-runtime-evaluation.v1",
        "trusted_audit_sha256": sha256_file(args.trusted_audit),
        "dataset_tree_sha256": trusted_audit.get("dataset_tree_sha256"),
        "case_count": len(rows),
        "exact_diagnosis_count": sum(row["exact_diagnosis"] for row in rows),
        "safe_non_conclusion_count": sum(row["safe_non_conclusion"] for row in rows),
        "false_diagnosis_count": sum(row["false_diagnosis"] for row in rows),
        "data_error_count": sum(row["status"] == "DATA_ERROR" for row in rows),
        "label_or_truth_use_count": sum(row["labels_used"] or row["truth_used"] for row in rows),
        "cli_api_mismatch_count": len(cli_api_mismatches),
        "batch_api_mismatch_count": len(batch_mismatches),
        "deterministic_replay_mismatch_count": len(replay_mismatches),
        "status_counts": dict(sorted(status_counts.items())),
        "cli_api_mismatches": cli_api_mismatches,
        "batch_api_mismatches": batch_mismatches,
        "deterministic_replay_mismatches": replay_mismatches,
        "cases": rows,
        "claim_boundary": "Trusted minimum campaign runtime interoperability and fail-closed routing only; not broad diagnosis accuracy, open-set calibration, causal proof, flight qualification, or safety approval.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "case_count", "exact_diagnosis_count", "safe_non_conclusion_count", "false_diagnosis_count",
        "data_error_count", "cli_api_mismatch_count", "batch_api_mismatch_count",
        "deterministic_replay_mismatch_count", "status_counts")}, indent=2))
    return 0 if (
        payload["case_count"] >= 6
        and payload["exact_diagnosis_count"] >= 4
        and payload["false_diagnosis_count"] == 0
        and payload["data_error_count"] == 0
        and payload["label_or_truth_use_count"] == 0
        and payload["cli_api_mismatch_count"] == 0
        and payload["batch_api_mismatch_count"] == 0
        and payload["deterministic_replay_mismatch_count"] == 0
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
