"""R21 frozen failure analysis and post-execution governance validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.data_exposure import contamination_audit
from astrograph.evaluation.development_remediation import load_preregistration, validate_existing_development_reports
from astrograph.evaluation.real_data_failure_analysis import analyze_frozen_bundle


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r20-receipts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ops-development-report", type=Path, required=True)
    parser.add_argument("--esa-development-report", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    registry = json.loads((source / "configs/evaluation/r21_data_exposure_registry.v1.json").read_text(encoding="utf-8"))
    plan = load_preregistration()
    analysis = analyze_frozen_bundle(args.r20_receipts, registry)
    write(args.results / "dxc_failure_mechanism_report.json", analysis["DXC"])
    write(args.results / "ops_failure_mechanism_report.json", analysis["OPS"])
    ops = json.loads(args.ops_development_report.read_text(encoding="utf-8"))
    esa = json.loads(args.esa_development_report.read_text(encoding="utf-8"))
    validation = validate_existing_development_reports(ops, esa)
    audit = contamination_audit([ops, esa], plan)
    write(args.results / "R21_CONTAMINATION_AUDIT.json", audit)
    gate = {"schema_version": "astrograph-r21-gate-summary.v1",
            "status": "DEVELOPMENT_REMEDIATION_READY_FOR_NEW_REPLICATION" if audit["status"] == "PASS" else audit["status"],
            "frozen_files": analysis["verification"]["original_frozen_files"],
            "inference_rerun": False, "external_predictions_generated": False,
            "new_external_validation_receipt_generated": False,
            "development_report_validation": validation, "contamination": audit["status"],
            "qualification": "NOT_FLIGHT_QUALIFIED"}
    write(args.results / "R21_GATE_SUMMARY.json", gate)
    print(json.dumps(gate, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
