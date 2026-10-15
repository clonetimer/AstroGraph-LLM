#!/usr/bin/env python3
"""Evaluate the frozen AstroGraph G1 protocol without converting pending approval into PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root: Path, sat_root: Path) -> dict:
    protocol = root / "protocols/acceptance"
    names = ("diagnosis_scope", "fault_taxonomy", "coverage_matrix", "metric_policy", "reasoning_rubric")
    checks: dict[str, bool] = {}
    approvals = _read(protocol / "approvals.json")
    for name in names:
        payload_path = protocol / f"{name}.json"
        schema_path = protocol / f"{name}.schema.json"
        Draft202012Validator(_read(schema_path)).validate(_read(payload_path))
        checks[f"{name}_hash_frozen"] = approvals["artifacts"].get(payload_path.name) == _sha(payload_path)
    scope = _read(protocol / "diagnosis_scope.json")
    coverage = _read(protocol / "coverage_matrix.json")
    checks["object_count_31"] = scope.get("object_count") == 31 and len(scope.get("objects", [])) == 31
    checks["coverage_rows_93"] = len(coverage.get("rows", [])) == 93
    checks["three_modes_per_object"] = all(
        {row["scenario_type"] for row in coverage["rows"] if row["object_id"] == item["object_id"]} == {"nominal", "fault", "degradation"}
        for item in scope["objects"]
    )
    checks["sat_contracts_present"] = (sat_root / "configs/acceptance/object_scope.json").is_file() and (sat_root / "configs/acceptance/scenario_baseline.json").is_file()
    checks["project_owner_approved"] = approvals["project_owner"]["status"] == "APPROVED"
    checks["independent_evaluator_approved"] = approvals["independent_evaluator"]["status"] == "APPROVED"
    technical = all(value for key, value in checks.items() if key != "independent_evaluator_approved")
    status = "PASS" if technical and checks["independent_evaluator_approved"] else "BLOCKED_APPROVAL" if technical else "FAIL"
    return {"schema_version": "astrograph-scope-gate.v1", "technical_status": "PASS" if technical else "FAIL", "status": status, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sat-root", type=Path, default=os.environ.get("ASTROGRAPH_SAT_ROOT"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sat_root is None:
        parser.error("--sat-root or ASTROGRAPH_SAT_ROOT is required")
    result = evaluate(args.project_root.resolve(), args.sat_root.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["status"] in {"PASS", "BLOCKED_APPROVAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
