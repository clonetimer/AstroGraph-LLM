#!/usr/bin/env python3
"""Externally pinned Round-2 trusted-data unlock gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAL = {"training", "calibration", "validation", "acceptance"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks: list[dict] = []
    blockers: list[str] = []
    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in args.manifest_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        audit, rows = {}, []
        blockers.append(f"EVIDENCE_UNREADABLE:{type(exc).__name__}")

    def check(criterion: str, condition: bool, observed: object, expected: object) -> None:
        checks.append({
            "criterion_id": criterion, "status": "PASS" if condition else "FAIL",
            "observed": observed, "expected": expected, "severity": "HARD",
        })
        if not condition:
            blockers.append(criterion)

    total = int(audit.get("total_run_count", -1))
    accepted = int(audit.get("acceptance_valid_count", -1))
    pairs = int(audit.get("acceptance_valid_pair_count", -1))
    check("AUDIT_PROVENANCE", bool(audit.get("dataset_tree_sha256") and audit.get("sat_agent_tree_sha256")), True, True)
    check("MANIFEST_INTEGRITY", len(rows) == total and total >= 12, f"rows={len(rows)};total={total}", ">=12 and equal")
    check("ACCEPTANCE_VALID_RUNS", accepted >= 12, accepted, ">=12")
    check("ACCEPTANCE_VALID_PAIRS", pairs >= 3, pairs, ">=3")
    check("NO_LABEL_LEAKAGE", int(audit.get("public_trace_leakage_count", -1)) == 0, audit.get("public_trace_leakage_count"), 0)
    false_acceptance = [
        row.get("run_id") for row in rows
        if row.get("acceptance_level") == "ACCEPTANCE_VALID"
        and (row.get("full_gate_status") != "PASS" or row.get("effect_evidence_status") != "PASS")
    ]
    check("NO_FALSE_ACCEPTANCE", not false_acceptance, false_acceptance[:5], [])
    invalid_usage = [
        row.get("run_id") for row in rows
        if row.get("acceptance_level") != "ACCEPTANCE_VALID"
        and FORMAL.intersection(row.get("allowed_uses", []))
    ]
    check("USE_RESTRICTIONS", not invalid_usage, invalid_usage[:5], [])
    check("REPRODUCTION_READY", audit.get("reproduction_ready") is True, audit.get("reproduction_ready"), True)
    check("AUDIT_STATUS", audit.get("status") == "PASS", audit.get("status"), "PASS")
    status = "PASS" if not blockers else "BLOCKED"
    output = {
        "schema_version": "astrograph-trusted-data-unlock-gate.v2",
        "status": status,
        "audit_sha256": sha256(args.audit) if args.audit.is_file() else "",
        "manifest_sha256": sha256(args.manifest_jsonl) if args.manifest_jsonl.is_file() else "",
        "checks": checks,
        "blockers": blockers,
        "claim_boundary": "Trusted data provenance and reproducibility only; not diagnosis accuracy or flight qualification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
