#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_identity(root: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_digest = _sha256(path)
        file_size = path.stat().st_size
        digest.update(f"{relative}\0{file_digest}\0{file_size}\n".encode("utf-8"))
        count += 1
        size += file_size
    return {"sha256": digest.hexdigest(), "file_count": count, "size_bytes": size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, action="append", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    truth = args.truth.resolve()
    public_roots = [item.resolve() for item in args.public_root]
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    identity_checks = {
        "truth_manifest_identity": (
            truth.is_file()
            and _sha256(truth) == identity["truth_manifest"]["sha256"]
            and truth.stat().st_size == identity["truth_manifest"]["size_bytes"]
        ),
        "formal_public_identity": public_roots[0].is_dir() and _tree_identity(public_roots[0]) == identity["formal_public"],
        "research_public_identity": public_roots[1].is_dir() and _tree_identity(public_roots[1]) == identity["research_public"],
    }
    raw_output = args.output.with_suffix(".evaluation.json")
    command = [
        sys.executable,
        str(project / "scripts" / "evaluate_expanded_single_fault.py"),
        "--truth", str(truth),
    ]
    for root in public_roots:
        command.extend(["--public-root", str(root)])
    command.extend(["--output", str(raw_output.resolve())])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = None
    payload: dict = {}
    if all(identity_checks.values()):
        completed = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=180)
        payload = json.loads(raw_output.read_text(encoding="utf-8")) if raw_output.exists() else {}
    checks = {
        **identity_checks,
        "process_exit_zero": completed is not None and completed.returncode == 0,
        "evaluation_pass": payload.get("status") == "PASS",
        "formal_132_complete": payload.get("formal", {}).get("class_matches") == 132,
        "holdout_69_complete": payload.get("holdout", {}).get("class_matches") == 69,
        "formal_exact_identifiable_complete": (
            payload.get("formal", {}).get("exact_mode_matches")
            == payload.get("formal", {}).get("exact_mode_expected")
        ),
        "candidate_sets_complete": payload.get("formal", {}).get("candidate_complete") == 132,
        "research_only_6_separate": payload.get("research_only", {}).get("count") == 6,
        "no_public_leakage": not payload.get("leakage_findings"),
        "no_case_id_dependence": bool(payload.get("results")) and all(
            item.get("case_id_permutation_invariant") for item in payload.get("results", [])
        ),
        "no_labels_or_truth_used": not any(
            item.get("labels_used") or item.get("truth_used") for item in payload.get("results", [])
        ),
    }
    decision = {
        "schema_version": "astrograph-expanded-single-fault-gate.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "command": command,
        "returncode": completed.returncode if completed is not None else None,
        "stdout": completed.stdout if completed is not None else "",
        "stderr": completed.stderr if completed is not None else "IDENTITY_CHECK_FAILED",
        "evaluation_path": str(raw_output),
        "identity_path": str(args.identity.resolve()),
        "claim_boundary": (
            "4C paired single-fault capability gate. Formal and research-only data remain separated; "
            "observational-equivalence classes are not counted as exact root-mode diagnoses. Restricted truth "
            "remains outside the candidate source tree and is accepted only when its frozen identity matches."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "checks": checks}, indent=2))
    return 0 if decision["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
