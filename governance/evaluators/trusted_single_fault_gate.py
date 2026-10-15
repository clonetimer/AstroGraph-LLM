#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    checks = []

    def add(criterion_id, observed, expected, passed):
        checks.append({
            "criterion_id": criterion_id,
            "observed": observed,
            "expected": expected,
            "status": "PASS" if passed else "FAIL",
            "severity": "HARD",
        })

    add("CAMPAIGN_STATUS", evaluation.get("campaign_status"), "PASS", evaluation.get("campaign_status") == "PASS")
    add("TRUSTED_RUN_COUNT", evaluation.get("acceptance_valid_count"), 48, evaluation.get("acceptance_valid_count") == 48)
    add("TRUSTED_PAIR_COUNT", evaluation.get("acceptance_valid_pair_count"), 24, evaluation.get("acceptance_valid_pair_count") == 24)
    add("NO_PUBLIC_TRUTH_LEAKAGE", evaluation.get("public_trace_leakage_count"), 0, evaluation.get("public_trace_leakage_count") == 0)
    add("MINIMUM_EXACT_DIAGNOSIS", evaluation.get("exact_diagnosis_count"), ">=22/24", evaluation.get("exact_diagnosis_count", 0) >= 22)
    add("NO_FALSE_DIAGNOSIS", evaluation.get("false_diagnosis_count"), 0, evaluation.get("false_diagnosis_count") == 0)
    add("HOLDOUT_EXACT", evaluation.get("holdout_exact_count"), "6/6", evaluation.get("holdout_case_count") == 6 and evaluation.get("holdout_exact_count") == 6)
    add("MACRO_F1", evaluation.get("macro_f1"), ">=0.90", float(evaluation.get("macro_f1", 0.0)) >= 0.90)
    add("SUBSYSTEM_LOCALIZATION", evaluation.get("subsystem_exact_count"), ">=23/24", evaluation.get("subsystem_exact_count", 0) >= 23)
    add("COMPONENT_LOCALIZATION", evaluation.get("component_exact_count"), ">=23/24", evaluation.get("component_exact_count", 0) >= 23)
    add("ONSET_ERROR", evaluation.get("onset_max_error_s"), "<=10s", evaluation.get("onset_max_error_s") is not None and float(evaluation["onset_max_error_s"]) <= 10.0)
    add("SEVERITY_ERROR", evaluation.get("severity_max_error"), "<=0.10", evaluation.get("severity_max_error") is not None and float(evaluation["severity_max_error"]) <= 0.10)
    add("NO_LABEL_OR_TRUTH_USE", evaluation.get("label_or_truth_use_count"), 0, evaluation.get("label_or_truth_use_count") == 0)
    add("DETERMINISTIC_REPLAY", evaluation.get("deterministic_replay_mismatch_count"), 0, evaluation.get("deterministic_replay_mismatch_count") == 0)
    add("CASE_ID_INDEPENDENCE", evaluation.get("case_id_permutation_mismatch_count"), 0, evaluation.get("case_id_permutation_mismatch_count") == 0)
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    result = {
        "schema_version": "astrograph-trusted-single-fault-gate.v1",
        "status": status,
        "checks": checks,
        "blockers": [item["criterion_id"] for item in checks if item["status"] != "PASS"],
        "claim_boundary": (
            "Frozen six-mode trusted paired single-fault campaign only; no broad coverage, open-set, "
            "causal, flight, or safety qualification claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
