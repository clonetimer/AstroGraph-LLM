#!/usr/bin/env python3
"""Evaluate the ADCS P2 candidate against explicit fail-closed production gates.

Missing governance evidence is a BLOCKED result, not an uncaught exception and
not an invitation to reconstruct historical metrics from prose.  The report
records every missing immutable artifact explicitly so source-only packages can
still execute the qualification gate deterministically without fabricating
research evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs/data_driven/adcs_production_qualification.json"
CLASS_MAP = ROOT / "configs/datasets/adcs_v5_2_2/fault_class_map.json"
HIERARCHY = ROOT / "configs/datasets/adcs_v5_2_2/diagnosis_hierarchy.json"

_METRIC_NAMES = (
    "detection_macro_f1",
    "localization_macro_f1",
    "classification_macro_f1",
    "e2e_macro_f1",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(
    metrics_evidence: dict[str, Any] | None, metric_name: str
) -> float | None:
    if metrics_evidence is None:
        return None
    value = (metrics_evidence.get("metrics") or {}).get(metric_name)
    if not isinstance(value, dict) or "mean" not in value:
        return None
    try:
        return float(value["mean"])
    except (TypeError, ValueError):
        return None


def evaluate(root: Path = ROOT) -> dict[str, Any]:
    policy_path = root / POLICY.relative_to(ROOT)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    metrics_path = root / policy["metrics_evidence_path"]
    metrics_evidence = _load_json(metrics_path)
    class_map = json.loads(
        (root / CLASS_MAP.relative_to(ROOT)).read_text(encoding="utf-8")
    )
    hierarchy = json.loads(
        (root / HIERARCHY.relative_to(ROOT)).read_text(encoding="utf-8")
    )
    metrics = {
        metric_name: _metric_value(metrics_evidence, metric_name)
        for metric_name in _METRIC_NAMES
    }
    metric_gates = policy["metric_gates"]
    runtime_gates = policy["runtime_gates"]
    class_count = len(hierarchy["fault_to_component"])
    mapped_count = len(class_map["mappings"])
    mapping_coverage = mapped_count / class_count

    package_path = root / runtime_gates["model_package_path"]
    receipt_path = root / runtime_gates["model_receipt_path"]
    package_hash_matches = False
    if package_path.is_file() and receipt_path.is_file():
        digest = hashlib.sha256(package_path.read_bytes()).hexdigest()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        package_hash_matches = digest == receipt.get("package_sha256")

    chain_path = root / runtime_gates["runtime_chain_evidence_path"]
    chain_statuses: dict[str, str] = {
        name: "MISSING" for name in runtime_gates["required_chain_statuses"]
    }
    chain = _load_json(chain_path)
    if chain is not None:
        chain_statuses = {
            name: str((chain.get("chains") or {}).get(name, {}).get("status", "MISSING"))
            for name in runtime_gates["required_chain_statuses"]
        }

    folds = None
    if metrics_evidence is not None:
        try:
            folds = int((metrics_evidence.get("protocol") or {}).get("folds"))
        except (TypeError, ValueError):
            folds = None

    def metric_pass(metric_name: str, gate_name: str) -> bool:
        value = metrics[metric_name]
        return value is not None and value >= float(metric_gates[gate_name])

    checks = {
        "metrics_evidence_present": metrics_evidence is not None,
        "fivefold_protocol": (
            folds is not None and folds >= metric_gates["minimum_fold_count"]
        ),
        "detection_macro_f1": metric_pass(
            "detection_macro_f1", "minimum_detection_macro_f1"
        ),
        "localization_macro_f1": metric_pass(
            "localization_macro_f1", "minimum_localization_macro_f1"
        ),
        "classification_macro_f1": metric_pass(
            "classification_macro_f1", "minimum_classification_macro_f1"
        ),
        "e2e_macro_f1": metric_pass("e2e_macro_f1", "minimum_e2e_macro_f1"),
        "runtime_provider_present": (root / runtime_gates["provider_path"]).is_file(),
        "hash_pinned_model_package_present": package_hash_matches,
        "exact_mapping_coverage": (
            mapping_coverage >= runtime_gates["required_exact_mapping_coverage"]
        ),
        "candidate_runtime_chain_evidence_present": chain_path.is_file(),
        "candidate_runtime_chain_statuses": bool(chain_statuses)
        and all(value == "SUCCEEDED" for value in chain_statuses.values()),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    required_paths = {
        "metrics_evidence": metrics_path,
        "model_package": package_path,
        "model_receipt": receipt_path,
        "runtime_chain_evidence": chain_path,
    }
    missing_artifacts = {
        name: str(path.relative_to(root))
        for name, path in required_paths.items()
        if not path.is_file()
    }
    return {
        "schema_version": "astrograph-adcs-production-qualification-report.v2",
        "policy_id": policy["policy_id"],
        "dataset_id": policy["dataset_id"],
        "candidate_id": policy["candidate_id"],
        "status": "QUALIFIED" if not blockers else "BLOCKED",
        "evidence_status": (
            "COMPLETE" if not missing_artifacts else "MISSING_REQUIRED_ARTIFACTS"
        ),
        "missing_artifacts": missing_artifacts,
        "metrics": metrics,
        "metric_gates": metric_gates,
        "runtime": {
            "exact_mapping_count": mapped_count,
            "dataset_class_count": class_count,
            "exact_mapping_coverage": mapping_coverage,
            "registry_promoted": False,
            "chain_statuses": chain_statuses,
        },
        "checks": checks,
        "blockers": blockers,
        "claim_boundary": (
            policy["claim_boundary"]
            + " Missing metrics/model/runtime receipts remain explicit blockers and are not "
            "reconstructed from documentation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["status"] == "QUALIFIED" or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
