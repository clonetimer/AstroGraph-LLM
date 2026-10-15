#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.assurance.claim_registry import (
    ClaimRecord,
    ValidationSeverity,
    load_dataset_profiles,
    load_project_policy,
    validate_claim,
)
from astrograph.assurance.label_ontology_v3 import LabelOntologyBundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an AstroGraph claim record fail-closed.")
    parser.add_argument("claim")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--labels")
    args = parser.parse_args()

    record = ClaimRecord.from_dict(json.loads(Path(args.claim).read_text(encoding="utf-8")))
    policy = load_project_policy(args.policy)
    profiles = load_dataset_profiles(args.profiles)
    if record.dataset_id not in profiles:
        raise SystemExit(f"unknown dataset_id: {record.dataset_id}")
    label_record = None
    if args.labels:
        bundle = LabelOntologyBundle.from_dict(json.loads(Path(args.labels).read_text(encoding="utf-8")))
        label_record = bundle.by_sha256().get(record.label_ontology_record_sha256 or "")
    issues = validate_claim(record, profiles[record.dataset_id], policy, label_record)
    report = {
        "schema_version": "astrograph-claim-validation-report.v1",
        "claim_id": record.claim_id,
        "claim_sha256": record.sha256,
        "valid": not any(item.severity is ValidationSeverity.ERROR for item in issues),
        "issues": [item.to_dict() for item in issues],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
