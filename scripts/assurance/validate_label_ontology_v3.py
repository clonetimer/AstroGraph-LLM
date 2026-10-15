#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.assurance.label_ontology_v3 import LabelOntologyBundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AstroGraph label ontology v3 records.")
    parser.add_argument("bundle")
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args()
    bundle = LabelOntologyBundle.from_dict(json.loads(Path(args.bundle).read_text(encoding="utf-8")))
    tasks = args.task or ["ANOMALY_DETECTION", "TEMPORAL_LOCALIZATION", "FAULT_IDENTIFICATION", "ROOT_CAUSE_ATTRIBUTION"]
    results = []
    valid = True
    for record in bundle.records:
        support = {task: record.support_for_task(task).to_dict() for task in tasks}
        if any(not value["supported"] for value in support.values()):
            valid = False
        results.append({"label_id": record.label_id, "record_sha256": record.sha256, "task_support": support})
    report = {
        "schema_version": "astrograph-label-ontology-validation-report.v3",
        "bundle_sha256": bundle.sha256,
        "valid_for_all_requested_tasks": valid,
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
