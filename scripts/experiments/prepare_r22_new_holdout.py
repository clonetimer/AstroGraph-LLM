from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.replication_r22 import (
    admit_new_holdout_groups,
    build_new_holdout_intake_receipt,
    build_r22_readiness,
    load_replication_plan,
)


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze R22 new-holdout bytes before truth/prediction work.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--family", choices=["OPS-SAT", "ESA"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attest-pre-truth", action="store_true", required=True)
    parser.add_argument("--group-ids-json", type=Path, help="JSON list of group IDs prepared without truth-label inspection")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_replication_plan()
    intake = build_new_holdout_intake_receipt(
        args.source,
        dataset_id=args.dataset_id,
        dataset_family=args.family,
        operator_attests_pre_truth_inspection=args.attest_pre_truth,
        plan=plan,
    )
    write(args.output_dir / "new_holdout_intake_receipt.json", intake)
    admissions = {}
    if args.group_ids_json:
        group_ids = json.loads(args.group_ids_json.read_text(encoding="utf-8"))
        if not isinstance(group_ids, list):
            raise ValueError("R22_GROUP_IDS_JSON_MUST_BE_LIST")
        spec = plan["families"][args.family]
        admission = admit_new_holdout_groups(
            intake,
            raw_group_ids=[str(value) for value in group_ids],
            evaluation_task_id=spec["evaluation_task_id"],
            label_semantics_sha256=spec["label_semantics_sha256"],
            target_universe_sha256=spec["target_universe_sha256"],
            plan=plan,
        )
        write(args.output_dir / "new_holdout_group_admission.json", admission)
        admissions[args.family] = admission
    readiness = build_r22_readiness(holdout_admissions=admissions, required_families=[args.family], plan=plan)
    write(args.output_dir / "r22_replication_readiness.json", readiness)
    print(json.dumps({"intake_status": intake["status"], "readiness": readiness["status"], "blockers": readiness["blockers"]}, ensure_ascii=False))
    return 0 if readiness["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
