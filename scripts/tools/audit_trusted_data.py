#!/usr/bin/env python3
"""Audit trusted Sat-Agent datasets under the AstroGraph data policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.data.trusted import (
    audit_round6_dataset,
    audit_trusted_campaign,
    write_trusted_audit_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("round6", "trusted-campaign"), default="round6")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sat-agent-root", type=Path, required=True)
    parser.add_argument("--dataset-archive-sha256", default="")
    parser.add_argument("--sat-agent-archive-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    auditor = audit_trusted_campaign if args.mode == "trusted-campaign" else audit_round6_dataset
    audit, rows, leakage = auditor(
        args.dataset_root,
        args.sat_agent_root,
        dataset_archive_sha256=args.dataset_archive_sha256,
        sat_agent_archive_sha256=args.sat_agent_archive_sha256,
    )
    output = args.output_dir
    write_trusted_audit_outputs(
        audit,
        rows,
        leakage,
        audit_path=output / "trusted_dataset_audit.json",
        manifest_jsonl_path=output / "trusted_case_manifest.jsonl",
        matrix_csv_path=output / "trusted_case_matrix.csv",
        leakage_path=output / "public_trace_leakage.json",
    )
    print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
    return 0 if audit.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
