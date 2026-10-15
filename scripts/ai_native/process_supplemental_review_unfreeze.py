#!/usr/bin/env python3
"""Source-adjudicate and unfreeze the returned supplemental KG review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.knowledge.coverage_review import (
    build_source_adjudication_package,
    sha256_file,
    unfreeze_supplemental_human_review,
    write_source_adjudication_review_sheet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcs-generator", type=Path, required=True)
    parser.add_argument("--adcs-base-generator", type=Path, required=True)
    parser.add_argument("--phm-generator", type=Path, required=True)
    parser.add_argument("--sota-audit", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / "governance/evidence/ai_native/kg_coverage_review_20260831"
    adjudication = build_source_adjudication_package(
        package_dir=directory,
        supplemental_review_path=directory / "human_review_decisions_with_supplemental_review_zh.csv",
        adjudication_draft_path=directory / "source_adjudication_draft_20260901.json",
        generator_source_paths={
            "adcs_v522_generator": args.adcs_generator,
            "adcs_imported_base_generator": args.adcs_base_generator,
            "phm_v6_generator": args.phm_generator,
        },
        output_path=directory / "source_adjudication_20260901.json",
    )
    receipt = unfreeze_supplemental_human_review(
        package_dir=directory,
        supplemental_review_path=directory / "human_review_decisions_with_supplemental_review_zh.csv",
        freeze_receipt_path=directory / "supplemental_review_temporary_freeze.json",
        sota_comparison_audit_path=args.sota_audit,
        source_adjudication_path=directory / "source_adjudication_20260901.json",
        output_path=directory / "supplemental_review_unfreeze_20260901.json",
    )
    handoff = write_source_adjudication_review_sheet(
        supplemental_review_path=directory / "human_review_decisions_with_supplemental_review_zh.csv",
        source_adjudication_path=directory / "source_adjudication_20260901.json",
        output_path=directory / "human_review_decisions_source_adjudicated_20260901.csv",
    )
    print(json.dumps({
        "source_adjudication_id": adjudication.adjudication_id,
        "source_adjudication_status": adjudication.status,
        "unfreeze_id": receipt.unfreeze_id,
        "unfreeze_status": receipt.status,
        "human_review_handoff": str(handoff),
        "human_review_handoff_sha256": sha256_file(handoff),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
