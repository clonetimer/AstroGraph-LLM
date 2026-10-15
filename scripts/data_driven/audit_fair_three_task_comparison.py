#!/usr/bin/env python3
"""Build a content-addressed anti-suppression audit for two comparison reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.data_driven.fair_comparison import build_fair_three_task_comparison_ledger


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcs-report", type=Path, required=True)
    parser.add_argument("--phm-report", type=Path, required=True)
    parser.add_argument("--adcs-dataset-index", type=Path, required=True)
    parser.add_argument("--phm-dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_fair_three_task_comparison_ledger(
        adcs_report_path=args.adcs_report,
        phm_report_path=args.phm_report,
        adcs_script_path=ROOT / "scripts/data_driven/compare_adcs_deep_encoders.py",
        phm_script_path=ROOT / "scripts/data_driven/compare_phm_deep_models.py",
        adcs_dataset_index_path=args.adcs_dataset_index,
        phm_dataset_index_path=args.phm_dataset_index,
        output_path=args.output,
    )
    print(json.dumps({
        "comparison_id": ledger.comparison_id,
        "status": ledger.status,
        "ledger_sha256": ledger.ledger_sha256,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
