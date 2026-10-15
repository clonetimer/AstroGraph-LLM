#!/usr/bin/env python3
"""Build the governed ADCS/PHM domain-SOTA comparison audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.sota_comparison import build_domain_sota_comparison_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcs-report", type=Path, required=True)
    parser.add_argument("--phm-astrograph", type=Path, required=True)
    parser.add_argument("--phm-inception-time", type=Path, required=True)
    parser.add_argument("--phm-timesnet", type=Path, required=True)
    parser.add_argument("--phm-modern-tcn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_domain_sota_comparison_audit(
        adcs_report_path=args.adcs_report,
        phm_report_paths={
            "astrograph_specialist": args.phm_astrograph,
            "inception_time": args.phm_inception_time,
            "timesnet": args.phm_timesnet,
            "modern_tcn": args.phm_modern_tcn,
        },
        source_paths=[
            ROOT / "src/astrograph/data_driven/sota_backbones.py",
            ROOT / "scripts/data_driven/run_hierarchical_group_cv.py",
            ROOT / "scripts/data_driven/run_phm_specialist_group_cv.py",
        ],
        output_path=args.output,
    )
    print(json.dumps({
        "comparison_id": result["comparison_id"],
        "status": result["status"],
        "audit_sha256": result["audit_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
