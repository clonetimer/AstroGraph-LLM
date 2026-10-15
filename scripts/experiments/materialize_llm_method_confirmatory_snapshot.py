#!/usr/bin/env python3
"""Materialize a frozen LLM-method confirmatory dataset snapshot outside the source tree.

Copies only the frozen case manifest, each dataset sample index, and the payload files
referenced by the admitted 38 fault cases and paired nominal samples. The copied bytes
must reproduce the path-independent dataset snapshot identity before the output is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.llm_method_campaign import materialize_confirmatory_dataset_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adcs-root", type=Path, help="Optional remounted ADCS source root; content identity must still match the frozen receipt.")
    parser.add_argument("--phm-root", type=Path, help="Optional remounted PHM source root; content identity must still match the frozen receipt.")
    parser.add_argument(
        "--keep-write-bits",
        action="store_true",
        help="Do not remove write permission bits after materialization. This never creates a read-only mount.",
    )
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot_receipt.read_text(encoding="utf-8"))
    dataset_roots = {}
    if args.adcs_root:
        dataset_roots["adcs"] = args.adcs_root
    if args.phm_root:
        dataset_roots["phm"] = args.phm_root
    receipt = materialize_confirmatory_dataset_snapshot(
        ROOT,
        snapshot_receipt=snapshot,
        case_manifest_path=args.case_manifest,
        output_root=args.output_root,
        dataset_roots=(dataset_roots or None),
        harden_readonly_permissions=not args.keep_write_bits,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
