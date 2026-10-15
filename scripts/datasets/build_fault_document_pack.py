#!/usr/bin/env python3
"""Build the P0.5 multi-format fault design/FMEA/test document pack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.document_pack_builder import build_fault_document_pack  # noqa: E402


TIMER_ROOT = REPO_ROOT.parents[1]
DEFAULT_COVERAGE = (
    REPO_ROOT
    / "governance/evidence/ai_native/kg_coverage_review_20260831/fault_coverage_matrix.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / ".astrograph-state/research/document_corpus/fault_pack_r001"
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/fault_pack_r001/build_receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-matrix", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = build_fault_document_pack(
        coverage_matrix_path=args.coverage_matrix.resolve(),
        output_root=args.output.resolve(),
        generator_paths={
            "adcs_v522_wrapper": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_adcs_eps_dataset_v522.py",
            "adcs_base_conditions": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_adcs_eps_dataset.py",
            "phm_v6": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_phm_dataset_v6.py",
        },
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
