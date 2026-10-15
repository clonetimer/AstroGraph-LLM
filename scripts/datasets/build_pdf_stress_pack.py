#!/usr/bin/env python3
"""Generate the P0.5 scanned/rotated PDF pressure sample and receipt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.pdf_stress_builder import build_pdf_stress_pack  # noqa: E402


DEFAULT_SOURCE = (
    REPO_ROOT
    / ".astrograph-state/research/document_corpus/fault_pack_r001/fault_design_fmea_test_handbook.pdf"
)
DEFAULT_OUTPUT = REPO_ROOT / ".astrograph-state/research/document_corpus/pdf_stress_r001"
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/pdf_stress_r001/build_receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = build_pdf_stress_pack(
        source_pdf=args.source.resolve(), output_root=args.output.resolve()
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
