#!/usr/bin/env python3
"""Build the deterministic DOCX table/image/OMML parser contract fixture."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.multimodal_contract_builder import (  # noqa: E402
    build_multimodal_docx_contract,
)


OUTPUT_ROOT = REPO_ROOT / ".astrograph-state/research/document_corpus/multimodal_contract_r001"
RECEIPT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/multimodal_contract_r001/build_receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    receipt = build_multimodal_docx_contract(output_root=args.output_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
