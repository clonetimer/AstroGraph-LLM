#!/usr/bin/env python3
"""Build deterministic ambiguous-structure review-gate evidence."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.retrieval.document_structure_stress import (  # noqa: E402
    build_document_structure_stress_pack,
)


def main() -> int:
    output = ROOT / ".astrograph-state/research/document_corpus/document_structure_stress_r001"
    receipt = (
        ROOT
        / "governance/evidence/research/document_corpus/document_structure_stress_r001/build_receipt.json"
    )
    result = build_document_structure_stress_pack(output_root=output)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "build_receipt.json", receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
