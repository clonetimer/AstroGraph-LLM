#!/usr/bin/env python3
"""Run the frozen section-aware versus fixed-character machine benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.chunking_benchmark import run_chunking_machine_benchmark  # noqa: E402


PREREG = REPO_ROOT / "configs/research/document_chunking_machine_benchmark_r001.json"
PACK = REPO_ROOT / ".astrograph-state/research/document_corpus/fault_pack_r001"
RAW = REPO_ROOT / ".astrograph-state/research/document_corpus/chunking_benchmark_r001/results.json"
RECEIPT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/chunking_benchmark_r001/receipt.json"
)


def main() -> int:
    raw, receipt = run_chunking_machine_benchmark(
        preregistration_path=PREREG,
        handbook_markdown_path=PACK / "fault_design_fmea_test_handbook.md",
        canonical_facts_path=PACK / "canonical_fault_document_facts.json",
    )
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
