#!/usr/bin/env python3
"""Build the local r014 document-graph human gold handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.knowledge.document_graph_gold import build_document_graph_gold_handoff


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / ".astrograph-state/research/document_corpus/document_graph_scope_pilot_r015"
OUTPUT = ROOT / ".astrograph-state/research/document_corpus/document_graph_gold_handoff_r001"
RECEIPT = ROOT / "governance/evidence/research/document_corpus/document_graph_gold_handoff_r001/build_receipt.json"


def main() -> int:
    result = build_document_graph_gold_handoff(pilot_root=PILOT, output_root=OUTPUT)
    compact = dict(result)
    compact["local_handoff_receipt_path"] = str((OUTPUT / "handoff_receipt.json").resolve())
    compact["local_handoff_receipt_file_sha256"] = hashlib.sha256(
        (OUTPUT / "handoff_receipt.json").read_bytes()
    ).hexdigest()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
