#!/usr/bin/env python3
"""Build the blank review handoff for confusion-document KG candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.knowledge.document_graph_review import build_document_graph_review_handoff


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".astrograph-state/research/document_corpus"
DEFAULT_PRIMARY = STATE / "confusion_candidate_graph_r001"
DEFAULT_SUPPLEMENTAL = STATE / "confusion_candidate_graph_exception_replay_r001"
DEFAULT_OUTPUT = ROOT / "governance/evidence/research/document_corpus/confusion_candidate_graph_review_r001"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--supplemental", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    supplementals = args.supplemental if args.supplemental is not None else [DEFAULT_SUPPLEMENTAL]
    receipt = build_document_graph_review_handoff(
        primary_root=args.primary,
        supplemental_roots=supplementals,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
