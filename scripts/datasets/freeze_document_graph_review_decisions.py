#!/usr/bin/env python3
"""Freeze the current document-KG human review sheet without publishing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.knowledge.document_graph_review_decisions import (
    freeze_document_graph_review_decisions,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-sheet", type=Path,
        default=ROOT / "governance/evidence/research/document_corpus/confusion_candidate_graph_review_r001/human_review_decisions.csv",
    )
    parser.add_argument(
        "--graph-root", type=Path, action="append",
        default=[
            ROOT / ".astrograph-state/research/document_corpus/confusion_candidate_graph_r001",
            ROOT / ".astrograph-state/research/document_corpus/confusion_candidate_graph_exception_replay_r001",
        ],
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "governance/evidence/research/document_corpus/confusion_candidate_graph_review_r001/decision_freeze.json",
    )
    args = parser.parse_args()
    receipt = freeze_document_graph_review_decisions(
        review_sheet_path=args.review_sheet,
        graph_roots=args.graph_root,
        output_path=args.output,
    )
    summary = {key: value for key, value in receipt.items() if key not in {"approved_items", "blockers"}}
    summary["blocker_count"] = len(receipt["blockers"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
