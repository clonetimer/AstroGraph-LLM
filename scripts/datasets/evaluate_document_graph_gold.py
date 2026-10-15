#!/usr/bin/env python3
"""Validate and evaluate the local r015 document graph gold handoff."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge.document_graph_gold_evaluation import evaluate_document_graph_gold


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    result = evaluate_document_graph_gold(
        pilot_root=ROOT / ".astrograph-state/research/document_corpus/document_graph_scope_pilot_r015",
        handoff_root=ROOT / ".astrograph-state/research/document_corpus/document_graph_gold_handoff_r001",
        output_path=ROOT / "governance/evidence/research/document_corpus/document_graph_gold_handoff_r001/evaluation.json",
    )
    summary = {key: value for key, value in result.items() if key != "blockers"}
    summary["blocker_count"] = len(result["blockers"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
