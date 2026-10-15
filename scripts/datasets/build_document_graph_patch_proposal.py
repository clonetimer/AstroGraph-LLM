#!/usr/bin/env python3
"""Build the current document-review to V3 KnowledgePatch proposal receipt."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge.document_graph_patch import build_document_graph_patch_proposal


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    result = build_document_graph_patch_proposal(
        review_freeze_path=(
            ROOT
            / "governance/evidence/research/document_corpus/"
            "confusion_candidate_graph_review_r001/decision_freeze.json"
        ),
        base_graph_path=(
            ROOT
            / "governance/evidence/ai_native/reusable_kg_construction_20260901/"
            "candidate_graph.json"
        ),
        output_path=(
            ROOT
            / "governance/evidence/research/document_corpus/"
            "document_graph_patch_r001/proposal.json"
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
