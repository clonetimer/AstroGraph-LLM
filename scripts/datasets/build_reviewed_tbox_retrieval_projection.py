#!/usr/bin/env python3
"""Build the owner-approved TBox retrieval-only graph projection."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge.tbox_retrieval_projection import (
    build_reviewed_tbox_retrieval_projection,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    result = build_reviewed_tbox_retrieval_projection(
        source_graph_path=(
            ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/candidate_graph.json"
        ),
        approval_path=(
            ROOT / "governance/evidence/ai_native/tbox_vocabulary_approval_20260902/approval.json"
        ),
        output_dir=(
            ROOT / "governance/evidence/research/document_corpus/reviewed_tbox_retrieval_r001"
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
