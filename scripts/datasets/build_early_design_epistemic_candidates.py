#!/usr/bin/env python3
"""Build fail-closed epistemic-span suggestions for the early design DOCX."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval import SectionFirstDocumentIngestor  # noqa: E402
from astrograph.retrieval.epistemic_annotation import suggest_epistemic_spans  # noqa: E402


SOURCE = REPO_ROOT.parent / "智能故障检测软件_设计报告.docx"
OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/early_design_report_r001/machine_epistemic_candidates.json"
)
EXPECTED_SHA = "3c36e1f563d1790217068b51d6e26d133f69b09c5bf8a370bd4bc68fc64d4ee0"


def main() -> int:
    elements, _, audit = SectionFirstDocumentIngestor().parse_docx(SOURCE)
    if audit.source_sha256 != EXPECTED_SHA:
        raise ValueError("early design source hash mismatch")
    candidates = suggest_epistemic_spans(elements)
    counts = Counter(item.suggested_status.value for item in candidates)
    payload = {
        "schema_version": "astrograph-machine-epistemic-span-candidates.v1",
        "status": "MACHINE_CANDIDATES_REVIEW_REQUIRED",
        "source_sha256": audit.source_sha256,
        "candidate_count": len(candidates),
        "suggested_status_counts": dict(sorted(counts.items())),
        "source_fact_candidate_count": counts.get("SOURCE_FACT", 0),
        "llm_used": False,
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "claim_boundary": (
            "Fail-closed review suggestions for an early design proposal. No span is promoted "
            "to SOURCE_FACT, human gold, KG fact, implementation evidence, or runtime evidence."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "status", "candidate_count", "suggested_status_counts", "llm_used"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
