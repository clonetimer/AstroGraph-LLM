#!/usr/bin/env python3
"""Build the real r015 document-claim branch and prove it contributes no vote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.fusion.document_claims import build_document_claim_forward_ranking
from astrograph.retrieval.document_governance import DocumentEvidenceClaim
from astrograph.retrieval.models import DocumentClaimAdmission


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "governance/evidence/research/document_corpus/document_graph_retrieval_bridge_r001"
    / "document_claim_admission_trace.json"
)
FAULT_MAP = ROOT / "configs/datasets/adcs_v5_2_2/fault_class_map.json"
OUTPUT = (
    ROOT
    / "governance/evidence/research/document_corpus/document_claim_fusion_adapter_r001"
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    claims = [DocumentEvidenceClaim.model_validate(item) for item in source["claims"]]
    admissions = [DocumentClaimAdmission.model_validate(item) for item in source["admissions"]]
    fault_map = json.loads(FAULT_MAP.read_text(encoding="utf-8"))
    registered = sorted(set(fault_map["mappings"].values()))
    build = build_document_claim_forward_ranking(
        episode_id="episode:0656424589768116497a0104",
        claims=claims,
        admissions=admissions,
        mappings=[],
        registered_fault_ids=registered,
        source_artifact_id="artifact:r015-document-claim-admission-trace",
        source_artifact_sha256=_file_sha(SOURCE),
    )
    if any(item.contributes_to_fusion for item in build.ranking.candidates):
        raise ValueError("ineligible real document claims unexpectedly contributed to fusion")
    if build.ranking.kg_fault_coverage is None or (
        build.ranking.kg_fault_coverage.supported_fault_ids
    ):
        raise ValueError("real document branch unexpectedly declared executable fault coverage")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ranking_path = OUTPUT / "shadow_document_claim_ranking.json"
    ranking_payload = build.model_dump(mode="json")
    ranking_path.write_text(
        json.dumps(ranking_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema_version": "astrograph-document-claim-fusion-adapter-audit.v1",
        "status": "PASS_CONTEXT_ONLY",
        "claim_trace_file_sha256": _file_sha(SOURCE),
        "fault_map_file_sha256": _file_sha(FAULT_MAP),
        "registered_fault_count": len(registered),
        "input_claim_count": len(claims),
        "formally_eligible_claim_count": sum(
            item.formal_fusion_eligible for item in admissions
        ),
        "mapping_count": 0,
        "contributing_candidate_count": sum(
            item.contributes_to_fusion for item in build.ranking.candidates
        ),
        "declared_executable_fault_count": len(
            build.ranking.kg_fault_coverage.supported_fault_ids
        ),
        "shadow_ranking_file_sha256": _file_sha(ranking_path),
        "shadow_ranking_semantic_sha256": _canonical_sha(ranking_payload),
        "builder_source_sha256": _file_sha(Path(__file__).resolve()),
        "implementation_source_sha256": {
            relative: _file_sha(ROOT / relative)
            for relative in (
                "src/astrograph/fusion/document_claims.py",
                "src/astrograph/fusion/models.py",
                "src/astrograph/retrieval/document_claims.py",
                "src/astrograph/retrieval/models.py",
            )
        },
        "positive_path_test": (
            "tests/test_document_claim_fusion_adapter.py proves an independently reviewed, "
            "published, calibrated, applicable exact mapping can be consumed by existing BiRRF."
        ),
        "claim_boundary": (
            "The current r015 document claims are unapproved and uncalibrated, so this artifact "
            "is a zero-vote shadow ranking. It is not diagnosis-performance evidence."
        ),
    }
    receipt_path = OUTPUT / "build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
