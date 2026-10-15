#!/usr/bin/env python3
"""Audit raw LLM proposals in a document graph scope pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal, KnowledgeSourceProfile
from astrograph.knowledge.document_grounding import audit_document_proposal_grounding


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / ".astrograph-state/research/document_corpus/document_graph_scope_pilot_r009"
DEFAULT_OUTPUT = ROOT / "governance/evidence/research/document_corpus/document_graph_scope_pilot_r009/grounding_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = json.loads((args.pilot / "receipt.json").read_text(encoding="utf-8"))
    results = []
    for row in receipt["results"]:
        target = row["document_id"].replace(":", "_")
        directories = list(args.pilot.glob(f"{row['lane']}/{row['dataset_id']}/{target}"))
        if len(directories) != 1:
            raise ValueError(f"pilot artifact directory is missing for {row['document_id']}")
        directory = directories[0]
        proposal = KnowledgeExtractionProposal.model_validate_json(
            (directory / "raw_extraction_proposal.json").read_text(encoding="utf-8")
        )
        bundle = json.loads((directory / "construction_bundle.json").read_text(encoding="utf-8"))
        profile = KnowledgeSourceProfile.model_validate(bundle["source_profiles"][0])
        audit = audit_document_proposal_grounding(
            source_profile=profile, proposal=proposal
        )
        results.append({
            "document_id": row["document_id"],
            "raw_proposal_sha256": sha(directory / "raw_extraction_proposal.json"),
            **audit,
        })
    body = {
        "schema_version": "astrograph-document-llm-grounding-batch-audit.v1",
        "pilot_receipt_sha256": sha(args.pilot / "receipt.json"),
        "document_count": len(results),
        "llm_candidate_document_count": sum(item["proposal_node_count"] > 0 for item in results),
        "status": "BLOCKED" if any(item["status"] == "BLOCKED" for item in results) else "PASS",
        "finding_counts": {
            code: sum(
                finding["code"] == code
                for item in results
                for finding in item["findings"]
            )
            for code in sorted({
                finding["code"] for item in results for finding in item["findings"]
            })
        },
        "results": results,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
    }
    body["audit_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in body.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
