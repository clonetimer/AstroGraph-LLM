from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.agents import VllmKnowledgeExtractionBackend
from astrograph.knowledge import (
    SupplementalReviewRefreezeReceipt,
    build_generic_fault_ontology_bundle,
    write_construction_bundle,
)


DEFAULT_REFREEZE = (
    ROOT
    / "governance/evidence/ai_native/kg_coverage_review_20260831"
    / "supplemental_review_refreeze_20260901.json"
)
DEFAULT_OUTPUT = (
    ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild AstroGraph's governed cross-domain TBox through the reusable KG pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refreeze-receipt", type=Path, default=DEFAULT_REFREEZE)
    parser.add_argument(
        "--verify-vllm",
        action="store_true",
        help="Require model discovery from the local OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refreeze = SupplementalReviewRefreezeReceipt.model_validate_json(
        args.refreeze_receipt.read_text(encoding="utf-8")
    )
    if not refreeze.temporary_frozen or refreeze.formal_decision_count != 0:
        raise RuntimeError("supplemental review branch is not safely refrozen")

    vllm_model: str | None = None
    if args.verify_vllm:
        backend = VllmKnowledgeExtractionBackend(base_url=args.vllm_base_url)
        vllm_model = backend.discover_model()

    bundle = build_generic_fault_ontology_bundle(
        ROOT / "configs/knowledge/generic_fault_ontology.json",
        ROOT / "configs/semantics/core_component_archetypes.json",
        ROOT / "configs/semantics/core_semantic_schema.json",
        excluded_review_refreeze_sha256=refreeze.refreeze_sha256,
    )
    paths = write_construction_bundle(bundle, output_dir=args.output_dir)
    receipt = {
        "schema_version": "astrograph-reusable-kg-build-receipt.v1",
        "build_id": bundle.build_id,
        "bundle_sha256": bundle.bundle_sha256,
        "build_mode": bundle.build_mode,
        "status": bundle.status,
        "candidate_validation_status": bundle.candidate_validation.status,
        "quality_status": bundle.quality.status,
        "node_count": bundle.quality.node_count,
        "edge_count": bundle.quality.edge_count,
        "connected_component_count": bundle.quality.connected_component_count,
        "orphan_node_count": bundle.quality.orphan_node_count,
        "expansion_suggestion_count": len(bundle.expansion_suggestions),
        "excluded_review_candidate_count": refreeze.candidate_count,
        "excluded_review_refreeze_sha256": refreeze.refreeze_sha256,
        "vllm_endpoint": args.vllm_base_url,
        "vllm_model_verified": vllm_model,
        "deterministic_migration_used_llm": False,
        "publication_allowed": bundle.publication_allowed,
        "active_graph_modified": bundle.active_graph_modified,
        "claim_boundary": (
            "This receipt covers deterministic migration of existing governed TBox assets. "
            "It does not claim an LLM call and cannot publish or modify Active KG."
        ),
    }
    receipt_path = args.output_dir / "build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        **receipt,
        "artifacts": {key: str(value) for key, value in paths.items()},
        "build_receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2))
    return 0 if bundle.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
