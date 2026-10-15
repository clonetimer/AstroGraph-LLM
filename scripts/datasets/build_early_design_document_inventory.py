#!/usr/bin/env python3
"""Build the non-normative machine inventory for the first real DOCX seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval import (  # noqa: E402
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    SectionFirstDocumentIngestor,
)
from astrograph.knowledge.construction import (  # noqa: E402
    knowledge_source_profile_set_sha256,
    process_governed_docx,
)


DEFAULT_SOURCE = REPO_ROOT.parent / "智能故障检测软件_设计报告.docx"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/early_design_report_r001/machine_inventory.json"
)
DEFAULT_KG_STAGING_OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/early_design_report_r001/knowledge_staging_receipt.json"
)
DEFAULT_CROSSWALK_OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/early_design_report_r001/document_processing_crosswalk.json"
)
EXPECTED_SOURCE_SHA256 = "3c36e1f563d1790217068b51d6e26d133f69b09c5bf8a370bd4bc68fc64d4ee0"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(source_sha256: str) -> GovernedDocumentDescriptor:
    return GovernedDocumentDescriptor(
        document_id="astrograph:early-design-proposal:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="external://AstroGraph/智能故障检测软件_设计报告.docx",
        source_sha256=source_sha256,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        runtime_vote_allowed=False,
    )


def build_inventory(source: Path, *, enforce_known_source: bool = True) -> dict[str, object]:
    elements, warnings, audit = SectionFirstDocumentIngestor().parse_docx(source)
    if enforce_known_source and audit.source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "early design source hash mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {audit.source_sha256}"
        )
    descriptor = _descriptor(audit.source_sha256)
    kind_counts: dict[str, int] = {}
    summaries: list[dict[str, object]] = []
    for element in elements:
        kind_counts[element.kind.value] = kind_counts.get(element.kind.value, 0) + 1
        summaries.append(
            {
                "element_id": element.element_id,
                "kind": element.kind.value,
                "section_path": element.section_path,
                "source_locator": element.source_locator,
                "text_sha256": _sha_text(element.text_representation),
                "asset_uri": element.asset_uri,
                "asset_sha256": element.asset_sha256,
                "representation_method": element.representation_method,
                "representation_reviewed": element.representation_reviewed,
                "style_id": element.style_id,
                "style_name": element.style_name,
            }
        )
    expected_structure = {
        "nonempty_paragraph_count": 67,
        "heading_count": 13,
        "table_count": 3,
        "table_row_count": 21,
        "table_cell_count": 72,
        "image_count": 3,
        "native_formula_count": 0,
    }
    actual_structure = {
        key: getattr(audit, key) for key in expected_structure
    }
    structure_matches = actual_structure == expected_structure
    return {
        "schema_version": "astrograph-document-corpus-machine-inventory.v1",
        "inventory_id": "early-design-report-r001",
        "status": "DRAFT_REVIEW_REQUIRED",
        "gold_status": "MACHINE_INVENTORY_ONLY_HUMAN_GOLD_PENDING",
        "descriptor": descriptor.model_dump(mode="json"),
        "parser": {
            "id": "astrograph-stdlib-docx-native",
            "version": "1.0.0",
            "input_mode": "DOCX_NATIVE_OOXML",
            "pdf_or_markdown_intermediate": False,
        },
        "expected_structure": expected_structure,
        "actual_structure": actual_structure,
        "structure_matches_expected": structure_matches,
        "headings": audit.headings,
        "style_counts": audit.style_counts,
        "media_sha256": audit.media_sha256,
        "element_kind_counts": dict(sorted(kind_counts.items())),
        "element_summaries": summaries,
        "warnings": warnings,
        "image_policy": {
            "default_epistemic_status": "ILLUSTRATIVE",
            "runtime_evidence_eligible": False,
            "diagnostic_cards_are_examples_not_historical_cases": True,
        },
        "required_human_gold": [
            "heading hierarchy",
            "table 5 cell structure and nine runtime stages",
            "table 6 training sample types",
            "table 7 deployment framework comparison",
            "three image-region/caption/section links",
            "epistemic status: fact/proposal/estimate/future assumption",
            "terminology mapping without silent replacement",
        ],
        "claim_boundary": (
            "This is a machine inventory of a non-normative early design proposal. "
            "It does not approve claims, update the active knowledge graph, create "
            "runtime evidence, or establish implementation status."
        ),
    }


def build_knowledge_staging_receipt(
    source: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError("early design source hash mismatch before KG staging")
    processed = process_governed_docx(
        descriptor=_descriptor(source_sha),
        source_path=source,
        independence_group="early-design-report-origin-r001",
        license_id="project-provided-internal-document",
    )
    staged = processed.knowledge_staging
    modality_counts: dict[str, int] = {}
    for section in staged.source_tree.sections:
        for modality in ("TEXT", "TABLE", "FORMULA", "IMAGE"):
            marker = f"[{modality};"
            modality_counts[modality] = modality_counts.get(modality, 0) + section.text.count(marker)
    receipt = {
        "schema_version": "astrograph-document-dual-processing-receipt.v2",
        "document_id": staged.document_descriptor.document_id,
        "source_sha256": source_sha,
        "source_profile_sha256": knowledge_source_profile_set_sha256(
            [staged.source_profile]
        ),
        "source_profile_artifact_sha256": hashlib.sha256(
            staged.source_profile.model_dump_json().encode("utf-8")
        ).hexdigest(),
        "source_tree_sha256": staged.source_tree.tree_sha256,
        "ingestion_sha256": staged.ingestion_sha256,
        "staging_sha256": staged.staging_sha256,
        "processing_sha256": processed.processing_sha256,
        "processing_content_sha256": processed.content_sha256,
        "retrieval_ingestion_sha256": processed.retrieval_ingestion.ingestion_sha256,
        "graph_scope": staged.graph_scope.value,
        "default_epistemic_status": staged.default_epistemic_status.value,
        "section_count": len(staged.source_tree.sections),
        "extraction_task_count": len(staged.extraction_tasks),
        "retrieval_chunk_count": len(processed.retrieval_ingestion.chunks),
        "parsed_element_count": len(processed.retrieval_ingestion.elements),
        "crosswalk_count": len(processed.crosswalk),
        "crosswalk_complete": len(processed.crosswalk) == len(processed.retrieval_ingestion.elements),
        "extraction_pass_counts": {
            pass_kind: sum(task.pass_kind.value == pass_kind for task in staged.extraction_tasks)
            for pass_kind in ("ENTITY", "RELATION", "ATTRIBUTE")
        },
        "modality_counts": modality_counts,
        "publication_allowed": staged.publication_allowed,
        "runtime_vote_allowed": staged.runtime_vote_allowed,
        "requires_human_review": staged.requires_human_review,
        "candidate_graph_built": processed.candidate_graph_built,
        "next_gate": (
            "Run controlled extraction into a DRAFT candidate graph; review every claim "
            "and use KnowledgePatch validation/replay before any release proposal."
        ),
        "claim_boundary": (
            "The real DOCX has entered both governed retrieval and the document-to-candidate-KG "
            "path with an element/chunk/fragment crosswalk. This receipt is not an extracted "
            "graph, human approval, Active KG update, causal evidence, or runtime voting authority."
        ),
    }
    return receipt, [item.model_dump(mode="json") for item in processed.crosswalk]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kg-staging-output", type=Path, default=DEFAULT_KG_STAGING_OUTPUT)
    parser.add_argument("--crosswalk-output", type=Path, default=DEFAULT_CROSSWALK_OUTPUT)
    parser.add_argument("--allow-source-hash-change", action="store_true")
    args = parser.parse_args()
    payload = build_inventory(
        args.source.resolve(),
        enforce_known_source=not args.allow_source_hash_change,
    )
    if not payload["structure_matches_expected"]:
        raise ValueError("early design DOCX structure no longer matches the frozen expectation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staging, crosswalk = build_knowledge_staging_receipt(args.source.resolve())
    args.kg_staging_output.parent.mkdir(parents=True, exist_ok=True)
    args.kg_staging_output.write_text(
        json.dumps(staging, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.crosswalk_output.parent.mkdir(parents=True, exist_ok=True)
    args.crosswalk_output.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-document-processing-crosswalk.v1",
                "document_id": staging["document_id"],
                "source_sha256": staging["source_sha256"],
                "processing_sha256": staging["processing_sha256"],
                "processing_content_sha256": staging["processing_content_sha256"],
                "items": crosswalk,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "kg_staging_output": str(args.kg_staging_output),
        "crosswalk_output": str(args.crosswalk_output),
        "status": payload["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
