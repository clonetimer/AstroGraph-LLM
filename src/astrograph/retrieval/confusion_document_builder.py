"""Build contrastive fault documents and stage every document for RAG and KG use."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from astrograph.knowledge.construction import process_governed_document

from .document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_inputs(
    config_path: Path, fault_facts_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    config = _load_json(config_path)
    facts = _load_json(fault_facts_path)
    if config.get("schema_version") != "astrograph-fault-confusion-groups.v1":
        raise ValueError("unsupported fault confusion configuration schema")
    if config.get("review_state") != "HUMAN_REVIEW_REQUIRED":
        raise ValueError("fault confusion groups must remain human-review-required")
    if not isinstance(facts, list) or not facts:
        raise ValueError("canonical fault facts must be a nonempty list")

    fact_keys = {(row["dataset_id"], row["fault_class_id"]) for row in facts}
    if len(fact_keys) != len(facts):
        raise ValueError("canonical fault facts contain duplicate dataset/fault keys")
    groups_by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_keys: set[tuple[str, str]] = set()
    group_ids: set[str] = set()
    for group in config.get("groups", []):
        group_id = str(group.get("group_id", "")).strip()
        dataset_id = str(group.get("dataset_id", "")).strip()
        fault_classes = group.get("fault_classes")
        questions = group.get("discriminator_questions")
        if not group_id or group_id in group_ids:
            raise ValueError("confusion group IDs must be nonempty and unique")
        group_ids.add(group_id)
        if not isinstance(fault_classes, list) or len(fault_classes) < 2:
            raise ValueError(f"confusion group {group_id} requires at least two faults")
        if len(fault_classes) != len(set(fault_classes)):
            raise ValueError(f"confusion group {group_id} contains duplicate faults")
        if not isinstance(questions, list) or not questions or any(
            not str(question).strip() for question in questions
        ):
            raise ValueError(f"confusion group {group_id} requires discriminator questions")
        for fault_id in fault_classes:
            key = (dataset_id, str(fault_id))
            if key not in fact_keys:
                raise ValueError(f"confusion group references unknown fault: {key}")
            grouped_keys.add(key)
            groups_by_fault[f"{dataset_id}\0{fault_id}"].append(group)
    if grouped_keys != fact_keys:
        missing = sorted(fact_keys - grouped_keys)
        unknown = sorted(grouped_keys - fact_keys)
        raise ValueError(
            f"confusion groups must exactly cover the canonical fault universe; "
            f"missing={missing}, unknown={unknown}"
        )
    return config, facts, groups_by_fault


def _document_markdown(
    fact: dict[str, Any], groups: list[dict[str, Any]], *, claim_boundary: str
) -> str:
    fault_id = str(fact["fault_class_id"])
    alternatives = sorted(
        {
            str(candidate)
            for group in groups
            for candidate in group["fault_classes"]
            if candidate != fault_id
        }
    )
    questions = list(
        dict.fromkeys(
            str(question).strip()
            for group in groups
            for question in group["discriminator_questions"]
        )
    )
    evidence_ids = [str(item) for item in fact.get("evidence_definition_ids", [])]
    summaries = fact.get("generator_condition_summaries", [])
    descriptions = sorted(
        {
            str(item.get("description", "")).strip()
            for item in summaries
            if isinstance(item, dict) and str(item.get("description", "")).strip()
        }
    )
    lines = [
        f"# Contrastive diagnosis candidate: {fault_id}",
        "",
        "> Review state: HUMAN_REVIEW_REQUIRED",
        "> Epistemic status: DESIGN_PROPOSAL",
        "> Runtime voting: PROHIBITED",
        "",
        "## Scope and provenance",
        "",
        f"- Dataset: `{fact['dataset_id']}`",
        f"- Target fault: `{fault_id}`",
        f"- Component candidate: `{fact.get('component_id') or 'UNRESOLVED'}`",
        f"- Mechanism association: `{fact.get('mechanism_id') or 'UNRESOLVED'}`",
        f"- Generator description: {'; '.join(descriptions) or 'not supplied'}",
        "- Source family: synthetic dataset taxonomy; alternatives are not independent evidence.",
        "",
        "## Confusion set",
        "",
    ]
    lines.extend(f"- Alternative candidate: `{item}`" for item in alternatives)
    lines.extend(["", "## Discriminator questions", ""])
    lines.extend(f"{index}. {question}" for index, question in enumerate(questions, 1))
    lines.extend(["", "## Executable evidence and counterevidence boundary", ""])
    if evidence_ids:
        lines.extend(f"- Registered evidence definition: `{item}`" for item in evidence_ids)
        lines.extend(
            [
                "- Only the registered definitions above may be executed; this document must not invent a threshold.",
                "- Supporting, contradicting, missing, and inapplicable results must be recorded separately.",
                "- Absence of supporting evidence is not exclusion unless a registered definition declares counter or exclusion polarity.",
            ]
        )
    else:
        lines.extend(
            [
                "- `NO_REGISTERED_EXECUTABLE_COUNTEREVIDENCE`",
                "- Required behavior: abstain or escalate when this distinction depends on missing executable evidence.",
                "- Do not invent a threshold, exclusion rule, or causal relation from this document.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            claim_boundary,
            "This document is a contrastive review aid, not an observed episode, confirmed causal path, calibrated rule, or Active-KG assertion.",
            "",
        ]
    )
    return "\n".join(lines)


def build_fault_confusion_document_pack(
    *,
    config_path: str | Path,
    fault_facts_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Generate one contrast document per fault and dual-stage it deterministically.

    The function deliberately stops before LLM extraction, candidate graph
    compilation, review, publication, or runtime voting.
    """

    config_path = Path(config_path).resolve()
    fault_facts_path = Path(fault_facts_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty fault confusion pack directory")
    output_root.mkdir(parents=True, exist_ok=True)
    config, facts, groups_by_fault = _validated_inputs(config_path, fault_facts_path)
    generator_sha = _sha256(Path(__file__).resolve())

    document_rows: list[dict[str, Any]] = []
    total_elements = total_chunks = total_sections = total_tasks = total_crosswalk = 0
    for fact in sorted(facts, key=lambda row: (row["dataset_id"], row["fault_class_id"])):
        dataset_id = str(fact["dataset_id"])
        fault_id = str(fact["fault_class_id"])
        groups = groups_by_fault[f"{dataset_id}\0{fault_id}"]
        relative_path = Path("comparisons") / dataset_id / f"{fault_id}.md"
        document_path = output_root / relative_path
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(
            _document_markdown(
                fact, groups, claim_boundary=str(config["claim_boundary"])
            ),
            encoding="utf-8",
        )
        source_sha = _sha256(document_path)
        descriptor = GovernedDocumentDescriptor(
            document_id=f"astrograph:confusion:{dataset_id}:{fault_id}:r001",
            document_type=DocumentType.DESIGN_PROPOSAL,
            visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
            source_uri=f"corpus://confusion-pack-r001/{relative_path.as_posix()}",
            source_sha256=source_sha,
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            authority=DocumentAuthority.NON_NORMATIVE,
            provenance_group_id="synthetic-dataset-taxonomy-confusion-r001",
            template_version=str(config["configuration_id"]),
            generator_sha256=generator_sha,
            synthetic_from_simulation=True,
            runtime_vote_allowed=False,
        )
        processed = process_governed_document(
            descriptor=descriptor,
            source_path=document_path,
            independence_group="synthetic-dataset-taxonomy-confusion-r001",
            license_id="project-generated",
            knowledge_version="fault-confusion-document-r001",
        )
        ingestion = processed.retrieval_ingestion
        staging = processed.knowledge_staging
        total_elements += len(ingestion.elements)
        total_chunks += len(ingestion.chunks)
        total_sections += len(staging.source_tree.sections)
        total_tasks += len(staging.extraction_tasks)
        total_crosswalk += len(processed.crosswalk)
        alternatives = sorted(
            {
                str(candidate)
                for group in groups
                for candidate in group["fault_classes"]
                if candidate != fault_id
            }
        )
        document_rows.append(
            {
                "dataset_id": dataset_id,
                "fault_class_id": fault_id,
                "path": relative_path.as_posix(),
                "source_sha256": source_sha,
                "descriptor": descriptor.model_dump(mode="json"),
                "confusion_group_ids": sorted(str(group["group_id"]) for group in groups),
                "alternative_fault_ids": alternatives,
                "registered_evidence_definition_ids": list(
                    fact.get("evidence_definition_ids", [])
                ),
                "requires_abstention_for_missing_executable_evidence": not bool(
                    fact.get("evidence_definition_ids")
                ),
                "retrieval_ingestion_sha256": ingestion.ingestion_sha256,
                "knowledge_staging_sha256": staging.staging_sha256,
                "processing_sha256": processed.processing_sha256,
                "parsed_element_count": len(ingestion.elements),
                "retrieval_chunk_count": len(ingestion.chunks),
                "knowledge_section_count": len(staging.source_tree.sections),
                "knowledge_extraction_task_count": len(staging.extraction_tasks),
                "crosswalk_count": len(processed.crosswalk),
                "crosswalk_complete": len(processed.crosswalk) == len(ingestion.elements),
                "candidate_kg_staged": processed.candidate_kg_staged,
                "candidate_graph_built": processed.candidate_graph_built,
                "publication_allowed": processed.publication_allowed,
                "runtime_vote_allowed": processed.runtime_vote_allowed,
            }
        )

    content_release_sha256 = _canonical_sha256(
        {
            "configuration_sha256": _sha256(config_path),
            "fault_facts_sha256": _sha256(fault_facts_path),
            "generator_sha256": generator_sha,
            "documents": [
                {
                    "dataset_id": row["dataset_id"],
                    "fault_class_id": row["fault_class_id"],
                    "path": row["path"],
                    "source_sha256": row["source_sha256"],
                }
                for row in document_rows
            ],
        }
    )
    manifest_body = {
        "schema_version": "astrograph-fault-confusion-document-pack.v1",
        "configuration_id": config["configuration_id"],
        "configuration_sha256": _sha256(config_path),
        "fault_facts_sha256": _sha256(fault_facts_path),
        "generator_sha256": generator_sha,
        "content_release_sha256": content_release_sha256,
        "document_count": len(document_rows),
        "fault_coverage_count": len(document_rows),
        "documents": document_rows,
        "totals": {
            "parsed_elements": total_elements,
            "retrieval_chunks": total_chunks,
            "knowledge_sections": total_sections,
            "knowledge_extraction_tasks": total_tasks,
            "crosswalks": total_crosswalk,
        },
        "all_documents_dual_processed": all(
            row["candidate_kg_staged"] and row["crosswalk_complete"]
            for row in document_rows
        ),
        "candidate_graph_built": False,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "review_state": "HUMAN_REVIEW_REQUIRED",
        "claim_boundary": (
            "Every contrast document entered both retrieval ingestion and candidate-KG "
            "staging. No LLM extraction, graph compilation, human approval, Active-KG "
            "publication, calibrated threshold, or runtime voting authority is claimed."
        ),
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": _canonical_sha256(manifest_body),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["build_fault_confusion_document_pack"]
