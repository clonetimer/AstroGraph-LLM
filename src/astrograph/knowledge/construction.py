"""Reusable, source-structured and governed knowledge-graph construction.

The module stops at a content-addressed candidate graph.  It deliberately has
no Active-KG publication capability; approved candidates must continue through
the existing KnowledgePatch and governance workflow.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.agents.knowledge_curation import (
    KnowledgeCandidateCriticAgent,
    KnowledgeCandidateValidationReceipt,
    KnowledgeExtractionProposal,
    KnowledgeNodeCandidate,
    KnowledgeRelationCandidate,
    KnowledgeSourceProfile,
    KnowledgeSourceProfilerAgent,
    SourceFragmentInput,
    knowledge_source_profile_set_sha256 as _portable_profile_set_sha256,
    knowledge_source_profile_sha256,
)
from astrograph.retrieval.document_governance import (
    ClaimEpistemicStatus,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from astrograph.retrieval.structured_document import (
    DocumentElement,
    SectionFirstDocumentIngestor,
    StructuredDocumentIngestion,
)

from .schema import AssertionStatus, ReviewStatus
from .schema_v3 import (
    GraphLayer,
    GraphEdgeV3,
    GraphNodeV3,
    GraphShapeReport,
    GraphShapeValidator,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
    NODE_LAYERS,
    RELATION_ALLOWLIST,
)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_sha256(value: BaseModel) -> str:
    return canonical_sha256(value.model_dump(mode="json"))


def knowledge_source_profile_set_sha256(
    profiles: list[KnowledgeSourceProfile],
) -> str:
    """Return the public, canonical identity used by curation proposals."""
    return _portable_profile_set_sha256(profiles)


def _portable_staging_identity(value: dict[str, Any]) -> dict[str, Any]:
    """Drop machine-local locators already covered by immutable content hashes."""

    payload = json.loads(json.dumps(value, ensure_ascii=False))
    payload.get("source_profile", {}).pop("source_path", None)
    payload.get("document_descriptor", {}).pop("source_uri", None)
    # Native ingestion hashes include source_uri. Profile fragments, the source
    # file hash and tree hash already cover the bytes and extracted structure.
    payload.pop("ingestion_sha256", None)
    return payload


def _without_machine_local_document_locators(value: Any) -> Any:
    """Recursively remove locator-only fields while retaining content/provenance hashes."""

    if isinstance(value, dict):
        return {
            key: _without_machine_local_document_locators(item)
            for key, item in value.items()
            if key not in {"source_path", "source_uri", "ingestion_sha256"}
        }
    if isinstance(value, list):
        return [_without_machine_local_document_locators(item) for item in value]
    return value


def _document_processing_content_identity(value: dict[str, Any]) -> dict[str, Any]:
    payload = _without_machine_local_document_locators(value)
    payload.pop("processing_sha256", None)
    payload.pop("content_sha256", None)
    return payload


def _construction_content_identity(value: dict[str, Any]) -> dict[str, Any]:
    payload = _without_machine_local_document_locators(value)
    payload.pop("bundle_sha256", None)
    payload.pop("content_sha256", None)
    return payload


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return cleaned or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in normalized if char.isalnum())


class KnowledgeSourceRole(str, Enum):
    DOMAIN_DOCUMENT = "DOMAIN_DOCUMENT"
    ENGINEERING_STANDARD = "ENGINEERING_STANDARD"
    GENERATOR_SOURCE = "GENERATOR_SOURCE"
    DATASET_INDEX = "DATASET_INDEX"
    ASTROGRAPH_ASSET = "ASTROGRAPH_ASSET"
    EXPERIMENT_EVIDENCE = "EXPERIMENT_EVIDENCE"


class DocumentKnowledgeGraphScope(str, Enum):
    TBOX_CANDIDATE = "TBOX_CANDIDATE"
    EPISODE_EVIDENCE_CANDIDATE = "EPISODE_EVIDENCE_CANDIDATE"
    HISTORICAL_CASE_CANDIDATE = "HISTORICAL_CASE_CANDIDATE"
    QUARANTINE_EVALUATOR_CANDIDATE = "QUARANTINE_EVALUATOR_CANDIDATE"


class SourceSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(pattern=r"^section:[^\s]+$")
    source_id: str = Field(pattern=r"^source:[^\s]+$")
    parent_section_id: str | None = None
    level: int = Field(ge=0)
    ordinal: int = Field(ge=0)
    title: str = Field(min_length=1)
    location: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=8_000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fragment_id: str = Field(pattern=r"^fragment:[^\s]+$")
    ancestor_section_ids: list[str] = Field(default_factory=list)
    context_summary: str = Field(min_length=1, max_length=2_000)


class SourceStructureTree(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-source-tree.v1"] = (
        "astrograph-knowledge-source-tree.v1"
    )
    source_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sections: list[SourceSection] = Field(min_length=1)
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_tree(self) -> "SourceStructureTree":
        by_id = {item.section_id: item for item in self.sections}
        if len(by_id) != len(self.sections):
            raise ValueError("source tree section IDs must be unique")
        fragment_ids = [item.fragment_id for item in self.sections]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("source tree fragment IDs must be unique")
        for section in self.sections:
            if section.parent_section_id is not None:
                parent = by_id.get(section.parent_section_id)
                if parent is None:
                    raise ValueError("source tree parent is missing")
                if parent.source_id != section.source_id or parent.level >= section.level:
                    raise ValueError("source tree parent crosses source or level boundary")
            expected_ancestors: list[str] = []
            cursor = section.parent_section_id
            seen: set[str] = set()
            while cursor is not None:
                if cursor in seen:
                    raise ValueError("source tree contains a cycle")
                seen.add(cursor)
                expected_ancestors.append(cursor)
                cursor = by_id[cursor].parent_section_id
            if section.ancestor_section_ids != list(reversed(expected_ancestors)):
                raise ValueError("source tree ancestor lineage mismatch")
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"tree_sha256"})
        )
        if self.tree_sha256 != expected:
            raise ValueError("source tree_sha256 mismatch")
        return self


class KnowledgeExtractionPassKind(str, Enum):
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    ATTRIBUTE = "ATTRIBUTE"


class KnowledgeExtractionTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    pass_kind: KnowledgeExtractionPassKind
    section_id: str
    fragment_id: str
    ancestor_section_ids: list[str]
    ancestor_context_summaries: list[str]
    section_context_summary: str = Field(min_length=1, max_length=2_000)
    allowed_node_types: list[KnowledgeNodeTypeV3]
    allowed_relation_types: list[KnowledgeRelationTypeV3]
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CanonicalizationAction(str, Enum):
    NEW_CANONICAL = "NEW_CANONICAL"
    ALIGN_EXISTING = "ALIGN_EXISTING"
    MERGE_EXACT = "MERGE_EXACT"
    KEEP_DISTINCT_REVIEW_SUGGESTED = "KEEP_DISTINCT_REVIEW_SUGGESTED"
    BLOCK_TYPE_CONFLICT = "BLOCK_TYPE_CONFLICT"
    BLOCK_PROPERTY_CONFLICT = "BLOCK_PROPERTY_CONFLICT"


class CanonicalizationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: CanonicalizationAction
    candidate_ids: list[str] = Field(min_length=1)
    canonical_id: str | None = None
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class CanonicalConcept(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_id: str
    node_type: KnowledgeNodeTypeV3
    canonical_name: str
    definition: str
    aliases: list[str] = Field(default_factory=list)
    source_candidate_ids: list[str] = Field(min_length=1)
    source_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)


class CanonicalRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_edge_id: str
    source_id: str
    relation_type: KnowledgeRelationTypeV3
    target_id: str
    predicate_definition: str
    rationale: str
    source_candidate_edge_ids: list[str] = Field(min_length=1)
    source_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeCanonicalizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-canonicalization.v1"] = (
        "astrograph-knowledge-canonicalization.v1"
    )
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    concepts: list[CanonicalConcept]
    relations: list[CanonicalRelation]
    decisions: list[CanonicalizationDecision]
    blocker_codes: list[str] = Field(default_factory=list)
    review_suggestion_count: int = Field(ge=0)
    status: Literal["PASS", "BLOCKED"]
    canonicalization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content(self) -> "KnowledgeCanonicalizationReport":
        if (self.status == "BLOCKED") != bool(self.blocker_codes):
            raise ValueError("canonicalization status and blockers differ")
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"canonicalization_sha256"})
        )
        if self.canonicalization_sha256 != expected:
            raise ValueError("canonicalization_sha256 mismatch")
        return self


class KnowledgeExpansionSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suggestion_id: str
    source_id: str
    candidate_relation_types: list[KnowledgeRelationTypeV3] = Field(min_length=1)
    target_id: str
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    common_neighbor_score: float = Field(ge=0.0, le=1.0)
    common_source_ancestor_score: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    operator: Literal["EDGE_SUGGEST"] = "EDGE_SUGGEST"
    auto_applied: Literal[False] = False
    requires_llm_verification: Literal[True] = True
    requires_human_review: Literal[True] = True


class KnowledgeConstructionQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-construction-quality.v1"] = (
        "astrograph-knowledge-construction-quality.v1"
    )
    status: Literal["PASS", "BLOCKED"]
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    connected_component_count: int = Field(ge=0)
    orphan_node_count: int = Field(ge=0)
    traced_node_fraction: float = Field(ge=0.0, le=1.0)
    traced_edge_fraction: float = Field(ge=0.0, le=1.0)
    used_fragment_fraction: float = Field(ge=0.0, le=1.0)
    duplicate_node_id_count: int = Field(ge=0)
    duplicate_edge_id_count: int = Field(ge=0)
    shape_report: GraphShapeReport
    blocker_codes: list[str] = Field(default_factory=list)


class KnowledgeConstructionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-reusable-knowledge-construction.v2"] = (
        "astrograph-reusable-knowledge-construction.v2"
    )
    build_id: str
    build_mode: Literal["NEW_EXTRACTION", "MIGRATE_EXISTING_GOVERNED_ONTOLOGY"]
    source_profiles: list[KnowledgeSourceProfile] = Field(min_length=1)
    source_trees: list[SourceStructureTree] = Field(min_length=1)
    extraction_tasks: list[KnowledgeExtractionTask] = Field(min_length=1)
    proposal: KnowledgeExtractionProposal
    candidate_validation: KnowledgeCandidateValidationReceipt
    canonicalization: KnowledgeCanonicalizationReport
    candidate_graph: KnowledgeGraphV3 | None
    quality: KnowledgeConstructionQualityReport
    expansion_suggestions: list[KnowledgeExpansionSuggestion] = Field(default_factory=list)
    expansion_operators: list[str] = Field(
        default_factory=lambda: [
            "CONTEXT_ENRICH", "CANONICALIZE", "EDGE_SUGGEST", "MERGE_SUBGRAPH"
        ]
    )
    excluded_review_refreeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["BLOCKED", "CANDIDATE_READY_FOR_HUMAN_REVIEW"]
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "A reproducible candidate graph and review suggestions only. LLM extraction, canonicalization "
        "and expansion cannot publish a KnowledgeRelease, change Runtime voting or modify Active KG."
    )

    @model_validator(mode="after")
    def validate_bundle(self) -> "KnowledgeConstructionBundle":
        blocked = (
            self.candidate_validation.status == "BLOCKED"
            or self.canonicalization.status == "BLOCKED"
            or self.quality.status == "BLOCKED"
        )
        if blocked != (self.status == "BLOCKED"):
            raise ValueError("construction bundle status does not match stage gates")
        if blocked != (self.candidate_graph is None):
            raise ValueError("blocked construction bundle cannot carry a candidate graph")
        expected = canonical_sha256(self.model_dump(
            mode="json", exclude={"bundle_sha256", "content_sha256"}
        ))
        if self.bundle_sha256 != expected:
            raise ValueError("construction bundle_sha256 mismatch")
        expected_content = canonical_sha256(_construction_content_identity(
            self.model_dump(mode="json")
        ))
        if self.content_sha256 is not None and self.content_sha256 != expected_content:
            raise ValueError("construction content_sha256 mismatch")
        return self


class DocumentKnowledgeStagingBundle(BaseModel):
    """Bridge from one governed document to the candidate-graph workflow.

    This bundle is deliberately not a ``KnowledgeGraphV3``.  Extraction and
    canonicalization still have to run, and publication still has to pass the
    existing KnowledgePatch review/replay workflow.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-document-knowledge-staging.v1"] = (
        "astrograph-document-knowledge-staging.v1"
    )
    document_descriptor: GovernedDocumentDescriptor
    source_profile: KnowledgeSourceProfile
    source_tree: SourceStructureTree
    extraction_tasks: list[KnowledgeExtractionTask] = Field(min_length=1)
    graph_scope: DocumentKnowledgeGraphScope
    default_epistemic_status: ClaimEpistemicStatus
    ingestion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_allowed: Literal[False] = False
    runtime_vote_allowed: Literal[False] = False
    requires_human_review: Literal[True] = True
    staging_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_staging_identity(self) -> "DocumentKnowledgeStagingBundle":
        descriptor = self.document_descriptor
        if descriptor.source_sha256 != self.source_profile.source_sha256:
            raise ValueError("document descriptor and source profile hashes differ")
        expected = canonical_sha256(_portable_staging_identity(
            self.model_dump(mode="json", exclude={"staging_sha256"})
        ))
        if self.staging_sha256 != expected:
            raise ValueError("staging_sha256 mismatch")
        return self


class DocumentProcessingCrosswalk(BaseModel):
    """Trace one parsed document element into both retrieval and KG artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    element_id: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    retrieval_chunk_id: str = Field(min_length=1)
    knowledge_section_id: str = Field(pattern=r"^section:[^\s]+$")
    knowledge_fragment_id: str = Field(pattern=r"^fragment:[^\s]+$")


class GovernedDocumentProcessingBundle(BaseModel):
    """One governed document, materialized for retrieval and candidate-KG use.

    The two branches share a source hash and an explicit element crosswalk.  The
    KG branch stops before extraction/publication, while the retrieval branch
    remains subject to the document access gate at query time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-governed-document-processing.v1"] = (
        "astrograph-governed-document-processing.v1"
    )
    retrieval_ingestion: StructuredDocumentIngestion
    knowledge_staging: DocumentKnowledgeStagingBundle
    crosswalk: list[DocumentProcessingCrosswalk] = Field(min_length=1)
    retrieval_ready: Literal[True] = True
    candidate_kg_staged: Literal[True] = True
    candidate_graph_built: Literal[False] = False
    publication_allowed: Literal[False] = False
    runtime_vote_allowed: Literal[False] = False
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    processing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_shared_provenance(self) -> "GovernedDocumentProcessingBundle":
        ingestion = self.retrieval_ingestion
        staging = self.knowledge_staging
        if ingestion.source_sha256 != staging.source_profile.source_sha256:
            raise ValueError("retrieval and candidate-KG source hashes differ")
        if ingestion.document_descriptor != staging.document_descriptor:
            raise ValueError("retrieval and candidate-KG descriptors differ")
        element_ids = {item.element_id for item in ingestion.elements}
        chunk_ids = {item.chunk_id for item in ingestion.chunks}
        sections = {item.section_id: item for item in staging.source_tree.sections}
        if (
            len(self.crosswalk) != len(element_ids)
            or {item.element_id for item in self.crosswalk} != element_ids
        ):
            raise ValueError("crosswalk must cover every parsed document element exactly once")
        if any(item.retrieval_chunk_id not in chunk_ids for item in self.crosswalk):
            raise ValueError("crosswalk references an unknown retrieval chunk")
        if any(
            item.knowledge_section_id not in sections
            or sections[item.knowledge_section_id].fragment_id != item.knowledge_fragment_id
            for item in self.crosswalk
        ):
            raise ValueError("crosswalk references an unknown knowledge section/fragment")
        expected = canonical_sha256(self.model_dump(
            mode="json", exclude={"processing_sha256", "content_sha256"}
        ))
        if self.processing_sha256 != expected:
            raise ValueError("processing_sha256 mismatch")
        expected_content = canonical_sha256(_document_processing_content_identity(
            self.model_dump(mode="json")
        ))
        if self.content_sha256 is not None and self.content_sha256 != expected_content:
            raise ValueError("processing content_sha256 mismatch")
        return self


class _SectionDraft(BaseModel):
    location: str
    title: str
    level: int
    parent_location: str | None
    text: str


def _bounded_drafts(drafts: list[_SectionDraft], max_chars: int = 7_500) -> list[_SectionDraft]:
    bounded: list[_SectionDraft] = []
    split_locations = {item.location for item in drafts if len(item.text) > max_chars}
    for item in drafts:
        if len(item.text) <= max_chars:
            if item.parent_location in split_locations:
                item = item.model_copy(
                    update={"parent_location": f"{item.parent_location}#part-1"}
                )
            bounded.append(item)
            continue
        chunks = [item.text[index:index + max_chars] for index in range(0, len(item.text), max_chars)]
        for index, text in enumerate(chunks, 1):
            bounded.append(_SectionDraft(
                location=f"{item.location}#part-{index}",
                title=f"{item.title} (part {index}/{len(chunks)})",
                level=item.level,
                parent_location=(
                    f"{item.parent_location}#part-1"
                    if item.parent_location in split_locations
                    else item.parent_location
                ),
                text=text,
            ))
    return bounded


def _markdown_drafts(path: Path) -> list[_SectionDraft]:
    content = path.read_text(encoding="utf-8")
    heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    drafts: list[_SectionDraft] = []
    stack: list[tuple[int, str]] = []
    current_title = path.stem
    current_level = 0
    current_location = "root"
    current_parent: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            drafts.append(_SectionDraft(
                location=current_location,
                title=current_title,
                level=current_level,
                parent_location=current_parent,
                text=text,
            ))

    heading_counts: dict[str, int] = defaultdict(int)
    for line in content.splitlines():
        match = heading.match(line)
        if match is None:
            buffer.append(line)
            continue
        flush()
        buffer = [line]
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        key = "/".join([item[1] for item in stack] + [_slug(title)]) or "root"
        heading_counts[key] += 1
        location = key if heading_counts[key] == 1 else f"{key}-{heading_counts[key]}"
        stack.append((level, location))
        current_title = title
        current_level = level
        current_location = location
        current_parent = parent
    flush()
    if not drafts:
        drafts = [_SectionDraft(
            location="root", title=path.stem, level=0, parent_location=None, text=content,
        )]
    return _bounded_drafts(drafts)


def _json_drafts(path: Path, max_sections: int = 2_000) -> list[_SectionDraft]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    drafts: list[_SectionDraft] = []

    def visit(value: Any, location: str, title: str, level: int, parent: str | None) -> None:
        if len(drafts) >= max_sections:
            raise ValueError("structured source exceeds max_sections")
        scalar: Any
        children: list[tuple[Any, str, str]] = []
        if isinstance(value, dict):
            scalar = {}
            for key, child in value.items():
                if isinstance(child, (dict, list)) and child:
                    children.append((child, f"{location}/{key}", str(key)))
                else:
                    scalar[key] = child
        elif isinstance(value, list):
            scalar = {"item_count": len(value)}
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    item_name = None
                    if isinstance(child, dict):
                        for key, item_value in child.items():
                            if key.endswith("_id") or key in {"id", "name", "title"}:
                                if isinstance(item_value, (str, int, float)):
                                    item_name = str(item_value)
                                    break
                    item_title = item_name or f"item-{index}"
                    children.append((child, f"{location}/{index}-{_slug(item_title)}", item_title))
                else:
                    scalar.setdefault("items", []).append(child)
        else:
            scalar = value
        text = json.dumps(scalar, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        drafts.append(_SectionDraft(
            location=location,
            title=title,
            level=level,
            parent_location=parent,
            text=text,
        ))
        for child, child_location, child_title in children:
            visit(child, child_location, child_title, level + 1, location)

    visit(payload, "root", path.stem, 0, None)
    return _bounded_drafts(drafts)


def _csv_drafts(path: Path, rows_per_section: int = 50) -> list[_SectionDraft]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    drafts = [_SectionDraft(
        location="root",
        title=path.stem,
        level=0,
        parent_location=None,
        text=json.dumps({"columns": list(rows[0]) if rows else [], "row_count": len(rows)},
                        ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )]
    for index in range(0, len(rows), rows_per_section):
        batch = rows[index:index + rows_per_section]
        drafts.append(_SectionDraft(
            location=f"rows/{index}-{index + len(batch) - 1}",
            title=f"rows {index}-{index + len(batch) - 1}",
            level=1,
            parent_location="root",
            text=json.dumps(batch, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ))
    return _bounded_drafts(drafts)


def _document_element_drafts(
    elements: list[DocumentElement], *, format_label: str = "Document"
) -> list[_SectionDraft]:
    """Preserve a parser's section hierarchy and modality/source locators."""

    def knowledge_representation(element: DocumentElement) -> str:
        if element.representation_reviewed:
            return element.text_representation.rstrip()
        asset = (
            f"; asset_sha256={element.asset_sha256}"
            if element.asset_sha256 is not None
            else ""
        )
        return (
            f"[{element.kind.value} representation withheld pending human review"
            f"; source_locator={element.source_locator}{asset}]"
        )

    grouped: dict[tuple[str, ...], list[DocumentElement]] = defaultdict(list)
    section_order: list[tuple[str, ...]] = []
    for element in elements:
        key = tuple(element.section_path)
        if key not in grouped:
            section_order.append(key)
        grouped[key].append(element)

    drafts: list[_SectionDraft] = []
    location_by_path: dict[tuple[str, ...], str] = {}
    known_paths = {
        prefix
        for path_key in section_order
        for depth in range(1, len(path_key) + 1)
        for prefix in [path_key[:depth]]
    }
    for path_key in sorted(known_paths, key=lambda item: (len(item), section_order.index(next(
        section for section in section_order if section[: len(item)] == item
    )))):
        digest = canonical_sha256(list(path_key))[:10]
        location = "/".join(_slug(item) for item in path_key) + f"-{digest}"
        location_by_path[path_key] = location
        parent = location_by_path.get(path_key[:-1])
        drafts.append(
            _SectionDraft(
                location=location,
                title=path_key[-1],
                level=len(path_key) - 1,
                parent_location=parent,
                text=f"{format_label} section: {path_key[-1]}",
            )
        )
        if path_key in grouped:
            content = "\n\n".join(
                f"[{element.kind.value}; {element.source_locator}]\n"
                f"{knowledge_representation(element)}"
                for element in grouped[path_key]
            )
            drafts.append(
                _SectionDraft(
                    location=f"{location}/content",
                    title=f"{path_key[-1]} content",
                    level=len(path_key),
                    parent_location=location,
                    text=content,
                )
            )
    return _bounded_drafts(drafts)


def _docx_drafts(
    path: Path,
    *,
    parsed_elements: list[DocumentElement] | None = None,
    source_sha256: str | None = None,
) -> tuple[list[_SectionDraft], str]:
    """Preserve DOCX heading hierarchy and non-text modalities for KG tasks."""

    if parsed_elements is None:
        elements, _, audit = SectionFirstDocumentIngestor().parse_docx(path)
        parsed_source_sha256 = audit.source_sha256
    else:
        elements = parsed_elements
        parsed_source_sha256 = source_sha256 or ""
        if not parsed_source_sha256:
            raise ValueError("preparsed DOCX elements require source_sha256")
    return _document_element_drafts(elements, format_label="DOCX"), parsed_source_sha256


def profile_structured_source(
    *,
    source_id: str,
    source_path: str | Path,
    independence_group: str,
    license_id: str,
    source_role: KnowledgeSourceRole = KnowledgeSourceRole.DOMAIN_DOCUMENT,
    parsed_docx_elements: list[DocumentElement] | None = None,
    parsed_document_elements: list[DocumentElement] | None = None,
    parsed_source_sha256: str | None = None,
) -> tuple[KnowledgeSourceProfile, SourceStructureTree]:
    """Adapt DOCX/Markdown/TXT/JSON/CSV into a stable structure tree and profile."""
    path = Path(source_path).resolve()
    suffix = path.suffix.lower()
    if parsed_docx_elements is not None and parsed_document_elements is not None:
        raise ValueError("provide only one preparsed document element collection")
    preparsed = parsed_document_elements or parsed_docx_elements
    if preparsed is not None:
        if suffix not in {".docx", ".md", ".txt", ".pdf"}:
            raise ValueError("preparsed elements use an unsupported document suffix")
        if not parsed_source_sha256:
            raise ValueError("preparsed document elements require source_sha256")
        drafts = _document_element_drafts(
            preparsed,
            format_label={
                ".docx": "DOCX", ".md": "Markdown", ".txt": "Text", ".pdf": "PDF"
            }[suffix],
        )
        extraction_method = {
            ".docx": "astrograph-docx-native-section-v1",
            ".md": "astrograph-markdown-section-v1",
            ".txt": "astrograph-text-section-v1",
            ".pdf": "astrograph-pdf-layout-section-v1",
        }[suffix]
    elif suffix in {".md", ".txt"}:
        drafts = _markdown_drafts(path)
        extraction_method = "astrograph-structured-markdown-v1"
    elif suffix == ".json":
        drafts = _json_drafts(path)
        extraction_method = "astrograph-structured-json-v1"
    elif suffix == ".csv":
        drafts = _csv_drafts(path)
        extraction_method = "astrograph-structured-csv-v1"
    elif suffix == ".docx":
        drafts, _ = _docx_drafts(
            path,
            parsed_elements=parsed_docx_elements,
            source_sha256=parsed_source_sha256,
        )
        extraction_method = "astrograph-docx-native-section-v1"
    else:
        raise ValueError(f"unsupported reusable knowledge source suffix: {suffix}")

    profiler = KnowledgeSourceProfilerAgent()
    profile = profiler.profile(
        source_id=source_id,
        source_path=path,
        independence_group=independence_group,
        source_role=source_role.value,
        license_id=license_id,
        extraction_method=extraction_method,
        fragments=[
            SourceFragmentInput(location=item.location, text=item.text) for item in drafts
        ],
    )
    fragment_by_location = {
        (fragment.location, fragment.text): fragment for fragment in profile.fragments
    }
    location_to_id: dict[str, str] = {}
    sections: list[SourceSection] = []
    by_location = {item.location: item for item in drafts}
    for ordinal, item in enumerate(drafts):
        digest = canonical_sha256({
            "source_id": profile.source_id,
            "location": item.location,
            "text": item.text,
        })[:20]
        location_to_id[item.location] = f"section:{profile.source_id.removeprefix('source:')}:{digest}"
    for ordinal, item in enumerate(drafts):
        ancestors: list[str] = []
        cursor = item.parent_location
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen or cursor not in by_location:
                raise ValueError("source adapter produced an invalid parent chain")
            seen.add(cursor)
            ancestors.append(location_to_id[cursor])
            cursor = by_location[cursor].parent_location
        child_titles = [
            child.title for child in drafts if child.parent_location == item.location
        ]
        summary_parts = [item.title, item.text[:600]]
        if child_titles:
            summary_parts.append("children: " + ", ".join(child_titles[:20]))
        fragment = fragment_by_location[(item.location, item.text)]
        sections.append(SourceSection(
            section_id=location_to_id[item.location],
            source_id=profile.source_id,
            parent_section_id=(
                location_to_id[item.parent_location] if item.parent_location is not None else None
            ),
            level=item.level,
            ordinal=ordinal,
            title=item.title,
            location=item.location,
            text=item.text,
            text_sha256=fragment.text_sha256,
            fragment_id=fragment.fragment_id,
            ancestor_section_ids=list(reversed(ancestors)),
            context_summary=" | ".join(summary_parts)[:2_000],
        ))
    body = {
        "source_profile_sha256": knowledge_source_profile_sha256(profile),
        "sections": [item.model_dump(mode="json") for item in sections],
    }
    tree = SourceStructureTree(**body, tree_sha256=canonical_sha256({
        "schema_version": "astrograph-knowledge-source-tree.v1", **body,
    }))
    return profile, tree


def stage_governed_document_source(
    *,
    descriptor: GovernedDocumentDescriptor,
    source_path: str | Path,
    independence_group: str,
    license_id: str,
    document_ingestion: StructuredDocumentIngestion | None = None,
) -> DocumentKnowledgeStagingBundle:
    """Stage a governed document for KG extraction without publishing facts."""

    if descriptor.review_state == DocumentReviewState.REJECTED:
        raise ValueError("rejected documents cannot enter knowledge staging")
    if descriptor.visibility_class == DocumentVisibilityClass.TRUTH_QUARANTINED:
        raise ValueError("truth-quarantined documents require a separate evaluator graph")
    if (
        descriptor.contains_episode_truth
        and descriptor.visibility_class
        is not DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
    ):
        raise ValueError(
            "episode truth may enter only the isolated historical-case candidate graph"
        )

    source_path = Path(source_path).resolve()
    if document_ingestion is not None:
        if source_path.suffix.lower() not in {".docx", ".md", ".txt", ".pdf"}:
            raise ValueError("preparsed ingestion uses an unsupported document suffix")
        if document_ingestion.document_descriptor != descriptor:
            raise ValueError("preparsed ingestion and governed descriptor differ")
        if document_ingestion.source_sha256 != descriptor.source_sha256:
            raise ValueError("preparsed ingestion and source descriptor hashes differ")
    profile, tree = profile_structured_source(
        source_id=f"source:{descriptor.document_id}",
        source_path=source_path,
        independence_group=independence_group,
        license_id=license_id,
        source_role=KnowledgeSourceRole.DOMAIN_DOCUMENT,
        parsed_document_elements=(
            document_ingestion.elements if document_ingestion is not None else None
        ),
        parsed_source_sha256=(
            document_ingestion.source_sha256 if document_ingestion is not None else None
        ),
    )
    if profile.source_sha256 != descriptor.source_sha256:
        raise ValueError("governed descriptor does not match the source file")
    ingestion = document_ingestion
    if ingestion is None and source_path.suffix.lower() == ".docx":
        ingestion, _ = SectionFirstDocumentIngestor().ingest_docx(
            source_path,
            source_id=descriptor.document_id,
            knowledge_version="document-kg-staging-v1",
            document_descriptor=descriptor,
        )
    if ingestion is None:
        ingestion_sha = canonical_sha256(
            {
                "source_sha256": profile.source_sha256,
                "tree_sha256": tree.tree_sha256,
                "extraction_method": profile.extraction_method,
            }
        )
    else:
        ingestion_sha = ingestion.ingestion_sha256

    if descriptor.visibility_class == DocumentVisibilityClass.DESIGN_PUBLIC:
        scope = DocumentKnowledgeGraphScope.TBOX_CANDIDATE
        epistemic = (
            ClaimEpistemicStatus.DESIGN_PROPOSAL
            if descriptor.document_type == DocumentType.DESIGN_PROPOSAL
            else ClaimEpistemicStatus.SOURCE_FACT
        )
    elif descriptor.visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL:
        scope = DocumentKnowledgeGraphScope.EPISODE_EVIDENCE_CANDIDATE
        epistemic = ClaimEpistemicStatus.SOURCE_FACT
    else:
        scope = DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE
        epistemic = ClaimEpistemicStatus.HISTORICAL_RESOLUTION

    body = {
        "document_descriptor": descriptor,
        "source_profile": profile,
        "source_tree": tree,
        "extraction_tasks": build_extraction_tasks([tree]),
        "graph_scope": scope,
        "default_epistemic_status": epistemic,
        "ingestion_sha256": ingestion_sha,
    }
    serial = {
        key: value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else [item.model_dump(mode="json") for item in value]
        if isinstance(value, list)
        else value.value
        if isinstance(value, Enum)
        else value
        for key, value in body.items()
    }
    serial.update(
        {
            "schema_version": "astrograph-document-knowledge-staging.v1",
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "requires_human_review": True,
        }
    )
    return DocumentKnowledgeStagingBundle(
        **body,
        staging_sha256=canonical_sha256(_portable_staging_identity(serial)),
    )


def stage_quarantined_document_source(
    *,
    descriptor: GovernedDocumentDescriptor,
    source_path: str | Path,
    independence_group: str,
    license_id: str,
) -> DocumentKnowledgeStagingBundle:
    """Stage evaluator truth into an isolated graph-only candidate lane.

    Unlike ``process_governed_document``, this function deliberately creates no
    retrieval chunks or crosswalk and cannot feed the Runtime index.
    """

    if descriptor.visibility_class is not DocumentVisibilityClass.TRUTH_QUARANTINED:
        raise ValueError("quarantine staging requires TRUTH_QUARANTINED visibility")
    if not descriptor.contains_episode_truth:
        raise ValueError("quarantine staging requires contains_episode_truth=true")
    if descriptor.runtime_vote_allowed:
        raise ValueError("quarantined truth can never vote at runtime")
    path = Path(source_path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor.source_sha256:
        raise ValueError("quarantined descriptor does not match the source file")
    profile, tree = profile_structured_source(
        source_id=f"source:{descriptor.document_id}",
        source_path=path,
        independence_group=independence_group,
        license_id=license_id,
        source_role=KnowledgeSourceRole.EXPERIMENT_EVIDENCE,
    )
    body = {
        "document_descriptor": descriptor,
        "source_profile": profile,
        "source_tree": tree,
        "extraction_tasks": build_extraction_tasks([tree]),
        "graph_scope": DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE,
        "default_epistemic_status": ClaimEpistemicStatus.SOURCE_FACT,
        "ingestion_sha256": canonical_sha256(
            {
                "lane": "QUARANTINE_EVALUATOR_GRAPH_ONLY",
                "source_sha256": profile.source_sha256,
                "tree_sha256": tree.tree_sha256,
                "extraction_method": profile.extraction_method,
            }
        ),
    }
    serial = {
        key: value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else [item.model_dump(mode="json") for item in value]
        if isinstance(value, list)
        else value.value
        if isinstance(value, Enum)
        else value
        for key, value in body.items()
    }
    serial.update(
        {
            "schema_version": "astrograph-document-knowledge-staging.v1",
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "requires_human_review": True,
        }
    )
    return DocumentKnowledgeStagingBundle(
        **body, staging_sha256=canonical_sha256(_portable_staging_identity(serial))
    )


def process_governed_document(
    *,
    descriptor: GovernedDocumentDescriptor,
    source_path: str | Path,
    independence_group: str,
    license_id: str,
    knowledge_version: str = "document-dual-use-v1",
) -> GovernedDocumentProcessingBundle:
    """Create trace-linked RAG chunks and candidate-KG staging from one document.

    This is intentionally a preprocessing operation.  It neither asks an LLM
    to assert facts nor publishes anything to the Active KG.
    """

    path = Path(source_path).resolve()
    ingestor = SectionFirstDocumentIngestor()
    suffix = path.suffix.lower()
    if suffix == ".docx":
        ingestion, _ = ingestor.ingest_docx(
            path,
            source_id=descriptor.document_id,
            knowledge_version=knowledge_version,
            document_descriptor=descriptor,
        )
    elif suffix == ".pdf":
        ingestion, _ = ingestor.ingest_pdf(
            path,
            source_id=descriptor.document_id,
            knowledge_version=knowledge_version,
            document_descriptor=descriptor,
        )
    elif suffix in {".md", ".txt"}:
        source_bytes = path.read_bytes()
        text = source_bytes.decode("utf-8-sig")
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        elements, warnings = ingestor.parse_markdown(
            text, asset_root=path.parent if suffix == ".md" else None
        )
        ingestion = ingestor.ingest_elements(
            source_id=descriptor.document_id,
            source_uri=str(path),
            source_sha256=source_sha,
            knowledge_version=knowledge_version,
            elements=elements,
            warnings=warnings,
            document_descriptor=descriptor,
        )
    else:
        raise ValueError("governed dual-use processing supports DOCX, PDF, Markdown, and text")
    staging = stage_governed_document_source(
        descriptor=descriptor,
        source_path=path,
        independence_group=independence_group,
        license_id=license_id,
        document_ingestion=ingestion,
    )

    chunk_by_locator: dict[str, str] = {}
    for chunk in ingestion.chunks:
        for locator in chunk.source_locators:
            if locator in chunk_by_locator:
                raise ValueError(f"source locator occurs in multiple retrieval chunks: {locator}")
            chunk_by_locator[locator] = chunk.chunk_id

    section_by_locator: dict[str, SourceSection] = {}
    for section in staging.source_tree.sections:
        for element in ingestion.elements:
            marker = f"[{element.kind.value}; {element.source_locator}]"
            if marker in section.text:
                if element.source_locator in section_by_locator:
                    raise ValueError(
                        "source locator occurs in multiple candidate-KG sections: "
                        f"{element.source_locator}"
                    )
                section_by_locator[element.source_locator] = section

    missing = sorted(
        element.source_locator
        for element in ingestion.elements
        if element.source_locator not in chunk_by_locator
        or element.source_locator not in section_by_locator
    )
    if missing:
        raise ValueError(f"dual-use document crosswalk is incomplete: {missing[:10]}")
    crosswalk = [
        DocumentProcessingCrosswalk(
            element_id=element.element_id,
            source_locator=element.source_locator,
            retrieval_chunk_id=chunk_by_locator[element.source_locator],
            knowledge_section_id=section_by_locator[element.source_locator].section_id,
            knowledge_fragment_id=section_by_locator[element.source_locator].fragment_id,
        )
        for element in ingestion.elements
    ]
    body = {
        "retrieval_ingestion": ingestion,
        "knowledge_staging": staging,
        "crosswalk": crosswalk,
    }
    serial = {
        "schema_version": "astrograph-governed-document-processing.v1",
        **{
            key: value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [item.model_dump(mode="json") for item in value]
            for key, value in body.items()
        },
        "retrieval_ready": True,
        "candidate_kg_staged": True,
        "candidate_graph_built": False,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
    }
    return GovernedDocumentProcessingBundle(
        **body,
        content_sha256=canonical_sha256(_document_processing_content_identity(serial)),
        processing_sha256=canonical_sha256(serial),
    )


def process_governed_docx(
    *,
    descriptor: GovernedDocumentDescriptor,
    source_path: str | Path,
    independence_group: str,
    license_id: str,
    knowledge_version: str = "document-dual-use-v1",
) -> GovernedDocumentProcessingBundle:
    """Backward-compatible DOCX-only wrapper for dual-use processing."""

    if Path(source_path).suffix.lower() != ".docx":
        raise ValueError("process_governed_docx accepts DOCX sources only")
    return process_governed_document(
        descriptor=descriptor,
        source_path=source_path,
        independence_group=independence_group,
        license_id=license_id,
        knowledge_version=knowledge_version,
    )


def govern_document_knowledge_proposal(
    *,
    staging: DocumentKnowledgeStagingBundle,
    proposal: KnowledgeExtractionProposal,
) -> KnowledgeExtractionProposal:
    """Stamp immutable document governance onto extracted KG candidates."""

    expected_profile_sha = knowledge_source_profile_set_sha256([staging.source_profile])
    if proposal.source_profile_sha256 != expected_profile_sha:
        raise ValueError("knowledge proposal does not target the staged document profile")
    allowed_fragments = {item.fragment_id for item in staging.source_profile.fragments}
    cited_fragments = {
        fragment
        for item in [*proposal.nodes, *proposal.relations]
        for fragment in item.source_fragment_ids
    }
    if not cited_fragments.issubset(allowed_fragments):
        raise ValueError("knowledge proposal cites fragments outside the staged document")
    allowed_layers_by_scope = {
        DocumentKnowledgeGraphScope.TBOX_CANDIDATE: {GraphLayer.TBOX},
        DocumentKnowledgeGraphScope.EPISODE_EVIDENCE_CANDIDATE: {
            GraphLayer.ABOX,
            GraphLayer.EVIDENCE,
        },
        DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE: {
            GraphLayer.TBOX,
            GraphLayer.ABOX,
            GraphLayer.EVIDENCE,
            GraphLayer.DECISION,
        },
        DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE: {
            GraphLayer.ABOX,
            GraphLayer.EVIDENCE,
            GraphLayer.DECISION,
        },
    }
    allowed_layers = allowed_layers_by_scope[staging.graph_scope]
    out_of_scope = sorted(
            item.node_id
            for item in proposal.nodes
            if NODE_LAYERS[item.node_type] not in allowed_layers
        )
    if out_of_scope:
        raise ValueError(
            f"{staging.graph_scope.value} document proposal contains out-of-scope "
            f"candidate nodes: {out_of_scope}"
        )
    protected = {
        "source_document_id": staging.document_descriptor.document_id,
        "source_document_sha256": staging.document_descriptor.source_sha256,
        "document_graph_scope": staging.graph_scope.value,
        "epistemic_status": staging.default_epistemic_status.value,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
    }

    def stamp(properties: dict[str, Any]) -> dict[str, Any]:
        conflicts = {
            key: value for key, value in protected.items()
            if key in properties and properties[key] != value
        }
        if conflicts:
            raise ValueError(f"proposal attempts to override document governance: {sorted(conflicts)}")
        return {**properties, **protected}

    return proposal.model_copy(update={
        "nodes": [
            item.model_copy(update={"properties": stamp(item.properties)})
            for item in proposal.nodes
        ],
        "relations": [
            item.model_copy(update={"properties": stamp(item.properties)})
            for item in proposal.relations
        ],
    })


def build_extraction_tasks(trees: list[SourceStructureTree]) -> list[KnowledgeExtractionTask]:
    output_schema_sha = canonical_sha256(KnowledgeExtractionProposal.model_json_schema())
    tasks: list[KnowledgeExtractionTask] = []
    for tree in trees:
        sections_by_id = {item.section_id: item for item in tree.sections}
        parent_ids = {item.parent_section_id for item in tree.sections if item.parent_section_id}
        leaves = [item for item in tree.sections if item.section_id not in parent_ids]
        for section in leaves:
            for pass_kind in KnowledgeExtractionPassKind:
                task_id = "KGTASK-" + canonical_sha256({
                    "pass": pass_kind.value,
                    "section_id": section.section_id,
                    "schema": output_schema_sha,
                })[:20].upper()
                tasks.append(KnowledgeExtractionTask(
                    task_id=task_id,
                    pass_kind=pass_kind,
                    section_id=section.section_id,
                    fragment_id=section.fragment_id,
                    ancestor_section_ids=section.ancestor_section_ids,
                    ancestor_context_summaries=[
                        sections_by_id[section_id].context_summary
                        for section_id in section.ancestor_section_ids
                    ],
                    section_context_summary=section.context_summary,
                    allowed_node_types=list(KnowledgeNodeTypeV3),
                    allowed_relation_types=list(KnowledgeRelationTypeV3),
                    output_schema_sha256=output_schema_sha,
                ))
    return tasks


def _candidate_canonical_id(candidate: KnowledgeNodeCandidate) -> str:
    if candidate.canonical_id_hint:
        prefix, separator, local = candidate.canonical_id_hint.partition(":")
        if not separator or not prefix or not local or any(char.isspace() for char in candidate.canonical_id_hint):
            raise ValueError("canonical_id_hint must be a stable namespaced ID")
        return candidate.canonical_id_hint
    local = candidate.node_id.removeprefix("candidate:")
    if ":" in local and not any(char.isspace() for char in local):
        return local
    return (
        f"concept:{candidate.node_type.value.casefold()}:{_slug(candidate.canonical_name)}-"
        f"{canonical_sha256(candidate.canonical_name)[:8]}"
    )


def canonicalize_knowledge_proposal(
    proposal: KnowledgeExtractionProposal,
    *,
    existing_graph: KnowledgeGraphV3 | None = None,
    fuzzy_review_threshold: float = 0.86,
) -> KnowledgeCanonicalizationReport:
    """Definition-aware exact canonicalization; fuzzy matches remain review-only."""
    decisions: list[CanonicalizationDecision] = []
    blockers: list[str] = []
    existing_by_id = {node.node_id: node for node in (existing_graph.nodes if existing_graph else [])}
    existing_name_index: dict[tuple[KnowledgeNodeTypeV3, str], str] = {}
    for node in existing_by_id.values():
        names = [
            str(node.properties.get("canonical_name", "")),
            str(node.properties.get("name", "")),
            *[str(item) for item in node.properties.get("aliases", [])],
        ]
        for name in names:
            if name:
                existing_name_index[(node.node_type, _name_key(name))] = node.node_id

    candidates = sorted(proposal.nodes, key=lambda item: item.node_id)
    name_types: dict[str, set[KnowledgeNodeTypeV3]] = defaultdict(set)
    for candidate in candidates:
        name_types[_name_key(candidate.canonical_name)].add(candidate.node_type)
    conflict_names = {key for key, types in name_types.items() if len(types) > 1}
    for key in sorted(conflict_names):
        group = [item for item in candidates if _name_key(item.canonical_name) == key]
        blockers.append("CANONICAL_NAME_TYPE_CONFLICT")
        decisions.append(CanonicalizationDecision(
            action=CanonicalizationAction.BLOCK_TYPE_CONFLICT,
            candidate_ids=[item.node_id for item in group],
            rationale="The same normalized name was proposed with incompatible V3 node types.",
        ))

    grouped: dict[tuple[KnowledgeNodeTypeV3, str], list[KnowledgeNodeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.node_type, _name_key(candidate.canonical_name))].append(candidate)

    concepts: list[CanonicalConcept] = []
    candidate_to_canonical: dict[str, str] = {}
    for (node_type, name_key), group in sorted(grouped.items(), key=lambda item: (item[0][0].value, item[0][1])):
        canonical_ids = {_candidate_canonical_id(item) for item in group}
        hinted_ids = {
            item.canonical_id_hint for item in group if item.canonical_id_hint is not None
        }
        if len(hinted_ids) > 1:
            blockers.append("CANONICAL_ID_HINT_CONFLICT")
            decisions.append(CanonicalizationDecision(
                action=CanonicalizationAction.BLOCK_PROPERTY_CONFLICT,
                candidate_ids=[item.node_id for item in group],
                rationale="Exact-name candidates request incompatible stable canonical IDs.",
            ))
        existing_id = existing_name_index.get((node_type, name_key))
        hinted_existing = {
            item.canonical_id_hint for item in group
            if item.canonical_id_hint in existing_by_id
            and existing_by_id[item.canonical_id_hint].node_type == node_type
        }
        selected = sorted(hinted_existing)[0] if hinted_existing else existing_id
        if selected is None:
            selected = sorted(canonical_ids)[0]
        merged_properties: dict[str, Any] = {}
        property_conflict = False
        for item in group:
            for key, value in item.properties.items():
                if key in merged_properties and merged_properties[key] != value:
                    property_conflict = True
                else:
                    merged_properties[key] = value
        if property_conflict:
            blockers.append("CANONICAL_PROPERTY_CONFLICT")
            decisions.append(CanonicalizationDecision(
                action=CanonicalizationAction.BLOCK_PROPERTY_CONFLICT,
                candidate_ids=[item.node_id for item in group],
                canonical_id=selected,
                rationale="Exact-name candidates contain incompatible schema-safe properties.",
            ))
        else:
            action = (
                CanonicalizationAction.ALIGN_EXISTING
                if selected in existing_by_id
                else CanonicalizationAction.MERGE_EXACT
                if len(group) > 1
                else CanonicalizationAction.NEW_CANONICAL
            )
            decisions.append(CanonicalizationDecision(
                action=action,
                candidate_ids=[item.node_id for item in group],
                canonical_id=selected,
                rationale="Definition-bearing candidates were aligned by V3 type and exact normalized name/alias.",
            ))
        aliases = sorted({
            alias for item in group for alias in [item.canonical_name, *item.aliases]
            if _name_key(alias) != name_key
        })
        definitions = sorted(
            {item.description for item in group if item.description}, key=lambda value: (-len(value), value)
        )
        fragments = sorted({fragment for item in group for fragment in item.source_fragment_ids})
        concepts.append(CanonicalConcept(
            canonical_id=selected,
            node_type=node_type,
            canonical_name=group[0].canonical_name,
            definition=definitions[0] if definitions else "",
            aliases=aliases,
            source_candidate_ids=[item.node_id for item in group],
            source_fragment_ids=fragments,
            confidence=min(item.confidence for item in group),
            properties=merged_properties,
        ))
        for item in group:
            candidate_to_canonical[item.node_id] = selected

    review_suggestions = 0
    for index, left in enumerate(concepts):
        for right in concepts[index + 1:]:
            if left.node_type != right.node_type or left.canonical_id == right.canonical_id:
                continue
            name_score = SequenceMatcher(
                None, _name_key(left.canonical_name), _name_key(right.canonical_name)
            ).ratio()
            definition_score = SequenceMatcher(
                None, _name_key(left.definition), _name_key(right.definition)
            ).ratio() if left.definition and right.definition else 0.0
            similarity = 0.65 * name_score + 0.35 * definition_score
            if similarity >= fuzzy_review_threshold:
                review_suggestions += 1
                decisions.append(CanonicalizationDecision(
                    action=CanonicalizationAction.KEEP_DISTINCT_REVIEW_SUGGESTED,
                    candidate_ids=left.source_candidate_ids + right.source_candidate_ids,
                    canonical_id=left.canonical_id,
                    similarity=similarity,
                    rationale="Approximate name/definition similarity is review evidence only; no automatic merge.",
                ))

    relation_groups: dict[
        tuple[str, KnowledgeRelationTypeV3, str], list[KnowledgeRelationCandidate]
    ] = defaultdict(list)
    for relation in proposal.relations:
        source = candidate_to_canonical.get(relation.source_id, relation.source_id)
        target = candidate_to_canonical.get(relation.target_id, relation.target_id)
        relation_groups[(source, relation.relation_type, target)].append(relation)
    relations: list[CanonicalRelation] = []
    for (source, relation_type, target), group in sorted(
        relation_groups.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        relation_id = "edge:" + canonical_sha256({
            "source": source, "relation": relation_type.value, "target": target,
        })[:24]
        merged_relation_properties: dict[str, Any] = {}
        relation_property_conflict = False
        for candidate_relation in group:
            for key, value in candidate_relation.properties.items():
                if (
                    key in merged_relation_properties
                    and merged_relation_properties[key] != value
                ):
                    relation_property_conflict = True
                else:
                    merged_relation_properties[key] = value
        if relation_property_conflict:
            blockers.append("CANONICAL_RELATION_PROPERTY_CONFLICT")
            decisions.append(CanonicalizationDecision(
                action=CanonicalizationAction.BLOCK_PROPERTY_CONFLICT,
                candidate_ids=[item.edge_id for item in group],
                canonical_id=relation_id,
                rationale="Duplicate relation candidates contain incompatible schema-safe properties.",
            ))
        relations.append(CanonicalRelation(
            canonical_edge_id=relation_id,
            source_id=source,
            relation_type=relation_type,
            target_id=target,
            predicate_definition=max(
                (item.predicate_definition for item in group), key=len, default=""
            ),
            rationale=" | ".join(sorted({item.rationale for item in group})),
            source_candidate_edge_ids=[item.edge_id for item in group],
            source_fragment_ids=sorted({
                fragment for item in group for fragment in item.source_fragment_ids
            }),
            confidence=min(item.confidence for item in group),
            properties=merged_relation_properties,
        ))
    body = {
        "proposal_sha256": _model_sha256(proposal),
        "concepts": [item.model_dump(mode="json") for item in concepts],
        "relations": [item.model_dump(mode="json") for item in relations],
        "decisions": [item.model_dump(mode="json") for item in decisions],
        "blocker_codes": sorted(set(blockers)),
        "review_suggestion_count": review_suggestions,
        "status": "BLOCKED" if blockers else "PASS",
    }
    return KnowledgeCanonicalizationReport(
        **body,
        canonicalization_sha256=canonical_sha256({
            "schema_version": "astrograph-knowledge-canonicalization.v1", **body,
        }),
    )


def compile_candidate_graph(
    canonicalization: KnowledgeCanonicalizationReport,
    *,
    graph_id: str,
    lifecycle_status: LifecycleStatus = LifecycleStatus.DRAFT,
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED,
    assertion_status: AssertionStatus = AssertionStatus.EXTRACTED,
) -> KnowledgeGraphV3:
    if canonicalization.status != "PASS":
        raise ValueError("blocked canonicalization cannot compile a candidate graph")
    source_sha = canonicalization.canonicalization_sha256
    source_id = f"source:construction:{source_sha[:20]}"
    nodes = [GraphNodeV3(
        node_id=item.canonical_id,
        node_type=item.node_type,
        source_id=source_id,
        source_sha256=source_sha,
        lifecycle_status=lifecycle_status,
        properties={
            **item.properties,
            "canonical_name": item.canonical_name,
            "definition": item.definition,
            "aliases": item.aliases,
            "source_candidate_ids": item.source_candidate_ids,
            "source_fragment_ids": item.source_fragment_ids,
            "construction_confidence": item.confidence,
        },
    ) for item in canonicalization.concepts]
    edges = [GraphEdgeV3(
        edge_id=item.canonical_edge_id,
        source_id=item.source_id,
        relation_type=item.relation_type,
        target_id=item.target_id,
        provenance_id=source_id,
        provenance_sha256=canonical_sha256({
            "fragments": item.source_fragment_ids,
            "rationale": item.rationale,
        }),
        lifecycle_status=lifecycle_status,
        assertion_status=assertion_status,
        review_status=review_status,
        properties={
            **item.properties,
            "predicate_definition": item.predicate_definition,
            "rationale": item.rationale,
            "source_candidate_edge_ids": item.source_candidate_edge_ids,
            "source_fragment_ids": item.source_fragment_ids,
            "construction_confidence": item.confidence,
        },
    ) for item in canonicalization.relations]
    return KnowledgeGraphV3(graph_id=graph_id, nodes=nodes, edges=edges)


def _source_ancestor_sets(trees: list[SourceStructureTree]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for tree in trees:
        for section in tree.sections:
            result[section.fragment_id] = set(section.ancestor_section_ids + [section.section_id])
    return result


def suggest_hidden_edges(
    graph: KnowledgeGraphV3,
    *,
    trees: list[SourceStructureTree],
    threshold: float = 0.35,
    limit: int = 50,
) -> list[KnowledgeExpansionSuggestion]:
    """Rank Tree-KG-style hidden-edge candidates without applying any edge."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    existing_directed_pairs: set[tuple[str, str]] = set()
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)
        existing_directed_pairs.add((edge.source_id, edge.target_id))
    ancestors_by_fragment = _source_ancestor_sets(trees)
    fragments_by_node = {
        node.node_id: [str(item) for item in node.properties.get("source_fragment_ids", [])]
        for node in graph.nodes
    }
    node_ancestors = {
        node_id: set().union(*(ancestors_by_fragment.get(fragment, set()) for fragment in fragments))
        if fragments else set()
        for node_id, fragments in fragments_by_node.items()
    }
    suggestions: list[KnowledgeExpansionSuggestion] = []
    nodes = sorted(graph.nodes, key=lambda item: item.node_id)
    for left in nodes:
        for right in nodes:
            if left.node_id == right.node_id:
                continue
            allowed_relations = sorted(
                {
                    relation
                    for source_type, relation, target_type in RELATION_ALLOWLIST
                    if source_type == left.node_type and target_type == right.node_type
                },
                key=lambda item: item.value,
            )
            if not allowed_relations:
                continue
            if (left.node_id, right.node_id) in existing_directed_pairs:
                continue
            left_name = str(left.properties.get("canonical_name", left.node_id))
            right_name = str(right.properties.get("canonical_name", right.node_id))
            semantic = SequenceMatcher(None, _name_key(left_name), _name_key(right_name)).ratio()
            union_neighbors = neighbors[left.node_id] | neighbors[right.node_id]
            common_neighbors = neighbors[left.node_id] & neighbors[right.node_id]
            neighbor_score = len(common_neighbors) / max(1, len(union_neighbors))
            ancestor_union = node_ancestors[left.node_id] | node_ancestors[right.node_id]
            ancestor_common = node_ancestors[left.node_id] & node_ancestors[right.node_id]
            ancestor_score = len(ancestor_common) / max(1, len(ancestor_union))
            combined = 0.6 * semantic + 0.3 * neighbor_score + 0.1 * ancestor_score
            if combined < threshold:
                continue
            suggestion_id = "KGEXP-" + canonical_sha256({
                "source": left.node_id,
                "candidate_relations": [item.value for item in allowed_relations],
                "target": right.node_id,
                "score": round(combined, 12),
            })[:20].upper()
            suggestions.append(KnowledgeExpansionSuggestion(
                suggestion_id=suggestion_id,
                source_id=left.node_id,
                candidate_relation_types=allowed_relations,
                target_id=right.node_id,
                semantic_similarity=semantic,
                common_neighbor_score=neighbor_score,
                common_source_ancestor_score=ancestor_score,
                combined_score=combined,
            ))
    return sorted(
        suggestions,
        key=lambda item: (
            -item.combined_score, item.source_id, item.target_id
        ),
    )[:limit]


def assess_candidate_graph(
    graph: KnowledgeGraphV3 | None,
    *,
    trees: list[SourceStructureTree],
    canonicalization: KnowledgeCanonicalizationReport,
) -> KnowledgeConstructionQualityReport:
    if graph is None:
        empty_shape = GraphShapeReport(status="FAIL", node_count=0, edge_count=0, issues=[])
        return KnowledgeConstructionQualityReport(
            status="BLOCKED", node_count=0, edge_count=0, connected_component_count=0,
            orphan_node_count=0, traced_node_fraction=0.0, traced_edge_fraction=0.0,
            used_fragment_fraction=0.0, duplicate_node_id_count=0, duplicate_edge_id_count=0,
            shape_report=empty_shape,
            blocker_codes=canonicalization.blocker_codes or ["UPSTREAM_STAGE_BLOCKED"],
        )
    shape = GraphShapeValidator().validate(graph)
    node_ids = [item.node_id for item in graph.nodes]
    edge_ids = [item.edge_id for item in graph.edges]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    unseen = set(node_ids)
    component_count = 0
    while unseen:
        component_count += 1
        queue = deque([unseen.pop()])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current] & unseen:
                unseen.remove(neighbor)
                queue.append(neighbor)
    all_fragments = {
        section.fragment_id for tree in trees for section in tree.sections
    }
    used_fragments = {
        str(fragment)
        for node in graph.nodes
        for fragment in node.properties.get("source_fragment_ids", [])
    } | {
        str(fragment)
        for edge in graph.edges
        for fragment in edge.properties.get("source_fragment_ids", [])
    }
    traced_nodes = sum(bool(node.properties.get("source_fragment_ids")) for node in graph.nodes)
    traced_edges = sum(bool(edge.properties.get("source_fragment_ids")) for edge in graph.edges)
    blockers = [] if shape.status == "PASS" else ["V3_GRAPH_SHAPE_FAILED"]
    if used_fragments - all_fragments:
        blockers.append("UNKNOWN_COMPILED_FRAGMENT")
    return KnowledgeConstructionQualityReport(
        status="BLOCKED" if blockers else "PASS",
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        connected_component_count=component_count,
        orphan_node_count=sum(not adjacency[node_id] for node_id in node_ids),
        traced_node_fraction=traced_nodes / max(1, len(graph.nodes)),
        traced_edge_fraction=traced_edges / max(1, len(graph.edges)),
        used_fragment_fraction=len(used_fragments & all_fragments) / max(1, len(all_fragments)),
        duplicate_node_id_count=len(node_ids) - len(set(node_ids)),
        duplicate_edge_id_count=len(edge_ids) - len(set(edge_ids)),
        shape_report=shape,
        blocker_codes=blockers,
    )


def build_knowledge_construction_bundle(
    *,
    source_profiles: list[KnowledgeSourceProfile],
    source_trees: list[SourceStructureTree],
    proposal: KnowledgeExtractionProposal,
    approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str],
    approved_extraction_methods: list[str],
    excluded_review_refreeze_sha256: str,
    graph_id: str,
    build_mode: Literal["NEW_EXTRACTION", "MIGRATE_EXISTING_GOVERNED_ONTOLOGY"] = "NEW_EXTRACTION",
    minimum_independent_causal_sources: int = 2,
    lifecycle_status: LifecycleStatus = LifecycleStatus.DRAFT,
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED,
    assertion_status: AssertionStatus = AssertionStatus.EXTRACTED,
) -> KnowledgeConstructionBundle:
    if build_mode == "NEW_EXTRACTION" and minimum_independent_causal_sources < 2:
        raise ValueError("new extraction cannot weaken the two-group causal evidence gate")
    if build_mode == "NEW_EXTRACTION" and lifecycle_status == LifecycleStatus.ACTIVE:
        raise ValueError("new extraction cannot compile directly to ACTIVE lifecycle status")
    validation = KnowledgeCandidateCriticAgent().review(
        source_profiles=source_profiles,
        proposal=proposal,
        approved_existing_nodes=approved_existing_nodes,
        approved_extraction_methods=approved_extraction_methods,
        minimum_independent_causal_sources=minimum_independent_causal_sources,
    )
    canonicalization = canonicalize_knowledge_proposal(proposal)
    graph: KnowledgeGraphV3 | None = None
    if validation.status != "BLOCKED" and canonicalization.status != "BLOCKED":
        graph = compile_candidate_graph(
            canonicalization,
            graph_id=graph_id,
            lifecycle_status=lifecycle_status,
            review_status=review_status,
            assertion_status=assertion_status,
        )
    quality = assess_candidate_graph(
        graph, trees=source_trees, canonicalization=canonicalization
    )
    if quality.status == "BLOCKED":
        graph = None
    suggestions = suggest_hidden_edges(graph, trees=source_trees) if graph is not None else []
    tasks = build_extraction_tasks(source_trees)
    build_identity = {
        "source_profile_sha256": [
            knowledge_source_profile_sha256(item) for item in source_profiles
        ],
        "source_tree_sha256": [item.tree_sha256 for item in source_trees],
        "proposal_sha256": _model_sha256(proposal),
        "canonicalization_sha256": canonicalization.canonicalization_sha256,
        "graph_id": graph_id,
        "build_mode": build_mode,
    }
    build_id = "KGBUILD-" + canonical_sha256(build_identity)[:20].upper()
    status = (
        "CANDIDATE_READY_FOR_HUMAN_REVIEW"
        if graph is not None
        else "BLOCKED"
    )
    body = {
        "build_id": build_id,
        "build_mode": build_mode,
        "source_profiles": source_profiles,
        "source_trees": source_trees,
        "extraction_tasks": tasks,
        "proposal": proposal,
        "candidate_validation": validation,
        "canonicalization": canonicalization,
        "candidate_graph": graph,
        "quality": quality,
        "expansion_suggestions": suggestions,
        "excluded_review_refreeze_sha256": excluded_review_refreeze_sha256,
        "status": status,
    }
    hash_body = {
        "schema_version": "astrograph-reusable-knowledge-construction.v2",
        **{
            key: (
                [entry.model_dump(mode="json") for entry in value]
                if isinstance(value, list) and value and isinstance(value[0], BaseModel)
                else value.model_dump(mode="json")
                if isinstance(value, BaseModel)
                else value
            )
            for key, value in body.items()
        },
        "expansion_operators": [
            "CONTEXT_ENRICH", "CANONICALIZE", "EDGE_SUGGEST", "MERGE_SUBGRAPH"
        ],
        "publication_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "A reproducible candidate graph and review suggestions only. LLM extraction, canonicalization "
            "and expansion cannot publish a KnowledgeRelease, change Runtime voting or modify Active KG."
        ),
    }
    return KnowledgeConstructionBundle(
        **body,
        content_sha256=canonical_sha256(_construction_content_identity(hash_body)),
        bundle_sha256=canonical_sha256(hash_body),
    )


def write_construction_bundle(
    bundle: KnowledgeConstructionBundle,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "bundle": output / "construction_bundle.json",
        "candidate_graph": output / "candidate_graph.json",
        "quality": output / "quality_report.json",
        "canonicalization": output / "canonicalization_report.json",
        "expansion_suggestions": output / "expansion_suggestions.json",
    }
    paths["bundle"].write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if bundle.candidate_graph is not None:
        paths["candidate_graph"].write_text(
            bundle.candidate_graph.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    else:
        paths["candidate_graph"].write_text("null\n", encoding="utf-8")
    paths["quality"].write_text(bundle.quality.model_dump_json(indent=2) + "\n", encoding="utf-8")
    paths["canonicalization"].write_text(
        bundle.canonicalization.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    paths["expansion_suggestions"].write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in bundle.expansion_suggestions],
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return paths


__all__ = [
    "CanonicalConcept", "CanonicalRelation", "CanonicalizationAction",
    "CanonicalizationDecision", "KnowledgeCanonicalizationReport",
    "KnowledgeConstructionBundle", "KnowledgeConstructionQualityReport",
    "KnowledgeExpansionSuggestion", "KnowledgeExtractionPassKind",
    "KnowledgeExtractionTask", "KnowledgeSourceRole", "SourceSection",
    "SourceStructureTree", "DocumentKnowledgeGraphScope",
    "DocumentKnowledgeStagingBundle", "DocumentProcessingCrosswalk",
    "GovernedDocumentProcessingBundle", "assess_candidate_graph", "build_extraction_tasks",
    "build_knowledge_construction_bundle", "canonicalize_knowledge_proposal",
    "compile_candidate_graph", "process_governed_document", "process_governed_docx",
    "govern_document_knowledge_proposal", "profile_structured_source",
    "stage_governed_document_source", "stage_quarantined_document_source", "suggest_hidden_edges",
    "write_construction_bundle",
]
