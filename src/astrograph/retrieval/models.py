from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .document_governance import (
    DocumentAccessContext,
    DocumentEvidenceClaim,
    GovernedDocumentDescriptor,
)


class RetrievalStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    LOW_MATCH = "LOW_MATCH"
    NO_MATCH = "NO_MATCH"


class RetrievalLane(str, Enum):
    LOCAL_DOCUMENT = "LOCAL_DOCUMENT"
    EXPERIENCE = "EXPERIENCE"
    ONTOLOGY = "ONTOLOGY"


class DiagnosticRetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-diagnostic-retrieval-query.v1"
    query_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    required_concept_ids: list[str] = Field(default_factory=list)
    seed_concept_ids: list[str] = Field(default_factory=list)
    caller_access_tags: list[str] = Field(default_factory=lambda: ["PUBLIC"])
    top_k: int = Field(default=8, ge=1, le=32)
    maximum_ontology_hops: int = Field(default=2, ge=0, le=2)
    construction_mode: Literal["CALLER_SUPPLIED", "EVIDENCE_AWARE"] = "CALLER_SUPPLIED"
    grounding_fault_ids: list[str] = Field(default_factory=list)
    grounding_evidence_fact_ids: list[str] = Field(default_factory=list)
    document_access_context: DocumentAccessContext | None = None

    @field_validator(
        "required_concept_ids",
        "seed_concept_ids",
        "caller_access_tags",
        "grounding_fault_ids",
        "grounding_evidence_fact_ids",
    )
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("retrieval query lists must contain unique values")
        return value


class GovernedTextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-governed-text-chunk.v1"
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_version: str = Field(min_length=1)
    text: str = Field(min_length=1)
    concept_ids: list[str] = Field(default_factory=list)
    required_access_tags: list[str] = Field(default_factory=lambda: ["PUBLIC"])
    lifecycle_status: Literal["ACTIVE", "SUPERSEDED", "RETIRED"] = "ACTIVE"
    review_status: Literal["MACHINE_VALIDATED", "HUMAN_REVIEWED"] = "HUMAN_REVIEWED"
    content_safety_status: Literal["PASS", "REVIEW_REQUIRED", "REJECT"] = "PASS"
    lane: RetrievalLane = RetrievalLane.LOCAL_DOCUMENT
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    modalities: list[Literal["TEXT", "TABLE", "FORMULA", "IMAGE"]] = Field(
        default_factory=lambda: ["TEXT"]
    )
    source_locators: list[str] = Field(default_factory=list)
    structure_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    document_descriptor: GovernedDocumentDescriptor | None = None

    @field_validator("concept_ids", "required_access_tags", "section_path", "modalities", "source_locators")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("chunk lists must contain unique values")
        return value

    @model_validator(mode="after")
    def content_hash_matches(self) -> "GovernedTextChunk":
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.chunk_sha256:
            raise ValueError("chunk_sha256 does not match chunk text")
        if self.structure_sha256 is not None:
            structured = {
                "text": self.text,
                "section_id": self.section_id,
                "section_path": self.section_path,
                "modalities": self.modalities,
                "source_locators": self.source_locators,
            }
            expected = hashlib.sha256(
                json.dumps(
                    structured, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if expected != self.structure_sha256:
                raise ValueError("structure_sha256 does not match chunk structure")
        return self


class OntologyConcept(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_version: str = Field(min_length=1)
    lifecycle_status: Literal["ACTIVE", "SUPERSEDED", "RETIRED"] = "ACTIVE"
    review_status: Literal["MACHINE_VALIDATED", "HUMAN_REVIEWED"] = "HUMAN_REVIEWED"
    required_access_tags: list[str] = Field(default_factory=lambda: ["PUBLIC"])

    @field_validator("labels", "aliases", "required_access_tags")
    @classmethod
    def unique_nonempty_text(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("ontology lists must contain unique nonempty strings")
        return value


class OntologyRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str = Field(min_length=1)
    source_concept_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    target_concept_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_version: str = Field(min_length=1)
    lifecycle_status: Literal["ACTIVE", "SUPERSEDED", "RETIRED"] = "ACTIVE"
    review_status: Literal["MACHINE_VALIDATED", "HUMAN_REVIEWED"] = "HUMAN_REVIEWED"
    required_access_tags: list[str] = Field(default_factory=lambda: ["PUBLIC"])
    source_fragment_ids: list[str] = Field(default_factory=list)
    epistemic_status: str | None = None
    association_semantics: str | None = None
    retrieval_only: bool = False
    diagnostic_evidence_allowed: bool = False
    runtime_vote_allowed: bool = False

    @field_validator("required_access_tags", "source_fragment_ids")
    @classmethod
    def unique_access_tags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("ontology access tags must be unique nonempty strings")
        return value

    @model_validator(mode="after")
    def retrieval_boundary_is_coherent(self) -> "OntologyRelation":
        if self.retrieval_only and (
            self.diagnostic_evidence_allowed or self.runtime_vote_allowed
        ):
            raise ValueError("retrieval-only ontology relations cannot be evidence or vote")
        return self


class PreReasoningRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-pre-reasoning-retrieval-config.v1"
    bm25_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    term_coverage_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    ontology_concept_bonus: float = Field(default=0.10, ge=0.0, le=0.25)
    include_section_path_in_lexical_index: bool = False
    minimum_local_score: float = Field(default=0.20, ge=0.0, le=1.25)
    minimum_concept_coverage: float = Field(default=0.50, ge=0.0, le=1.0)
    require_local_and_ontology: bool = True
    maximum_ontology_paths: int = Field(default=128, ge=1, le=1024)
    maximum_relations_per_concept: int = Field(default=32, ge=1, le=256)
    allowed_relation_types: list[str] = Field(
        default_factory=lambda: [
            "EQUIVALENT_TO",
            "IS_A",
            "PART_OF",
            "SUSCEPTIBLE_TO",
            "PROPAGATES_TO",
            "MANIFESTS_AS",
            "INDICATED_BY",
        ]
    )

    @model_validator(mode="after")
    def weights_are_nonzero(self) -> "PreReasoningRetrievalConfig":
        if self.bm25_weight + self.term_coverage_weight + self.dense_weight <= 0.0:
            raise ValueError("at least one local retrieval weight must be positive")
        if len(self.allowed_relation_types) != len(set(self.allowed_relation_types)):
            raise ValueError("allowed_relation_types must be unique")
        return self


class LocalRetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    source_id: str
    source_uri: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_version: str
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0)
    bm25_score: float = Field(ge=0.0, le=1.0)
    term_coverage: float = Field(ge=0.0, le=1.0)
    dense_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_concept_ids: list[str] = Field(default_factory=list)
    snippet: str
    lane: RetrievalLane = RetrievalLane.LOCAL_DOCUMENT
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    modalities: list[Literal["TEXT", "TABLE", "FORMULA", "IMAGE"]] = Field(default_factory=list)
    source_locators: list[str] = Field(default_factory=list)
    structure_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class OntologyPathHop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str
    source_concept_id: str
    relation_type: str
    target_concept_id: str
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_version: str
    source_fragment_ids: list[str] = Field(default_factory=list)
    epistemic_status: str | None = None
    association_semantics: str | None = None
    retrieval_only: bool = False
    diagnostic_evidence_allowed: bool = False
    runtime_vote_allowed: bool = False

    @field_validator("source_fragment_ids")
    @classmethod
    def unique_fragment_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("ontology source fragments must be unique nonempty strings")
        return value

    @model_validator(mode="after")
    def retrieval_boundary_is_coherent(self) -> "OntologyPathHop":
        if self.retrieval_only and (
            self.diagnostic_evidence_allowed or self.runtime_vote_allowed
        ):
            raise ValueError("retrieval-only ontology hops cannot be evidence or vote")
        return self


class DocumentGraphTraceLink(BaseModel):
    """Auditable, non-voting bridge from a retrieval chunk to a KG fragment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-graph-trace-link.v1"
    retrieval_chunk_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_section_id: str = Field(pattern=r"^section:[^\s]+$")
    knowledge_fragment_id: str = Field(pattern=r"^fragment:[^\s]+$")
    source_locators: list[str] = Field(min_length=1)
    graph_scope: str = Field(min_length=1)
    link_status: Literal[
        "STAGING_TRACE_ONLY", "CANDIDATE_GRAPH_LINK", "REVIEWED_GRAPH_LINK"
    ]
    graph_id: str | None = Field(default=None, min_length=1)
    graph_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    graph_item_ids: list[str] = Field(default_factory=list)
    runtime_vote_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False

    @field_validator("graph_item_ids", "source_locators")
    @classmethod
    def unique_graph_items(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("trace link lists must contain unique nonempty strings")
        return value

    @model_validator(mode="after")
    def status_matches_items(self) -> "DocumentGraphTraceLink":
        if self.link_status == "STAGING_TRACE_ONLY" and self.graph_item_ids:
            raise ValueError("staging-only trace links cannot claim reviewed graph items")
        if self.link_status == "STAGING_TRACE_ONLY" and (
            self.graph_id is not None or self.graph_sha256 is not None
        ):
            raise ValueError("staging-only trace links cannot claim graph identity")
        if self.link_status != "STAGING_TRACE_ONLY" and (
            not self.graph_item_ids or self.graph_id is None or self.graph_sha256 is None
        ):
            raise ValueError("graph trace links require graph identity and at least one item")
        return self


class DocumentClaimAdmissionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-claim-admission-policy.v1"
    formal_document_fusion_enabled: bool = False
    policy_review_status: Literal["UNREVIEWED", "HUMAN_REVIEWED"] = "UNREVIEWED"
    reviewer_id: str | None = None
    fusion_approval_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    calibration_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_independence_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    calibrated_claim_ids: list[str] = Field(default_factory=list)
    independent_provenance_group_ids: list[str] = Field(default_factory=list)
    applicability_context: list[str] = Field(default_factory=list)

    @field_validator(
        "calibrated_claim_ids", "independent_provenance_group_ids", "applicability_context"
    )
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("document claim policy lists must be unique and nonempty")
        return value

    @model_validator(mode="after")
    def enabled_policy_requires_reviewed_receipts(self) -> "DocumentClaimAdmissionPolicy":
        if self.formal_document_fusion_enabled and (
            self.policy_review_status != "HUMAN_REVIEWED"
            or not self.reviewer_id
            or self.fusion_approval_receipt_sha256 is None
            or self.calibration_receipt_sha256 is None
            or self.source_independence_receipt_sha256 is None
        ):
            raise ValueError(
                "formal document fusion requires named review and approval, calibration, "
                "and source-independence receipts"
            )
        return self

    @property
    def policy_sha256(self) -> str:
        raw = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DocumentClaimAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-claim-admission.v1"
    claim_id: str = Field(min_length=1)
    retrieval_context_allowed: Literal[True] = True
    formal_fusion_eligible: bool = False
    reason_codes: list[str] = Field(min_length=1)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fusion_approval_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    calibration_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_independence_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    active_graph_modified: Literal[False] = False

    @model_validator(mode="after")
    def eligibility_matches_reasons(self) -> "DocumentClaimAdmission":
        blocking = [item for item in self.reason_codes if item != "FORMAL_FUSION_ELIGIBLE"]
        if self.formal_fusion_eligible != (not blocking):
            raise ValueError("document claim eligibility does not match its reason codes")
        if self.formal_fusion_eligible and any(
            value is None
            for value in (
                self.fusion_approval_receipt_sha256,
                self.calibration_receipt_sha256,
                self.source_independence_receipt_sha256,
            )
        ):
            raise ValueError("eligible document claim admission must retain all approval receipts")
        return self


class OntologyRetrievalPath(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed_concept_id: str
    target_concept_id: str
    hops: list[OntologyPathHop] = Field(min_length=1, max_length=2)
    lane: RetrievalLane = RetrievalLane.ONTOLOGY


class PreReasoningRetrievalPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-pre-reasoning-retrieval-package.v1"
    package_id: str = Field(min_length=1)
    query_id: str
    episode_id: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RetrievalStatus
    local_hits: list[LocalRetrievalHit] = Field(default_factory=list)
    ontology_paths: list[OntologyRetrievalPath] = Field(default_factory=list)
    ontology_seed_concept_ids: list[str] = Field(default_factory=list)
    ontology_asset_audits: list[dict] = Field(default_factory=list)
    document_graph_trace_links: list[DocumentGraphTraceLink] = Field(default_factory=list)
    document_evidence_claims: list[DocumentEvidenceClaim] = Field(default_factory=list)
    document_claim_admissions: list[DocumentClaimAdmission] = Field(default_factory=list)
    matched_concept_ids: list[str] = Field(default_factory=list)
    required_concept_coverage: float = Field(ge=0.0, le=1.0)
    top_local_score: float = Field(ge=0.0)
    dense_retrieval_available: bool = False
    access_filtered_chunk_count: int = Field(default=0, ge=0)
    document_access_reason_counts: dict[str, int] = Field(default_factory=dict)
    governance_filtered_chunk_count: int = Field(default=0, ge=0)
    access_filtered_ontology_item_count: int = Field(default=0, ge=0)
    ontology_paths_truncated: bool = False
    deduplicated_chunk_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    paperseek_fallback_recommended: bool = False
    active_graph_modified: bool = False
    query_construction_mode: Literal["CALLER_SUPPLIED", "EVIDENCE_AWARE"] = "CALLER_SUPPLIED"
    grounding_fault_ids: list[str] = Field(default_factory=list)
    grounding_evidence_fact_ids: list[str] = Field(default_factory=list)
    experience_hit_count: int = Field(default=0, ge=0)
    experience_source_ids: list[str] = Field(default_factory=list)
    experience_reuse_receipt: dict | None = None
    knowledge_update_candidates: list[dict] = Field(default_factory=list)
    claim_boundary: str = (
        "Pre-generation retrieval context only. It cannot change BiRRF, publish knowledge, establish "
        "causality, or turn retrieved text into formal diagnosis evidence."
    )

    @model_validator(mode="after")
    def trace_links_reference_returned_hits(self) -> "PreReasoningRetrievalPackage":
        hits = {item.chunk_id: item for item in self.local_hits}
        for link in self.document_graph_trace_links:
            hit = hits.get(link.retrieval_chunk_id)
            if hit is None:
                raise ValueError("document graph trace link must reference a returned local hit")
            if (
                link.source_document_id != hit.source_id
                or link.source_document_sha256 != hit.source_sha256
            ):
                raise ValueError("document graph trace link source does not match its returned hit")
        claim_ids = {item.claim_id for item in self.document_evidence_claims}
        if len(claim_ids) != len(self.document_evidence_claims):
            raise ValueError("document evidence claim IDs must be unique")
        links_by_key = {
            (item.retrieval_chunk_id, item.knowledge_fragment_id): item
            for item in self.document_graph_trace_links
        }
        for claim in self.document_evidence_claims:
            link = links_by_key.get((claim.source_chunk_id, claim.source_fragment_id))
            if link is None:
                raise ValueError("returned document claim is not bound to a returned trace link")
            if (
                claim.source_document_id != link.source_document_id
                or claim.source_document_sha256 != link.source_document_sha256
                or claim.source_locator not in link.source_locators
            ):
                raise ValueError("returned document claim provenance differs from its trace link")
        admission_ids = [item.claim_id for item in self.document_claim_admissions]
        if len(admission_ids) != len(set(admission_ids)):
            raise ValueError("document claim admissions must be unique")
        if set(admission_ids) != claim_ids:
            raise ValueError("document claim admissions must cover returned claims exactly")
        return self


class PreReasoningRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: DiagnosticRetrievalQuery
    chunks: list[GovernedTextChunk] = Field(default_factory=list)
    concepts: list[OntologyConcept] = Field(default_factory=list)
    relations: list[OntologyRelation] = Field(default_factory=list)
    ontology_asset_audits: list[dict] = Field(default_factory=list)
    document_graph_trace_links: list[DocumentGraphTraceLink] = Field(default_factory=list)
    document_evidence_claims: list[DocumentEvidenceClaim] = Field(default_factory=list)
    document_claim_admission_policy: DocumentClaimAdmissionPolicy = Field(
        default_factory=DocumentClaimAdmissionPolicy
    )
    config: PreReasoningRetrievalConfig = Field(default_factory=PreReasoningRetrievalConfig)
    experience_reuse_receipt: dict | None = None
    knowledge_update_candidates: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def references_are_closed(self) -> "PreReasoningRetrievalRequest":
        chunks = {item.chunk_id: item for item in self.chunks}
        link_keys: set[tuple[str, str]] = set()
        for link in self.document_graph_trace_links:
            chunk = chunks.get(link.retrieval_chunk_id)
            if chunk is None:
                raise ValueError("document graph trace link references an unknown chunk")
            if (
                link.source_document_id != chunk.source_id
                or link.source_document_sha256 != chunk.source_sha256
            ):
                raise ValueError("document graph trace link source does not match its chunk")
            key = (link.retrieval_chunk_id, link.knowledge_fragment_id)
            if key in link_keys:
                raise ValueError("document graph trace links must be unique per chunk/fragment")
            link_keys.add(key)
        claim_ids: set[str] = set()
        for claim in self.document_evidence_claims:
            if claim.claim_id in claim_ids:
                raise ValueError("document evidence claim IDs must be unique")
            claim_ids.add(claim.claim_id)
            matching_links = [
                link for link in self.document_graph_trace_links
                if link.retrieval_chunk_id == claim.source_chunk_id
                and link.knowledge_fragment_id == claim.source_fragment_id
            ]
            if not matching_links:
                raise ValueError("document evidence claim is not bound to a trace link")
            if any(
                claim.source_document_id != link.source_document_id
                or claim.source_document_sha256 != link.source_document_sha256
                or claim.source_locator not in link.source_locators
                for link in matching_links
            ):
                raise ValueError("document evidence claim provenance differs from its trace link")
        for audit in self.ontology_asset_audits:
            if (
                audit.get("active_graph_modified") is not False
                or audit.get("diagnostic_evidence_allowed") is not False
                or audit.get("runtime_vote_allowed") is not False
            ):
                raise ValueError(
                    "ontology asset audits must preserve retrieval-only non-voting boundaries"
                )
        for candidate in self.knowledge_update_candidates:
            if candidate.get("active_graph_modified") is not False:
                raise ValueError("experience-derived knowledge candidates must not modify the active graph")
        concept_ids = [item.concept_id for item in self.concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("ontology concept IDs must be unique")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("ontology relation IDs must be unique")
        known = set(concept_ids)
        unknown_query = set(self.query.required_concept_ids + self.query.seed_concept_ids) - known
        if unknown_query:
            raise ValueError(f"retrieval query references unknown concepts: {sorted(unknown_query)}")
        unknown_relations = {
            endpoint
            for item in self.relations
            for endpoint in (item.source_concept_id, item.target_concept_id)
            if endpoint not in known
        }
        if unknown_relations:
            raise ValueError(f"ontology relations reference unknown concepts: {sorted(unknown_relations)}")
        return self
