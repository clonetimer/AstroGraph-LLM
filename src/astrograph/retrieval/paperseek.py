from __future__ import annotations

from enum import Enum
import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import DiagnosticRetrievalQuery, PreReasoningRetrievalPackage, RetrievalStatus


class PaperSeekFallbackStatus(str, Enum):
    NOT_TRIGGERED = "NOT_TRIGGERED"
    OFFLINE_DISCOVERY_REQUESTED = "OFFLINE_DISCOVERY_REQUESTED"


class PaperSeekCandidateDisposition(str, Enum):
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"
    CORPUS_REVIEW_READY = "CORPUS_REVIEW_READY"


class LiteratureAnnotationDecision(str, Enum):
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    QUARANTINE = "QUARANTINE"


class LiteratureMethodRole(str, Enum):
    DIAGNOSTIC_BASELINE = "DIAGNOSTIC_BASELINE"
    METHOD_ANCHOR = "METHOD_ANCHOR"
    TOOL_PLANNING = "TOOL_PLANNING"
    KNOWLEDGE_GROUNDING = "KNOWLEDGE_GROUNDING"
    SAFETY_GOVERNANCE = "SAFETY_GOVERNANCE"
    EVALUATION_DESIGN = "EVALUATION_DESIGN"


class PaperSeekFallbackPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paperseek-fallback-policy.v1"
    minimum_relevance_score: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_source_quality_score: float = Field(default=0.60, ge=0.0, le=1.0)
    require_doi_or_stable_uri: bool = True
    allowed_license_statuses: list[str] = Field(default_factory=lambda: ["ALLOWED"])
    maximum_results: int = Field(default=50, ge=1, le=200)
    preferred_sources: list[str] = Field(
        default_factory=lambda: ["OpenAlex", "Semantic Scholar", "Crossref", "arXiv"]
    )
    claim_boundary: str = (
        "Engineering admission defaults. They are not calibrated retrieval thresholds and cannot authorize "
        "automatic corpus or graph publication."
    )


class PaperSeekDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paperseek-discovery-request.v1"
    request_id: str
    query_id: str
    episode_id: str
    query_text: str
    trigger_retrieval_package_id: str
    trigger_status: RetrievalStatus
    trigger_reason_codes: list[str]
    preferred_sources: list[str]
    maximum_results: int
    execution_mode: str = "OFFLINE_ASYNC"
    online_diagnosis_waits_for_result: bool = False
    active_corpus_modified: bool = False
    active_graph_modified: bool = False


class PaperSeekFallbackDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paperseek-fallback-decision.v1"
    status: PaperSeekFallbackStatus
    discovery_request: PaperSeekDiscoveryRequest | None = None
    reason_codes: list[str] = Field(default_factory=list)
    active_corpus_modified: bool = False
    active_graph_modified: bool = False


class PaperSeekLiteratureCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    title: str = Field(min_length=1)
    abstract: str = ""
    doi: str | None = None
    stable_uri: str | None = None
    provider: str = Field(min_length=1)
    provider_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_quality_score: float = Field(ge=0.0, le=1.0)
    license_status: str
    domain_applicability_status: str
    duplicate_of_candidate_id: str | None = None


class PaperSeekCandidateAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    disposition: PaperSeekCandidateDisposition
    reason_codes: list[str]
    corpus_candidate_created: bool = False
    active_corpus_modified: bool = False
    graph_candidate_created: bool = False
    active_graph_modified: bool = False
    requires_human_review: bool = True


class PaperSeekLiteratureAnnotation(BaseModel):
    """Governed title/abstract annotation, deliberately separate from corpus admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paperseek-literature-annotation.v1"
    candidate_id: str
    title: str = Field(min_length=1)
    stable_uri: str = Field(min_length=1)
    source_locator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    discovery_run_id: str = Field(min_length=1)
    decision: LiteratureAnnotationDecision
    method_roles: list[LiteratureMethodRole] = Field(default_factory=list)
    review_depth: str = "TITLE_ABSTRACT"
    annotation_status: str = "MODEL_ASSISTED_SELECTED"
    supported_design_claims: list[str] = Field(default_factory=list)
    unsupported_project_claims: list[str] = Field(default_factory=list)
    astrograph_compatibility: str = Field(min_length=1)
    selection_rationale: str = Field(min_length=1)
    runtime_use_allowed: bool = False
    active_corpus_modified: bool = False
    active_graph_modified: bool = False
    requires_human_confirmation: bool = True

    @model_validator(mode="after")
    def validate_source_locator_hash(self) -> "PaperSeekLiteratureAnnotation":
        expected = hashlib.sha256(self.stable_uri.encode("utf-8")).hexdigest()
        if self.source_locator_sha256 != expected:
            raise ValueError("source_locator_sha256 does not match stable_uri")
        return self

    @classmethod
    def select(
        cls,
        candidate: PaperSeekLiteratureCandidate,
        *,
        discovery_run_id: str,
        decision: LiteratureAnnotationDecision,
        method_roles: list[LiteratureMethodRole],
        supported_design_claims: list[str],
        unsupported_project_claims: list[str],
        astrograph_compatibility: str,
        selection_rationale: str,
    ) -> "PaperSeekLiteratureAnnotation":
        if decision == LiteratureAnnotationDecision.INCLUDE:
            if not method_roles:
                raise ValueError("included literature requires at least one method role")
            if not supported_design_claims:
                raise ValueError("included literature requires an explicit supported design claim")
            if not unsupported_project_claims:
                raise ValueError("included literature must record unsupported AstroGraph claims")
        stable_uri = candidate.stable_uri or (
            f"https://doi.org/{candidate.doi}" if candidate.doi else ""
        )
        if not stable_uri:
            raise ValueError("literature annotation requires a stable URI or DOI")
        return cls(
            candidate_id=candidate.candidate_id,
            title=candidate.title,
            stable_uri=stable_uri,
            source_locator_sha256=hashlib.sha256(stable_uri.encode("utf-8")).hexdigest(),
            provider=candidate.provider,
            discovery_run_id=discovery_run_id,
            decision=decision,
            method_roles=method_roles,
            supported_design_claims=supported_design_claims,
            unsupported_project_claims=unsupported_project_claims,
            astrograph_compatibility=astrograph_compatibility,
            selection_rationale=selection_rationale,
        )


class PaperSeekGovernedBridge:
    bridge_id = "paperseek-governed-offline-discovery"
    bridge_version = "0.1.0"

    @staticmethod
    def _normalized_identifier(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().casefold()
        return normalized or None

    @staticmethod
    def _normalized_title(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value.casefold()))

    @staticmethod
    def decide_fallback(
        query: DiagnosticRetrievalQuery,
        retrieval: PreReasoningRetrievalPackage,
        *,
        policy: PaperSeekFallbackPolicy | None = None,
    ) -> PaperSeekFallbackDecision:
        policy = policy or PaperSeekFallbackPolicy()
        if query.query_id != retrieval.query_id or query.episode_id != retrieval.episode_id:
            raise ValueError("PaperSeek fallback query does not match retrieval package")
        if retrieval.status == RetrievalStatus.SUFFICIENT:
            return PaperSeekFallbackDecision(
                status=PaperSeekFallbackStatus.NOT_TRIGGERED,
                reason_codes=["LOCAL_AND_ONTOLOGY_RETRIEVAL_SUFFICIENT"],
            )
        request = PaperSeekDiscoveryRequest(
            request_id=f"PAPERSEEK-{retrieval.package_id}",
            query_id=query.query_id,
            episode_id=query.episode_id,
            query_text=query.text,
            trigger_retrieval_package_id=retrieval.package_id,
            trigger_status=retrieval.status,
            trigger_reason_codes=retrieval.reason_codes,
            preferred_sources=policy.preferred_sources,
            maximum_results=policy.maximum_results,
        )
        return PaperSeekFallbackDecision(
            status=PaperSeekFallbackStatus.OFFLINE_DISCOVERY_REQUESTED,
            discovery_request=request,
            reason_codes=["PRE_REASONING_RETRIEVAL_BELOW_GATE"],
        )

    @staticmethod
    def assess_candidate(
        candidate: PaperSeekLiteratureCandidate,
        *,
        policy: PaperSeekFallbackPolicy | None = None,
    ) -> PaperSeekCandidateAdmission:
        policy = policy or PaperSeekFallbackPolicy()
        reasons: list[str] = []
        if candidate.duplicate_of_candidate_id is not None:
            reasons.append("DUPLICATE_CANDIDATE")
        if candidate.relevance_score < policy.minimum_relevance_score:
            reasons.append("RELEVANCE_BELOW_THRESHOLD")
        if candidate.source_quality_score < policy.minimum_source_quality_score:
            reasons.append("SOURCE_QUALITY_BELOW_THRESHOLD")
        if candidate.license_status not in policy.allowed_license_statuses:
            reasons.append("LICENSE_NOT_APPROVED")
        if candidate.domain_applicability_status != "IN_SCOPE":
            reasons.append("DOMAIN_APPLICABILITY_NOT_APPROVED")
        if policy.require_doi_or_stable_uri and not (candidate.doi or candidate.stable_uri):
            reasons.append("STABLE_IDENTIFIER_MISSING")
        if any(
            item in reasons
            for item in (
                "DUPLICATE_CANDIDATE",
                "LICENSE_NOT_APPROVED",
                "DOMAIN_APPLICABILITY_NOT_APPROVED",
            )
        ):
            disposition = PaperSeekCandidateDisposition.REJECT
        elif reasons:
            disposition = PaperSeekCandidateDisposition.QUARANTINE
        else:
            disposition = PaperSeekCandidateDisposition.CORPUS_REVIEW_READY
        return PaperSeekCandidateAdmission(
            candidate_id=candidate.candidate_id,
            disposition=disposition,
            reason_codes=reasons or ["MET_ENGINEERING_CORPUS_REVIEW_GATE"],
            corpus_candidate_created=disposition == PaperSeekCandidateDisposition.CORPUS_REVIEW_READY,
        )

    @classmethod
    def assess_candidates(
        cls,
        candidates: list[PaperSeekLiteratureCandidate],
        *,
        policy: PaperSeekFallbackPolicy | None = None,
    ) -> list[PaperSeekCandidateAdmission]:
        """Mark DOI, stable-URI, and normalized-title duplicates before admission."""
        seen: dict[tuple[str, str], str] = {}
        admissions: list[PaperSeekCandidateAdmission] = []
        for candidate in candidates:
            keys = [
                ("doi", value)
                for value in [cls._normalized_identifier(candidate.doi)]
                if value is not None
            ]
            keys.extend(
                ("uri", value)
                for value in [cls._normalized_identifier(candidate.stable_uri)]
                if value is not None
            )
            normalized_title = cls._normalized_title(candidate.title)
            if normalized_title:
                keys.append(("title", normalized_title))
            duplicate_of = next((seen[key] for key in keys if key in seen), None)
            assessed = (
                candidate
                if duplicate_of is None or candidate.duplicate_of_candidate_id is not None
                else candidate.model_copy(update={"duplicate_of_candidate_id": duplicate_of})
            )
            admissions.append(cls.assess_candidate(assessed, policy=policy))
            if duplicate_of is None:
                for key in keys:
                    seen[key] = candidate.candidate_id
        return admissions
