"""Truth-free OAG contracts layered on native AstroGraph evidence and fusion."""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.fusion import FaultRanking, BiRRFConfig, RankingSource
from .three_task import ThreeTaskResult
from .assessment import DiagnosticAssessment


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class OAGInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["astrograph-oag-input.v1"] = "astrograph-oag-input.v1"
    bundle: EpisodeEvidenceBundle
    diagnostic_ranking: FaultRanking
    fusion_config: BiRRFConfig = Field(default_factory=BiRRFConfig)
    decision_policy: Literal["legacy", "evidence_aware"] = "legacy"
    top_k: int = Field(default=5, ge=1, le=50)
    require_paperseek: bool = False
    require_ontology_extraction: bool = False
    require_llm: bool = False
    three_task_result: ThreeTaskResult | None = None
    review_evidence: bool = False
    resolve_supported_ties: bool = False
    resolve_unsupported_top: bool = False
    complementary_diagnosis_enabled: bool = False
    mechanism_confirmation_enabled: bool = False
    maximum_diagnostic_candidates: int = Field(default=3, ge=2, le=10)
    registered_fault_ids: list[str] | None = None

    @classmethod
    def from_prediction(cls, bundle, prediction, *, evidence_projection=None, registered_fault_ids=None, **kwargs):
        """Bridge an existing three-task provider result without reading labels."""
        heads = ThreeTaskResult.from_prediction(
            prediction, registered_fault_ids=registered_fault_ids or sorted({d.target_fault_id for d in bundle.evidence_definitions}),
            start_s=bundle.episode.start_s, end_s=bundle.episode.end_s, evidence_projection=evidence_projection)
        return cls(bundle=bundle, diagnostic_ranking=heads.classification_ranking,
                   three_task_result=heads, registered_fault_ids=registered_fault_ids, **kwargs)

    @model_validator(mode="after")
    def validate_inference_inputs(self):
        if self.mechanism_confirmation_enabled and not self.complementary_diagnosis_enabled:
            raise ValueError('mechanism confirmation requires complementary diagnosis')
        if not self.bundle.episode.source_is_label_free:
            raise ValueError("OAG requires label-free episode evidence")
        if self.diagnostic_ranking.source != RankingSource.DATA_DRIVEN:
            raise ValueError("diagnostic_ranking must be the upstream DATA_DRIVEN branch")
        if self.registered_fault_ids is not None:
            ids = self.registered_fault_ids
            if not ids or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
                raise ValueError("registered fault IDs must be nonempty and unique")
            if not {d.target_fault_id for d in self.bundle.evidence_definitions} <= set(ids):
                raise ValueError("model fault universe omits configured evidence faults")
        episode_id = self.bundle.episode.episode_id
        if self.diagnostic_ranking.episode_id != episode_id:
            raise ValueError("diagnostic ranking and evidence must reference the same episode")
        if self.three_task_result is not None:
            heads = self.three_task_result
            if heads.episode_id != episode_id:
                raise ValueError("three-task result episode mismatch")
            if (heads.window_start_s, heads.window_end_s) != (self.bundle.episode.start_s, self.bundle.episode.end_s):
                raise ValueError("three-task result window mismatch")
            if heads.evidence_projection is not None:
                binding = heads.evidence_projection
                if binding.evidence_bundle_sha256 != content_hash(self.bundle.model_dump(mode="json")):
                    raise ValueError("native projection evidence bundle hash mismatch")
                if binding.feature_input_hashes != {f.feature_result_id:f.input_data_sha256 for f in self.bundle.feature_results}:
                    raise ValueError("native projection feature input hashes mismatch")
            elif heads.input_data_sha256 not in {f.input_data_sha256 for f in self.bundle.feature_results}:
                raise ValueError("three-task result telemetry hash is not bound to the evidence")
            if heads.classification_ranking.model_dump() != self.diagnostic_ranking.model_dump():
                raise ValueError("three-task classification must be the exact fusion input ranking")
        facts = self.bundle.evidence_facts
        if len({f.evidence_fact_id for f in facts}) != len(facts):
            raise ValueError("duplicate evidence fact IDs")
        definitions = {d.evidence_definition_id: d for d in self.bundle.evidence_definitions}
        if len(definitions) != len(self.bundle.evidence_definitions):
            raise ValueError("duplicate evidence definition IDs")
        features = {f.feature_result_id for f in self.bundle.feature_results}
        if len(features) != len(self.bundle.feature_results):
            raise ValueError("duplicate feature result IDs")
        for fact in facts:
            if fact.episode_id != episode_id:
                raise ValueError("evidence fact episode mismatch")
            definition = definitions.get(fact.evidence_definition_id)
            if definition is None or definition.target_fault_id != fact.target_fault_id or definition.role != fact.role:
                raise ValueError("evidence fact does not match its registered definition")
            if fact.feature_result_id not in features and fact.feature_result_id != "MISSING":
                raise ValueError("evidence fact references an unknown feature result")
        # Evaluation metadata never enters a prompt. Reject nonempty private truth
        # rather than attempting to clean it or letting it influence retrieval.
        if self.bundle.evaluation_metadata:
            raise ValueError("remove evaluation_metadata before invoking OAG; labels belong to evaluator")
        return self


class LiteratureDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["EXTERNAL_KNOWLEDGE_NOT_OBSERVATION"] = "EXTERNAL_KNOWLEDGE_NOT_OBSERVATION"

    @model_validator(mode="after")
    def check_content_hash(self):
        if hashlib.sha256(self.text.encode()).hexdigest() != self.source_content_sha256:
            raise ValueError("literature content hash mismatch")
        from urllib.parse import urlsplit
        parsed = urlsplit(self.source_url)
        if self.source_url == "urn:sha256:" + self.source_content_sha256:
            return self
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("literature needs an HTTP(S) source URL")
        return self


class KnowledgeTriple(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject: str = Field(min_length=1)
    predicate: Literal["part_of", "affects", "associated_with", "measured_by", "has_symptom", "requires_evidence"]
    object: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)


class OntologyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    triples: list[KnowledgeTriple] = Field(default_factory=list)
    # Live literature does not silently change approved diagnostic knowledge.
    lifecycle: Literal["REVIEW_REQUIRED"] = "REVIEW_REQUIRED"
    active_kg_modified: Literal[False] = False


class ExplainedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["OBSERVATION", "KNOWLEDGE", "HYPOTHESIS"]
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("citation_ids")
    @classmethod
    def unique_citations(cls, value):
        return sorted(set(value))


class OAGExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_fault_id: str | None = None
    abstained: bool
    claims: list[ExplainedClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    assessment: DiagnosticAssessment | None = None

    @model_validator(mode="after")
    def validate_decision(self):
        if self.abstained != (self.candidate_fault_id is None):
            raise ValueError("abstention must exactly correspond to no selected fault")
        return self


class ReviewedOAGExplanation(OAGExplanation):
    """A requested evidence assessment must be required in the model's schema."""
    assessment: DiagnosticAssessment = Field(...)
