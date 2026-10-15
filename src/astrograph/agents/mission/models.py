"""Host policy and model-visible task contracts are deliberately separate."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

WorkMode = Literal["normal", "thinking", "creative"]

TaskKind = Literal["diagnose", "build_knowledge", "train_model", "research", "infer_candidate", "replay"]

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class ModeBudget(Strict):
    """Host-owned opt-in overrides; these grant no tools or release permissions."""
    max_checks: int | None = Field(default=None, ge=0, le=20)
    max_optional_actions: int | None = Field(default=None, ge=0, le=5)
    max_llm_calls: int | None = Field(default=None, ge=0, le=20)
    max_proposals: int = Field(default=3, ge=1, le=6)

class CaseIdentity(Strict):
    case_id: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    operating_mode: str = Field(min_length=1)
    observed_end: datetime

    @model_validator(mode="after")
    def aware(self):
        if self.observed_end.tzinfo is None or self.observed_end.utcoffset() is None:
            raise ValueError("memory requires an absolute timezone-aware observation time")
        return self

class AgentRequest(Strict):
    schema_version: Literal["astrograph-mission-request.v1"] = "astrograph-mission-request.v1"
    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
    task: TaskKind = "diagnose"
    work_mode: WorkMode = "normal"
    goal: str = Field(default="Diagnose spacecraft faults using data and verified mechanism evidence.", min_length=1, max_length=4000)
    # File paths are resolved by the HOST. The model receives only logical keys.
    artifacts: dict[str, str] = Field(default_factory=dict)
    documents: list[str] = Field(default_factory=list, max_length=32)
    case: CaseIdentity | None = None

class AgentSettings(Strict):
    """Administrator-owned settings. Never accept these from a model tool call."""
    schema_version: Literal["astrograph-mission-settings.v1"] = "astrograph-mission-settings.v1"
    project_root: str = "."
    read_roots: list[str] = Field(default_factory=lambda:["."])
    output_root: str = "runs/mission"
    allowed_work_modes: list[WorkMode] = Field(default_factory=lambda: ["normal", "thinking", "creative"])
    work_mode_budgets: dict[Literal["thinking", "creative"], ModeBudget] = Field(default_factory=dict)
    planner: Literal["deterministic", "llm_assisted"] = "deterministic"
    llm_config: str | None = None
    max_llm_calls: int = Field(default=6, ge=0, le=20)
    max_checks: int = Field(default=3, ge=0, le=20)
    max_optional_actions: int = Field(default=2, ge=0, le=5)
    max_artifact_bytes: int = Field(default=64_000_000, ge=1024, le=512_000_000)
    subprocess_timeout_s: float = Field(default=300, gt=0, le=7200)
    task_timeout_s: float = Field(default=600, gt=0, le=14400)
    llm_timeout_s: float = Field(default=120, gt=0, le=3600)
    max_http_attempts: int = Field(default=6, ge=0, le=100)
    context_token_limit: int = Field(default=16384, ge=4096, le=1048576)
    max_plan_revisions: int = Field(default=1, ge=0, le=4)
    ontology_batch_chars: int = Field(default=12000, ge=512, le=32000)
    ontology_max_batches: int = Field(default=16, ge=1, le=64)
    resume_read_only: bool = False
    llm_trace_mode: Literal["full", "hashes"] = "full"
    domain: Literal["adcs", "phm", "generic"] = "adcs"
    evidence_profile: Literal["sensor-checks", "rw-window", "legacy"] = "sensor-checks"
    # Explicit pins prevent the Agent from training then silently replacing production weights.
    approved_model_sha256: list[str] = Field(default_factory=list)
    approved_training_manifest_sha256: list[str] = Field(default_factory=list)
    memory_db: str | None = None
    memory_policy: str | None = None
    paperseek_url: str | None = None
    paperseek_source_root: str | None = None
    allow_external_literature: bool = False
    a2o_source_root: str | None = None
    allow_a2o: bool = False
    mineru_executable: str | None = None
    allow_mineru: bool = False
    # Optional offline model locations, never model-supplied URLs or IDs.
    embedding_model_path: str | None = None
    reranker_model_path: str | None = None
    # Capability discovery is host-owned and never authorizes model execution.
    model_capability_registry: str | None = None
    # Separate host-owned binding catalog bridges compatible capabilities to strict admitted providers.
    model_execution_bindings: str | None = None
    # Optional expert-governed document/feedback lifecycle.
    lifecycle_db: str | None = None
    lifecycle_policy: str | None = None
    use_approved_document_library: bool = False
    # Optional hash-pinned reviewed-claim assets for contextual RAG stance enrichment.
    # These do not authorize document votes or graph publication.
    reviewed_claim_trace_path: str | None = None
    reviewed_claim_trace_sha256: str | None = None
    reviewed_claim_mapping_path: str | None = None
    reviewed_claim_mapping_sha256: str | None = None

    @model_validator(mode="after")
    def check_pairing(self):
        if not self.allowed_work_modes or len(set(self.allowed_work_modes)) != len(self.allowed_work_modes):
            raise ValueError("unique nonempty allowed_work_modes required")
        if bool(self.memory_db) != bool(self.memory_policy):
            raise ValueError("memory_db and memory_policy must be configured together")
        if bool(self.lifecycle_db) != bool(self.lifecycle_policy):
            raise ValueError("lifecycle_db and lifecycle_policy must be configured together")
        if self.use_approved_document_library and not self.lifecycle_db:
            raise ValueError("approved document library requires lifecycle_db/lifecycle_policy")
        reviewed_assets=(self.reviewed_claim_trace_path,self.reviewed_claim_trace_sha256,
                         self.reviewed_claim_mapping_path,self.reviewed_claim_mapping_sha256)
        if any(reviewed_assets) and not all(reviewed_assets):
            raise ValueError("reviewed claim RAG binding requires paths and SHA-256 digests for both assets")
        import re
        for digest in (self.reviewed_claim_trace_sha256,self.reviewed_claim_mapping_sha256):
            if digest is not None and not re.fullmatch(r"[0-9a-f]{64}",digest):
                raise ValueError("reviewed claim RAG binding requires SHA-256 digests")
        if self.domain == "phm" and self.evidence_profile != "legacy":
            raise ValueError("PHM currently supports the legacy evidence profile only")
        for digest in self.approved_model_sha256 + self.approved_training_manifest_sha256:
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("approval must bind a SHA-256 digest")
        return self

class PreparatoryAction(Strict):
    tool_id: Literal["rag.retrieve", "literature.search"]
    query: str = Field(min_length=1, max_length=1200)
    rationale: str = Field(min_length=1, max_length=600)

class InvestigationPlan(Strict):
    task: TaskKind
    actions: list[PreparatoryAction] = Field(default_factory=list, max_length=5)
    rationale: str = Field(min_length=1, max_length=800)

class CheckChoice(Strict):
    tool_id: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=600)

class CitedParagraph(Strict):
    text: str = Field(min_length=1, max_length=2000)
    citation_ids: list[str] = Field(min_length=1, max_length=16)

class AgentNarrative(Strict):
    candidate_fault_id: str | None
    # LLM selects visible passage IDs; host renders exact text. Legacy full quotes
    # remain validated, but never mix both representations in one response.
    excerpt_ids: list[str] = Field(default_factory=list, max_length=8)
    paragraphs: list[CitedParagraph] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=12)

class ExplorationHypothesis(Strict):
    hypothesis: str = Field(min_length=1, max_length=1600)
    citation_ids: list[str] = Field(min_length=1, max_length=12)
    required_observations: list[str] = Field(min_length=1, max_length=12)
    falsification_test: str = Field(min_length=1, max_length=1600)
    status: Literal["UNVERIFIED"] = "UNVERIFIED"
    diagnostic_vote_allowed: Literal[False] = False

class ExplorationProposal(Strict):
    hypotheses: list[ExplorationHypothesis] = Field(default_factory=list, max_length=6)
    limitations: list[str] = Field(default_factory=list, max_length=12)
    active_release_allowed: Literal[False] = False
    confirmed_memory_write_allowed: Literal[False] = False

class AgentResult(Strict):
    schema_version: Literal["astrograph-mission-result.v1"] = "astrograph-mission-result.v1"
    task_id: str
    task: TaskKind
    status: Literal["SUCCEEDED", "PARTIAL", "BLOCKED", "FAILED"]
    work_mode: WorkMode = "normal"
    mode_receipt: dict[str, Any] = Field(default_factory=dict)
    request_sha256: str
    settings_sha256: str
    plan: dict[str, Any]
    output: dict[str, Any]
    tool_receipts: list[dict[str, Any]]
    limitations: list[str]
    capability_status: dict[str, Any] = Field(default_factory=dict)
    execution_budget: dict[str, Any] = Field(default_factory=dict)
    active_model_modified: Literal[False] = False
    active_kg_modified: Literal[False] = False
    control_actions_authorized: Literal[False] = False
