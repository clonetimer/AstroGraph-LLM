from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20].upper()}"


class ClaimType(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    INFERRED = "INFERRED"
    HYPOTHETICAL = "HYPOTHETICAL"
    UNKNOWN = "UNKNOWN"


class HypothesisType(str, Enum):
    REGISTERED_FAULT = "REGISTERED_FAULT"
    NO_REGISTERED_FAULT = "NO_REGISTERED_FAULT"
    RESEARCH_HYPOTHESIS = "RESEARCH_HYPOTHESIS"


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    UNDER_VERIFICATION = "UNDER_VERIFICATION"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class DecisionDisposition(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    NO_REGISTERED_FAULT = "NO_REGISTERED_FAULT"


class CritiqueSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKING = "BLOCKING"


class InvestigationBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_reflection_rounds: int = Field(default=3, ge=0, le=10)
    max_graph_queries: int = Field(default=10, ge=0, le=100)
    max_tool_calls: int = Field(default=30, ge=1, le=500)
    max_simulation_calls: int = Field(default=0, ge=0, le=100)
    stop_after_no_new_evidence_rounds: int = Field(default=2, ge=1, le=10)


class InvestigationStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    expected_output_protocol: str = Field(min_length=1)


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-investigation-plan.v1"
    plan_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    candidate_fault_ids: list[str] = Field(default_factory=list)
    steps: list[InvestigationStep]
    budget: InvestigationBudget = Field(default_factory=InvestigationBudget)
    stop_conditions: list[str] = Field(default_factory=list)
    shadow_mode: bool = True
    created_at: str = Field(default_factory=_now)
    claim_boundary: str = (
        "Agentic investigation output is advisory and cannot replace the V0.4.4 deterministic diagnosis result."
    )

    @field_validator("steps")
    @classmethod
    def unique_steps(cls, value: list[InvestigationStep]) -> list[InvestigationStep]:
        ids = [item.step_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique")
        return value

    @classmethod
    def deterministic_id(cls, *, task_id: str, objective: str, candidate_fault_ids: list[str]) -> str:
        return _stable_id(
            "IP",
            {
                "task_id": task_id,
                "objective": objective,
                "candidate_fault_ids": sorted(candidate_fault_ids),
            },
        )


class GraphQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-graph-query-plan.v1"
    query_plan_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    start_entity_ids: list[str] = Field(default_factory=list)
    allowed_relation_types: list[str] = Field(default_factory=list)
    target_entity_types: list[str] = Field(default_factory=list)
    include_support: bool = True
    include_counter_evidence: bool = True
    max_depth: int = Field(default=3, ge=1, le=6)
    max_results: int = Field(default=100, ge=1, le=1000)
    operating_condition_filters: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True

    @model_validator(mode="after")
    def enforce_read_only(self) -> "GraphQueryPlan":
        if not self.read_only:
            raise ValueError("GraphQueryPlan must remain read-only")
        return self


class DiagnosticHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(min_length=1)
    hypothesis_type: HypothesisType
    target_fault_id: str | None = None
    description: str = Field(min_length=1)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    supporting_evidence_fact_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_fact_ids: list[str] = Field(default_factory=list)
    missing_evidence_definition_ids: list[str] = Field(default_factory=list)
    not_applicable_evidence_definition_ids: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    claim_type: ClaimType = ClaimType.INFERRED
    member_fault_ids: list[str] = Field(default_factory=list)
    interaction_hypothesis: str = ""
    verification_requirements: list[str] = Field(default_factory=list)
    research_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target(self) -> "DiagnosticHypothesis":
        if self.hypothesis_type == HypothesisType.REGISTERED_FAULT and not self.target_fault_id:
            raise ValueError("registered fault hypothesis requires target_fault_id")
        if self.hypothesis_type == HypothesisType.NO_REGISTERED_FAULT and self.target_fault_id is not None:
            raise ValueError("no-registered-fault hypothesis cannot carry a fault ID")
        if self.hypothesis_type == HypothesisType.RESEARCH_HYPOTHESIS and self.status == HypothesisStatus.SUPPORTED:
            raise ValueError("research hypothesis cannot be marked SUPPORTED as a formal diagnosis")
        if len(self.member_fault_ids) != len(set(self.member_fault_ids)):
            raise ValueError("member_fault_ids must be unique")
        if self.hypothesis_type != HypothesisType.RESEARCH_HYPOTHESIS and self.member_fault_ids:
            raise ValueError("member_fault_ids are reserved for research hypotheses")
        return self


class HypothesisSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-hypothesis-set.v1"
    hypothesis_set_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    hypotheses: list[DiagnosticHypothesis]
    generated_by_agent_id: str = Field(min_length=1)
    generated_at: str = Field(default_factory=_now)
    validation_metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Registered hypotheses remain shadow conclusions until Decision Governor and deterministic Core comparison complete."
    )

    @field_validator("hypotheses")
    @classmethod
    def unique_hypotheses(cls, value: list[DiagnosticHypothesis]) -> list[DiagnosticHypothesis]:
        ids = [item.hypothesis_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis_id values must be unique")
        return value


class VerificationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_task_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    feature_definition_id: str | None = None
    evidence_definition_id: str | None = None
    input_refs: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    success_criteria: str = "tool_execution_succeeded_and_output_validated"


class VerificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-verification-plan.v1"
    verification_plan_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    tasks: list[VerificationTask]
    generated_by_agent_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=_now)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-verification-result.v1"
    verification_task_id: str = Field(min_length=1)
    status: str = Field(pattern=r"^(SUCCEEDED|FAILED|SKIPPED)$")
    output_refs: dict[str, str] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    data_hashes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    executed_at: str = Field(default_factory=_now)


class CritiqueFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    severity: CritiqueSeverity
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    hypothesis_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    recommended_action: str = ""


class CritiqueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-critique-result.v1"
    critique_id: str = Field(min_length=1)
    hypothesis_set_id: str = Field(min_length=1)
    findings: list[CritiqueFinding] = Field(default_factory=list)
    blocking: bool = False
    generated_by_agent_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=_now)
    claim_boundary: str = "LLM or rule-based critique is advisory; it is not the final safety release authority."


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-decision-record.v1"
    decision_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    disposition: DecisionDisposition
    accepted_hypothesis_id: str | None = None
    accepted_fault_id: str | None = None
    rejected_hypothesis_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)
    formal_release_allowed: bool = False
    requires_human_review: bool = False
    shadow_mode: bool = True
    deterministic_core_result: dict[str, Any] = Field(default_factory=dict)
    comparison_status: str = "NOT_COMPARED"
    protocol_versions: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    claim_boundary: str = (
        "This decision is a shadow-mode engineering result and is not a flight-qualified diagnosis or automatic control command."
    )

    @model_validator(mode="after")
    def enforce_shadow_release(self) -> "DecisionRecord":
        if self.shadow_mode and self.formal_release_allowed:
            raise ValueError("shadow-mode decision cannot allow formal release")
        return self


class AgentTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    agent_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    tool_id: str | None = None
    tool_version: str | None = None
    status: str = Field(default="SUCCEEDED")
    created_at: str = Field(default_factory=_now)


class AgentTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-agent-trace.v1"
    trace_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    model_id: str = "deterministic-reference-agent"
    model_version: str = "0.5.2"
    prompt_protocol_version: str = "astrograph-agentic-shadow-v2"
    events: list[AgentTraceEvent] = Field(default_factory=list)
    termination_reason: str = ""
    created_at: str = Field(default_factory=_now)

    def append(self, event: AgentTraceEvent) -> None:
        expected = len(self.events) + 1
        if event.sequence != expected:
            raise ValueError(f"trace event sequence must be {expected}")
        self.events.append(event)
