from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id

from .base import ControlledAgent


class DiagnosisStage(str, Enum):
    VALIDATE_SEMANTICS = "VALIDATE_SEMANTICS"
    QUALIFY_DIAGNOSTIC_CONTROL_PLANE = "QUALIFY_DIAGNOSTIC_CONTROL_PLANE"
    BUILD_RUNTIME_EVIDENCE = "BUILD_RUNTIME_EVIDENCE"
    QUERY_KNOWLEDGE_GRAPH = "QUERY_KNOWLEDGE_GRAPH"
    RUN_DATA_DRIVEN_MODEL = "RUN_DATA_DRIVEN_MODEL"
    FUSE_CANDIDATES = "FUSE_CANDIDATES"
    BUILD_VERIFICATION_PLAN = "BUILD_VERIFICATION_PLAN"
    GENERATE_HYPOTHESES = "GENERATE_HYPOTHESES"
    SAFETY_CRITIQUE = "SAFETY_CRITIQUE"
    RUN_DETERMINISTIC_CORE = "RUN_DETERMINISTIC_CORE"
    GOVERN_DECISION = "GOVERN_DECISION"
    APPLY_DIAGNOSIS_POLICY = "APPLY_DIAGNOSIS_POLICY"
    BUILD_CAUSAL_EVIDENCE_GRAPH = "BUILD_CAUSAL_EVIDENCE_GRAPH"
    PERSIST_REPORT_AND_TRACE = "PERSIST_REPORT_AND_TRACE"


class DiagnosisPlanningContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoner_mode: Literal["rule", "llm", "hybrid"] = "rule"
    shadow_mode: bool = True
    has_data_driven_prediction: bool = False
    has_data_driven_inference_request: bool = False
    data_model_registry_available: bool = False
    has_kg_evidence_binding: bool = False
    has_dor_control_plane_audit: bool = False


class DiagnosisPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    stage: DiagnosisStage
    enabled: bool = True
    rationale: str = Field(min_length=1)


class DiagnosisExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-diagnosis-execution-plan.v1"
    plan_id: str
    planner_id: str
    planner_version: str
    steps: list[DiagnosisPlanStep]
    warnings: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    claim_boundary: str = (
        "The plan controls tool sequencing and branch eligibility; it does not override evidence or release gates."
    )

    @property
    def enabled_stages(self) -> list[DiagnosisStage]:
        return [item.stage for item in self.steps if item.enabled]


class DiagnosisPlannerAgent(ControlledAgent):
    """Low-permission deterministic planner for the online diagnosis workflow.

    It makes branch decisions explicit and auditable while leaving all diagnostic facts
    to deterministic tools, registered models, the knowledge layer, and policy gates.
    """

    agent_id = "diagnosis-planner-agent"
    agent_version = "0.6.5"

    def plan(self, context: DiagnosisPlanningContext) -> DiagnosisExecutionPlan:
        data_requested = context.has_data_driven_prediction or context.has_data_driven_inference_request
        data_executable = context.has_data_driven_prediction or (
            context.has_data_driven_inference_request and context.data_model_registry_available
        )
        warnings: list[str] = []
        requires_human_review = False

        if context.has_data_driven_inference_request and not context.data_model_registry_available:
            warnings.append("DATA_MODEL_REGISTRY_REQUIRED")
            requires_human_review = True
        if not data_requested:
            warnings.append("DATA_DRIVEN_BRANCH_NOT_REQUESTED")
            requires_human_review = True
        if not context.has_kg_evidence_binding:
            warnings.append("LIVE_KG_EVIDENCE_NOT_BOUND")
            requires_human_review = True
        if context.reasoner_mode in {"llm", "hybrid"}:
            warnings.append("LLM_REASONING_REMAINS_ADVISORY_AND_POLICY_GATED")

        raw_steps = [
            (DiagnosisStage.VALIDATE_SEMANTICS, True, "Validate the telemetry protocol and identifier semantics."),
            (
                DiagnosisStage.QUALIFY_DIAGNOSTIC_CONTROL_PLANE,
                context.has_dor_control_plane_audit,
                "Audit dataset roles, leakage guards, reference applicability, and engine admissions with DOR."
                if context.has_dor_control_plane_audit
                else "No DOR dataset-level control-plane audit was requested.",
            ),
            (DiagnosisStage.BUILD_RUNTIME_EVIDENCE, True, "Convert telemetry into typed features and evidence facts."),
            (
                DiagnosisStage.QUERY_KNOWLEDGE_GRAPH,
                context.has_kg_evidence_binding,
                "Query the bound read-only knowledge release." if context.has_kg_evidence_binding else "No live KG binding was supplied.",
            ),
            (
                DiagnosisStage.RUN_DATA_DRIVEN_MODEL,
                data_executable,
                "Use the supplied prediction or invoke a registered model provider."
                if data_executable
                else "No executable data-driven source is available.",
            ),
            (
                DiagnosisStage.FUSE_CANDIDATES,
                data_executable,
                "Fuse data-driven ranking with graph/evidence rankings."
                if data_executable
                else "Candidate fusion requires an executable data-driven ranking.",
            ),
            (DiagnosisStage.BUILD_VERIFICATION_PLAN, True, "Compile missing and contradicting evidence checks."),
            (DiagnosisStage.GENERATE_HYPOTHESES, True, "Generate constrained hypotheses from registered facts."),
            (DiagnosisStage.SAFETY_CRITIQUE, True, "Reject invented identifiers and unsupported claims."),
            (DiagnosisStage.RUN_DETERMINISTIC_CORE, True, "Run the independent deterministic diagnostic core."),
            (DiagnosisStage.GOVERN_DECISION, True, "Compare reasoner and deterministic-core results."),
            (DiagnosisStage.APPLY_DIAGNOSIS_POLICY, True, "Apply formal evidence and release gates."),
            (DiagnosisStage.BUILD_CAUSAL_EVIDENCE_GRAPH, True, "Persist the traceable episode evidence/causal graph."),
            (DiagnosisStage.PERSIST_REPORT_AND_TRACE, True, "Persist report artifacts, decisions, and the agent trace."),
        ]
        steps = [
            DiagnosisPlanStep(sequence=index, stage=stage, enabled=enabled, rationale=rationale)
            for index, (stage, enabled, rationale) in enumerate(raw_steps, start=1)
        ]
        plan_id = stable_id(
            "DPLAN",
            {
                "planner_id": self.agent_id,
                "planner_version": self.agent_version,
                "context": context.model_dump(mode="json"),
                "steps": [item.model_dump(mode="json") for item in steps],
            },
        )
        return DiagnosisExecutionPlan(
            plan_id=plan_id,
            planner_id=self.agent_id,
            planner_version=self.agent_version,
            steps=steps,
            warnings=warnings,
            requires_human_review=requires_human_review,
        )
