from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from astrograph.data_driven import DataDrivenPrediction
from astrograph.diagnosis import DiagnosisEvidenceBundle, DiagnosisPolicyDecision, KGEvidenceReference
from astrograph.evidence import EpisodeEvidenceBundle, SoftEvidenceBundle
from astrograph.fusion import CandidateFusionPackage
from astrograph.fusion.document_claims import DocumentClaimOnlineShadowResult
from astrograph.investigation import CritiqueResult, DecisionRecord, HypothesisSet, VerificationPlan
from astrograph.investigation.shadow_candidate_revision import ShadowCandidateRevision
from astrograph.knowledge import (
    ComponentSpatiotemporalFaultGraph,
    EpisodeEvidenceGraph,
    GraphEvidenceAssociation,
    SoftGraphInferenceResult,
    SpatiotemporalTemplateMatch,
)
from astrograph.simulation import OnlineDataGateReport
from .llm_diagnostic_loop import LLMControlledDiagnosticTrajectory
from astrograph.runtime.artifacts import RuntimeStageArtifact, build_runtime_stage_artifact
from astrograph.retrieval import PaperSeekFallbackDecision, PreReasoningRetrievalPackage


class TraceEventDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str
    action: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    tool_id: str | None = None
    tool_version: str | None = None


class _StageData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_events: list[TraceEventDraft] = Field(default_factory=list)


class PrepareStageData(_StageData):
    normalized_request: dict[str, Any]
    dataset_profile_receipt: dict[str, Any] | None = None
    profile_registered_fault_ids: list[str] = Field(default_factory=list)
    runtime_policy_receipts: list[dict[str, Any]] = Field(default_factory=list)
    diagnosis_execution_plan: dict[str, Any]


class SemanticsStageData(_StageData):
    registry_version: str
    validation: dict[str, Any]


class ControlPlaneStageData(_StageData):
    dor_control_plane_audit: dict[str, Any] | None = None


class EvidenceStageData(_StageData):
    bundle: EpisodeEvidenceBundle
    soft_evidence: SoftEvidenceBundle | None = None
    data_gate_report: OnlineDataGateReport
    builder_version: str


class SpatiotemporalStageData(_StageData):
    graph: ComponentSpatiotemporalFaultGraph | None = None
    matches: list[SpatiotemporalTemplateMatch] = Field(default_factory=list)


class SemanticExtensionStageData(_StageData):
    dor_semantic_audit: dict[str, Any] | None = None
    dor_multitask_audit: dict[str, Any] | None = None
    dor_vertical_freeze_audit: dict[str, Any] | None = None


class MechanismExtensionStageData(_StageData):
    isgn_routing_audit: dict[str, Any] | None = None
    isgn_observation_audit: dict[str, Any] | None = None


class KGEvidenceStageData(_StageData):
    kg_evidence: KGEvidenceReference | None = None
    forward_associations: list[GraphEvidenceAssociation] = Field(default_factory=list)
    backward_associations: list[GraphEvidenceAssociation] = Field(default_factory=list)
    selected_backend: str | None = None
    degraded_to_reference: bool = False
    fallback_reason: str | None = None


class DataDrivenStageData(_StageData):
    prediction: DataDrivenPrediction | None = None


class CalibrationStageData(_StageData):
    fec_evidence_audit: dict[str, Any] | None = None


class FusionStageData(_StageData):
    candidate_fusion: CandidateFusionPackage | None = None
    soft_graph_shadow: SoftGraphInferenceResult | None = None
    document_claim_shadow: DocumentClaimOnlineShadowResult = Field(
        default_factory=lambda: DocumentClaimOnlineShadowResult(
            status="DISABLED", reason_codes=["DOCUMENT_CLAIM_SHADOW_DISABLED"]
        )
    )


class VerificationStageData(_StageData):
    verification_plan: VerificationPlan


class ReasoningStageData(_StageData):
    hypothesis_set: HypothesisSet
    pre_reasoning_retrieval: PreReasoningRetrievalPackage | None = None
    paperseek_fallback_decision: PaperSeekFallbackDecision | None = None
    reasoning_backend: dict[str, Any] = Field(default_factory=dict)
    activation_decision: dict[str, Any] = Field(default_factory=dict)
    llm_diagnostic_trajectory: LLMControlledDiagnosticTrajectory | None = None
    shadow_candidate_revision: ShadowCandidateRevision | None = None


class CritiqueStageData(_StageData):
    critique_result: CritiqueResult


class DeterministicCoreStageData(_StageData):
    deterministic_core_result: dict[str, Any]


class GovernorStageData(_StageData):
    decision_record: DecisionRecord


class ReleaseQualificationStageData(_StageData):
    diagnosis_evidence_bundle: DiagnosisEvidenceBundle
    diagnosis_policy_decision: DiagnosisPolicyDecision


class EvidenceGraphStageData(_StageData):
    evidence_graph: EpisodeEvidenceGraph


class PersistStageData(_StageData):
    execution_result: dict[str, Any]
    artifact_manifest: dict[str, Any] | None = None


T = TypeVar("T", bound=BaseModel)


def build_typed_artifact(
    *,
    stage_id: str,
    stage_version: str,
    execution_scope: str,
    data: BaseModel,
    upstream: dict[str, RuntimeStageArtifact],
) -> RuntimeStageArtifact:
    return build_runtime_stage_artifact(
        artifact_type=f"astrograph.{stage_id}.artifact",
        artifact_version="1.0",
        producer_stage_id=stage_id,
        producer_stage_version=stage_version,
        execution_scope=execution_scope,
        content=data.model_dump(mode="json"),
        upstream_artifacts=upstream,
    )


def parse_typed_data(artifact: RuntimeStageArtifact, model: type[T]) -> T:
    return model.model_validate(artifact.content)


def parse_trace_events(artifact: RuntimeStageArtifact) -> list[TraceEventDraft]:
    events = artifact.content.get("trace_events", [])
    return [TraceEventDraft.model_validate(item) for item in events]
