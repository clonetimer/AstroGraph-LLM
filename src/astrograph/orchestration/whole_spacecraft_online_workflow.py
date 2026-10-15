from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.agents.agentic import (
    DiagnosticReasonerAgent,
    EvidenceNavigatorAgent,
    SafetyCriticAgent,
    StructuredReasoningBackend,
    VerificationScientistAgent,
)
from astrograph.agents.diagnosis_planner import (
    DiagnosisPlannerAgent,
    DiagnosisPlanningContext,
)
from astrograph.data_driven import (
    DataDrivenModelRegistry,
    DataDrivenPrediction,
    ModelInferenceRequest,
    OpenSetPolicy,
    assess_open_set,
    default_v0522_activity_axis_manifest,
    manifest_sha256,
)
from astrograph.dataset_profiles import DatasetProfileReceipt, DatasetProfileRegistry
from astrograph.diagnosis import (
    CapabilityAxesEvidence,
    CoreComparisonReference,
    DiagnosisEvidenceBundle,
    ReleaseQualificationContext,
    ReleaseQualificationGate,
    KGEvidenceReference,
    ModelEvidenceReference,
)
from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.fusion import (
    BiRRFConfig,
    CandidateFusionPackage,
    RuntimeFusionService,
)
from astrograph.fusion.document_claims import DocumentClaimOnlineShadowConfig
from astrograph.evidence.executable import stable_id
from astrograph.investigation import (
    AgentTrace,
    AgentTraceEvent,
    CritiqueFinding,
    CritiqueResult,
    CritiqueSeverity,
    ClaimType,
    DiagnosticDecisionGovernor,
    DiagnosticHypothesis,
    HypothesisSet,
    HypothesisStatus,
    HypothesisType,
)
from astrograph.integrations.dor_semantic import (
    DORSemanticAudit,
    DORSemanticAuditRequest,
    DORSemanticAuditService,
    semantic_release_blockers,
)
from astrograph.integrations.dor_control import (
    DORControlPlaneAudit,
    DORControlPlaneAuditRequest,
    DORControlPlaneAuditService,
    control_plane_release_blockers,
)
from astrograph.integrations.dor_multitask import (
    DORMultitaskAudit,
    DORMultitaskAuditRequest,
    DORMultitaskAuditService,
    multitask_release_blockers,
)
from astrograph.integrations.dor_vertical import (
    DORVerticalFreezeAudit,
    DORVerticalFreezeAuditRequest,
    DORVerticalFreezeAuditService,
    vertical_freeze_release_blockers,
)
from astrograph.integrations.fec_evidence import (
    FECEvidenceAudit,
    FECEvidenceAuditRequest,
    FECEvidenceAuditService,
    fec_evidence_release_blockers,
)
from astrograph.integrations.isgn_routing import (
    ISGNRoutingAudit,
    ISGNRoutingAuditRequest,
    ISGNRoutingAuditService,
    isgn_routing_release_blockers,
)
from astrograph.integrations.isgn_observation import (
    ISGNObservationAudit,
    ISGNObservationAuditRequest,
    ISGNObservationAuditService,
    isgn_observation_release_blockers,
)
from astrograph.knowledge import EpisodeEvidenceGraph
from astrograph.knowledge.runtime import (
    KGEvidenceRuntimeResult,
    KGEvidenceRuntimeService,
    Neo4jKGEvidenceBindingConfig,
)
from astrograph.knowledge import (
    ComponentSpatiotemporalFaultGraph,
    CsvEvidenceGraphReferenceExecutor,
    GraphEvidenceAssociation,
    ReadOnlyCypherCompiler,
    ReadOnlyGraphQueryExecutor,
    SpatiotemporalFaultTemplateCatalog,
    SpatiotemporalTemplateMatch,
    associations_from_query_result,
    build_backward_fault_query_plan,
    build_forward_evidence_query_plan,
    build_component_spatiotemporal_graph,
    match_spatiotemporal_templates,
)
from astrograph.anomaly import AnomalyEpisode
from astrograph.orchestration.models import (
    ExecutionResult,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.simulation import OnlineDataGateReport, SimulationRunCompleted
from astrograph.tools import ToolRegistry
from astrograph.tools.online import build_online_tool_registry
from astrograph.policy_profiles import (
    FusionPolicyProfile,
    GovernedGraphQueryCompiler,
    RuntimePolicyReceipt,
    RuntimePolicyRegistry,
)
from astrograph.retrieval import PaperSeekFallbackPolicy, PreReasoningRetrievalRequest
from .realtime_reasoning import RealtimeReasoningActivationPolicy


class WholeSpacecraftOnlineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulation_event: SimulationRunCompleted
    dataset_profile_id: str | None = None
    protocol_root: str | None = None
    evidence_package_path: str | None = None
    soft_evidence_calibration_policy_path: str | None = None
    soft_evidence_calibration_policy_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    soft_graph_provenance_policy_path: str | None = None
    soft_graph_provenance_policy_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    runtime_evidence_builder_id: Literal[
        "configured.package.v1", "legacy.whole_spacecraft.v4"
    ] = "configured.package.v1"
    artifact_output_dir: str = Field(min_length=1)
    shadow_mode: bool = True
    reasoner_mode: Literal["rule", "llm", "hybrid"] = "rule"
    reasoning_activation_policy: RealtimeReasoningActivationPolicy = Field(
        default_factory=RealtimeReasoningActivationPolicy
    )
    operator_reasoning_request: bool = False
    data_driven_prediction: DataDrivenPrediction | None = None
    data_driven_inference_request: ModelInferenceRequest | None = None
    birrf_config: BiRRFConfig = Field(default_factory=BiRRFConfig)
    graph_query_policy_profile_id: str | None = None
    fusion_policy_profile_id: str | None = None
    open_set_policy: OpenSetPolicy | None = None
    kg_evidence_binding: "Neo4jKGEvidenceBindingConfig | None" = None
    dor_semantic_audit: DORSemanticAuditRequest | None = None
    dor_control_plane_audit: DORControlPlaneAuditRequest | None = None
    dor_multitask_audit: DORMultitaskAuditRequest | None = None
    dor_vertical_freeze_audit: DORVerticalFreezeAuditRequest | None = None
    fec_evidence_audit: FECEvidenceAuditRequest | None = None
    isgn_routing_audit: ISGNRoutingAuditRequest | None = None
    isgn_observation_audit: ISGNObservationAuditRequest | None = None
    spatiotemporal_episodes: list[AnomalyEpisode] = Field(default_factory=list)
    spatiotemporal_template_catalog_path: str | None = None
    spatiotemporal_rerank_weight: float = Field(default=0.0, ge=0.0, le=0.25)
    pre_reasoning_retrieval_request: PreReasoningRetrievalRequest | None = None
    pre_reasoning_fault_catalog_path: str | None = None
    pre_reasoning_experience_ledger_path: str | None = None
    pre_reasoning_memory_store_path: str | None = None
    paperseek_fallback_policy: PaperSeekFallbackPolicy = Field(default_factory=PaperSeekFallbackPolicy)
    llm_diagnostic_loop_enabled: bool = True
    llm_diagnostic_loop_max_steps: int = Field(default=4, ge=3, le=8)
    llm_diagnostic_capability_ids: list[str] | None = None
    llm_diagnostic_observation_feedback_enabled: bool = True
    llm_active_telemetry_observation_enabled: bool = False
    llm_shadow_candidate_revision_enabled: bool = False
    llm_retrieval_context_enabled: bool = True
    llm_experience_context_enabled: bool = True
    document_claim_shadow: DocumentClaimOnlineShadowConfig = Field(
        default_factory=DocumentClaimOnlineShadowConfig
    )

    @model_validator(mode="after")
    def one_data_driven_source(self) -> "WholeSpacecraftOnlineInput":
        if self.llm_diagnostic_capability_ids is not None and len(
            self.llm_diagnostic_capability_ids
        ) != len(set(self.llm_diagnostic_capability_ids)):
            raise ValueError("llm_diagnostic_capability_ids must be unique")
        if self.dataset_profile_id is not None:
            if (
                self.protocol_root is not None
                or self.evidence_package_path is not None
                or self.soft_evidence_calibration_policy_path is not None
                or self.soft_evidence_calibration_policy_sha256 is not None
                or self.soft_graph_provenance_policy_path is not None
                or self.soft_graph_provenance_policy_sha256 is not None
            ):
                raise ValueError(
                    "dataset_profile_id cannot be combined with direct protocol/evidence assets"
                )
            if self.fusion_policy_profile_id is not None:
                raise ValueError(
                    "dataset_profile_id already binds a reviewed fusion policy profile and cannot "
                    "be combined with fusion_policy_profile_id"
                )
        elif self.protocol_root is None:
            raise ValueError("direct runtime evidence configuration requires protocol_root")
        if self.dataset_profile_id is None and self.runtime_evidence_builder_id == "configured.package.v1":
            if self.evidence_package_path is None:
                raise ValueError(
                    "configured runtime evidence requires evidence_package_path; "
                    "legacy evidence must be selected explicitly"
                )
        elif self.evidence_package_path is not None:
            raise ValueError("legacy runtime evidence must not receive evidence_package_path")
        if bool(self.soft_evidence_calibration_policy_path) != bool(
            self.soft_evidence_calibration_policy_sha256
        ):
            raise ValueError(
                "soft evidence calibration policy path and SHA-256 must be provided together"
            )
        if bool(self.soft_graph_provenance_policy_path) != bool(
            self.soft_graph_provenance_policy_sha256
        ):
            raise ValueError(
                "soft graph provenance policy path and SHA-256 must be provided together"
            )
        if self.data_driven_prediction is not None and self.data_driven_inference_request is not None:
            raise ValueError("provide either data_driven_prediction or data_driven_inference_request, not both")
        if self.fec_evidence_audit is not None and (
            self.data_driven_prediction is None and self.data_driven_inference_request is None
        ):
            raise ValueError("fec_evidence_audit requires a data-driven prediction source")
        configured = bool(self.spatiotemporal_episodes) or bool(self.spatiotemporal_template_catalog_path)
        if configured and not (self.spatiotemporal_episodes and self.spatiotemporal_template_catalog_path):
            raise ValueError(
                "spatiotemporal_episodes and spatiotemporal_template_catalog_path must be provided together"
            )
        if self.spatiotemporal_rerank_weight > 0.0 and not configured:
            raise ValueError("spatiotemporal_rerank_weight requires episodes and a template catalog")
        if self.fusion_policy_profile_id is not None and "birrf_config" in self.model_fields_set:
            raise ValueError("fusion_policy_profile_id cannot be combined with inline birrf_config")
        if (
            self.pre_reasoning_experience_ledger_path is not None
            or self.pre_reasoning_memory_store_path is not None
        ) and self.pre_reasoning_fault_catalog_path is None and self.dataset_profile_id is None:
            raise ValueError("experience reuse requires evidence-aware retrieval and a fault catalog")
        if self.pre_reasoning_retrieval_request is not None and (
            self.pre_reasoning_experience_ledger_path is not None
            or self.pre_reasoning_memory_store_path is not None
        ):
            raise ValueError("caller-supplied retrieval cannot be combined with automatic experience reuse")
        return self


class LiveKGEvidenceQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kg_evidence: KGEvidenceReference
    forward_associations: list[GraphEvidenceAssociation]
    backward_associations: list[GraphEvidenceAssociation]
    selected_backend: str = "neo4j"
    degraded_to_reference: bool = False
    fallback_reason: str | None = None


class OnlineDagCheckpoint(BaseModel):
    """Observable boundary emitted by the whole-spacecraft online DAG executor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(min_length=1)
    status: Literal["SUCCEEDED", "SKIPPED"] = "SUCCEEDED"
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class WholeSpacecraftOnlineDiagnosisWorkflow:
    workflow_id = "simulation.whole_spacecraft.online_diagnosis.v0.5.37"
    workflow_version = "0.5.40"

    def __init__(
        self,
        *,
        reasoning_backend: StructuredReasoningBackend | None = None,
        tool_registry: ToolRegistry | None = None,
        data_model_registry: DataDrivenModelRegistry | None = None,
        dataset_profile_registry: DatasetProfileRegistry | None = None,
        runtime_policy_registry: RuntimePolicyRegistry | None = None,
        post_diagnosis_knowledge_service: Any | None = None,
        post_diagnosis_document_paths: tuple[Path, ...] = (),
        post_diagnosis_material_review: dict[str, Any] | None = None,
    ) -> None:
        self.backend = reasoning_backend
        self.post_diagnosis_knowledge_service = post_diagnosis_knowledge_service
        self.post_diagnosis_document_paths = tuple(post_diagnosis_document_paths)
        self.post_diagnosis_material_review = post_diagnosis_material_review
        self.tools = tool_registry or build_online_tool_registry()
        self.data_models = data_model_registry
        self.dataset_profiles = dataset_profile_registry
        self.runtime_policies = runtime_policy_registry
        self.planner = DiagnosisPlannerAgent(tool_registry=self.tools)
        self.navigator = EvidenceNavigatorAgent(tool_registry=self.tools)
        self.reasoner = DiagnosticReasonerAgent(tool_registry=self.tools, backend=reasoning_backend)
        self.verifier = VerificationScientistAgent(tool_registry=self.tools)
        self.critic = SafetyCriticAgent(tool_registry=self.tools)
        self.governor = DiagnosticDecisionGovernor()
        self.diagnosis_policy_gate = ReleaseQualificationGate()
        self.fusion_service = RuntimeFusionService()

    @staticmethod
    def _build_policy_evidence_bundle(
        *,
        simulation_event: SimulationRunCompleted,
        bundle: EpisodeEvidenceBundle,
        decision: Any,
        candidate_fusion: CandidateFusionPackage | None,
        kg_evidence: KGEvidenceReference | None,
        fec_evidence_audit: FECEvidenceAudit | None = None,
    ) -> DiagnosisEvidenceBundle:
        model_evidence = None
        if candidate_fusion is not None:
            ranking = candidate_fusion.data_ranking
            model_evidence = ModelEvidenceReference(
                model_id=ranking.producer_id,
                model_version=ranking.producer_version,
                prediction_id=str(ranking.metadata.get("prediction_id") or ranking.ranking_id),
                probability_ranking_id=ranking.ranking_id,
                top_fault_id=ranking.candidates[0].fault_id if ranking.candidates else None,
                calibrated=(
                    fec_evidence_audit.calibration_supported
                    if fec_evidence_audit is not None
                    else bool(ranking.metadata.get("calibrated_probabilities", False))
                ),
            )
        capability_axes = WholeSpacecraftOnlineDiagnosisWorkflow._capability_axes_evidence(
            simulation_event=simulation_event,
        )
        return DiagnosisEvidenceBundle(
            bundle_id=stable_id(
                "DEB",
                {
                    "run_id": simulation_event.run_id,
                    "episode_id": bundle.episode.episode_id,
                    "decision_id": decision.decision_id,
                    "candidate_fusion_id": (
                        candidate_fusion.birrf_result.fusion_id if candidate_fusion is not None else None
                    ),
                },
            ),
            sample_id=simulation_event.run_id,
            episode_id=bundle.episode.episode_id,
            split=str(simulation_event.metadata.get("split", "UNKNOWN")),
            model_evidence=model_evidence,
            kg_evidence=kg_evidence,
            safety_replay=None,
            core_comparison=CoreComparisonReference(
                comparison_id=decision.decision_id,
                status=decision.comparison_status,
                reference_core_version=str(decision.deterministic_core_result.get("core_version", "UNKNOWN")),
                core_diagnosis_status=str(decision.deterministic_core_result.get("diagnosis_status", "UNKNOWN")),
                core_predicted_case_id=decision.deterministic_core_result.get("predicted_case_id"),
                accepted_fault_id=decision.accepted_fault_id,
                independence_level=str(
                    decision.deterministic_core_result.get(
                        "independence_level",
                        "LLM_INDEPENDENT_SHARED_REFERENCE_IMPLEMENTATION",
                    )
                ),
            ),
            capability_axes=capability_axes,
            release_context=ReleaseQualificationContext(
                runtime_mode="SHADOW" if decision.shadow_mode else "QUALIFICATION",
                formal_release_authorization=(
                    "GRANTED" if decision.formal_release_allowed else "NOT_GRANTED"
                ),
            ),
            upstream_formal_release_allowed=bool(decision.formal_release_allowed),
            metadata={
                "workflow_id": WholeSpacecraftOnlineDiagnosisWorkflow.workflow_id,
                "workflow_version": WholeSpacecraftOnlineDiagnosisWorkflow.workflow_version,
                "shadow_mode": decision.shadow_mode,
                "decision_disposition": decision.disposition.value,
                "accepted_fault_id": decision.accepted_fault_id,
                "kg_evidence_status": "BOUND" if kg_evidence is not None else "NOT_BOUND_IN_LIVE_WORKFLOW",
            },
        )

    @staticmethod
    def _load_run_manifest(simulation_event: SimulationRunCompleted) -> dict[str, Any]:
        embedded = simulation_event.metadata.get("run_manifest")
        if isinstance(embedded, dict):
            return embedded
        path = Path(simulation_event.run_dir) / "run_manifest.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8-sig"))
        return {}

    @staticmethod
    def _requested_axes_from_run_manifest(run_manifest: dict[str, Any]) -> list[str]:
        activity_axes = run_manifest.get("activity_axes")
        requested: set[str] = set()
        if isinstance(activity_axes, dict):
            for axis, record in activity_axes.items():
                if isinstance(record, dict) and record.get("controlled") is True:
                    requested.add(str(axis))
        if not requested and "seed" in run_manifest:
            requested.add("seed")
        return sorted(requested)

    @staticmethod
    def _capability_axes_evidence(
        *,
        simulation_event: SimulationRunCompleted,
    ) -> CapabilityAxesEvidence:
        capability_manifest = default_v0522_activity_axis_manifest()
        run_manifest = WholeSpacecraftOnlineDiagnosisWorkflow._load_run_manifest(simulation_event)
        requested_axes = WholeSpacecraftOnlineDiagnosisWorkflow._requested_axes_from_run_manifest(run_manifest)
        return CapabilityAxesEvidence(
            supported_axes=[axis.value for axis in capability_manifest.supported_axes],
            unsupported_axes=[axis.value for axis in capability_manifest.unsupported_axes],
            requested_axes=requested_axes,
            capability_manifest_id=capability_manifest.manifest_id,
            capability_manifest_sha256=manifest_sha256(capability_manifest),
        )

    @staticmethod
    def _adapt_kg_runtime_result(result: KGEvidenceRuntimeResult) -> LiveKGEvidenceQueryResult:
        return LiveKGEvidenceQueryResult(
            kg_evidence=KGEvidenceReference.model_validate(result.reference.model_dump(mode="json")),
            forward_associations=result.forward_associations,
            backward_associations=result.backward_associations,
            selected_backend=result.selected_backend,
            degraded_to_reference=result.degraded_to_reference,
            fallback_reason=result.fallback_reason,
        )

    @staticmethod
    def _query_live_kg_evidence(
        *,
        config: Neo4jKGEvidenceBindingConfig,
        bundle: EpisodeEvidenceBundle,
        compiler: ReadOnlyCypherCompiler | GovernedGraphQueryCompiler | None = None,
    ) -> LiveKGEvidenceQueryResult:
        result = KGEvidenceRuntimeService().query_live(
            config=config, bundle=bundle, compiler=compiler
        )
        return WholeSpacecraftOnlineDiagnosisWorkflow._adapt_kg_runtime_result(result)

    @staticmethod
    def _query_csv_reference_kg_evidence(
        *,
        config: Neo4jKGEvidenceBindingConfig,
        bundle: EpisodeEvidenceBundle,
        fallback_reason: str,
        compiler: ReadOnlyCypherCompiler | GovernedGraphQueryCompiler | None = None,
    ) -> LiveKGEvidenceQueryResult:
        result = KGEvidenceRuntimeService().query_csv_reference(
            config=config,
            bundle=bundle,
            fallback_reason=fallback_reason,
            compiler=compiler,
        )
        return WholeSpacecraftOnlineDiagnosisWorkflow._adapt_kg_runtime_result(result)

    @staticmethod
    def _query_bound_kg_evidence(
        *,
        config: Neo4jKGEvidenceBindingConfig,
        bundle: EpisodeEvidenceBundle,
        compiler: ReadOnlyCypherCompiler | GovernedGraphQueryCompiler | None = None,
    ) -> LiveKGEvidenceQueryResult:
        service = KGEvidenceRuntimeService()
        return service.query_bound(
            config=config,
            bundle=bundle,
            compiler=compiler,
            live_query=lambda: WholeSpacecraftOnlineDiagnosisWorkflow._query_live_kg_evidence(
                config=config, bundle=bundle, compiler=compiler
            ),
            fallback_query=lambda reason: WholeSpacecraftOnlineDiagnosisWorkflow._query_csv_reference_kg_evidence(
                config=config, bundle=bundle, fallback_reason=reason, compiler=compiler
            ),
        )

    @staticmethod
    def _add_out_of_scope_hypothesis(
        hypotheses: HypothesisSet,
        bundle: EpisodeEvidenceBundle,
    ) -> HypothesisSet:
        registered_supported = any(
            item.hypothesis_type == HypothesisType.REGISTERED_FAULT
            and item.status == HypothesisStatus.SUPPORTED
            for item in hypotheses.hypotheses
        )
        anomaly_count = int(bundle.analysis_metadata.get("generic_anomaly_field_count", 0))
        if registered_supported or anomaly_count == 0:
            return hypotheses
        updated = []
        for item in hypotheses.hypotheses:
            if item.hypothesis_type == HypothesisType.NO_REGISTERED_FAULT:
                updated.append(
                    item.model_copy(
                        update={
                            "status": HypothesisStatus.CONTRADICTED,
                            "reasoning_summary": (
                                "Paired-run residuals show unexplained anomalies, so "
                                "no-registered-fault cannot be confirmed."
                            ),
                        }
                    )
                )
            else:
                updated.append(item)
        updated.append(
            DiagnosticHypothesis(
                hypothesis_id=stable_id(
                    "HYP",
                    {"episode": bundle.episode.episode_id, "type": "OUT_OF_SCOPE"},
                ),
                hypothesis_type=HypothesisType.RESEARCH_HYPOTHESIS,
                target_fault_id=None,
                description="检测到当前正式故障目录之外的整星异常",
                status=HypothesisStatus.OUT_OF_SCOPE,
                reasoning_summary=(
                    f"有 {anomaly_count} 个协议字段相对配对正常运行发生变化，"
                    "但当前16类正式单故障证据均未满足。"
                ),
            )
        )
        return hypotheses.model_copy(update={"hypotheses": updated})

    @staticmethod
    def _add_composite_research_hypothesis(
        hypotheses: HypothesisSet,
        bundle: EpisodeEvidenceBundle,
    ) -> HypothesisSet:
        """Add an explicit, non-formal research object for declared multi-event runs.

        Member fault candidates come only from executable evidence.  Simulation event
        descriptors are retained as E0 evaluation context and never used to turn the
        research object into a formal diagnosis.
        """
        event_count = int(bundle.evaluation_metadata.get("simulation_event_count_context", 0))
        if event_count <= 1:
            return hypotheses
        supported = sorted(
            [
                item
                for item in hypotheses.hypotheses
                if item.hypothesis_type == HypothesisType.REGISTERED_FAULT
                and item.status == HypothesisStatus.SUPPORTED
                and item.target_fault_id
            ],
            key=lambda item: item.target_fault_id or "",
        )
        member_fault_ids = [item.target_fault_id for item in supported if item.target_fault_id]
        support_ids = sorted(
            {
                evidence_id
                for item in supported
                for evidence_id in item.supporting_evidence_fact_ids
            }
        )
        descriptors = list(
            bundle.evaluation_metadata.get("simulation_event_descriptors_context", [])
        )
        research = DiagnosticHypothesis(
            hypothesis_id=stable_id(
                "RH",
                {
                    "episode": bundle.episode.episode_id,
                    "type": "COMPOSITE_INTERACTION",
                    "member_fault_ids": member_fault_ids,
                    "event_count": event_count,
                },
            ),
            hypothesis_type=HypothesisType.RESEARCH_HYPOTHESIS,
            target_fault_id=None,
            description="多事件整星场景的复合故障交互研究假设",
            status=HypothesisStatus.UNDER_VERIFICATION,
            supporting_evidence_fact_ids=support_ids,
            reasoning_summary=(
                f"可执行证据支持 {len(member_fault_ids)} 个单故障成员；"
                "当前尚无经审核的复合故障FaultMode和交互判据，不能正式确诊。"
            ),
            claim_type=ClaimType.HYPOTHETICAL,
            member_fault_ids=member_fault_ids,
            interaction_hypothesis=(
                "多个已支持单故障可能在时间重叠区间产生耦合或叠加响应；"
                "该关系尚未经过消融仿真和独立金标准验证。"
            ),
            verification_requirements=[
                "核验各成员故障的机制Episode及时间重叠区间",
                "分别运行单成员消融场景并比较联合响应",
                "检查是否存在超出线性叠加的交互残差",
                "由知识治理流程审核复合FaultMode与必要反证",
            ],
            research_metadata={
                "simulation_event_count_context": event_count,
                "simulation_event_descriptors_context": descriptors,
                "formal_release_allowed": False,
                "research_status": "UNVERIFIED",
            },
        )
        updated = [
            item
            for item in hypotheses.hypotheses
            if not (
                item.hypothesis_type == HypothesisType.RESEARCH_HYPOTHESIS
                and item.research_metadata.get("research_status") == "UNVERIFIED"
                and item.member_fault_ids
            )
        ]
        updated.append(research)
        metadata = dict(hypotheses.validation_metadata)
        metadata.update(
            {
                "composite_research_hypothesis_added": True,
                "composite_member_fault_ids": member_fault_ids,
                "composite_event_count_context": event_count,
            }
        )
        return hypotheses.model_copy(
            update={"hypotheses": updated, "validation_metadata": metadata}
        )

    @staticmethod
    def _augment_critique(
        critique: CritiqueResult,
        hypotheses: HypothesisSet,
        bundle: EpisodeEvidenceBundle,
    ) -> CritiqueResult:
        findings = list(critique.findings)
        out_scope = any(item.status == HypothesisStatus.OUT_OF_SCOPE for item in hypotheses.hypotheses)
        if out_scope:
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id(
                        "CF",
                        {"episode": bundle.episode.episode_id, "code": "UNREGISTERED_RUNTIME_ANOMALY"},
                    ),
                    severity=CritiqueSeverity.WARNING,
                    code="UNREGISTERED_RUNTIME_ANOMALY",
                    message=(
                        "Runtime telemetry differs from the paired nominal run, but no current "
                        "registered fault explains it."
                    ),
                    recommended_action=(
                        "Route to ResearchHypothesis/knowledge curation; do not publish a formal diagnosis."
                    ),
                )
            )
        composite_context = int(bundle.evaluation_metadata.get("simulation_event_count_context", 0)) > 1
        if composite_context:
            composite_hypothesis = next(
                (
                    item
                    for item in hypotheses.hypotheses
                    if item.hypothesis_type == HypothesisType.RESEARCH_HYPOTHESIS
                    and item.member_fault_ids
                ),
                None,
            )
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id(
                        "CF",
                        {
                            "episode": bundle.episode.episode_id,
                            "code": "COMPOSITE_SIMULATION_CONTEXT_NOT_FORMALLY_SUPPORTED",
                        },
                    ),
                    severity=CritiqueSeverity.BLOCKING,
                    code="COMPOSITE_SIMULATION_CONTEXT_NOT_FORMALLY_SUPPORTED",
                    message=(
                        "The simulation task declared multiple simultaneous events, while formal "
                        "composite-fault diagnosis is not registered."
                    ),
                    hypothesis_id=(
                        composite_hypothesis.hypothesis_id if composite_hypothesis else None
                    ),
                    evidence_refs=(
                        composite_hypothesis.supporting_evidence_fact_ids
                        if composite_hypothesis else []
                    ),
                    recommended_action=(
                        "Keep detected single-fault members as evidence only, execute the composite "
                        "verification requirements, and route the run to human review."
                    ),
                )
            )
        return critique.model_copy(
            update={"findings": findings, "blocking": critique.blocking or composite_context}
        )

    def execute_runtime_stage(
        self,
        *,
        stage_id: str,
        envelope: TaskEnvelope,
        context: WorkflowContext,
        artifact_store: Any,
    ) -> Any:
        """Execute one stateless online DAG node from explicit upstream artifacts."""
        from .whole_spacecraft_stage_executor import WholeSpacecraftOnlineStageExecutor

        return WholeSpacecraftOnlineStageExecutor(self).execute(
            stage_id=stage_id, envelope=envelope, context=context, store=artifact_store
        )

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        """Execute the canonical artifact-backed DAG in stable topological order.

        Compatibility callers use the same per-stage executor as AstroGraphRuntime; there
        is no second workflow implementation and no shared generator session.
        """
        from astrograph.runtime.artifacts import RuntimeArtifactStore
        from .whole_spacecraft_stage_executor import (
            ONLINE_STAGE_IDS,
            OnlineStageExecutionError,
        )

        store = RuntimeArtifactStore(execution_scope=envelope.task_id)
        try:
            final: ExecutionResult | None = None
            for stage_id in ONLINE_STAGE_IDS:
                outcome = self.execute_runtime_stage(
                    stage_id=stage_id, envelope=envelope, context=context, artifact_store=store
                )
                store.put(stage_id, outcome.artifact)
                if outcome.execution_result is not None:
                    final = outcome.execution_result
            if final is None:
                return ExecutionResult.failed(
                    task_id=envelope.task_id, workflow_id=self.workflow_id,
                    code="ASTROGRAPH_ONLINE_DAG_NO_RESULT",
                    message="Whole-spacecraft online DAG terminated without an ExecutionResult.",
                )
            # Optional post-persistence read-only explanation. It must never
            # alter the online DAG's structured result or decision governance.
            if self.post_diagnosis_knowledge_service is not None:
                from .online_post_diagnosis import run_online_post_diagnosis_explanation
                try:
                    run_online_post_diagnosis_explanation(
                        result=final,
                        knowledge_service=self.post_diagnosis_knowledge_service,
                        output_root=Path(str(final.metadata.get("artifact_dir", ""))),
                        document_paths=list(self.post_diagnosis_document_paths),
                        material_review_context=self.post_diagnosis_material_review,
                    )
                except Exception:
                    # A completed diagnosis survives even a sidecar I/O failure.
                    pass
            return final
        except OnlineStageExecutionError as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id, workflow_id=self.workflow_id,
                code=exc.code, message=exc.message, details=exc.details,
            )
        except Exception as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id, workflow_id=self.workflow_id,
                code="ASTROGRAPH_ONLINE_WORKFLOW_FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )

    def _iter_runtime_dag(self, envelope: TaskEnvelope, context: WorkflowContext):
        """Compatibility checkpoint iterator backed by independent Stage execution.

        Canonical Runtime stages never consume this iterator. It remains only for older
        direct callers that inspect checkpoint order.
        """
        from astrograph.runtime.artifacts import RuntimeArtifactStore
        from .whole_spacecraft_stage_executor import (
            ONLINE_STAGE_IDS,
            OnlineStageExecutionError,
        )

        store = RuntimeArtifactStore(execution_scope=envelope.task_id)
        try:
            for stage_id in ONLINE_STAGE_IDS:
                outcome = self.execute_runtime_stage(
                    stage_id=stage_id, envelope=envelope, context=context, artifact_store=store
                )
                store.put(stage_id, outcome.artifact)
                yield OnlineDagCheckpoint(
                    stage_id=stage_id, status=outcome.status,
                    metadata=outcome.metadata, warnings=outcome.warnings,
                )
                if outcome.execution_result is not None:
                    return outcome.execution_result
            return ExecutionResult.failed(
                task_id=envelope.task_id, workflow_id=self.workflow_id,
                code="ASTROGRAPH_ONLINE_DAG_NO_RESULT",
                message="Whole-spacecraft online DAG terminated without an ExecutionResult.",
            )
        except OnlineStageExecutionError as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id, workflow_id=self.workflow_id,
                code=exc.code, message=exc.message, details=exc.details,
            )
        except Exception as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id, workflow_id=self.workflow_id,
                code="ASTROGRAPH_ONLINE_WORKFLOW_FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
