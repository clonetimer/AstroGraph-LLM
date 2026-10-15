from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from astrograph.agents.diagnosis_planner import DiagnosisPlanningContext
from astrograph.data_driven import assess_open_set
from astrograph.dataset_profiles import DatasetProfileReceipt
from astrograph.diagnosis import KGEvidenceReference, build_structured_diagnosis_output
from astrograph.evidence import EpisodeEvidenceBundle, SoftEvidenceBundle
from astrograph.evidence.configured_runtime import ConfiguredRuntimeEvidenceBuilder
from astrograph.fusion.document_claims import build_online_document_claim_shadow
from astrograph.integrations.dor_control import (
    DORControlPlaneAudit,
    DORControlPlaneAuditService,
    control_plane_release_blockers,
)
from astrograph.integrations.dor_semantic import (
    DORSemanticAudit,
    DORSemanticAuditService,
    semantic_release_blockers,
)
from astrograph.integrations.dor_multitask import (
    DORMultitaskAudit,
    DORMultitaskAuditService,
    multitask_release_blockers,
)
from astrograph.integrations.dor_vertical import (
    DORVerticalFreezeAudit,
    DORVerticalFreezeAuditService,
    vertical_freeze_release_blockers,
)
from astrograph.integrations.fec_evidence import (
    FECEvidenceAudit,
    FECEvidenceAuditService,
    fec_evidence_release_blockers,
)
from astrograph.integrations.isgn_observation import (
    ISGNObservationAudit,
    ISGNObservationAuditService,
    isgn_observation_release_blockers,
)
from astrograph.integrations.isgn_routing import (
    ISGNRoutingAudit,
    ISGNRoutingAuditService,
    isgn_routing_release_blockers,
)
from astrograph.investigation import AgentTrace, AgentTraceEvent, HypothesisSet
from astrograph.investigation.active_telemetry_probes import (
    ActiveTelemetryProbeExecutor,
    build_admitted_telemetry_change_probe_catalog,
    is_admitted_telemetry_probe,
)
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeCatalog,
    build_evidence_margin_probe_catalog,
)
from astrograph.investigation.shadow_candidate_revision import (
    ShadowCandidateRevision,
    validate_shadow_candidate_revision,
)
from astrograph.knowledge import (
    ComponentSpatiotemporalFaultGraph,
    EpisodeEvidenceGraph,
    SpatiotemporalFaultTemplateCatalog,
    SpatiotemporalTemplateMatch,
    build_soft_graph_inference,
    load_soft_graph_provenance_policy,
    build_component_spatiotemporal_graph,
    match_spatiotemporal_templates,
)
from astrograph.policy_profiles import FusionPolicyProfile, GovernedGraphQueryCompiler, RuntimePolicyReceipt
from astrograph.provider_lifecycle import BUILTIN_PROVIDER_QUALIFICATIONS
from astrograph.retrieval import (
    EvidenceAwarePreGenerationContextBuilder,
    PaperSeekGovernedBridge,
    PreReasoningRetrievalService,
)
from astrograph.runtime.artifacts import RuntimeArtifactStore, RuntimeStageArtifact
from astrograph.onboarding import sha256_file
from astrograph.simulation import OnlineDataGateReport

from .models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext
from .llm_diagnostic_loop import ControlledDiagnosticDeliberationService
from .online_stage_artifacts import (
    CalibrationStageData,
    ControlPlaneStageData,
    CritiqueStageData,
    DataDrivenStageData,
    DeterministicCoreStageData,
    EvidenceGraphStageData,
    EvidenceStageData,
    FusionStageData,
    GovernorStageData,
    KGEvidenceStageData,
    MechanismExtensionStageData,
    PersistStageData,
    PrepareStageData,
    ReasoningStageData,
    ReleaseQualificationStageData,
    SemanticExtensionStageData,
    SemanticsStageData,
    SpatiotemporalStageData,
    TraceEventDraft,
    VerificationStageData,
    build_typed_artifact,
    parse_trace_events,
    parse_typed_data,
)


def _stable_evidence_payload(bundle: EpisodeEvidenceBundle) -> dict[str, Any]:
    """Compare recomputed evidence without wall-clock construction timestamps."""
    payload = bundle.model_dump(mode="json")
    for row in payload["feature_results"]:
        row.pop("created_at", None)
    for row in payload["evidence_facts"]:
        row.pop("created_at", None)
    return payload


def _revalidated_telemetry_probe_catalog(
    *, request: Any, bundle: EpisodeEvidenceBundle, gate: OnlineDataGateReport,
    base_catalog: DiagnosticProbeCatalog, priority_fault_ids: tuple[str, ...] = (),
) -> DiagnosticProbeCatalog:
    """Re-admit copied telemetry before exposing any time-local samples to the LLM."""
    if (
        request.runtime_evidence_builder_id != "configured.package.v1"
        or request.evidence_package_path is None
        or not gate.passed
    ):
        return base_catalog.model_copy(update={
            "telemetry_admission_status": "UNAVAILABLE_CONFIGURED_RUNTIME"
        })
    try:
        builder = ConfiguredRuntimeEvidenceBuilder(request.evidence_package_path)
        source_path = Path(request.simulation_event.run_dir) / "telemetry.csv"
        nominal_path = Path(request.simulation_event.nominal_run_dir) / "telemetry.csv"
        source_sha = sha256_file(source_path)
        nominal_sha = sha256_file(nominal_path)
        source = pd.read_csv(source_path, low_memory=False)
        nominal = pd.read_csv(nominal_path, low_memory=False)
        rebuilt, rebuilt_gate = builder.build_from_frames(
            request.simulation_event, source=source, nominal=nominal
        )
        if (
            not rebuilt_gate.passed
            or _stable_evidence_payload(rebuilt) != _stable_evidence_payload(bundle)
            or sha256_file(source_path) != source_sha
            or sha256_file(nominal_path) != nominal_sha
        ):
            raise ValueError("telemetry is inconsistent with formal evidence or changed during read")
        active_catalog = build_admitted_telemetry_change_probe_catalog(
            bundle=bundle,
            mapped_frame=builder._map_frame(source),
            approved_channels=builder.package.channels,
            source_sha256=source_sha,
            mapping_sha256=builder.package_sha256,
            priority_fault_ids=priority_fault_ids,
            paired_nominal_mapped_frame=builder._map_frame(nominal),
            nominal_sha256=nominal_sha,
        )
        return DiagnosticProbeCatalog(
            episode_id=bundle.episode.episode_id,
            probes=(*base_catalog.probes, *active_catalog.probes),
            telemetry_admission_status=active_catalog.telemetry_admission_status,
            claim_boundary=active_catalog.claim_boundary,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return base_catalog.model_copy(update={
            "telemetry_admission_status": "REJECTED_TELEMETRY_REVALIDATION"
        })


ONLINE_STAGE_IDS: tuple[str, ...] = (
    "online.prepare.v1",
    "online.semantics.v1",
    "online.extension.control_plane.v1",
    "online.evidence.v1",
    "online.spatiotemporal.v1",
    "online.extension.semantic.v1",
    "online.extension.mechanism.v1",
    "online.kg_evidence.v1",
    "online.data_driven.v1",
    "online.extension.calibration.v1",
    "online.fusion.v1",
    "online.verification.v1",
    "online.deterministic_core.v1",
    "online.reasoning.v1",
    "online.critique.v1",
    "online.governor.v1",
    "online.policy_gate.v1",
    "online.evidence_graph.v1",
    "online.persist.v1",
)

# Duplicated intentionally from the stable Runtime factory: orchestration must not be
# imported statically by astrograph.runtime. A post6 architecture test asserts equality.
ONLINE_STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "online.prepare.v1": (),
    "online.semantics.v1": ("online.prepare.v1",),
    "online.extension.control_plane.v1": ("online.prepare.v1", "online.semantics.v1"),
    "online.evidence.v1": (
        "online.prepare.v1",
        "online.semantics.v1",
        "online.extension.control_plane.v1",
    ),
    "online.spatiotemporal.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.extension.semantic.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.extension.mechanism.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.kg_evidence.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.data_driven.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.extension.calibration.v1": (
        "online.prepare.v1",
        "online.data_driven.v1",
    ),
    "online.fusion.v1": (
        "online.prepare.v1",
        "online.evidence.v1",
        "online.spatiotemporal.v1",
        "online.kg_evidence.v1",
        "online.data_driven.v1",
        "online.extension.calibration.v1",
    ),
    "online.verification.v1": ("online.evidence.v1",),
    "online.reasoning.v1": (
        "online.prepare.v1",
        "online.evidence.v1",
        "online.fusion.v1",
        "online.verification.v1",
        "online.deterministic_core.v1",
    ),
    "online.critique.v1": ("online.evidence.v1", "online.reasoning.v1"),
    "online.deterministic_core.v1": ("online.evidence.v1",),
    "online.governor.v1": (
        "online.evidence.v1",
        "online.reasoning.v1",
        "online.critique.v1",
        "online.deterministic_core.v1",
    ),
    "online.policy_gate.v1": (
        "online.prepare.v1",
        "online.evidence.v1",
        "online.governor.v1",
        "online.extension.control_plane.v1",
        "online.extension.semantic.v1",
        "online.extension.mechanism.v1",
        "online.extension.calibration.v1",
        "online.kg_evidence.v1",
        "online.fusion.v1",
    ),
    "online.evidence_graph.v1": (
        "online.evidence.v1",
        "online.reasoning.v1",
        "online.governor.v1",
        "online.policy_gate.v1",
    ),
    "online.persist.v1": tuple(ONLINE_STAGE_IDS[:-1]),
}


class OnlineStageExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class OnlineStageOutcome:
    status: Literal["SUCCEEDED", "SKIPPED"]
    artifact: RuntimeStageArtifact
    metadata: dict[str, Any]
    warnings: list[str]
    execution_result: ExecutionResult | None = None


class WholeSpacecraftOnlineStageExecutor:
    """Stateless executor for one whole-spacecraft online DAG node.

    Every invocation reconstructs its domain inputs from explicit upstream artifacts.
    No generator frame, mutable workflow-local variables, or request session is shared
    between stages. This is the post6 independent-computation boundary.
    """

    stage_version = "3.0"

    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def _upstream(self, stage_id: str, store: RuntimeArtifactStore) -> dict[str, RuntimeStageArtifact]:
        return {dep: store.get(dep) for dep in ONLINE_STAGE_DEPENDENCIES[stage_id]}

    def _artifact(self, stage_id: str, store: RuntimeArtifactStore, data: Any) -> RuntimeStageArtifact:
        return build_typed_artifact(
            stage_id=stage_id,
            stage_version=self.stage_version,
            execution_scope=store.execution_scope,
            data=data,
            upstream=self._upstream(stage_id, store),
        )

    def _prepare_data(self, store: RuntimeArtifactStore) -> PrepareStageData:
        return parse_typed_data(store.get("online.prepare.v1"), PrepareStageData)

    def _request(self, store: RuntimeArtifactStore) -> Any:
        from .whole_spacecraft_online_workflow import WholeSpacecraftOnlineInput
        return WholeSpacecraftOnlineInput.model_validate(self._prepare_data(store).normalized_request)

    @staticmethod
    def _trace(
        action: str,
        output: dict[str, Any],
        *,
        agent_id: str = "astrograph-v0510-online",
        tool_id: str | None = None,
        tool_version: str | None = None,
        input_summary: dict[str, Any] | None = None,
    ) -> TraceEventDraft:
        return TraceEventDraft(
            agent_id=agent_id,
            action=action,
            input_summary=input_summary or {},
            output_summary=output,
            tool_id=tool_id,
            tool_version=tool_version,
        )

    def execute(
        self,
        *,
        stage_id: str,
        envelope: TaskEnvelope,
        context: WorkflowContext,
        store: RuntimeArtifactStore,
    ) -> OnlineStageOutcome:
        if stage_id not in ONLINE_STAGE_IDS:
            raise OnlineStageExecutionError("ASTROGRAPH_ONLINE_DAG_STAGE_UNKNOWN", f"Unknown online stage: {stage_id}")
        # Accessing the declared inputs here makes missing-artifact failures deterministic
        # before any external provider or side effect runs.
        self._upstream(stage_id, store)
        method = getattr(self, "_" + stage_id.removeprefix("online.").removesuffix(".v1").replace(".", "_") )
        return method(envelope, context, store)

    def _prepare(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        from .whole_spacecraft_online_workflow import WholeSpacecraftOnlineInput

        required = {
            PermissionScope.READ_DATA,
            PermissionScope.READ_KNOWLEDGE,
            PermissionScope.RUN_DIAGNOSIS,
            PermissionScope.WRITE_AUDIT,
            PermissionScope.WRITE_ARTIFACTS,
        }
        missing = sorted(required - set(context.permissions), key=lambda item: item.value)
        if missing:
            raise OnlineStageExecutionError(
                "ASTROGRAPH_ONLINE_PERMISSION_DENIED",
                "Online simulation diagnosis permissions are incomplete.",
                details={"missing_permissions": [item.value for item in missing]},
            )
        try:
            request = WholeSpacecraftOnlineInput.model_validate(envelope.input_payload)
        except ValidationError as exc:
            raise OnlineStageExecutionError(
                "ASTROGRAPH_ONLINE_INPUT_INVALID", f"{type(exc).__name__}: {exc}"
            ) from exc

        requested_dataset_profile_id = request.dataset_profile_id
        dataset_profile_receipt: DatasetProfileReceipt | None = None
        profile_registered_fault_ids: list[str] = []
        if request.dataset_profile_id is not None:
            if self.workflow.dataset_profiles is None:
                raise OnlineStageExecutionError(
                    "ASTROGRAPH_DATASET_PROFILE_REGISTRY_REQUIRED",
                    "dataset_profile_id requires an injected DatasetProfileRegistry.",
                )
            try:
                resolved_profile = self.workflow.dataset_profiles.resolve(request.dataset_profile_id)
                if (
                    request.pre_reasoning_experience_ledger_path is not None
                    or request.pre_reasoning_memory_store_path is not None
                ) and "fault_catalog" not in resolved_profile.asset_paths:
                    raise ValueError(
                        "Dataset Profile experience reuse requires a verified fault_catalog asset"
                    )
                candidate_path = resolved_profile.asset_paths.get("candidate_universe")
                if candidate_path is not None:
                    candidate_universe = json.loads(candidate_path.read_text(encoding="utf-8"))
                    profile_registered_fault_ids = sorted(
                        {str(value) for value in candidate_universe.get("runtime_fault_ids", [])}
                    )
                dataset_reference_scope = resolved_profile.profile.reference_scope.to_dor_manifest()
                dor_control_request = request.dor_control_plane_audit
                fec_request = request.fec_evidence_audit
                if dor_control_request is not None:
                    if (
                        dor_control_request.expected_reference_scope is not None
                        and dor_control_request.expected_reference_scope != dataset_reference_scope
                    ):
                        raise ValueError("inline DOR expected_reference_scope conflicts with Dataset Profile")
                    dor_control_request = dor_control_request.model_copy(
                        update={"expected_reference_scope": dataset_reference_scope}
                    )
                if fec_request is not None:
                    profile_scope = resolved_profile.profile.reference_scope
                    if (
                        fec_request.expected_reference_scope is not None
                        and fec_request.expected_reference_scope != profile_scope
                    ):
                        raise ValueError("inline FEC expected_reference_scope conflicts with Dataset Profile")
                    fec_request = fec_request.model_copy(update={"expected_reference_scope": profile_scope})
                request = request.model_copy(
                    update={
                        "protocol_root": str(resolved_profile.protocol_root),
                        "evidence_package_path": str(resolved_profile.asset_paths["runtime_evidence"]),
                        "soft_evidence_calibration_policy_path": (
                            str(resolved_profile.asset_paths["soft_evidence_calibration_policy"])
                            if "soft_evidence_calibration_policy" in resolved_profile.asset_paths
                            else None
                        ),
                        "soft_evidence_calibration_policy_sha256": (
                            resolved_profile.receipt.verified_asset_sha256.get(
                                "soft_evidence_calibration_policy"
                            )
                        ),
                        "soft_graph_provenance_policy_path": (
                            str(resolved_profile.asset_paths["soft_graph_provenance_policy"])
                            if "soft_graph_provenance_policy" in resolved_profile.asset_paths
                            else None
                        ),
                        "soft_graph_provenance_policy_sha256": (
                            resolved_profile.receipt.verified_asset_sha256.get(
                                "soft_graph_provenance_policy"
                            )
                        ),
                        "runtime_evidence_builder_id": resolved_profile.profile.runtime_evidence_builder_id,
                        "fusion_policy_profile_id": resolved_profile.profile.fusion_policy_profile_id,
                        "dor_control_plane_audit": dor_control_request,
                        "fec_evidence_audit": fec_request,
                        "pre_reasoning_fault_catalog_path": (
                            str(resolved_profile.asset_paths["fault_catalog"])
                            if "fault_catalog" in resolved_profile.asset_paths
                            else None
                        ),
                    }
                )
                dataset_profile_receipt = resolved_profile.receipt
            except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, KeyError, ValueError) as exc:
                raise OnlineStageExecutionError(
                    "ASTROGRAPH_DATASET_PROFILE_INVALID",
                    f"{type(exc).__name__}: {exc}",
                    details={"dataset_profile_id": request.dataset_profile_id},
                ) from exc

        policy_receipts: list[RuntimePolicyReceipt] = []
        if request.graph_query_policy_profile_id or request.fusion_policy_profile_id:
            if self.workflow.runtime_policies is None:
                raise OnlineStageExecutionError(
                    "ASTROGRAPH_RUNTIME_POLICY_REGISTRY_REQUIRED",
                    "runtime policy profile IDs require an injected RuntimePolicyRegistry.",
                )
            try:
                if request.graph_query_policy_profile_id:
                    resolved_query = self.workflow.runtime_policies.resolve(
                        profile_type="GRAPH_QUERY", profile_id=request.graph_query_policy_profile_id
                    )
                    policy_receipts.append(resolved_query.receipt)
                if request.fusion_policy_profile_id:
                    resolved_fusion = self.workflow.runtime_policies.resolve(
                        profile_type="FUSION", profile_id=request.fusion_policy_profile_id
                    )
                    if not isinstance(resolved_fusion.profile, FusionPolicyProfile):
                        raise TypeError("resolved FUSION profile has the wrong schema")
                    # Persist the concrete reviewed configuration. Clearing the profile ID
                    # prevents the replay payload from violating the input model's
                    # 'profile OR inline config' ownership rule.
                    request = request.model_copy(
                        update={
                            "birrf_config": resolved_fusion.profile.config,
                            "fusion_policy_profile_id": None,
                        }
                    )
                    policy_receipts.append(resolved_fusion.receipt)
            except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, KeyError, ValueError) as exc:
                raise OnlineStageExecutionError(
                    "ASTROGRAPH_RUNTIME_POLICY_INVALID", f"{type(exc).__name__}: {exc}"
                ) from exc

        if request.reasoner_mode == "rule" and self.workflow.backend is not None:
            raise OnlineStageExecutionError(
                "ASTROGRAPH_REASONER_MODE_MISMATCH", "rule mode must not receive an LLM backend."
            )
        if request.reasoner_mode in {"llm", "hybrid"} and self.workflow.backend is None:
            raise OnlineStageExecutionError(
                "ASTROGRAPH_REASONER_BACKEND_REQUIRED",
                f"{request.reasoner_mode} mode requires a structured reasoning backend.",
            )
        if self.workflow.backend is not None and (
            envelope.execution_policy.deterministic_only or not envelope.execution_policy.allow_llm
        ):
            raise OnlineStageExecutionError(
                "ASTROGRAPH_LLM_POLICY_BLOCKED",
                "LLM backend requires deterministic_only=false and allow_llm=true.",
            )

        plan = self.workflow.planner.plan(
            DiagnosisPlanningContext(
                reasoner_mode=request.reasoner_mode,
                shadow_mode=request.shadow_mode,
                has_data_driven_prediction=request.data_driven_prediction is not None,
                has_data_driven_inference_request=request.data_driven_inference_request is not None,
                data_model_registry_available=self.workflow.data_models is not None,
                has_kg_evidence_binding=request.kg_evidence_binding is not None,
                has_dor_control_plane_audit=request.dor_control_plane_audit is not None,
            )
        )
        traces: list[TraceEventDraft] = []
        if dataset_profile_receipt is not None:
            traces.append(self._trace(
                "resolve_dataset_profile",
                dataset_profile_receipt.model_dump(mode="json"),
                agent_id="dataset-profile-registry",
            ))
        for receipt in policy_receipts:
            traces.append(self._trace(
                "resolve_runtime_policy_profile",
                receipt.model_dump(mode="json"),
                agent_id="runtime-policy-registry",
            ))
        traces.extend([
            self._trace(
                "plan_diagnosis_workflow",
                {
                    "plan_id": plan.plan_id,
                    "enabled_stages": [item.value for item in plan.enabled_stages],
                    "warnings": plan.warnings,
                    "requires_human_review": plan.requires_human_review,
                },
                agent_id=self.workflow.planner.agent_id,
            ),
            self._trace(
                "simulation_completed_event",
                {
                    "event_id": request.simulation_event.event_id,
                    "run_dir": request.simulation_event.run_dir,
                    "reasoner_mode": request.reasoner_mode,
                    "candidate_fusion_enabled": (
                        request.data_driven_prediction is not None
                        or request.data_driven_inference_request is not None
                    ),
                },
            ),
        ])
        normalized_request = request.model_dump(mode="json")
        # The profile has already been resolved into concrete, hash-verified paths.
        # Replay stages validate this payload as a direct request, so retaining the
        # selector as well would violate the public "profile OR direct paths" rule.
        normalized_request["dataset_profile_id"] = None
        data = PrepareStageData(
            normalized_request=normalized_request,
            dataset_profile_receipt=(dataset_profile_receipt.model_dump(mode="json") if dataset_profile_receipt else None),
            profile_registered_fault_ids=profile_registered_fault_ids,
            runtime_policy_receipts=[item.model_dump(mode="json") for item in policy_receipts],
            diagnosis_execution_plan=plan.model_dump(mode="json"),
            trace_events=traces,
        )
        artifact = self._artifact("online.prepare.v1", store, data)
        return OnlineStageOutcome(
            status="SUCCEEDED",
            artifact=artifact,
            metadata={
                "reasoner_mode": request.reasoner_mode,
                "dataset_profile_id": requested_dataset_profile_id,
                "runtime_policy_profile_ids": [item.profile_id for item in policy_receipts],
                "plan_id": plan.plan_id,
            },
            warnings=list(plan.warnings),
        )

    def _semantics(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        result = self.workflow.tools.invoke("semantics.validate", {"protocol_root": request.protocol_root}, context)
        data = SemanticsStageData(
            registry_version=str(result["registry_version"]),
            validation=result["validation"],
            trace_events=[self._trace(
                "validate_semantics", result["validation"], tool_id="semantics.validate",
                tool_version=str(result["registry_version"]),
            )],
        )
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.semantics.v1", store, data), {
            "registry_version": str(result["registry_version"]), "validation": result["validation"]
        }, [])

    def _extension_control_plane(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        audit = None
        traces: list[TraceEventDraft] = []
        if request.dor_control_plane_audit is not None:
            audit = DORControlPlaneAuditService().audit(request.dor_control_plane_audit)
            traces.append(self._trace(
                "audit_diagnostic_control_plane_with_dor",
                {
                    "audit_id": audit.audit_id,
                    "status": audit.status.value,
                    "task": audit.task,
                    "eligible_for_diagnostic_execution": audit.eligible_for_diagnostic_execution,
                    "release_gate_enabled": audit.release_gate_enabled,
                },
                agent_id="dor-control-plane-audit-service",
            ))
        data = ControlPlaneStageData(
            dor_control_plane_audit=(audit.model_dump(mode="json") if audit else None), trace_events=traces
        )
        return OnlineStageOutcome(
            "SUCCEEDED" if audit else "SKIPPED",
            self._artifact("online.extension.control_plane.v1", store, data),
            {"provider": "DOR", "configured": request.dor_control_plane_audit is not None,
             "status": audit.status.value if audit else "NOT_CONFIGURED",
             "provider_lifecycle": BUILTIN_PROVIDER_QUALIFICATIONS.get("external-cli-managed.v1").model_dump(mode="json") if audit else None},
            [],
        )

    def _evidence(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        built = self.workflow.tools.invoke(
            "online.runtime_evidence.build",
            {
                "simulation_event": request.simulation_event.model_dump(mode="json"),
                "protocol_root": request.protocol_root,
                "evidence_package_path": request.evidence_package_path,
                "runtime_evidence_builder_id": request.runtime_evidence_builder_id,
                "soft_evidence_calibration_policy_path": (
                    request.soft_evidence_calibration_policy_path
                ),
                "soft_evidence_calibration_policy_sha256": (
                    request.soft_evidence_calibration_policy_sha256
                ),
            },
            context,
        )
        bundle = EpisodeEvidenceBundle.model_validate(built["bundle"])
        soft_evidence = SoftEvidenceBundle.model_validate(built["soft_evidence"])
        gate = OnlineDataGateReport.model_validate(built["data_gate_report"])
        trace = self._trace(
            "build_online_runtime_evidence",
            {
                "data_gate_passed": gate.passed,
                "feature_count": len(bundle.feature_results),
                "evidence_count": len(bundle.evidence_facts),
                "soft_evidence_count": len(soft_evidence.facts),
                "soft_calibrated_fact_count": soft_evidence.calibrated_fact_count,
                "soft_evidence_bundle_sha256": soft_evidence.bundle_sha256,
                "soft_calibration_policy_id": soft_evidence.calibration_policy_id,
                "soft_calibration_policy_sha256": soft_evidence.calibration_policy_sha256,
                "episode_id": bundle.episode.episode_id,
            },
            tool_id="online.runtime_evidence.build",
            tool_version=str(built["builder_version"]),
        )
        data = EvidenceStageData(
            bundle=bundle,
            soft_evidence=soft_evidence,
            data_gate_report=gate,
            builder_version=str(built["builder_version"]),
            trace_events=[trace],
        )
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.evidence.v1", store, data), {
            "data_gate_passed": gate.passed, "episode_id": bundle.episode.episode_id,
            "feature_count": len(bundle.feature_results), "evidence_count": len(bundle.evidence_facts),
            "soft_evidence_count": len(soft_evidence.facts),
            "soft_calibrated_fact_count": soft_evidence.calibrated_fact_count,
            "soft_evidence_bundle_sha256": soft_evidence.bundle_sha256,
            "soft_calibration_policy_id": soft_evidence.calibration_policy_id,
            "soft_calibration_policy_sha256": soft_evidence.calibration_policy_sha256,
        }, [])

    def _spatiotemporal(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        graph: ComponentSpatiotemporalFaultGraph | None = None
        matches: list[SpatiotemporalTemplateMatch] = []
        traces: list[TraceEventDraft] = []
        if request.spatiotemporal_template_catalog_path is not None:
            catalog = SpatiotemporalFaultTemplateCatalog.from_json(request.spatiotemporal_template_catalog_path)
            graph = build_component_spatiotemporal_graph(request.spatiotemporal_episodes, catalog)
            if graph.run_id != request.simulation_event.run_id:
                raise ValueError("spatiotemporal episode run_id must match simulation_event.run_id")
            matches = match_spatiotemporal_templates(graph, catalog)
            traces.append(self._trace(
                "match_component_spatiotemporal_fault_graph",
                {
                    "catalog_version": catalog.catalog_version,
                    "node_count": len(graph.nodes), "edge_count": len(graph.edges),
                    "unmapped_episode_count": len(graph.unmapped_episode_ids),
                    "eligible_match_count": sum(item.eligible_for_rerank for item in matches),
                    "rerank_weight": request.spatiotemporal_rerank_weight,
                },
                agent_id="spatiotemporal-fault-graph-matcher",
            ))
        data = SpatiotemporalStageData(graph=graph, matches=matches, trace_events=traces)
        return OnlineStageOutcome(
            "SUCCEEDED" if graph else "SKIPPED",
            self._artifact("online.spatiotemporal.v1", store, data),
            {"configured": request.spatiotemporal_template_catalog_path is not None, "match_count": len(matches)},
            [],
        )

    def _extension_semantic(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        semantic = multitask = vertical = None
        traces: list[TraceEventDraft] = []
        if request.dor_semantic_audit is not None:
            semantic = DORSemanticAuditService().audit(bundle, request.dor_semantic_audit)
            traces.append(self._trace("audit_evidence_semantics_with_dor", {
                "audit_id": semantic.audit_id, "status": semantic.status.value,
                "eligible_for_formal_fault_release": semantic.eligible_for_formal_fault_release,
                "supported_fault_ids": semantic.supported_fault_ids,
                "release_gate_enabled": semantic.release_gate_enabled,
            }, agent_id="dor-semantic-audit-service"))
        if request.dor_multitask_audit is not None:
            multitask_request = request.dor_multitask_audit
            if multitask_request.case_id is not None and multitask_request.case_id != request.simulation_event.run_id:
                raise ValueError("DOR multitask case_id must match simulation_event.run_id")
            multitask_request = multitask_request.model_copy(update={"case_id": request.simulation_event.run_id})
            multitask = DORMultitaskAuditService().audit(multitask_request)
            traces.append(self._trace("audit_cross_task_diagnosis_with_dor", {
                "audit_id": multitask.audit_id, "status": multitask.status.value,
                "diagnostic_depth": multitask.diagnostic_depth,
                "strongest_fault_ids": multitask.strongest_fault_ids,
                "physical_change_claimed": multitask.physical_change_claimed,
                "release_gate_enabled": multitask.release_gate_enabled, "birrf_branch_count_delta": 0,
            }, agent_id="dor-multitask-audit-service"))
        if request.dor_vertical_freeze_audit is not None:
            vertical = DORVerticalFreezeAuditService().audit(request.dor_vertical_freeze_audit)
            traces.append(self._trace("audit_vertical_evidence_and_freeze_with_dor", {
                "audit_id": vertical.audit_id, "status": vertical.status.value,
                "target_domain": vertical.target_domain, "vertical_status": vertical.vertical_status,
                "freeze_status": vertical.freeze_status, "missing_vertical_gates": vertical.missing_vertical_gates,
                "blocking_freeze_gates": vertical.blocking_freeze_gates,
                "release_gate_enabled": vertical.release_gate_enabled,
            }, agent_id="dor-vertical-freeze-audit-service"))
        data = SemanticExtensionStageData(
            dor_semantic_audit=semantic.model_dump(mode="json") if semantic else None,
            dor_multitask_audit=multitask.model_dump(mode="json") if multitask else None,
            dor_vertical_freeze_audit=vertical.model_dump(mode="json") if vertical else None,
            trace_events=traces,
        )
        configured = any((semantic, multitask, vertical))
        return OnlineStageOutcome("SUCCEEDED" if configured else "SKIPPED", self._artifact("online.extension.semantic.v1", store, data), {
            "provider": "DOR", "semantic_configured": request.dor_semantic_audit is not None,
            "multitask_configured": request.dor_multitask_audit is not None,
            "vertical_freeze_configured": request.dor_vertical_freeze_audit is not None,
            "provider_lifecycle": BUILTIN_PROVIDER_QUALIFICATIONS.get("external-cli-managed.v1").model_dump(mode="json") if configured else None,
        }, [])

    def _extension_mechanism(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        routing = observation = None
        traces: list[TraceEventDraft] = []
        if request.isgn_observation_audit is not None:
            observation = ISGNObservationAuditService().audit(request.isgn_observation_audit)
            traces.append(self._trace("qualify_mechanism_observation_with_isgn", {
                "audit_id": observation.audit_id, "status": observation.status.value,
                "dataset_id": observation.dataset_id, "system_id": observation.system_id,
                "eligible_for_mechanism_certification": observation.eligible_for_mechanism_certification,
                "sample_time_present": observation.sample_time_present,
                "timebase_status": observation.timebase_status,
                "temporal_sufficiency_status": observation.temporal_sufficiency_status,
            }, agent_id="isgn-observation-audit-service"))
        if request.isgn_routing_audit is not None:
            routing = ISGNRoutingAuditService().audit(bundle, request.isgn_routing_audit)
            traces.append(self._trace("route_evidence_through_isgn_atlas", {
                "audit_id": routing.audit_id, "status": routing.status.value,
                "system_id": routing.system_id, "active_evidence_count": routing.active_evidence_count,
                "mapped_evidence_count": routing.mapped_evidence_count,
                "eligible_for_mechanism_evidence": routing.eligible_for_mechanism_evidence,
            }, agent_id="isgn-mechanism-routing-audit-service"))
        data = MechanismExtensionStageData(
            isgn_routing_audit=routing.model_dump(mode="json") if routing else None,
            isgn_observation_audit=observation.model_dump(mode="json") if observation else None,
            trace_events=traces,
        )
        return OnlineStageOutcome(
            "SUCCEEDED" if routing or observation else "SKIPPED",
            self._artifact("online.extension.mechanism.v1", store, data),
            {"provider": "ISGN", "routing_configured": request.isgn_routing_audit is not None,
             "observation_configured": request.isgn_observation_audit is not None,
             "provider_lifecycle": BUILTIN_PROVIDER_QUALIFICATIONS.get("external-cli-managed.v1").model_dump(mode="json") if (routing or observation) else None}, [],
        )

    def _kg_evidence(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        result = None
        traces: list[TraceEventDraft] = []
        if request.kg_evidence_binding is not None:
            compiler = None
            if request.graph_query_policy_profile_id:
                if self.workflow.runtime_policies is None:
                    raise ValueError("graph query policy requires RuntimePolicyRegistry")
                compiler = GovernedGraphQueryCompiler(self.workflow.runtime_policies.resolve(
                    profile_type="GRAPH_QUERY", profile_id=request.graph_query_policy_profile_id
                ))
            result = self.workflow._query_bound_kg_evidence(
                config=request.kg_evidence_binding, bundle=bundle, compiler=compiler
            )
            kg = result.kg_evidence
            traces.append(self._trace("bind_kg_evidence", {
                "release_id": kg.release_id, "knowledge_version": kg.knowledge_version,
                "row_count": kg.row_count, "query_ids": kg.query_ids,
                "forward_association_count": len(result.forward_associations),
                "backward_association_count": len(result.backward_associations),
                "selected_backend": result.selected_backend,
                "degraded_to_reference": result.degraded_to_reference,
                "fallback_reason": result.fallback_reason,
            }, agent_id="read-only-kg-evidence-service"))
        data = KGEvidenceStageData(
            kg_evidence=(result.kg_evidence if result else None),
            forward_associations=(result.forward_associations if result else []),
            backward_associations=(result.backward_associations if result else []),
            selected_backend=(result.selected_backend if result else None),
            degraded_to_reference=(result.degraded_to_reference if result else False),
            fallback_reason=(result.fallback_reason if result else None),
            trace_events=traces,
        )
        return OnlineStageOutcome("SUCCEEDED" if result else "SKIPPED", self._artifact("online.kg_evidence.v1", store, data), {
            "configured": request.kg_evidence_binding is not None,
            "selected_backend": result.selected_backend if result else None,
            "degraded_to_reference": result.degraded_to_reference if result else False,
            "provider_lifecycle": (
                BUILTIN_PROVIDER_QUALIFICATIONS.get(
                    "kg-csv-reference.v1" if result.selected_backend == "csv_reference_fallback" else "neo4j-readonly.v1"
                ).model_dump(mode="json") if result is not None else None
            ),
        }, [])

    def _data_driven(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        prediction = request.data_driven_prediction
        traces: list[TraceEventDraft] = []
        provider_lifecycle = None
        if request.data_driven_inference_request is not None:
            if self.workflow.data_models is None:
                raise ValueError("data_driven_inference_request requires a configured DataDrivenModelRegistry")
            if request.data_driven_inference_request.episode_id != bundle.episode.episode_id:
                raise ValueError("model inference request episode ID does not match runtime evidence episode")
            prediction = self.workflow.data_models.predict(request.data_driven_inference_request)
            provider_lifecycle = self.workflow.data_models.provider_lifecycle(
                request.data_driven_inference_request.model_id, request.data_driven_inference_request.model_version
            )
            traces.append(self._trace("run_registered_data_driven_model", {
                "prediction_id": prediction.prediction_id, "model_id": prediction.model_id,
                "model_version": prediction.model_version, "model_family": prediction.model_family,
                "task_type": prediction.task_type, "output_semantics": prediction.output_semantics,
            }, agent_id="data-driven-model-registry"))
        data = DataDrivenStageData(prediction=prediction, trace_events=traces)
        return OnlineStageOutcome("SUCCEEDED" if prediction else "SKIPPED", self._artifact("online.data_driven.v1", store, data), {
            "configured": request.data_driven_prediction is not None or request.data_driven_inference_request is not None,
            "prediction_id": prediction.prediction_id if prediction else None,
            "model_id": prediction.model_id if prediction else None,
            "provider_lifecycle": provider_lifecycle,
        }, [])

    def _extension_calibration(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        prediction = parse_typed_data(store.get("online.data_driven.v1"), DataDrivenStageData).prediction
        audit = None
        traces: list[TraceEventDraft] = []
        if request.fec_evidence_audit is not None:
            if prediction is None:
                raise ValueError("FEC evidence audit requires a resolved data-driven prediction")
            audit = FECEvidenceAuditService().audit(prediction, request.fec_evidence_audit)
            traces.append(self._trace("qualify_model_evidence_with_fec", {
                "audit_id": audit.audit_id, "status": audit.status.value, "mode": audit.mode.value,
                "calibration_supported": audit.calibration_supported,
                "eligible_for_model_evidence": audit.eligible_for_model_evidence,
                "prediction_set": audit.prediction_set, "anomaly_decision": audit.anomaly_decision,
                "release_gate_enabled": audit.release_gate_enabled,
            }, agent_id="fec-model-evidence-audit-service"))
        data = CalibrationStageData(fec_evidence_audit=audit.model_dump(mode="json") if audit else None, trace_events=traces)
        return OnlineStageOutcome("SUCCEEDED" if audit else "SKIPPED", self._artifact("online.extension.calibration.v1", store, data), {
            "provider": "FEC", "configured": request.fec_evidence_audit is not None,
            "status": audit.status.value if audit else "NOT_CONFIGURED",
            "provider_lifecycle": BUILTIN_PROVIDER_QUALIFICATIONS.get("external-cli-managed.v1").model_dump(mode="json") if audit else None,
        }, [])

    def _fusion(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        evidence = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData)
        prediction = parse_typed_data(store.get("online.data_driven.v1"), DataDrivenStageData).prediction
        spatial = parse_typed_data(store.get("online.spatiotemporal.v1"), SpatiotemporalStageData)
        kg = parse_typed_data(store.get("online.kg_evidence.v1"), KGEvidenceStageData)
        candidate = None
        traces: list[TraceEventDraft] = []
        registered_fault_ids = sorted(
            {item.target_fault_id for item in evidence.bundle.evidence_definitions}
            | set(self._prepare_data(store).profile_registered_fault_ids)
        )
        if prediction is not None:
            data_ranking = prediction.to_fault_ranking(registered_fault_ids)
            open_set = assess_open_set(prediction, request.open_set_policy) if request.open_set_policy is not None else None
            runtime_metadata: dict[str, Any] = {}
            fused_metadata: dict[str, Any] = {}
            quality_warnings: list[str] = []
            if kg.kg_evidence is not None:
                runtime_metadata.update({
                    "kg_evidence_release_id": kg.kg_evidence.release_id,
                    "kg_evidence_knowledge_version": kg.kg_evidence.knowledge_version,
                    "kg_forward_association_count": len(kg.forward_associations),
                    "kg_backward_association_count": len(kg.backward_associations),
                    "kg_selected_backend": kg.selected_backend,
                    "degraded_to_reference": kg.degraded_to_reference,
                    "kg_fallback_reason": kg.fallback_reason,
                })
                fused_metadata.update({
                    "kg_evidence_release_id": kg.kg_evidence.release_id,
                    "kg_evidence_query_ids": kg.kg_evidence.query_ids,
                    "graph_query_backed": True,
                    "kg_selected_backend": kg.selected_backend,
                    "degraded_to_reference": kg.degraded_to_reference,
                    "kg_fallback_reason": kg.fallback_reason,
                })
                if kg.degraded_to_reference:
                    quality_warnings.append("KG_BACKEND_FALLBACK_TO_CSV_REFERENCE")
            candidate = self.workflow.fusion_service.build(
                bundle=evidence.bundle,
                data_ranking=data_ranking,
                config=request.birrf_config,
                normal_probability=prediction.normal_probability,
                normal_rank=prediction.normal_rank,
                spatiotemporal_matches=spatial.matches,
                spatiotemporal_weight=request.spatiotemporal_rerank_weight,
                forward_associations=(kg.forward_associations or None),
                backward_associations=(kg.backward_associations or None),
                open_set_assessment=open_set,
                runtime_metadata=runtime_metadata,
                fused_metadata=fused_metadata,
                quality_warnings=quality_warnings,
                registered_fault_ids=registered_fault_ids,
            )
            traces.append(self._trace("fuse_data_and_graph_rankings", {
                "fusion_id": candidate.birrf_result.fusion_id,
                "top_fault_id": candidate.birrf_result.top_fault_id,
                "conflict_flags": candidate.birrf_result.conflict_flags,
                "data_top_fault_id": candidate.birrf_result.metadata.get("data_top_fault_id"),
                "forward_top_fault_id": candidate.birrf_result.metadata.get("forward_top_fault_id"),
                "backward_top_fault_id": candidate.birrf_result.metadata.get("backward_top_fault_id"),
                "graph_query_backed": candidate.metadata.get("graph_query_backed", False),
                "kg_evidence_release_id": candidate.metadata.get("kg_evidence_release_id"),
                "open_set_status": candidate.birrf_result.metadata.get("open_set_assessment", {}).get("status"),
            }, agent_id="birrf-fusion-service"))
        soft_graph_shadow = None
        if evidence.soft_evidence is not None:
            provenance_policy = None
            if request.soft_graph_provenance_policy_path is not None:
                provenance_policy = load_soft_graph_provenance_policy(
                    request.soft_graph_provenance_policy_path,
                    expected_sha256=request.soft_graph_provenance_policy_sha256,
                )
            soft_graph_shadow = build_soft_graph_inference(
                evidence.soft_evidence,
                knowledge_release_id=(kg.kg_evidence.release_id if kg.kg_evidence else None),
                candidate_fault_ids=(
                    candidate.registered_fault_ids
                    if candidate is not None
                    else sorted({
                        item.target_fault_id for item in evidence.bundle.evidence_definitions
                    })
                ),
                independence_groups_by_definition=(
                    provenance_policy.definition_groups() if provenance_policy else None
                ),
                provenance_policy_id=(provenance_policy.policy_id if provenance_policy else None),
                provenance_policy_sha256=(
                    provenance_policy.policy_sha256 if provenance_policy else None
                ),
            )
            traces.append(self._trace("project_soft_evidence_to_shadow_graph", {
                "inference_id": soft_graph_shadow.inference_id,
                "suggested_action": soft_graph_shadow.suggested_action.value,
                "candidate_fault_ids": soft_graph_shadow.candidate_fault_ids,
                "reason_codes": soft_graph_shadow.reason_codes,
                "runtime_effect_applied": soft_graph_shadow.runtime_effect_applied,
            }, agent_id="soft-evidence-graph-shadow"))
        document_claim_shadow = build_online_document_claim_shadow(
            config=request.document_claim_shadow,
            episode_id=evidence.bundle.episode.episode_id,
            registered_fault_ids=registered_fault_ids,
        )
        if document_claim_shadow.status != "DISABLED":
            traces.append(self._trace("rebuild_document_claim_ranking_in_shadow", {
                "status": document_claim_shadow.status,
                "execution_mode": document_claim_shadow.execution_mode,
                "candidate_fusion_modified": document_claim_shadow.candidate_fusion_modified,
                "consumed_claim_ids": (
                    document_claim_shadow.ranking_build.consumed_claim_ids
                    if document_claim_shadow.ranking_build else []
                ),
                "reason_codes": document_claim_shadow.reason_codes,
            }, agent_id="document-claim-shadow-service"))
        data = FusionStageData(
            candidate_fusion=candidate,
            soft_graph_shadow=soft_graph_shadow,
            document_claim_shadow=document_claim_shadow,
            trace_events=traces,
        )
        return OnlineStageOutcome("SUCCEEDED" if candidate else "SKIPPED", self._artifact("online.fusion.v1", store, data), {
            "fusion_id": candidate.birrf_result.fusion_id if candidate else None,
            "top_fault_id": candidate.birrf_result.top_fault_id if candidate else None,
            "soft_graph_inference_id": (
                soft_graph_shadow.inference_id if soft_graph_shadow else None
            ),
            "soft_graph_suggested_action": (
                soft_graph_shadow.suggested_action.value if soft_graph_shadow else "NOT_CONFIGURED"
            ),
            "soft_graph_runtime_effect_applied": False,
            "document_claim_shadow_status": document_claim_shadow.status,
            "document_claim_runtime_effect_applied": False,
        }, [])

    def _verification(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        evidence = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData)
        plan = self.workflow.verifier.build_plan(evidence.bundle)
        data = VerificationStageData(verification_plan=plan, trace_events=[self._trace(
            "create_verification_plan", {"verification_plan_id": plan.verification_plan_id, "task_count": len(plan.tasks)},
            agent_id=self.workflow.verifier.agent_id,
        )])
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.verification.v1", store, data), {
            "verification_plan_id": plan.verification_plan_id, "task_count": len(plan.tasks)
        }, [])

    def _reasoning(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        from .realtime_reasoning import decide_reasoning_activation

        request = self._request(store)
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        candidate = parse_typed_data(store.get("online.fusion.v1"), FusionStageData).candidate_fusion
        core = parse_typed_data(
            store.get("online.deterministic_core.v1"), DeterministicCoreStageData
        ).deterministic_core_result
        retrieval = None
        paperseek_fallback = None
        retrieval_request = request.pre_reasoning_retrieval_request
        if retrieval_request is None and request.pre_reasoning_fault_catalog_path is not None:
            if request.evidence_package_path is None:
                raise ValueError("automatic pre-reasoning retrieval requires runtime evidence package")
            retrieval_request = EvidenceAwarePreGenerationContextBuilder(
                runtime_evidence_path=request.evidence_package_path,
                fault_catalog_path=request.pre_reasoning_fault_catalog_path,
                experience_ledger_path=request.pre_reasoning_experience_ledger_path,
                memory_store_path=request.pre_reasoning_memory_store_path,
            ).build_request(bundle, candidate_fusion=candidate)
        if retrieval_request is not None:
            if retrieval_request.query.episode_id != bundle.episode.episode_id:
                raise ValueError("pre-reasoning retrieval query and evidence bundle episode IDs differ")
            retrieval = PreReasoningRetrievalService().retrieve(retrieval_request)
            paperseek_fallback = PaperSeekGovernedBridge.decide_fallback(
                retrieval_request.query,
                retrieval,
                policy=request.paperseek_fallback_policy,
            )
        llm_retrieval = retrieval if request.llm_retrieval_context_enabled else None
        if llm_retrieval is not None and not request.llm_experience_context_enabled:
            llm_retrieval = llm_retrieval.model_copy(update={
                "local_hits": [
                    item for item in llm_retrieval.local_hits if item.lane.value != "EXPERIENCE"
                ],
                "experience_hit_count": 0,
                "experience_source_ids": [],
                "experience_reuse_receipt": None,
                "knowledge_update_candidates": [],
            })
        activation = decide_reasoning_activation(
            request_mode=request.reasoner_mode,
            policy=request.reasoning_activation_policy,
            bundle=bundle,
            deterministic_core_result=core,
            candidate_fusion_available=candidate is not None,
            operator_requested=request.operator_reasoning_request,
        )
        backend_error: str | None = None
        deliberation = None
        probe_catalog: DiagnosticProbeCatalog | None = None
        try:
            if activation.llm_invoked:
                if request.llm_diagnostic_loop_enabled and self.workflow.backend is not None:
                    kg_stage = parse_typed_data(
                        store.get("online.kg_evidence.v1"), KGEvidenceStageData
                    )
                    verification_stage = parse_typed_data(
                        store.get("online.verification.v1"), VerificationStageData
                    )
                    probe_catalog = build_evidence_margin_probe_catalog(bundle)
                    if request.llm_active_telemetry_observation_enabled:
                        gate = parse_typed_data(
                            store.get("online.evidence.v1"), EvidenceStageData
                        ).data_gate_report
                        probe_catalog = _revalidated_telemetry_probe_catalog(
                            request=request, bundle=bundle, gate=gate, base_catalog=probe_catalog,
                            priority_fault_ids=(
                                tuple(item.fault_id for item in sorted(
                                    candidate.birrf_result.candidates,
                                    key=lambda item: (item.final_rank, item.fault_id),
                                )[:4]) if candidate else ()
                            ),
                        )
                    deliberation = ControlledDiagnosticDeliberationService().run(
                        backend=self.workflow.backend,
                        episode_id=bundle.episode.episode_id,
                        max_steps=request.llm_diagnostic_loop_max_steps,
                        allowed_capability_ids=request.llm_diagnostic_capability_ids,
                        observation_feedback_enabled=(
                            request.llm_diagnostic_observation_feedback_enabled
                        ),
                        probe_catalog=probe_catalog,
                        probe_executor=(
                            ActiveTelemetryProbeExecutor()
                            if request.llm_active_telemetry_observation_enabled else None
                        ),
                        artifacts={
                            "evidence": bundle.model_dump(mode="json"),
                            "fusion": candidate.model_dump(mode="json") if candidate else None,
                            "kg": kg_stage.model_dump(mode="json"),
                            "retrieval": (
                                llm_retrieval.model_dump(mode="json") if llm_retrieval else None
                            ),
                            "core": core,
                            "verification": verification_stage.verification_plan.model_dump(mode="json"),
                        },
                    )
                proposed = self.workflow.reasoner.reason(
                    bundle,
                    candidate_fusion=candidate.model_dump(mode="json") if candidate else None,
                    retrieval_context=llm_retrieval,
                    deliberation_context=(
                        deliberation.model_dump(mode="json") if deliberation else None
                    ),
                    capture_shadow_candidate_revision=request.llm_shadow_candidate_revision_enabled,
                )
            else:
                proposed = self.workflow.reasoner.reference_hypotheses(bundle)
        except (TimeoutError, ConnectionError, OSError) as exc:
            if not request.reasoning_activation_policy.deterministic_fallback_on_llm_error:
                raise
            backend_error = type(exc).__name__
            if deliberation is not None:
                deliberation = ControlledDiagnosticDeliberationService().record_synthesis_failure(
                    deliberation, error_type=backend_error
                )
            proposed = self.workflow.reasoner.reference_hypotheses(bundle)
            activation = activation.model_copy(update={
                "llm_invoked": True,
                "trigger_codes": sorted(set([*activation.trigger_codes, "LLM_ERROR_DETERMINISTIC_FALLBACK"])),
            })
        hypotheses = self.workflow._add_out_of_scope_hypothesis(proposed, bundle)
        hypotheses = self.workflow._add_composite_research_hypothesis(hypotheses, bundle)
        backend_meta = dict(
            getattr(self.workflow.backend, "last_call_metadata", {})
            if activation.llm_invoked and backend_error is None and self.workflow.backend is not None
            else {"mode": "deterministic-reference", "llm_error_type": backend_error,
                  "llm_skipped": not activation.llm_invoked}
        )
        if request.llm_active_telemetry_observation_enabled:
            backend_meta["active_telemetry_probe_catalog"] = {
                "admission_status": (
                    probe_catalog.telemetry_admission_status if probe_catalog else "NOT_CONSTRUCTED"
                ),
                "available_probe_count": (
                    sum(is_admitted_telemetry_probe(item, probe_catalog) for item in probe_catalog.probes)
                    if probe_catalog else 0
                ),
            }
        raw_shadow_proposal = backend_meta.pop("shadow_candidate_revision_proposal", None)
        shadow_revision: ShadowCandidateRevision | None = None
        if request.llm_shadow_candidate_revision_enabled:
            shadow_revision = validate_shadow_candidate_revision(
                raw_proposal=raw_shadow_proposal,
                trajectory=deliberation,
                registered_fault_ids={item.target_fault_id for item in bundle.evidence_definitions},
                baseline_candidate_fault_ids=backend_meta.get("selected_candidate_fault_ids"),
            )
            if (
                request.llm_active_telemetry_observation_enabled
                and deliberation is not None
                and shadow_revision.active_telemetry_attributable
            ):
                deliberation = ControlledDiagnosticDeliberationService().record_shadow_revision(
                    deliberation, shadow_revision,
                )
                hypotheses = hypotheses.model_copy(update={
                    "validation_metadata": {
                        **hypotheses.validation_metadata,
                        "llm_diagnostic_trajectory_sha256": deliberation.trajectory_sha256,
                    }
                })
        provider_lifecycle = (
            getattr(self.workflow.backend, "lifecycle_profile").model_dump(mode="json")
            if self.workflow.backend is not None and hasattr(self.workflow.backend, "lifecycle_profile")
            else None
        )
        navigation = None
        navigation_error_type = None
        try:
            navigation = self.workflow.navigator.navigate_episode(bundle)
        except Exception as exc:
            # Navigation is additive context; it must never block reasoning.
            navigation_error_type = type(exc).__name__
        navigation_summary = {
            "episode_id": bundle.episode.episode_id,
            "status": "SUCCEEDED" if navigation is not None else "FAILED_NON_BLOCKING",
            "error_type": navigation_error_type,
            "path_count": len(navigation.paths) if navigation else 0,
            "faults_with_support": (
                sorted(
                    fault
                    for fault, info in (navigation.fault_summary or {}).items()
                    if info.get("has_support")
                )
                if navigation
                else []
            ),
        }
        data = ReasoningStageData(
            hypothesis_set=hypotheses,
            pre_reasoning_retrieval=retrieval,
            paperseek_fallback_decision=paperseek_fallback,
            llm_diagnostic_trajectory=deliberation,
            shadow_candidate_revision=shadow_revision,
            reasoning_backend=backend_meta,
            activation_decision=activation.model_dump(mode="json"),
            trace_events=[
                self._trace(
                    "navigate_episode_evidence_graph",
                    navigation_summary,
                    agent_id=self.workflow.navigator.agent_id,
                ),
                self._trace(
                    "generate_hypotheses",
                    {
                        "hypothesis_count": len(hypotheses.hypotheses),
                        "validation_metadata": hypotheses.validation_metadata,
                        "reasoning_backend": backend_meta,
                        "activation_decision": activation.model_dump(mode="json"),
                        "pre_reasoning_retrieval_package_id": (
                            retrieval.package_id if retrieval else None
                        ),
                        "pre_reasoning_retrieval_status": (
                            retrieval.status.value if retrieval else "NOT_CONFIGURED"
                        ),
                        "paperseek_fallback_status": (
                            paperseek_fallback.status.value
                            if paperseek_fallback
                            else "NOT_CONFIGURED"
                        ),
                        "llm_diagnostic_loop_status": (
                            deliberation.status.value if deliberation else "NOT_EXECUTED"
                        ),
                        "llm_method_substance_status": (
                            deliberation.substance_audit.status.value
                            if deliberation else "NOT_EXECUTED"
                        ),
                        "llm_diagnostic_contribution_status": (
                            deliberation.contribution_audit.status.value
                            if deliberation else "NOT_EXECUTED"
                        ),
                    },
                    agent_id=self.workflow.reasoner.agent_id,
                )
            ],
        )
        return OnlineStageOutcome(
            "SUCCEEDED",
            self._artifact("online.reasoning.v1", store, data),
            {
                "hypothesis_count": len(hypotheses.hypotheses),
                "reasoner_mode": request.reasoner_mode,
                "reasoning_activation": activation.model_dump(mode="json"),
                "pre_reasoning_retrieval_package_id": (
                    retrieval.package_id if retrieval else None
                ),
                "pre_reasoning_retrieval_status": (
                    retrieval.status.value if retrieval else "NOT_CONFIGURED"
                ),
                "paperseek_fallback_status": (
                    paperseek_fallback.status.value
                    if paperseek_fallback
                    else "NOT_CONFIGURED"
                ),
                "provider_lifecycle": provider_lifecycle,
                "llm_diagnostic_loop_status": (
                    deliberation.status.value if deliberation else "NOT_EXECUTED"
                ),
                "llm_method_substance_status": (
                    deliberation.substance_audit.status.value if deliberation else "NOT_EXECUTED"
                ),
                "llm_diagnostic_contribution_status": (
                    deliberation.contribution_audit.status.value if deliberation else "NOT_EXECUTED"
                ),
            },
            [],
        )

    def _critique(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        hypotheses = parse_typed_data(store.get("online.reasoning.v1"), ReasoningStageData).hypothesis_set
        critique = self.workflow._augment_critique(self.workflow.critic.critique(hypotheses, bundle), hypotheses, bundle)
        data = CritiqueStageData(critique_result=critique, trace_events=[self._trace(
            "safety_critique", {"blocking": critique.blocking, "codes": [item.code for item in critique.findings]},
            agent_id=self.workflow.critic.agent_id,
        )])
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.critique.v1", store, data), {
            "blocking": critique.blocking, "finding_count": len(critique.findings)
        }, [])

    def _deterministic_core(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        core = self.workflow.tools.invoke("diagnosis.invoke_online_core", {"bundle": bundle.model_dump(mode="json")}, context)
        data = DeterministicCoreStageData(deterministic_core_result=core, trace_events=[self._trace(
            "run_independent_deterministic_core",
            {"diagnosis_status": core["diagnosis_status"], "predicted_case_id": core["predicted_case_id"],
             "supported_fault_ids": core["supported_fault_ids"]},
            tool_id="diagnosis.invoke_online_core", tool_version=str(core["core_version"]),
        )])
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.deterministic_core.v1", store, data), {
            "diagnosis_status": core["diagnosis_status"], "predicted_case_id": core["predicted_case_id"],
            "core_version": str(core["core_version"])
        }, [])

    def _governor(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        evidence = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData)
        hypotheses = parse_typed_data(store.get("online.reasoning.v1"), ReasoningStageData).hypothesis_set
        critique = parse_typed_data(store.get("online.critique.v1"), CritiqueStageData).critique_result
        core = parse_typed_data(store.get("online.deterministic_core.v1"), DeterministicCoreStageData).deterministic_core_result
        decision = self.workflow.governor.decide(
            hypothesis_set=hypotheses, critique=critique, deterministic_core_result=core,
            protocol_versions={"astrograph": self.workflow.workflow_version,
                               "telemetry_semantics": evidence.data_gate_report.protocol_version,
                               "runtime_core": core["core_version"]},
        )
        data = GovernorStageData(decision_record=decision, trace_events=[self._trace(
            "decision_governor", {"disposition": decision.disposition.value,
                                  "accepted_fault_id": decision.accepted_fault_id,
                                  "comparison_status": decision.comparison_status,
                                  "requires_human_review": decision.requires_human_review},
            agent_id=self.workflow.governor.governor_id,
        )])
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.governor.v1", store, data), {
            "role": "DIAGNOSTIC_DECISION", "disposition": decision.disposition.value,
            "accepted_fault_id": decision.accepted_fault_id,
            "requires_human_review": decision.requires_human_review,
        }, [])

    def _policy_gate(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        evidence = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData)
        decision = parse_typed_data(store.get("online.governor.v1"), GovernorStageData).decision_record
        fusion = parse_typed_data(store.get("online.fusion.v1"), FusionStageData).candidate_fusion
        kg_data = parse_typed_data(store.get("online.kg_evidence.v1"), KGEvidenceStageData)
        control_data = parse_typed_data(store.get("online.extension.control_plane.v1"), ControlPlaneStageData)
        semantic_data = parse_typed_data(store.get("online.extension.semantic.v1"), SemanticExtensionStageData)
        mechanism_data = parse_typed_data(store.get("online.extension.mechanism.v1"), MechanismExtensionStageData)
        calibration_data = parse_typed_data(store.get("online.extension.calibration.v1"), CalibrationStageData)
        control = DORControlPlaneAudit.model_validate(control_data.dor_control_plane_audit) if control_data.dor_control_plane_audit else None
        semantic = DORSemanticAudit.model_validate(semantic_data.dor_semantic_audit) if semantic_data.dor_semantic_audit else None
        multitask = DORMultitaskAudit.model_validate(semantic_data.dor_multitask_audit) if semantic_data.dor_multitask_audit else None
        vertical = DORVerticalFreezeAudit.model_validate(semantic_data.dor_vertical_freeze_audit) if semantic_data.dor_vertical_freeze_audit else None
        routing = ISGNRoutingAudit.model_validate(mechanism_data.isgn_routing_audit) if mechanism_data.isgn_routing_audit else None
        observation = ISGNObservationAudit.model_validate(mechanism_data.isgn_observation_audit) if mechanism_data.isgn_observation_audit else None
        fec = FECEvidenceAudit.model_validate(calibration_data.fec_evidence_audit) if calibration_data.fec_evidence_audit else None
        deb = self.workflow._build_policy_evidence_bundle(
            simulation_event=request.simulation_event, bundle=evidence.bundle, decision=decision,
            candidate_fusion=fusion, kg_evidence=kg_data.kg_evidence, fec_evidence_audit=fec,
        )
        fused = fusion.birrf_result if fusion else None
        policy = self.workflow.diagnosis_policy_gate.decide(
            deb,
            blocking_conflicts=fused.blocking_conflicts if fused else None,
            quality_warnings=fused.quality_warnings if fused else None,
            open_set_actions=fused.open_set_actions if fused else None,
            semantic_gate_blockers=[
                *control_plane_release_blockers(control),
                *semantic_release_blockers(semantic, decision.accepted_fault_id),
                *multitask_release_blockers(multitask, decision.accepted_fault_id),
                *vertical_freeze_release_blockers(vertical),
            ],
            calibration_gate_blockers=fec_evidence_release_blockers(fec, decision.accepted_fault_id),
            mechanism_gate_blockers=(
                isgn_routing_release_blockers(routing, decision.accepted_fault_id)
                + isgn_observation_release_blockers(observation, decision.accepted_fault_id)
            ),
        )
        data = ReleaseQualificationStageData(
            diagnosis_evidence_bundle=deb, diagnosis_policy_decision=policy,
            trace_events=[self._trace("diagnosis_policy_gate", {
                "policy_status": policy.status.value, "blocking_reasons": policy.blocking_reasons
            }, agent_id="diagnosis-policy-gate")],
        )
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.policy_gate.v1", store, data), {
            "role": "RELEASE_QUALIFICATION", "policy_status": policy.status.value,
            "blocking_reasons": policy.blocking_reasons,
        }, [])

    def _evidence_graph(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        bundle = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData).bundle
        hypotheses = parse_typed_data(store.get("online.reasoning.v1"), ReasoningStageData).hypothesis_set
        decision = parse_typed_data(store.get("online.governor.v1"), GovernorStageData).decision_record
        document = self.workflow.tools.invoke("online.evidence_graph.build", {
            "bundle": bundle.model_dump(mode="json"),
            "hypothesis_set": hypotheses.model_dump(mode="json"),
            "decision_record": decision.model_dump(mode="json"),
        }, context)
        graph = EpisodeEvidenceGraph.model_validate(document)
        data = EvidenceGraphStageData(evidence_graph=graph, trace_events=[self._trace(
            "build_dynamic_episode_evidence_graph", {"node_count": len(graph.nodes), "edge_count": len(graph.edges)},
            tool_id="online.evidence_graph.build", tool_version="0.5.10",
        )])
        return OnlineStageOutcome("SUCCEEDED", self._artifact("online.evidence_graph.v1", store, data), {
            "node_count": len(graph.nodes), "edge_count": len(graph.edges)
        }, [])

    def _persist(self, envelope: TaskEnvelope, context: WorkflowContext, store: RuntimeArtifactStore) -> OnlineStageOutcome:
        request = self._request(store)
        prepare = self._prepare_data(store)
        evidence = parse_typed_data(store.get("online.evidence.v1"), EvidenceStageData)
        control = parse_typed_data(store.get("online.extension.control_plane.v1"), ControlPlaneStageData)
        semantic = parse_typed_data(store.get("online.extension.semantic.v1"), SemanticExtensionStageData)
        mechanism = parse_typed_data(store.get("online.extension.mechanism.v1"), MechanismExtensionStageData)
        kg = parse_typed_data(store.get("online.kg_evidence.v1"), KGEvidenceStageData)
        data_driven = parse_typed_data(store.get("online.data_driven.v1"), DataDrivenStageData)
        calibration = parse_typed_data(store.get("online.extension.calibration.v1"), CalibrationStageData)
        fusion = parse_typed_data(store.get("online.fusion.v1"), FusionStageData)
        spatial = parse_typed_data(store.get("online.spatiotemporal.v1"), SpatiotemporalStageData)
        verification = parse_typed_data(store.get("online.verification.v1"), VerificationStageData)
        reasoning = parse_typed_data(store.get("online.reasoning.v1"), ReasoningStageData)
        critique = parse_typed_data(store.get("online.critique.v1"), CritiqueStageData)
        core = parse_typed_data(store.get("online.deterministic_core.v1"), DeterministicCoreStageData)
        governor = parse_typed_data(store.get("online.governor.v1"), GovernorStageData)
        qualification = parse_typed_data(store.get("online.policy_gate.v1"), ReleaseQualificationStageData)
        graph = parse_typed_data(store.get("online.evidence_graph.v1"), EvidenceGraphStageData)

        trace = AgentTrace(
            trace_id=f"TRACE-{context.run_id}", workflow_id=self.workflow.workflow_id, run_id=context.run_id,
            model_id=getattr(self.workflow.backend, "model_id", "deterministic-reference-agent"),
            model_version=getattr(self.workflow.backend, "model_version", self.workflow.workflow_version),
        )
        sequence = 0
        for stage_id in ONLINE_STAGE_IDS[:-1]:
            artifact = store.get(stage_id)
            for draft in parse_trace_events(artifact):
                sequence += 1
                trace.append(AgentTraceEvent(
                    sequence=sequence, agent_id=draft.agent_id, action=draft.action,
                    input_summary=draft.input_summary, output_summary=draft.output_summary,
                    tool_id=draft.tool_id, tool_version=draft.tool_version,
                ))
        trace.termination_reason = (
            "online_shadow_diagnosis_completed_with_human_review"
            if governor.decision_record.requires_human_review else "online_shadow_diagnosis_completed"
        )
        sequence += 1
        trace.append(AgentTraceEvent(
            sequence=sequence, agent_id="astrograph-v0510-online", action="persist_online_artifacts",
            input_summary={}, output_summary={"artifact_output_dir": request.artifact_output_dir},
            tool_id="online.artifacts.persist", tool_version="0.5.10",
        ))
        llm_snapshot = (
            {"status": "LLM_BACKEND", "runtime_snapshot": reasoning.reasoning_backend}
            if reasoning.activation_decision.get("llm_invoked")
            else {"status": "DETERMINISTIC_REFERENCE", "runtime_snapshot": {
                "schema_version": "astrograph-llm-runtime-snapshot.v1",
                "model": "deterministic-reference-agent", "prompt_profile": "deterministic",
            }}
        )
        persistence_payload = {
            "artifact_output_dir": request.artifact_output_dir,
            "dataset_profile_receipt": prepare.dataset_profile_receipt,
            "diagnosis_execution_plan": prepare.diagnosis_execution_plan,
            "simulation_event": request.simulation_event.model_dump(mode="json"),
            "data_gate_report": evidence.data_gate_report.model_dump(mode="json"),
            "episode_evidence_bundle": evidence.bundle.model_dump(mode="json"),
            "soft_evidence": (
                evidence.soft_evidence.model_dump(mode="json") if evidence.soft_evidence else None
            ),
            "dor_semantic_audit": semantic.dor_semantic_audit,
            "dor_control_plane_audit": control.dor_control_plane_audit,
            "dor_multitask_audit": semantic.dor_multitask_audit,
            "dor_vertical_freeze_audit": semantic.dor_vertical_freeze_audit,
            "fec_evidence_audit": calibration.fec_evidence_audit,
            "isgn_routing_audit": mechanism.isgn_routing_audit,
            "isgn_observation_audit": mechanism.isgn_observation_audit,
            "data_driven_prediction": data_driven.prediction.model_dump(mode="json") if data_driven.prediction else None,
            "candidate_fusion": fusion.candidate_fusion.model_dump(mode="json") if fusion.candidate_fusion else None,
            "soft_graph_shadow": (
                fusion.soft_graph_shadow.model_dump(mode="json") if fusion.soft_graph_shadow else None
            ),
            "document_claim_shadow": fusion.document_claim_shadow.model_dump(mode="json"),
            "component_spatiotemporal_fault_graph": spatial.graph.model_dump(mode="json") if spatial.graph else None,
            "spatiotemporal_template_matches": [item.model_dump(mode="json") for item in spatial.matches],
            "verification_plan": verification.verification_plan.model_dump(mode="json"),
            "pre_reasoning_retrieval": (
                reasoning.pre_reasoning_retrieval.model_dump(mode="json")
                if reasoning.pre_reasoning_retrieval else None
            ),
            "paperseek_fallback_decision": (
                reasoning.paperseek_fallback_decision.model_dump(mode="json")
                if reasoning.paperseek_fallback_decision else None
            ),
            "llm_diagnostic_trajectory": (
                reasoning.llm_diagnostic_trajectory.model_dump(mode="json")
                if reasoning.llm_diagnostic_trajectory else None
            ),
            "shadow_candidate_revision": (
                reasoning.shadow_candidate_revision.model_dump(mode="json")
                if reasoning.shadow_candidate_revision else None
            ),
            "hypothesis_set": reasoning.hypothesis_set.model_dump(mode="json"),
            "critique_result": critique.critique_result.model_dump(mode="json"),
            "deterministic_core_result": core.deterministic_core_result,
            "decision_record": governor.decision_record.model_dump(mode="json"),
            "diagnosis_evidence_bundle": qualification.diagnosis_evidence_bundle.model_dump(mode="json"),
            "diagnosis_policy_decision": qualification.diagnosis_policy_decision.model_dump(mode="json"),
            "episode_evidence_graph": graph.evidence_graph.model_dump(mode="json"),
            "agent_trace": trace.model_dump(mode="json"),
            "reasoner_mode": request.reasoner_mode,
            "reasoning_activation": reasoning.activation_decision,
            "model_id": trace.model_id,
            "llm_runtime_snapshot": llm_snapshot,
            "workflow_version": self.workflow.workflow_version,
            "runtime_stage_artifacts": {
                sid: artifact.model_dump(mode="json") for sid, artifact in store.snapshot().items()
            },
        }
        persistence_payload["structured_diagnosis_output"] = (
            build_structured_diagnosis_output(persistence_payload).model_dump(mode="json")
        )
        manifest = self.workflow.tools.invoke("online.artifacts.persist", persistence_payload, context)
        output_payload = {k: v for k, v in persistence_payload.items() if k != "runtime_stage_artifacts"}
        output_payload["artifact_manifest"] = manifest
        output_payload["runtime_policy_receipts"] = prepare.runtime_policy_receipts
        plan_warnings = list(prepare.diagnosis_execution_plan.get("warnings", []))
        result = ExecutionResult.succeeded(
            task_id=envelope.task_id, workflow_id=self.workflow.workflow_id,
            output_payload=output_payload,
            tool_versions={spec.tool_id: spec.tool_version for spec in self.workflow.tools.list_specs()},
            warnings=[
                "E0 synthetic online shadow result; all 16 single-fault scenarios are now registered.",
                "Composite-fault contexts remain UNVERIFIED research hypotheses and human-review only.",
                *plan_warnings,
            ],
            metadata={
                "run_id": request.simulation_event.run_id,
                "artifact_dir": request.artifact_output_dir,
                "reasoner_mode": request.reasoner_mode,
                "candidate_fusion_enabled": fusion.candidate_fusion is not None,
                "pre_reasoning_retrieval_status": (
                    reasoning.pre_reasoning_retrieval.status.value
                    if reasoning.pre_reasoning_retrieval else "NOT_CONFIGURED"
                ),
                "paperseek_fallback_status": (
                    reasoning.paperseek_fallback_decision.status.value
                    if reasoning.paperseek_fallback_decision else "NOT_CONFIGURED"
                ),
                "dataset_profile_id": (
                    prepare.dataset_profile_receipt or {}
                ).get("profile_id"),
                "runtime_policy_profile_ids": [item.get("profile_id") for item in prepare.runtime_policy_receipts],
                "dor_control_plane_status": (control.dor_control_plane_audit or {}).get("status", "NOT_CONFIGURED"),
                "dor_multitask_status": (semantic.dor_multitask_audit or {}).get("status", "NOT_CONFIGURED"),
                "dor_vertical_freeze_status": (semantic.dor_vertical_freeze_audit or {}).get("status", "NOT_CONFIGURED"),
                "fec_evidence_status": (calibration.fec_evidence_audit or {}).get("status", "NOT_CONFIGURED"),
                "isgn_routing_status": (mechanism.isgn_routing_audit or {}).get("status", "NOT_CONFIGURED"),
                "isgn_observation_status": (mechanism.isgn_observation_audit or {}).get("status", "NOT_CONFIGURED"),
            },
        )
        data = PersistStageData(execution_result=result.model_dump(mode="json"), artifact_manifest=manifest)
        artifact = self._artifact("online.persist.v1", store, data)
        return OnlineStageOutcome("SUCCEEDED", artifact, {
            "artifact_output_dir": request.artifact_output_dir, "artifact_manifest_present": manifest is not None
        }, [], execution_result=result)
