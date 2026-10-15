from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from .contracts import RuntimeStageContext, StageResult, StageStatus


class DeterministicDiagnosisCompatibilityStage:
    """Adapter around the historical evidence/KG deterministic workflow.

    The compatibility workflow remains available, but it no longer owns an application
    entrypoint. Only AstroGraphRuntime invokes it.
    """

    stage_id = "compat.deterministic_diagnosis.v1"
    stage_version = "1.0"

    def __init__(self, *, repository: Any, policy: Any, audit_store: Any, constraints: Any = None) -> None:
        self.repository = repository
        self.policy = policy
        self.audit_store = audit_store
        self.constraints = constraints

    def execute(self, payload: dict[str, Any], context: RuntimeStageContext) -> StageResult:
        schemas = import_module("astrograph.schemas")
        workflow = import_module("astrograph.diagnosis.workflow")
        conflicts = import_module("astrograph.diagnosis.conflicts")
        request = schemas.WorkflowInput.model_validate(payload)
        state = workflow.invoke_workflow(
            self.repository,
            request,
            policy=self.policy,
            audit_store=self.audit_store,
            constraints=self.constraints,
        )
        diagnosis_result = schemas.DiagnosisResult.model_validate(state["diagnosis"])
        response = {
            "schema_version": "astrograph-diagnosis-api-response.v3",
            "request_id": request.request_id,
            "case_id": request.case_id,
            "diagnosis": state["diagnosis"],
            "conflict_explanation": conflicts.explain_diagnostic_conflicts(diagnosis_result).model_dump(mode="json"),
            "report_markdown": state["report_markdown"],
            "audit": state["audit_record"],
            "runtime": {
                "pipeline_id": "diagnosis.compat.v1",
                "stage_id": self.stage_id,
            },
        }
        return StageResult.succeeded(response, metadata={"compatibility": True})


class UnifiedDiagnosisCompatibilityStage:
    """Adapter for the historical UnifiedDiagnosisService behind AstroGraphRuntime.

    Single-case and batch modes are selected explicitly in the runtime payload. The
    implementation import stays lazy so the canonical runtime contract is independent
    from the legacy diagnosis package.
    """

    stage_id = "compat.unified_diagnosis.v1"
    stage_version = "1.0"

    def execute(self, payload: dict[str, Any], context: RuntimeStageContext) -> StageResult:
        diagnosis = import_module("astrograph.diagnosis")
        mode = str(payload.get("mode", "single"))
        service = diagnosis.UnifiedDiagnosisService()
        if mode == "single":
            required = ("case_id", "telemetry_path")
            missing = [key for key in required if not payload.get(key)]
            if missing:
                return StageResult.failed(
                    code="ASTROGRAPH_UNIFIED_RUNTIME_INPUT_INVALID",
                    message=f"unified diagnosis requires fields: {', '.join(missing)}",
                    stage_id=self.stage_id,
                )
            result = service.diagnose_paths(
                case_id=str(payload["case_id"]),
                telemetry_path=payload["telemetry_path"],
                nominal_path=payload.get("nominal_path"),
                backend=str(payload.get("backend", "auto")),
                catalog_path=payload.get("catalog_path"),
                catalog_model_path=payload.get("catalog_model_path"),
                open_set_model_path=payload.get("open_set_model_path"),
            )
            output = result.model_dump(mode="json")
            return StageResult.succeeded(
                output,
                metadata={
                    "compatibility": True,
                    "mode": "single",
                    "diagnosis_status": output.get("status", ""),
                },
            )
        if mode == "batch":
            manifest_path = payload.get("manifest_path")
            if not manifest_path:
                return StageResult.failed(
                    code="ASTROGRAPH_UNIFIED_RUNTIME_INPUT_INVALID",
                    message="unified batch diagnosis requires manifest_path",
                    stage_id=self.stage_id,
                )
            result = service.diagnose_batch_path(manifest_path)
            output = result.model_dump(mode="json")
            return StageResult.succeeded(
                output,
                metadata={"compatibility": True, "mode": "batch"},
            )
        return StageResult.failed(
            code="ASTROGRAPH_UNIFIED_RUNTIME_MODE_INVALID",
            message=f"unsupported unified diagnosis mode: {mode}",
            stage_id=self.stage_id,
        )


class WholeSpacecraftOnlineCompatibilityStage:
    """Adapter exposing the feature-complete online chain through the canonical runtime.

    Heavy domain imports are intentionally lazy so astrograph.runtime has no static
    dependency on integrations, agents, simulation, DOR/FEC/ISGN/M2N, or LLM modules.
    """

    stage_id = "compat.whole_spacecraft_online.v1"
    stage_version = "1.0"

    def __init__(self, workflow_factory: Callable[[], Any] | None = None) -> None:
        self.workflow_factory = workflow_factory

    def _workflow(self) -> Any:
        if self.workflow_factory is not None:
            return self.workflow_factory()
        module = import_module("astrograph.orchestration.whole_spacecraft_online_workflow")
        return module.WholeSpacecraftOnlineDiagnosisWorkflow()

    def execute(self, payload: dict[str, Any], context: RuntimeStageContext) -> StageResult:
        orchestration_models = import_module("astrograph.orchestration.models")
        if "envelope" not in payload or "context" not in payload:
            return StageResult.failed(
                code="ASTROGRAPH_ONLINE_RUNTIME_INPUT_INVALID",
                message="whole_spacecraft.online.v1 requires envelope and context objects",
                stage_id=self.stage_id,
            )
        envelope = orchestration_models.TaskEnvelope.model_validate(payload["envelope"])
        workflow_context = orchestration_models.WorkflowContext.model_validate(payload["context"])
        execution = self._workflow().invoke(envelope, workflow_context)
        result_payload = execution.model_dump(mode="json")
        status_value = getattr(execution.status, "value", str(execution.status))
        if status_value != "SUCCEEDED":
            error = getattr(execution, "error", None)
            details = error.model_dump(mode="json") if error is not None and hasattr(error, "model_dump") else {}
            return StageResult.failed(
                code=details.get("code", "ASTROGRAPH_ONLINE_WORKFLOW_FAILED"),
                message=details.get("message", "Whole-spacecraft online workflow failed."),
                stage_id=self.stage_id,
                details={"execution_result": result_payload},
            )
        return StageResult.succeeded(
            result_payload,
            warnings=list(getattr(execution, "warnings", []) or []),
            metadata={"compatibility": True, "source_workflow_id": getattr(execution, "workflow_id", "")},
        )


WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS: tuple[str, ...] = (
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


class WholeSpacecraftOnlineDagStage:
    """One independent artifact-backed node in the online diagnosis DAG.

    Heavy domain imports remain lazy. Unlike post4/post5, this Stage never owns or
    advances a shared workflow generator; every invocation consumes only the explicit
    RuntimeArtifactStore inputs declared by the pipeline dependency graph.
    """

    stage_version = "3.0"

    def __init__(
        self,
        stage_id: str,
        workflow_factory: Callable[[], Any] | None = None,
        *,
        final_stage: bool = False,
    ) -> None:
        if stage_id not in WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS:
            raise ValueError(f"unknown whole-spacecraft online DAG stage: {stage_id}")
        self.stage_id = stage_id
        self.workflow_factory = workflow_factory
        self.final_stage = final_stage

    def _workflow(self) -> Any:
        if self.workflow_factory is not None:
            return self.workflow_factory()
        module = import_module("astrograph.orchestration.whole_spacecraft_online_workflow")
        return module.WholeSpacecraftOnlineDiagnosisWorkflow()

    def execute(self, payload: dict[str, Any], context: RuntimeStageContext) -> StageResult:
        orchestration_models = import_module("astrograph.orchestration.models")
        if "envelope" not in payload or "context" not in payload:
            return StageResult.failed(
                code="ASTROGRAPH_ONLINE_RUNTIME_INPUT_INVALID",
                message="whole_spacecraft.online.v1 requires envelope and context objects",
                stage_id=self.stage_id,
            )
        envelope = orchestration_models.TaskEnvelope.model_validate(payload["envelope"])
        workflow_context = orchestration_models.WorkflowContext.model_validate(payload["context"])
        workflow = self._workflow()
        try:
            outcome = workflow.execute_runtime_stage(
                stage_id=self.stage_id,
                envelope=envelope,
                context=workflow_context,
                artifact_store=context.artifacts,
            )
        except Exception as exc:
            code = getattr(exc, "code", "ASTROGRAPH_ONLINE_WORKFLOW_FAILED")
            message = getattr(exc, "message", f"{type(exc).__name__}: {exc}")
            details = getattr(exc, "details", {})
            return StageResult.failed(
                code=code, message=message, stage_id=self.stage_id, details=details
            )

        metadata = dict(outcome.metadata)
        if self.final_stage:
            execution = outcome.execution_result
            if execution is None:
                return StageResult.failed(
                    code="ASTROGRAPH_ONLINE_DAG_NO_RESULT",
                    message="online DAG persistence stage completed without an ExecutionResult",
                    stage_id=self.stage_id,
                )
            result_payload = execution.model_dump(mode="json")
            status_value = getattr(execution.status, "value", str(execution.status))
            if status_value != "SUCCEEDED":
                error = getattr(execution, "error", None)
                error_details = (
                    error.model_dump(mode="json")
                    if error is not None and hasattr(error, "model_dump") else {}
                )
                return StageResult.failed(
                    code=error_details.get("code", "ASTROGRAPH_ONLINE_WORKFLOW_FAILED"),
                    message=error_details.get("message", "Whole-spacecraft online DAG failed."),
                    stage_id=self.stage_id,
                    details={"execution_result": result_payload},
                )
            metadata.update({
                "workflow_id": getattr(execution, "workflow_id", ""),
                "dag_complete": True,
                "checkpoint_status": outcome.status,
            })
            return StageResult.succeeded(
                result_payload,
                warnings=list(dict.fromkeys([*outcome.warnings, *list(execution.warnings or [])])),
                metadata=metadata,
                artifact=outcome.artifact,
            )

        if outcome.status == "SKIPPED":
            return StageResult(
                status=StageStatus.SKIPPED,
                payload=payload,
                warnings=outcome.warnings,
                metadata=metadata,
                artifact=outcome.artifact,
            )
        return StageResult.succeeded(
            payload, warnings=outcome.warnings, metadata=metadata, artifact=outcome.artifact
        )

def build_whole_spacecraft_online_dag_stages(
    workflow_factory: Callable[[], Any] | None = None,
) -> tuple[WholeSpacecraftOnlineDagStage, ...]:
    return tuple(
        WholeSpacecraftOnlineDagStage(
            stage_id,
            workflow_factory,
            final_stage=(stage_id == WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS[-1]),
        )
        for stage_id in WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS
    )
