from __future__ import annotations

from pathlib import Path
from importlib import import_module
from typing import Any, Callable

from .contracts import RuntimeStageContext
from .execution_control import CancellationMode, ExecutionBoundary, StageExecutionProfile, StageSideEffect
from .core import AstroGraphRuntime, RuntimePipeline
from .stages import (
    DeterministicDiagnosisCompatibilityStage,
    UnifiedDiagnosisCompatibilityStage,
    WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS,
    build_whole_spacecraft_online_dag_stages,
)


DIAGNOSIS_PIPELINE_ID = "diagnosis.compat.v1"
UNIFIED_DIAGNOSIS_PIPELINE_ID = "diagnosis.unified.compat.v1"
WHOLE_SPACECRAFT_PIPELINE_ID = "whole_spacecraft.online.v1"


WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "online.prepare.v1": (),
    "online.semantics.v1": ("online.prepare.v1",),
    "online.extension.control_plane.v1": ("online.prepare.v1", "online.semantics.v1"),
    "online.evidence.v1": (
        "online.prepare.v1", "online.semantics.v1", "online.extension.control_plane.v1"
    ),
    "online.spatiotemporal.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.extension.semantic.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.extension.mechanism.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.kg_evidence.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.data_driven.v1": ("online.prepare.v1", "online.evidence.v1"),
    "online.extension.calibration.v1": ("online.prepare.v1", "online.data_driven.v1"),
    "online.fusion.v1": (
        "online.prepare.v1", "online.evidence.v1", "online.spatiotemporal.v1",
        "online.kg_evidence.v1", "online.data_driven.v1", "online.extension.calibration.v1",
    ),
    "online.verification.v1": ("online.evidence.v1",),
    "online.reasoning.v1": (
        "online.prepare.v1", "online.evidence.v1", "online.fusion.v1", "online.verification.v1",
        "online.deterministic_core.v1",
    ),
    "online.critique.v1": ("online.evidence.v1", "online.reasoning.v1"),
    "online.deterministic_core.v1": ("online.evidence.v1",),
    "online.governor.v1": (
        "online.evidence.v1", "online.reasoning.v1", "online.critique.v1",
        "online.deterministic_core.v1",
    ),
    "online.policy_gate.v1": (
        "online.prepare.v1", "online.evidence.v1", "online.governor.v1",
        "online.extension.control_plane.v1", "online.extension.semantic.v1",
        "online.extension.mechanism.v1", "online.extension.calibration.v1",
        "online.kg_evidence.v1", "online.fusion.v1",
    ),
    "online.evidence_graph.v1": (
        "online.evidence.v1", "online.reasoning.v1", "online.governor.v1", "online.policy_gate.v1"
    ),
    "online.persist.v1": tuple(WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS[:-1]),
}

if set(WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES) != set(WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS):
    raise RuntimeError("whole-spacecraft online DAG dependency map is incomplete")


WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES: tuple[str, ...] = (
    "online.spatiotemporal.v1",
    "online.verification.v1",
    "online.deterministic_core.v1",
)

WHOLE_SPACECRAFT_PROVIDER_QUALIFICATION_IDS: dict[str, tuple[str, ...]] = {
    "online.extension.control_plane.v1": ("external-cli-managed.v1",),
    "online.extension.semantic.v1": ("external-cli-managed.v1",),
    "online.extension.mechanism.v1": ("external-cli-managed.v1",),
    "online.extension.calibration.v1": ("external-cli-managed.v1",),
    "online.kg_evidence.v1": ("neo4j-readonly.v1", "kg-csv-reference.v1"),
    "online.data_driven.v1": ("inprocess-model-runtime.v1", "http-json-provider.v1"),
    "online.reasoning.v1": ("http-json-provider.v1",),
}



def build_whole_spacecraft_online_execution_profiles(
    *,
    hard_isolation_available: bool,
) -> dict[str, StageExecutionProfile]:
    profiles: dict[str, StageExecutionProfile] = {
        stage_id: StageExecutionProfile(
            stage_id=stage_id,
            parallel_safe=False,
            replay_equivalence_certified=False,
            payload_passthrough=True,
            cancellation_mode=CancellationMode.COOPERATIVE,
            side_effect=(
                StageSideEffect.WRITE_ARTIFACT
                if stage_id == "online.persist.v1"
                else StageSideEffect.READ_ONLY
            ),
            resource_units=1,
            provider_qualification_ids=WHOLE_SPACECRAFT_PROVIDER_QUALIFICATION_IDS.get(stage_id, ()),
            notes="post9 conservative default; provider lifecycle qualification gates any future parallel promotion",
        )
        for stage_id in WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS
    }
    for stage_id in WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES:
        if hard_isolation_available:
            profiles[stage_id] = StageExecutionProfile(
                stage_id=stage_id,
                parallel_safe=True,
                replay_equivalence_certified=True,
                payload_passthrough=True,
                cancellation_mode=CancellationMode.COOPERATIVE,
                execution_boundary=ExecutionBoundary.IN_PROCESS,
                parallel_execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
                isolation_adapter_id="whole_spacecraft.online.v1",
                cancellation_qualification_id="post8.whole_spacecraft.local-branch.hard-isolation.v1",
                provider_qualification_ids=("local-deterministic-isolated.v1",),
                side_effect=StageSideEffect.READ_ONLY,
                resource_units=1,
                notes=(
                    "post8 hard-isolated worker boundary; replay-equivalence certified. "
                    "Stable SAFE_PARALLEL is eligible when explicitly requested."
                ),
            )
        else:
            profiles[stage_id] = StageExecutionProfile(
                stage_id=stage_id,
                parallel_safe=True,
                replay_equivalence_certified=True,
                payload_passthrough=True,
                cancellation_mode=CancellationMode.COOPERATIVE,
                provider_qualification_ids=("local-deterministic-isolated.v1",),
                side_effect=StageSideEffect.READ_ONLY,
                resource_units=1,
                notes=(
                    "post8 isolation unavailable for custom/no-project-root workflow factory; "
                    "cooperative research-mode only"
                ),
            )
    return profiles


# Public reviewed-project profile map. Application runtime construction may downgrade these
# when a custom workflow factory cannot be reconstructed safely inside an isolated worker.
WHOLE_SPACECRAFT_ONLINE_EXECUTION_PROFILES = build_whole_spacecraft_online_execution_profiles(
    hard_isolation_available=True
)



def _reviewed_online_workflow_factory(project_root: Path) -> Callable[[], Any]:
    root = project_root.resolve()

    def factory() -> Any:
        workflow_module = import_module("astrograph.orchestration.whole_spacecraft_online_workflow")
        dataset_module = import_module("astrograph.dataset_profiles")
        policy_module = import_module("astrograph.policy_profiles")
        dataset_registry_path = root / "configs" / "dataset_profiles" / "registry.json"
        policy_registry_path = root / "configs" / "runtime_policies" / "registry.json"
        dataset_registry = (
            dataset_module.DatasetProfileRegistry(index_path=dataset_registry_path)
            if dataset_registry_path.is_file()
            else None
        )
        policy_registry = (
            policy_module.RuntimePolicyRegistry(policy_registry_path)
            if policy_registry_path.is_file()
            else None
        )
        return workflow_module.WholeSpacecraftOnlineDiagnosisWorkflow(
            dataset_profile_registry=dataset_registry,
            runtime_policy_registry=policy_registry,
        )

    return factory


def build_application_runtime(
    *,
    repository: Any,
    policy: Any,
    audit_store: Any,
    constraints: Any = None,
    project_root: str | Path | None = None,
    online_workflow_factory: Callable[[], Any] | None = None,
    include_online_pipeline: bool = True,
) -> AstroGraphRuntime:
    pipelines = [
        RuntimePipeline(
            pipeline_id=DIAGNOSIS_PIPELINE_ID,
            stages=(
                DeterministicDiagnosisCompatibilityStage(
                    repository=repository,
                    policy=policy,
                    audit_store=audit_store,
                    constraints=constraints,
                ),
            ),
            description="Legacy deterministic evidence/KG diagnosis behind the canonical runtime.",
        ),
        RuntimePipeline(
            pipeline_id=UNIFIED_DIAGNOSIS_PIPELINE_ID,
            stages=(UnifiedDiagnosisCompatibilityStage(),),
            description="Legacy source-specific unified diagnosis behind the canonical runtime compatibility perimeter.",
        ),
    ]
    if include_online_pipeline:
        effective_online_factory = online_workflow_factory
        if effective_online_factory is None and project_root is not None:
            effective_online_factory = _reviewed_online_workflow_factory(Path(project_root))
        pipelines.append(
            RuntimePipeline(
                pipeline_id=WHOLE_SPACECRAFT_PIPELINE_ID,
                stages=build_whole_spacecraft_online_dag_stages(effective_online_factory),
                description=(
                    "Fine-grained whole-spacecraft online diagnosis DAG behind the canonical runtime; "
                    "DOR/FEC/ISGN nodes are optional extension gates and M2N is not an online dependency."
                ),
                dependencies=WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES,
                execution_profiles=build_whole_spacecraft_online_execution_profiles(
                    hard_isolation_available=(
                        project_root is not None and online_workflow_factory is None
                    )
                ),
            )
        )
    return AstroGraphRuntime(
        pipelines,
        context=RuntimeStageContext(
            project_root=Path(project_root).resolve() if project_root is not None else None,
        ),
    )
