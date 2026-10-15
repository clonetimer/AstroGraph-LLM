from __future__ import annotations

from astrograph.agents import DiagnosisPlannerAgent, DiagnosisPlanningContext, DiagnosisStage
from astrograph.data_driven import (
    CallableRawModelProvider,
    DataDrivenModelManifest,
    DataDrivenModelRegistry,
    DataDrivenTaskType,
    FaultModelBinding,
    FaultModelBindingRegistry,
    ModelCapability,
    ModelOutputSemantics,
    ModelRuntime,
    ModelSelectionRequest,
    ModelSelectionStatus,
    RawModelOutput,
)


def _provider(model_id: str, faults: list[str], runtime: ModelRuntime = ModelRuntime.CUSTOM_CALLABLE):
    manifest = DataDrivenModelManifest(
        model_id=model_id,
        model_version="1.0.0",
        display_name=model_id,
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=runtime,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS],
        supported_fault_ids=faults,
        feature_schema_ids=["FEATURE-TS-V1"],
    )

    def predict(request):
        values = {fault_id: 0.0 for fault_id in faults}
        values["NORMAL"] = 1.0
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            class_values=values,
        )

    return CallableRawModelProvider(manifest, predict)


def test_fault_model_binding_selects_explicit_high_coverage_model() -> None:
    models = DataDrivenModelRegistry.for_research()
    models.register(_provider("adcs-specific", ["ADCS_RW_JAM", "ADCS_RW_FRICTION_INCREASE"]))
    models.register(_provider("generic", ["ADCS_RW_JAM", "ADCS_RW_FRICTION_INCREASE", "EPS_BATTERY_CAPACITY_LOSS"]))
    bindings = FaultModelBindingRegistry(
        [
            FaultModelBinding(fault_id="ADCS_RW_JAM", model_id="adcs-specific", priority=250),
            FaultModelBinding(fault_id="ADCS_RW_FRICTION_INCREASE", model_id="adcs-specific", priority=250),
        ]
    )

    decision = bindings.select(
        ModelSelectionRequest(
            candidate_fault_ids=["ADCS_RW_JAM", "ADCS_RW_FRICTION_INCREASE"],
            feature_schema_id="FEATURE-TS-V1",
        ),
        models,
    )

    assert decision.status == ModelSelectionStatus.SELECTED
    assert decision.selected_model_id == "adcs-specific"
    assert decision.candidates[0].coverage == 1.0
    assert decision.candidates[0].used_unbound_fallback is False
    assert decision.requires_human_review is False


def test_model_selection_marks_generic_unbound_fallback_for_review() -> None:
    models = DataDrivenModelRegistry.for_research()
    models.register(_provider("generic", ["EPS_BATTERY_CAPACITY_LOSS"]))
    decision = FaultModelBindingRegistry().select(
        ModelSelectionRequest(
            candidate_fault_ids=["EPS_BATTERY_CAPACITY_LOSS"],
            feature_schema_id="FEATURE-TS-V1",
        ),
        models,
    )

    assert decision.status == ModelSelectionStatus.SELECTED
    assert decision.selected_model_id == "generic"
    assert decision.candidates[0].used_unbound_fallback is True
    assert decision.requires_human_review is True
    assert "UNBOUND_GENERIC_FALLBACK_USED" in decision.reasons


def test_diagnosis_planner_enables_available_branches_and_preserves_gates() -> None:
    plan = DiagnosisPlannerAgent().plan(
        DiagnosisPlanningContext(
            reasoner_mode="hybrid",
            has_data_driven_inference_request=True,
            data_model_registry_available=True,
            has_kg_evidence_binding=True,
            has_dor_control_plane_audit=True,
        )
    )

    assert DiagnosisStage.RUN_DATA_DRIVEN_MODEL in plan.enabled_stages
    assert DiagnosisStage.QUERY_KNOWLEDGE_GRAPH in plan.enabled_stages
    assert DiagnosisStage.APPLY_DIAGNOSIS_POLICY in plan.enabled_stages
    assert DiagnosisStage.QUALIFY_DIAGNOSTIC_CONTROL_PLANE in plan.enabled_stages
    assert "LLM_REASONING_REMAINS_ADVISORY_AND_POLICY_GATED" in plan.warnings
    assert plan.requires_human_review is False


def test_diagnosis_planner_blocks_non_executable_model_branch() -> None:
    plan = DiagnosisPlannerAgent().plan(
        DiagnosisPlanningContext(
            has_data_driven_inference_request=True,
            data_model_registry_available=False,
            has_kg_evidence_binding=False,
        )
    )
    model_step = next(item for item in plan.steps if item.stage == DiagnosisStage.RUN_DATA_DRIVEN_MODEL)
    assert model_step.enabled is False
    assert "DATA_MODEL_REGISTRY_REQUIRED" in plan.warnings
    assert plan.requires_human_review is True
