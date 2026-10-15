from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astrograph.object_plugins import build_builtin_object_plugin_registry
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, OperatorInput
from astrograph.orchestration.models import PermissionScope, WorkflowContext
from astrograph.plugin_factory import (
    ConfigDrivenFeatureService,
    EvidenceServiceSpec,
    FeatureImplementationMode,
    FeatureServiceSpec,
    ObjectPluginBuildSpec,
    ObjectPluginFactoryService,
    PluginGenerationPlanner,
    TelemetryBindingSpec,
)
from astrograph.tools.object_plugins import build_object_plugin_tool_registry


def _build_spec() -> ObjectPluginBuildSpec:
    return ObjectPluginBuildSpec(
        plugin_id="generated.test.power_amplifier",
        plugin_version="0.1.0",
        object_type="solid_state_power_amplifier",
        subsystem="PAYLOAD",
        dataset_profiles=["SSPA_SIM"],
        object_parameters={"nominal_gain_db": 30.0},
        telemetry_bindings=[
            TelemetryBindingSpec(
                binding_id="output_power",
                source_field="payload.sspa.output_power_dbm",
                unit="dBm",
                semantic_role="response",
            ),
            TelemetryBindingSpec(
                binding_id="drain_current",
                source_field="payload.sspa.drain_current_a",
                unit="A",
                semantic_role="mechanism_support",
            ),
        ],
        features=[
            FeatureServiceSpec(
                feature_id="FD-SSPA-POWER-RESIDUAL",
                name="Output power residual",
                output_unit="dB",
                input_bindings={"current": "output_power", "baseline": "output_power"},
                operator_id="MEAN_ABS_PAIRED_RESIDUAL",
                source_refs=["SSPA engineering model"],
            ),
            FeatureServiceSpec(
                feature_id="FD-SSPA-COMPOSITE",
                name="Composite degradation indicator",
                output_unit="1",
                input_bindings={"current": "drain_current"},
                operator_pipeline=["MEAN_VALUE", "MAX_VALUE"],
                source_refs=["SSPA engineering model"],
            ),
            FeatureServiceSpec(
                feature_id="FD-SSPA-ELECTROTHERMAL",
                name="Electrothermal fitted parameter",
                output_unit="1",
                input_bindings={"current": "drain_current"},
                custom_formula="fit coupled electrothermal response parameter",
                source_refs=["SSPA engineering model"],
            ),
        ],
        evidence=[
            EvidenceServiceSpec(
                evidence_id="ED-SSPA-POWER-DEGRADATION",
                target_fault_id="SSPA_GAIN_DEGRADATION",
                feature_id="FD-SSPA-POWER-RESIDUAL",
                role="REQUIRED",
                comparison="GREATER_OR_EQUAL",
                threshold=0.5,
                threshold_unit="dB",
                critical=True,
                source_refs=["SSPA engineering model"],
            )
        ],
        source_refs=["SSPA engineering model"],
    )


def test_builtin_operator_registry_covers_existing_feature_aggregations():
    expected = {
        "MAX_ABS_PAIRED_RESIDUAL",
        "MEAN_ABS_PAIRED_RESIDUAL",
        "MEAN_PAIRED_RESIDUAL",
        "LINEAR_SLOPE",
        "LAST_PAIRED_RESIDUAL",
        "AVAILABILITY_DROP_FRACTION",
        "MAX_ABS_STATUS_RESIDUAL",
        "MAX_VALUE",
        "MIN_VALUE",
        "MEAN_VALUE",
        "TRUE_FRACTION",
        "FALSE_FRACTION",
        "RMS_VALUE",
        "STD_VALUE",
        "RANGE_VALUE",
        "HALF_WINDOW_MEAN_DIFFERENCE",
        "MEAN_PAIRED_RATIO",
        "WEIGHTED_MEAN",
        "JOINT_THRESHOLD_FRACTION",
    }
    assert {item.operator_id for item in BUILTIN_OPERATOR_REGISTRY.list_specs()} == expected
    assert BUILTIN_OPERATOR_REGISTRY.compute(
        "MAX_ABS_PAIRED_RESIDUAL",
        OperatorInput(current=[1.0, 4.0], baseline=[1.0, 1.0]),
    ) == pytest.approx(3.0)
    assert BUILTIN_OPERATOR_REGISTRY.compute(
        "LINEAR_SLOPE",
        OperatorInput(current=[0.0, 2.0, 4.0], time_s=[0.0, 1.0, 2.0]),
    ) == pytest.approx(2.0)


def test_same_config_driven_service_supports_different_object_types():
    service = ConfigDrivenFeatureService()
    rw_spec = FeatureServiceSpec(
        feature_id="FD-RW-SPEED",
        name="RW speed residual",
        output_unit="rad/s",
        input_bindings={"current": "speed", "baseline": "speed"},
        operator_id="MAX_ABS_PAIRED_RESIDUAL",
    )
    sspa_spec = FeatureServiceSpec(
        feature_id="FD-SSPA-POWER",
        name="SSPA power residual",
        output_unit="dB",
        input_bindings={"current": "power", "baseline": "power"},
        operator_id="MAX_ABS_PAIRED_RESIDUAL",
    )
    rw = service.compute(
        rw_spec,
        current_signals={"speed": np.array([10.0, 11.0, 13.0])},
        baseline_signals={"speed": np.array([10.0, 10.0, 10.0])},
    )
    sspa = service.compute(
        sspa_spec,
        current_signals={"power": np.array([30.0, 29.5, 28.0])},
        baseline_signals={"power": np.array([30.0, 30.0, 30.0])},
    )
    assert rw.value == pytest.approx(3.0)
    assert sspa.value == pytest.approx(2.0)
    assert rw.operator_id == sspa.operator_id


def test_generation_planner_distinguishes_config_composition_and_custom_code():
    plan = PluginGenerationPlanner().plan(_build_spec())
    modes = {item.feature_id: item.mode for item in plan.decisions}
    assert modes["FD-SSPA-POWER-RESIDUAL"] == FeatureImplementationMode.CONFIGURATION_ONLY
    assert modes["FD-SSPA-COMPOSITE"] == FeatureImplementationMode.OPERATOR_COMPOSITION
    assert modes["FD-SSPA-ELECTROTHERMAL"] == FeatureImplementationMode.CUSTOM_CODE_REQUIRED
    assert plan.configuration_only_feature_count == 1
    assert plan.composition_feature_count == 1
    assert plan.custom_code_feature_count == 1
    assert plan.requires_human_review is True
    assert plan.online_registration_allowed is False


def test_factory_generates_reviewable_non_executable_scaffold(tmp_path: Path):
    result = ObjectPluginFactoryService().generate(_build_spec(), tmp_path)
    package_root = Path(str(result["package_root"]))
    validation = result["validation"]
    assert validation["valid"] is True
    assert validation["online_registration_allowed"] is False
    assert validation["review_status"] == "REVIEW_REQUIRED"
    assert (package_root / "REVIEW_REQUIRED.md").exists()
    manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["plugin_id"] == "generated.test.power_amplifier"
    plugin_source = (package_root / "plugin.py").read_text(encoding="utf-8")
    assert "ONLINE_REGISTRATION_ALLOWED = False" in plugin_source
    assert "raise RuntimeError" in plugin_source


def test_builtin_reaction_wheel_plugin_is_registered_and_operator_validated():
    registry = build_builtin_object_plugin_registry()
    manifest = registry.get("builtin.d3.reaction_wheel").manifest
    assert manifest.object_type == "reaction_wheel"
    assert manifest.deterministic_runtime is True
    assert set(manifest.required_operator_ids).issubset(
        {item.operator_id for item in BUILTIN_OPERATOR_REGISTRY.list_specs()}
    )


def test_generic_object_plugin_tools_are_separate_read_only_allow_list():
    registry = build_object_plugin_tool_registry()
    specs = {item.tool_id: item for item in registry.list_specs()}
    assert set(specs) == {"object_plugin.list", "object_plugin.evidence.build"}
    assert all(item.deterministic for item in specs.values())
    assert all(item.side_effect_level.value == "READ_ONLY" for item in specs.values())
    context = WorkflowContext(
        workflow_id="object-plugin-test",
        run_id="run",
        permissions=[PermissionScope.READ_KNOWLEDGE],
    )
    listed = registry.invoke("object_plugin.list", {}, context)
    assert listed["plugins"][0]["plugin_id"] == "builtin.d3.reaction_wheel"

from astrograph.orchestration.models import ExecutionPolicy, ExecutionStatus, TaskEnvelope
from astrograph.orchestration.object_plugin_factory_workflow import ObjectPluginFactoryWorkflow


def _factory_context() -> WorkflowContext:
    return WorkflowContext(
        workflow_id=ObjectPluginFactoryWorkflow.workflow_id,
        run_id="factory-run",
        permissions=[PermissionScope.READ_KNOWLEDGE, PermissionScope.WRITE_ARTIFACTS],
    )


def test_object_plugin_factory_workflow_uses_agent_spec_but_never_executes_generated_code(tmp_path: Path):
    envelope = TaskEnvelope(
        task_id="plugin-factory",
        task_type="object_plugin_factory",
        input_payload={
            "output_root": str(tmp_path),
            "object_context": {"object_type": "solid_state_power_amplifier"},
            "draft_build_spec": _build_spec().model_dump(mode="json"),
        },
    )
    result = ObjectPluginFactoryWorkflow().invoke(envelope, _factory_context())
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["online_registration_allowed"] is False
    assert result.metadata["generated_code_executed"] is False
    assert result.metadata["design_backend"] == "deterministic-draft"


def test_object_plugin_design_backend_is_policy_gated(tmp_path: Path):
    class FakeBackend:
        model_id = "fake-designer"
        model_version = "1"

        def generate_structured(self, *, agent_id, output_model, payload):
            return output_model.model_validate(_build_spec().model_dump(mode="json"))

    workflow = ObjectPluginFactoryWorkflow(design_backend=FakeBackend())
    payload = {
        "output_root": str(tmp_path),
        "object_context": {"object_type": "solid_state_power_amplifier"},
    }
    blocked = workflow.invoke(
        TaskEnvelope(task_id="blocked", task_type="factory", input_payload=payload),
        _factory_context(),
    )
    assert blocked.status == ExecutionStatus.FAILED
    assert blocked.error.code == "ASTROGRAPH_OBJECT_PLUGIN_DESIGN_LLM_BLOCKED"

    allowed = workflow.invoke(
        TaskEnvelope(
            task_id="allowed",
            task_type="factory",
            input_payload=payload,
            execution_policy=ExecutionPolicy(deterministic_only=False, allow_llm=True),
        ),
        _factory_context(),
    )
    assert allowed.status == ExecutionStatus.SUCCEEDED
    assert allowed.metadata["design_backend"] == "fake-designer:1"
    assert allowed.metadata["generated_code_executed"] is False

import importlib.util


def test_generated_direct_feature_service_is_executable_but_not_registerable(tmp_path: Path):
    result = ObjectPluginFactoryService().generate(_build_spec(), tmp_path)
    package_root = Path(str(result["package_root"]))
    module_spec = importlib.util.spec_from_file_location(
        "generated_sspa_plugin", package_root / "plugin.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    value = module.compute_feature(
        "FD-SSPA-POWER-RESIDUAL",
        current_signals={"output_power": [30.0, 29.0, 28.0]},
        baseline_signals={"output_power": [30.0, 30.0, 30.0]},
    )
    assert value["value"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="not an approved online object plugin"):
        module.register(None)
    composite = module.compute_feature(
        "FD-SSPA-COMPOSITE",
        current_signals={"drain_current": [1.0, 2.0]},
    )
    assert composite["value"] == pytest.approx(1.5)
    assert composite["implementation_mode"] == "OPERATOR_COMPOSITION"
    with pytest.raises(RuntimeError, match="reviewed custom code"):
        module.compute_feature(
            "FD-SSPA-ELECTROTHERMAL",
            current_signals={"drain_current": [1.0, 2.0]},
        )
