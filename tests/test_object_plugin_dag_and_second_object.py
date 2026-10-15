from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.object_plugins import D3ThermalHeaterObjectPlugin, build_builtin_object_plugin_registry
from astrograph.plugin_factory import (
    ConfigDrivenFeatureService,
    FeatureImplementationMode,
    FeatureServiceSpec,
    ObjectPluginBuildSpec,
    OperatorDataRef,
    OperatorDataSourceKind,
    OperatorGraphCompiler,
    OperatorGraphNodeSpec,
    OperatorGraphSpec,
    PluginGenerationPlanner,
)


ROOT = Path(__file__).resolve().parents[1]


def _explicit_graph_feature() -> FeatureServiceSpec:
    return FeatureServiceSpec(
        feature_id="FD-GRAPH-FUSED-RESIDUAL",
        name="Fused output residual",
        output_unit="dB",
        input_bindings={"current": "power", "baseline": "power"},
        operator_graph=OperatorGraphSpec(
            nodes=[
                OperatorGraphNodeSpec(
                    node_id="mean_residual",
                    operator_id="MEAN_ABS_PAIRED_RESIDUAL",
                    current_inputs=[
                        OperatorDataRef(
                            source=OperatorDataSourceKind.CURRENT_SIGNAL,
                            ref="power",
                        )
                    ],
                    baseline_inputs=[
                        OperatorDataRef(
                            source=OperatorDataSourceKind.BASELINE_SIGNAL,
                            ref="power",
                        )
                    ],
                ),
                OperatorGraphNodeSpec(
                    node_id="max_residual",
                    operator_id="MAX_ABS_PAIRED_RESIDUAL",
                    current_inputs=[
                        OperatorDataRef(
                            source=OperatorDataSourceKind.CURRENT_SIGNAL,
                            ref="power",
                        )
                    ],
                    baseline_inputs=[
                        OperatorDataRef(
                            source=OperatorDataSourceKind.BASELINE_SIGNAL,
                            ref="power",
                        )
                    ],
                ),
                OperatorGraphNodeSpec(
                    node_id="fused",
                    operator_id="WEIGHTED_MEAN",
                    current_inputs=[
                        OperatorDataRef(
                            source=OperatorDataSourceKind.NODE_OUTPUT,
                            ref="mean_residual",
                        ),
                        OperatorDataRef(
                            source=OperatorDataSourceKind.NODE_OUTPUT,
                            ref="max_residual",
                        ),
                    ],
                    parameters={"weights": [0.25, 0.75]},
                ),
            ],
            output_node_id="fused",
        ),
    )


def test_explicit_operator_dag_executes_topologically_and_records_trace():
    feature = _explicit_graph_feature()
    result = ConfigDrivenFeatureService().compute(
        feature,
        current_signals={"power": [30.0, 29.0, 28.0]},
        baseline_signals={"power": [30.0, 30.0, 30.0]},
    )
    assert result.value == pytest.approx(1.75)
    assert result.implementation_mode == FeatureImplementationMode.OPERATOR_COMPOSITION
    assert [item.node_id for item in result.operator_trace] == [
        "max_residual",
        "mean_residual",
        "fused",
    ]
    assert result.quality_flags == []


def test_operator_dag_cycle_is_blocked_before_scaffolding():
    graph = OperatorGraphSpec(
        nodes=[
            OperatorGraphNodeSpec(
                node_id="a",
                operator_id="MEAN_VALUE",
                current_inputs=[
                    OperatorDataRef(source=OperatorDataSourceKind.NODE_OUTPUT, ref="b")
                ],
            ),
            OperatorGraphNodeSpec(
                node_id="b",
                operator_id="MAX_VALUE",
                current_inputs=[
                    OperatorDataRef(source=OperatorDataSourceKind.NODE_OUTPUT, ref="a")
                ],
            ),
        ],
        output_node_id="b",
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        OperatorGraphCompiler().compile(graph)


def test_thermal_heater_real_scene_replay_needs_no_custom_feature_code():
    spec = ObjectPluginBuildSpec.model_validate_json(
        (ROOT / "examples/object_plugin_specs/thermal_heater_build_spec.json").read_text(
            encoding="utf-8"
        )
    )
    plan = PluginGenerationPlanner().plan(spec)
    assert plan.configuration_only_feature_count == 3
    assert plan.composition_feature_count == 1
    assert plan.custom_code_feature_count == 0
    assert plan.blocked_feature_count == 0

    scene_root = ROOT / "tests/fixtures/spacecraft_scenarios"
    fault = pd.read_csv(scene_root / "heater_stuck_on/telemetry.csv")
    nominal = pd.read_csv(scene_root / "nominal/telemetry.csv")
    event = json.loads(
        (scene_root / "heater_stuck_on/run_manifest.json").read_text(encoding="utf-8")
    )["events"][0]
    start = float(event["start_s"])
    end = start + float(event["duration_s"])
    selected = fault[(fault.time_s >= start) & (fault.time_s <= end)]
    nominal_selected = nominal[nominal.time_s.isin(selected.time_s)]

    current_signals = {
        binding.binding_id: selected[binding.source_field].to_numpy()
        for binding in spec.telemetry_bindings
    }
    baseline_signals = {
        binding.binding_id: nominal_selected[binding.source_field].to_numpy()
        for binding in spec.telemetry_bindings
    }
    service = ConfigDrivenFeatureService()
    values = {
        feature.feature_id: service.compute(
            feature,
            current_signals=current_signals,
            baseline_signals=baseline_signals,
            time_s=selected.time_s.to_numpy(),
        ).value
        for feature in spec.features
    }
    assert values["FD-HEATER-CMD-ACTUAL-MISMATCH-FRACTION-V1"] == pytest.approx(1.0)
    assert values["FD-HEATER-CURRENT-MEAN-V1"] >= 0.2
    assert values["FD-PAYLOAD-TEMP-PAIRED-MAX-V1"] >= 0.05
    assert values["FD-BATTERY-CURRENT-PAIRED-MAX-V1"] >= 0.1


def test_second_builtin_object_plugin_filters_existing_engineering_evidence_bundle():
    examples = json.loads(
        (
            ROOT
            / "tests/fixtures/thermal_star/workflow_examples.json"
        ).read_text(encoding="utf-8")
    )
    source = EpisodeEvidenceBundle.model_validate(
        examples["thermal_heater_stuck_on"]["episode_evidence_bundle"]
    )
    plugin = D3ThermalHeaterObjectPlugin()
    filtered = plugin.filter_bundle(source)
    assert filtered.domain_id == "D3_THERMAL_HEATER"
    assert filtered.analysis_component_instance == "PAYLOAD_HEATER"
    assert len(filtered.feature_definitions) == 4
    assert len(filtered.feature_results) == 4
    assert len(filtered.evidence_definitions) == 4
    assert len(filtered.evidence_facts) == 4
    assert {item.target_fault_id for item in filtered.evidence_facts} == {
        "THERMAL_HEATER_STUCK_ON"
    }

    unsupported = EpisodeEvidenceBundle.model_validate(
        examples["star_tracker_dropout"]["episode_evidence_bundle"]
    )
    with pytest.raises(ValueError, match="not supported"):
        plugin.filter_bundle(unsupported)


def test_builtin_object_plugin_registry_now_contains_two_explicit_objects():
    manifests = build_builtin_object_plugin_registry().list_manifests()
    assert {item.plugin_id for item in manifests} == {
        "builtin.d3.reaction_wheel",
        "builtin.d3.thermal_heater",
    }
    heater = next(item for item in manifests if item.plugin_id.endswith("thermal_heater"))
    assert heater.object_type == "thermal_heater"
    assert set(heater.required_operator_ids) == {
        "JOINT_THRESHOLD_FRACTION",
        "MEAN_VALUE",
        "MAX_ABS_PAIRED_RESIDUAL",
    }


def test_generated_second_object_package_executes_explicit_dag(tmp_path: Path):
    import importlib.util

    from astrograph.plugin_factory import ObjectPluginFactoryService

    spec = ObjectPluginBuildSpec.model_validate_json(
        (ROOT / "examples/object_plugin_specs/thermal_heater_build_spec.json").read_text(
            encoding="utf-8"
        )
    )
    result = ObjectPluginFactoryService().generate(spec, tmp_path)
    package_root = Path(str(result["package_root"]))
    assert result["validation"]["composition_feature_count"] == 1
    assert result["validation"]["custom_code_feature_count"] == 0
    module_spec = importlib.util.spec_from_file_location(
        "generated_thermal_heater_plugin", package_root / "plugin.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    value = module.compute_feature(
        "FD-HEATER-CMD-ACTUAL-MISMATCH-FRACTION-V1",
        current_signals={
            "heater_on": [0.0, 1.0, 1.0],
            "heater_command_on": [0.0, 0.0, 1.0],
        },
    )
    assert value["value"] == pytest.approx(1.0 / 3.0)
    assert value["implementation_mode"] == "OPERATOR_COMPOSITION"
    assert value["operator_trace"][0]["operator_id"] == "JOINT_THRESHOLD_FRACTION"
