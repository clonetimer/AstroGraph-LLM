from __future__ import annotations

import numpy as np
import pytest

from astrograph.assurance import AdmissionError, canonical_sha256
from astrograph.data_driven import (
    DataDrivenModelManifest,
    DataDrivenTaskType,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ModelRuntime,
    ObservationScope,
)
from astrograph.data_driven.preprocessing import ChannelPreprocessor
from astrograph.integrations.phm_three_task_torch import (
    PHMThreeTaskTorchProvider,
    build_registry_from_phm_three_task_provider,
    load_phm_three_task_torch_package,
    save_phm_three_task_torch_package,
)


def _provider():
    import torch
    from torch import nn

    class FixedModel(nn.Module):
        def __init__(self, outputs):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.outputs = outputs

        def forward(self, value):
            return tuple(
                torch.as_tensor(output, dtype=value.dtype, device=value.device).repeat(len(value), 1)
                + self.anchor * 0.0
                for output in self.outputs
            )

    manifest = DataDrivenModelManifest(
        model_id="phm-three-task-test",
        model_version="1",
        display_name="PHM three-task test",
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION,
        runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS,
                      ModelCapability.TEMPORAL_CONTEXT],
        supported_fault_ids=["THERMAL_HEATER_STUCK_ON"],
        feature_schema_ids=["phm-v60-causal-window-v1"],
        input_contract={"window_length": 4, "feature_order": ["a", "b"]},
    )
    preprocessor = ChannelPreprocessor(
        method="standard", impute=np.zeros(2, np.float32), center=np.zeros(2, np.float32),
        scale=np.ones(2, np.float32), lower=np.full(2, -np.inf, np.float32),
        upper=np.full(2, np.inf, np.float32), normalized_clip=None,
    )
    return PHMThreeTaskTorchProvider(
        manifest=manifest,
        detector_model=FixedModel(([0.0, 2.0], [0.0, 0.0], [0.0, 0.0])),
        diagnosis_model=FixedModel(([0.0, 0.0], [0.0, 2.0], [0.0, 3.0])),
        preprocessor=preprocessor,
        selected_channels=("a", "b"),
        class_names=("thermal_other", "thermal_heater_stuck_on"),
        component_names=("battery", "heater"),
        fault_to_component={"thermal_other": "heater", "thermal_heater_stuck_on": "heater"},
        class_map={"thermal_heater_stuck_on": "THERMAL_HEATER_STUCK_ON"},
        runtime_fault_ids=frozenset({"THERMAL_HEATER_STUCK_ON"}),
        detection_selection={
            "detector": "deep", "detection_threshold": 0.6,
            "detection_score_normalization": "none", "alarm_minimum_consecutive": 1,
        },
        diagnosis_selection={"hierarchy_blend_alpha": 0.5},
    )


def _request(order=("a", "b")):
    values = [[0.0, 1.0]] * 4
    payload = {"channel_order": list(order), "telemetry_window": values}
    return ModelInferenceRequest(
        episode_id="episode-1", model_id="phm-three-task-test", model_version="1",
        feature_schema_id="phm-v60-causal-window-v1",
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=["THERMAL_HEATER_STUCK_ON"],
        payload=payload,
        observation_scope=ObservationScope(
            scope_id="phm-test-window-1", source_id="phm-test-source",
            window_start_s=0.0, window_end_s=3.0, trajectory_group_id="trajectory-1",
        ),
    )


def test_provider_runs_through_common_registry_adapter():
    prediction = build_registry_from_phm_three_task_provider(_provider()).predict(_request())
    probabilities = {item.class_id: item.probability for item in prediction.classes}
    assert probabilities["THERMAL_HEATER_STUCK_ON"] > probabilities["NORMAL"]
    assert prediction.metadata["raw_output_metadata"]["predicted_component"] == "heater"
    assert prediction.metadata["raw_output_metadata"]["specialist_contract"].startswith("causal")


def test_provider_rejects_runtime_channel_reordering():
    registry = build_registry_from_phm_three_task_provider(_provider())
    with pytest.raises(ValueError, match="channel order"):
        registry.predict(_request(("b", "a")))


def test_provider_rejects_a_selected_logistic_detector_without_its_artifact():
    provider = _provider()
    provider.detection_selection = {"detector": "logistic", "detection_threshold": 0.5}
    with pytest.raises(ValueError, match="selected deep detector"):
        provider.__post_init__()


def test_provider_abstains_when_dataset_class_has_no_runtime_mapping():
    provider = _provider()
    provider.class_map = {}
    prediction = build_registry_from_phm_three_task_provider(provider).predict(_request())
    probabilities = {item.class_id: item.probability for item in prediction.classes}
    assert probabilities == {"NORMAL": 1.0, "THERMAL_HEATER_STUCK_ON": 0.0}
    assert prediction.out_of_distribution_score == 1.0
    assert prediction.metadata["raw_output_metadata"]["fusion_abstained"] is True


def test_weights_only_package_round_trip_and_hash_pin(tmp_path):
    from astrograph.data_driven.three_task_tcn import (
        make_causal_multiview_detector_tcn,
        make_hierarchical_mechanism_gat_tcn,
    )

    provider = _provider()
    provider.detector_model = make_causal_multiview_detector_tcn(2, 2, 2, window_size=4)
    provider.diagnosis_model = make_hierarchical_mechanism_gat_tcn(
        2, 2, 2, component_groups=[[0], [1]], subsystem_groups=[[0, 1]],
        component_subsystem=[0, 0], class_component=[1, 1], subsystem_edges=[],
    )
    contract = {
        "component_groups": [[0], [1]], "subsystem_groups": [[0, 1]],
        "component_subsystem": [0, 0], "class_component": [1, 1],
        "subsystem_edges": [],
    }
    path = tmp_path / "model.pt"
    digest = save_phm_three_task_torch_package(
        provider, path, hierarchical_contract=contract
    )
    loaded = load_phm_three_task_torch_package(path, expected_sha256=digest)
    registry = build_registry_from_phm_three_task_provider(loaded)
    assert registry.require_admission is True
    prediction = registry.predict(_request())
    assert sum(item.probability for item in prediction.classes) == pytest.approx(1.0)
    assert prediction.metadata["model_admission"]["feature_contract_id"]
    assert prediction.metadata["evidence_lineage_by_fault"]["THERMAL_HEATER_STUCK_ON"]
    with pytest.raises(AdmissionError, match="frozen literal"):
        registry.predict(_request(("b", "a")))
    with pytest.raises(ValueError, match="SHA-256"):
        load_phm_three_task_torch_package(path, expected_sha256="0" * 64)


def test_strict_registry_rejects_unpinned_in_memory_provider():
    with pytest.raises(ValueError, match="hash-pinned"):
        build_registry_from_phm_three_task_provider(_provider(), require_admission=True)
