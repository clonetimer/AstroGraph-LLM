from __future__ import annotations

import numpy as np
import pytest

from astrograph.assurance import AdmissionError, canonical_sha256
from astrograph.data_driven import (
    DataDrivenModelManifest, DataDrivenTaskType, ModelCapability,
    ModelInferenceRequest, ModelOutputSemantics, ModelRuntime, ObservationScope,
)
from astrograph.data_driven.preprocessing import ChannelPreprocessor
from astrograph.data_driven.three_task_tcn import make_shared_residual_tcn
from astrograph.integrations.adcs_three_task_torch import (
    ADCSThreeTaskTorchProvider, build_registry_from_adcs_three_task_provider,
    load_adcs_three_task_torch_package, save_adcs_three_task_torch_package,
)


def _provider() -> ADCSThreeTaskTorchProvider:
    manifest = DataDrivenModelManifest(
        model_id="adcs-p2-test", model_version="1", display_name="ADCS P2 test",
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION, runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS,
                      ModelCapability.TEMPORAL_CONTEXT],
        supported_fault_ids=["ADCS_RW_JAM"],
        feature_schema_ids=["adcs-p2-window-v1"],
        input_contract={"window_length": 4, "feature_order": ["a", "b"]},
    )
    preprocessor = ChannelPreprocessor(
        method="standard", impute=np.zeros(2, np.float32), center=np.zeros(2, np.float32),
        scale=np.ones(2, np.float32), lower=np.full(2, -np.inf, np.float32),
        upper=np.full(2, np.inf, np.float32), normalized_clip=None,
    )
    return ADCSThreeTaskTorchProvider(
        manifest=manifest, model=make_shared_residual_tcn(2, 2, 2),
        preprocessor=preprocessor, selected_channels=("a", "b"),
        class_names=("adcs_other", "adcs_rw_jamming"),
        component_names=("imu", "reaction_wheel"),
        fault_to_component={"adcs_other": "imu", "adcs_rw_jamming": "reaction_wheel"},
        class_map={"adcs_rw_jamming": "ADCS_RW_JAM"},
        runtime_fault_ids=frozenset({"ADCS_RW_JAM"}),
        detection_selection={"detector_deep_blend_alpha": 0.0,
                             "detection_threshold": 0.6, "alarm_policy_mode": "point"},
        logistic_coefficient=np.zeros(8, np.float64), logistic_intercept=2.0,
        diagnosis_selection={"hierarchy_blend_alpha": 0.5},
    )


def _request(order=("a", "b")) -> ModelInferenceRequest:
    values = [[0.0, 1.0]] * 4
    payload = {"channel_order": list(order), "telemetry_window": values}
    return ModelInferenceRequest(
        episode_id="episode-1", model_id="adcs-p2-test", model_version="1",
        feature_schema_id="adcs-p2-window-v1",
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=["ADCS_RW_JAM"],
        payload=payload,
        observation_scope=ObservationScope(
            scope_id="adcs-test-window-1", source_id="adcs-test-source",
            window_start_s=0.0, window_end_s=3.0, trajectory_group_id="trajectory-1",
        ),
    )


def test_provider_runs_logistic_deep_blend_through_registry():
    prediction = build_registry_from_adcs_three_task_provider(_provider()).predict(_request())
    metadata = prediction.metadata["raw_output_metadata"]
    assert metadata["logistic_detection_probability"] > 0.8
    assert metadata["detected"] is True
    assert metadata["specialist_contract"] == "p2_residual_tcn_plus_logistic_detector"


def test_provider_rejects_channel_reordering_and_stateful_alarm():
    registry = build_registry_from_adcs_three_task_provider(_provider())
    with pytest.raises(ValueError, match="channel order"):
        registry.predict(_request(("b", "a")))
    provider = _provider()
    provider.detection_selection = {**provider.detection_selection, "alarm_policy_mode": "hysteresis"}
    with pytest.raises(ValueError, match="stateful"):
        provider.__post_init__()


def test_provider_abstains_when_predicted_class_is_not_mapped():
    provider = _provider()
    provider.class_map = {}
    prediction = build_registry_from_adcs_three_task_provider(provider).predict(_request())
    assert prediction.out_of_distribution_score == 1.0
    assert prediction.metadata["raw_output_metadata"]["fusion_abstained"] is True


def test_weights_only_package_round_trip_and_hash_pin(tmp_path):
    provider = _provider()
    path = tmp_path / "adcs.pt"
    digest = save_adcs_three_task_torch_package(provider, path)
    loaded = load_adcs_three_task_torch_package(path, expected_sha256=digest)
    registry = build_registry_from_adcs_three_task_provider(loaded)
    assert registry.require_admission is True
    prediction = registry.predict(_request())
    assert sum(item.probability for item in prediction.classes) == pytest.approx(1.0)
    assert prediction.metadata["model_admission"]["feature_contract_id"]
    assert prediction.metadata["evidence_lineage_by_fault"]["ADCS_RW_JAM"]
    with pytest.raises(AdmissionError, match="frozen literal"):
        registry.predict(_request(("b", "a")))
    with pytest.raises(ValueError, match="SHA-256"):
        load_adcs_three_task_torch_package(path, expected_sha256="0" * 64)


def test_strict_registry_rejects_unpinned_in_memory_provider():
    with pytest.raises(ValueError, match="hash-pinned"):
        build_registry_from_adcs_three_task_provider(_provider(), require_admission=True)


def test_provider_batch_matches_ordered_single_predictions():
    registry = build_registry_from_adcs_three_task_provider(_provider())
    first = _request()
    second = first.model_copy(
        update={
            "episode_id": "episode-2",
            "observation_scope": first.observation_scope.model_copy(
                update={"scope_id": "adcs-test-window-2"}
            ),
        }
    )
    singles = [registry.predict(first), registry.predict(second)]
    batch = registry.predict_many([first, second])
    assert [item.episode_id for item in batch] == ["episode-1", "episode-2"]
    for expected, actual in zip(singles, batch, strict=True):
        assert actual.metadata["raw_output_metadata"]["detection_probability"] == pytest.approx(
            expected.metadata["raw_output_metadata"]["detection_probability"]
        )
        assert [item.class_id for item in actual.classes] == [
            item.class_id for item in expected.classes
        ]
        assert [item.probability for item in actual.classes] == pytest.approx(
            [item.probability for item in expected.classes]
        )
        assert actual.metadata["raw_output_metadata"]["batch_size"] == 2


def test_hash_pinned_batch_preserves_per_request_admission_receipts(tmp_path):
    provider = _provider()
    path = tmp_path / "adcs-batch.pt"
    digest = save_adcs_three_task_torch_package(provider, path)
    registry = build_registry_from_adcs_three_task_provider(
        load_adcs_three_task_torch_package(path, expected_sha256=digest)
    )
    first = _request()
    second = first.model_copy(
        update={
            "episode_id": "episode-2",
            "observation_scope": first.observation_scope.model_copy(
                update={"scope_id": "adcs-test-window-2", "window_start_s": 1.0, "window_end_s": 4.0}
            ),
        }
    )
    predictions = registry.predict_many([first, second])
    receipts = [item.metadata["model_admission"] for item in predictions]
    assert receipts[0]["receipt_id"] != receipts[1]["receipt_id"]
    assert all(item["feature_contract_id"] for item in receipts)
    assert [item.episode_id for item in predictions] == ["episode-1", "episode-2"]


def test_strict_adcs_contract_allows_explicit_null_only_for_frozen_imputation(tmp_path):
    provider = _provider()
    path = tmp_path / "adcs-nullable.pt"
    digest = save_adcs_three_task_torch_package(provider, path)
    registry = build_registry_from_adcs_three_task_provider(
        load_adcs_three_task_torch_package(path, expected_sha256=digest)
    )
    base = _request()
    nullable_payload = {
        "channel_order": ["a", "b"],
        "telemetry_window": [[None, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    }
    nullable = base.model_copy(
        update={
            "input_data_sha256": canonical_sha256(nullable_payload),
            "payload": nullable_payload,
        }
    )
    prediction = registry.predict(nullable)
    assert prediction.classes

    nonfinite_payload = {
        "channel_order": ["a", "b"],
        "telemetry_window": [[float("nan"), 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    }
    nonfinite = base.model_copy(
        update={
            "input_data_sha256": canonical_sha256(nonfinite_payload),
            "payload": nonfinite_payload,
        }
    )
    with pytest.raises(AdmissionError, match="value kind"):
        registry.predict(nonfinite)


def test_loader_rejects_legacy_v1_package_without_positional_contract(tmp_path):
    import hashlib
    import torch

    provider = _provider()
    v2_path = tmp_path / "adcs-v2.pt"
    save_adcs_three_task_torch_package(provider, v2_path)
    bundle = torch.load(v2_path, map_location="cpu", weights_only=True)
    bundle["schema_version"] = "astrograph-adcs-three-task-torch-package.v1"
    bundle.pop("channel_order_contract", None)
    legacy_path = tmp_path / "adcs-v1.pt"
    torch.save(bundle, legacy_path)
    digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="no trustworthy positional channel-order contract"):
        load_adcs_three_task_torch_package(legacy_path, expected_sha256=digest)


def test_loader_rejects_tampered_channel_order_contract(tmp_path):
    import hashlib
    import torch

    provider = _provider()
    source = tmp_path / "adcs-source.pt"
    save_adcs_three_task_torch_package(provider, source)
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    bundle["channel_order_contract"] = {
        **bundle["channel_order_contract"],
        "tensor_channel_order": ["b", "a"],
    }
    tampered = tmp_path / "adcs-tampered.pt"
    torch.save(bundle, tampered)
    digest = hashlib.sha256(tampered.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="differs from provider channel order|SHA-256 mismatch"):
        load_adcs_three_task_torch_package(tampered, expected_sha256=digest)
