from __future__ import annotations

from typing import Any

import pytest

from astrograph.assurance import (
    AdmissionError,
    AdmissionReceipt,
    AdmissionSigner,
    FeatureContract,
    FeatureContractRegistry,
    FeatureValueKind,
    InputBinding,
    InputRole,
    ModelAdmissionManifest,
    canonical_sha256,
)
from astrograph.data_driven import (
    CallableRawModelProvider,
    DataDrivenModelManifest,
    DataDrivenModelRegistry,
    DataDrivenTaskType,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ModelRuntime,
    ObservationScope,
    RawModelOutput,
)


def _manifest() -> DataDrivenModelManifest:
    return DataDrivenModelManifest(
        model_id="strict-test-model",
        model_version="1.0.0",
        display_name="Strict test model",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.CUSTOM_CALLABLE,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS],
        supported_fault_ids=["F-A", "F-B"],
        normal_class_id="NORMAL",
        feature_schema_ids=["strict.features.v1"],
        input_contract={"feature_order": ["x", "y"]},
    )


def _contract() -> FeatureContract:
    return FeatureContract(
        contract_id="strict.contract.v1",
        feature_schema_id="strict.features.v1",
        bindings=(
            InputBinding(
                binding_id="x",
                payload_path=("features", "x"),
                value_kind=FeatureValueKind.NUMBER,
                source_fields=("sensor.x",),
                semantic_signal_ids=("signal.x",),
                transform_chain=("mean",),
            ),
            InputBinding(
                binding_id="y",
                payload_path=("features", "y"),
                value_kind=FeatureValueKind.NUMBER,
                source_fields=("sensor.y",),
                semantic_signal_ids=("signal.y",),
                transform_chain=("slope",),
            ),
        ),
    )


def _registry(captured: list[ModelInferenceRequest]) -> tuple[DataDrivenModelRegistry, AdmissionSigner]:
    manifest = _manifest()

    def predict(request: ModelInferenceRequest) -> RawModelOutput:
        captured.append(request)
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=request.model_id,
            model_version=request.model_version or manifest.model_version,
            output_semantics=ModelOutputSemantics.PROBABILITIES,
            class_values={"F-A": 0.7, "F-B": 0.2, "NORMAL": 0.1},
        )

    provider = CallableRawModelProvider(
        manifest=manifest,
        predict_fn=predict,
        model_artifact_sha256="a" * 64,
    )
    contract = _contract()
    admission = ModelAdmissionManifest(
        admission_id="strict.admission.v1",
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        provider_id=provider.provider_id,
        model_artifact_sha256="a" * 64,
        data_model_manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        feature_contract_id=contract.contract_id,
        feature_contract_sha256=contract.content_sha256,
        supported_fault_ids=tuple(manifest.supported_fault_ids),
        normal_class_id=manifest.normal_class_id,
    )
    contracts = FeatureContractRegistry()
    contracts.register_contract(contract)
    contracts.register_admission(admission)
    signer = AdmissionSigner(b"k" * 32, key_id="test-key")
    registry = DataDrivenModelRegistry(
        admission_registry=contracts,
        admission_signer=signer,
        require_admission=True,
    )
    registry.register(provider)
    return registry, signer


def _request(**updates: Any) -> ModelInferenceRequest:
    payload = updates.pop("payload", {"features": {"x": 1.5, "y": -0.25}})
    values: dict[str, Any] = {
        "episode_id": "REAL-EPISODE-001",
        "model_id": "strict-test-model",
        "model_version": "1.0.0",
        "feature_schema_id": "strict.features.v1",
        "input_data_sha256": canonical_sha256(payload),
        "registered_fault_ids": ["F-A", "F-B"],
        "payload": payload,
        "observation_scope": ObservationScope(
            scope_id="scope-001",
            source_id="telemetry-source",
            window_start_s=10.0,
            window_end_s=20.0,
            source_data_sha256="b" * 64,
            trajectory_group_id="trajectory-1",
        ),
        "metadata": {"fault_id": "F-A", "case_id": "case-answer-proxy"},
    }
    values.update(updates)
    return ModelInferenceRequest(**values)


def test_strict_registry_blinds_provider_and_attaches_trusted_lineage() -> None:
    captured: list[ModelInferenceRequest] = []
    registry, signer = _registry(captured)
    prediction = registry.predict(_request())

    assert prediction.episode_id == "REAL-EPISODE-001"
    assert len(captured) == 1
    provider_view = captured[0]
    assert provider_view.episode_id.startswith("OBS-")
    assert provider_view.episode_id != prediction.episode_id
    assert provider_view.metadata == {}
    assert provider_view.observation_scope is None
    assert provider_view.payload == {"features": {"x": 1.5, "y": -0.25}}
    assert provider_view.registered_fault_ids == ["F-A", "F-B"]
    assert provider_view.input_data_sha256 != canonical_sha256(provider_view.payload)
    assert provider_view.input_data_sha256 != _request().input_data_sha256

    assert "request_metadata" not in prediction.metadata
    assert prediction.metadata["model_admission"]["feature_contract_id"] == "strict.contract.v1"
    receipt = AdmissionReceipt.model_validate(
        prediction.metadata["model_admission"]["receipt"]
    )
    assert signer.verify(receipt)
    lineage = prediction.metadata["evidence_lineage_by_fault"]
    assert lineage["F-A"]["lineage_complete"] is True
    assert lineage["F-A"]["source_evidence_ids"]
    assert lineage["F-A"]["correlation_group_ids"] == lineage["F-B"]["correlation_group_ids"]


def test_admission_receipt_signature_is_verifiable_and_tamper_evident() -> None:
    captured: list[ModelInferenceRequest] = []
    registry, signer = _registry(captured)
    provider = registry.get("strict-test-model", "1.0.0")
    execution = registry._admission_registry.admit(  # trusted gate under test
        request=_request(),
        data_model_manifest=provider.manifest,
        provider_id=provider.provider_id,
        signer=signer,
    )
    assert signer.verify(execution.receipt)
    tampered = execution.receipt.model_copy(update={"admitted_payload_sha256": "0" * 64})
    assert not signer.verify(tampered)


@pytest.mark.parametrize(
    "case_request",
    [
        _request(payload={"features": {"x": 1.0, "y": 2.0, "fault_type": 1.0}}),
        _request(payload={"features": {"x": 1.0}}),
        _request(registered_fault_ids=["F-A"]),
        _request(input_data_sha256="0" * 64),
    ],
)
def test_strict_admission_rejects_unknown_missing_dynamic_or_hash_drift(
    case_request: ModelInferenceRequest,
) -> None:
    registry, _ = _registry([])
    with pytest.raises(AdmissionError):
        registry.predict(case_request)


def test_truth_or_identity_names_cannot_be_declared_as_feature_bindings() -> None:
    with pytest.raises(ValueError, match="prohibited truth/identity"):
        InputBinding(
            binding_id="bad",
            payload_path=("features", "condition_id"),
            value_kind=FeatureValueKind.NUMBER,
            source_fields=("condition_id",),
            semantic_signal_ids=("condition_id",),
        )


def test_mask_contract_requires_bool_array_and_exact_shape() -> None:
    with pytest.raises(ValueError, match="MASK role"):
        InputBinding(
            binding_id="mask",
            payload_path=("mask",),
            value_kind=FeatureValueKind.NUMBER_ARRAY,
            role=InputRole.MASK,
            shape=(4,),
            source_fields=("sensor.validity",),
            semantic_signal_ids=("signal.validity",),
        )


def test_provider_marked_as_admission_required_cannot_enter_legacy_registry() -> None:
    manifest = _manifest()

    def predict(request: ModelInferenceRequest) -> RawModelOutput:
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=ModelOutputSemantics.PROBABILITIES,
            class_values={"F-A": 0.7, "F-B": 0.2, "NORMAL": 0.1},
        )

    provider = CallableRawModelProvider(
        manifest=manifest,
        predict_fn=predict,
        requires_admission=True,
    )
    with pytest.raises(ValueError, match="requires strict admission"):
        DataDrivenModelRegistry.for_research().register(provider)


def test_strict_contract_cannot_enable_unbound_payload_leaves() -> None:
    with pytest.raises(ValueError, match="cannot allow unbound"):
        FeatureContract(
            contract_id="unsafe.contract.v1",
            feature_schema_id="strict.features.v1",
            bindings=_contract().bindings,
            allow_additional_payload_leaves=True,
        )


def test_strict_registry_detects_runtime_artifact_binding_drift() -> None:
    registry, _ = _registry([])
    provider = registry.get("strict-test-model", "1.0.0")
    provider.model_artifact_sha256 = "b" * 64
    with pytest.raises(AdmissionError, match="artifact hash"):
        registry.predict(_request())


def test_unadmitted_registry_requires_explicit_research_opt_in() -> None:
    with pytest.raises(ValueError, match="disabled by default"):
        DataDrivenModelRegistry()

    registry = DataDrivenModelRegistry.for_research()
    assert registry.require_admission is False
    assert registry.allow_unadmitted_research is True
