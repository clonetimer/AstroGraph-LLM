from __future__ import annotations

from typing import Any, Sequence

from astrograph.data_driven.registry import provider_identity

from .contract_registry import FeatureContractRegistry
from .feature_admission import (
    FeatureContract,
    FeatureValueKind,
    InputBinding,
    InputRole,
    ModelAdmissionManifest,
    canonical_sha256,
)


def build_temporal_window_contract_registry(
    *,
    provider: Any,
    model_artifact_sha256: str,
    selected_channels: Sequence[str],
    window_length: int,
    contract_id: str | None = None,
    allow_missing_values: bool = False,
) -> FeatureContractRegistry:
    """Build a strict contract for ``channel_order`` + ``telemetry_window`` providers."""

    manifest = provider.manifest
    schema_ids = tuple(manifest.feature_schema_ids)
    if len(schema_ids) != 1:
        raise ValueError("strict temporal-window admission requires exactly one feature_schema_id")
    channels = tuple(str(item) for item in selected_channels)
    if not channels:
        raise ValueError("strict temporal-window admission requires selected channels")
    if window_length <= 0:
        raise ValueError("strict temporal-window admission requires positive window length")
    resolved_contract_id = contract_id or (
        f"astrograph.{manifest.model_id}.{manifest.model_version}.temporal-window.v1"
    )
    contract = FeatureContract(
        contract_id=resolved_contract_id,
        feature_schema_id=schema_ids[0],
        bindings=(
            InputBinding(
                binding_id="channel-order",
                payload_path=("channel_order",),
                value_kind=FeatureValueKind.STRING_ARRAY,
                role=InputRole.CONTEXT,
                shape=(len(channels),),
                source_fields=("runtime.channel_order",),
                semantic_signal_ids=("contract.channel_order",),
                transform_chain=("identity",),
                expected_value_sha256=canonical_sha256(list(channels)),
            ),
            InputBinding(
                binding_id="telemetry-window",
                payload_path=("telemetry_window",),
                value_kind=(
                    FeatureValueKind.NULLABLE_NUMBER_ARRAY
                    if allow_missing_values
                    else FeatureValueKind.NUMBER_ARRAY
                ),
                role=InputRole.FEATURE,
                shape=(window_length, len(channels)),
                source_fields=channels,
                semantic_signal_ids=channels,
                transform_chain=(
                    "ordered-window",
                    *(
                        ("explicit-null-to-frozen-model-imputation",)
                        if allow_missing_values
                        else ()
                    ),
                ),
            ),
        ),
        require_observation_scope=True,
        require_input_hash_match=True,
    )
    admission = ModelAdmissionManifest(
        admission_id=f"{manifest.model_id}@{manifest.model_version}:admission.v1",
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        provider_id=provider_identity(provider),
        model_artifact_sha256=model_artifact_sha256,
        data_model_manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        feature_contract_id=contract.contract_id,
        feature_contract_sha256=contract.content_sha256,
        supported_fault_ids=tuple(manifest.supported_fault_ids),
        normal_class_id=manifest.normal_class_id,
    )
    registry = FeatureContractRegistry()
    registry.register_contract(contract)
    registry.register_admission(admission)
    return registry


def build_flat_feature_mapping_contract_registry(
    *,
    provider: Any,
    model_artifact_sha256: str,
    feature_order: Sequence[str],
    payload_container: str = "features",
    contract_id: str | None = None,
) -> FeatureContractRegistry:
    """Build a strict contract for a mapping of named finite numeric features."""

    manifest = provider.manifest
    schema_ids = tuple(manifest.feature_schema_ids)
    if len(schema_ids) != 1:
        raise ValueError("strict flat-feature admission requires exactly one feature_schema_id")
    features = tuple(str(item) for item in feature_order)
    if not features or len(features) != len(set(features)):
        raise ValueError("strict flat-feature admission requires a nonempty unique feature order")
    if not payload_container:
        raise ValueError("flat-feature payload container must be nonempty")
    resolved_contract_id = contract_id or (
        f"astrograph.{manifest.model_id}.{manifest.model_version}.flat-features.v1"
    )
    bindings = tuple(
        InputBinding(
            binding_id=f"feature-{index:04d}",
            payload_path=(payload_container, feature),
            value_kind=FeatureValueKind.NUMBER,
            role=InputRole.FEATURE,
            source_fields=(feature,),
            semantic_signal_ids=(feature,),
            transform_chain=("identity",),
        )
        for index, feature in enumerate(features)
    )
    contract = FeatureContract(
        contract_id=resolved_contract_id,
        feature_schema_id=schema_ids[0],
        bindings=bindings,
        require_observation_scope=True,
        require_input_hash_match=True,
    )
    admission = ModelAdmissionManifest(
        admission_id=f"{manifest.model_id}@{manifest.model_version}:admission.v1",
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        provider_id=provider_identity(provider),
        model_artifact_sha256=model_artifact_sha256,
        data_model_manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        feature_contract_id=contract.contract_id,
        feature_contract_sha256=contract.content_sha256,
        supported_fault_ids=tuple(manifest.supported_fault_ids),
        normal_class_id=manifest.normal_class_id,
    )
    registry = FeatureContractRegistry()
    registry.register_contract(contract)
    registry.register_admission(admission)
    return registry


__all__ = [
    "build_flat_feature_mapping_contract_registry",
    "build_temporal_window_contract_registry",
]
