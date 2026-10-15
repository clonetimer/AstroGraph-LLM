"""Hash-pinned runtime boundary for the ADCS v5.2.2 P2 three-task model."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from astrograph.assurance import (
    AdmissionSigner,
    build_temporal_window_contract_registry,
)
from astrograph.assurance import ArtifactRoot
from astrograph.data_driven import (
    DataDrivenModelManifest,
    DataDrivenModelRegistry,
    ModelInferenceRequest,
    RawModelOutput,
)
from astrograph.data_driven.candidate_mapping import (
    NoMappedFaultMassError,
    map_runtime_fault_attribution,
)
from astrograph.data_driven.preprocessing import ChannelPreprocessor
from astrograph.provider_lifecycle import BUILTIN_PROVIDER_QUALIFICATIONS


ADCS_PACKAGE_SCHEMA_V1 = "astrograph-adcs-three-task-torch-package.v1"
ADCS_PACKAGE_SCHEMA_V2 = "astrograph-adcs-three-task-torch-package.v2"
CHANNEL_ORDER_CONTRACT_SCHEMA = "astrograph-channel-order-contract.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def declared_channel_order_contract(
    selected_channels: Sequence[str],
    *,
    provenance: str = "DECLARED_ORDER_ENFORCED",
    legacy_package_sha256: str | None = None,
    source_header_order_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a content-addressed positional tensor contract."""

    channels = tuple(map(str, selected_channels))
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("selected channel order must be non-empty and duplicate-free")
    if not str(provenance).strip():
        raise ValueError("channel-order provenance is required")
    contract: dict[str, Any] = {
        "schema_version": CHANNEL_ORDER_CONTRACT_SCHEMA,
        "tensor_channel_order": list(channels),
        "order_enforced": True,
        "provenance": str(provenance),
    }
    for field_name, field_value in (
        ("legacy_package_sha256", legacy_package_sha256),
        ("source_header_order_sha256", source_header_order_sha256),
    ):
        if field_value is not None:
            normalized = str(field_value).lower()
            if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
            contract[field_name] = normalized
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _validate_channel_order_contract(
    contract: Mapping[str, Any], selected_channels: Sequence[str]
) -> dict[str, Any]:
    normalized = dict(contract)
    if normalized.get("schema_version") != CHANNEL_ORDER_CONTRACT_SCHEMA:
        raise ValueError("unsupported ADCS channel-order contract schema")
    if normalized.get("order_enforced") is not True:
        raise ValueError("ADCS channel-order contract must enforce positional order")
    if tuple(map(str, normalized.get("tensor_channel_order", ()))) != tuple(selected_channels):
        raise ValueError("channel-order contract differs from provider channel order")
    if not str(normalized.get("provenance", "")).strip():
        raise ValueError("ADCS channel-order contract provenance is required")
    claimed = normalized.pop("contract_sha256", None)
    if claimed is None or claimed != _canonical_sha256(normalized):
        raise ValueError("ADCS channel-order contract SHA-256 mismatch")
    normalized["contract_sha256"] = claimed
    return normalized


def _summary_features(window: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (window.mean(1), window.std(1), window[:, -1], window[:, -1] - window[:, 0]),
        axis=1,
    )


@dataclass
class ADCSThreeTaskTorchProvider:
    """Serve the frozen P2 Residual TCN plus its logistic detection specialist."""

    manifest: DataDrivenModelManifest
    model: Any
    preprocessor: ChannelPreprocessor
    selected_channels: tuple[str, ...]
    class_names: tuple[str, ...]
    component_names: tuple[str, ...]
    fault_to_component: Mapping[str, str]
    class_map: Mapping[str, str]
    runtime_fault_ids: frozenset[str]
    detection_selection: Mapping[str, Any]
    logistic_coefficient: np.ndarray
    logistic_intercept: float
    diagnosis_selection: Mapping[str, Any]
    model_artifact_sha256: str | None = None
    channel_order_contract: Mapping[str, Any] | None = None

    @property
    def lifecycle_profile(self):
        return BUILTIN_PROVIDER_QUALIFICATIONS.get("inprocess-model-runtime.v1")

    @property
    def requires_admission(self) -> bool:
        return self.model_artifact_sha256 is not None

    def __post_init__(self) -> None:
        if tuple(self.manifest.supported_fault_ids) != tuple(sorted(self.runtime_fault_ids)):
            raise ValueError("manifest/runtime candidate universes differ")
        if tuple(self.manifest.input_contract.get("feature_order", ())) != self.selected_channels:
            raise ValueError("manifest feature order differs from provider channel order")
        manifest_tensor_order = self.manifest.input_contract.get("tensor_channel_order")
        if manifest_tensor_order is not None and tuple(manifest_tensor_order) != self.selected_channels:
            raise ValueError("manifest tensor channel order differs from provider channel order")
        if set(self.class_names) != set(self.fault_to_component):
            raise ValueError("fault-to-component contract does not cover the class universe")
        expected = 4 * len(self.selected_channels)
        coefficient = np.asarray(self.logistic_coefficient, dtype=np.float64)
        if coefficient.shape != (expected,):
            raise ValueError(f"logistic coefficient must have shape ({expected},)")
        blend = float(self.detection_selection["detector_deep_blend_alpha"])
        if not 0.0 <= blend <= 1.0:
            raise ValueError("detector blend alpha must be in [0, 1]")
        if self.detection_selection.get("alarm_policy_mode", "point") != "point":
            raise ValueError("stateful alarm policies are not supported by this provider")
        if self.diagnosis_selection.get("classification_policy", "hard_component") not in {
            "hard_component", "unmasked", "soft_component"
        }:
            raise ValueError("unknown classification policy")
        if self.channel_order_contract is None:
            self.channel_order_contract = declared_channel_order_contract(self.selected_channels)
        self.channel_order_contract = _validate_channel_order_contract(
            self.channel_order_contract, self.selected_channels
        )
        self.logistic_coefficient = coefficient
        self.model.eval()

    @staticmethod
    def _softmax_batch(logits) -> np.ndarray:
        import torch

        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def _window(self, request: ModelInferenceRequest) -> np.ndarray:
        order = request.payload.get("channel_order")
        values = request.payload.get("telemetry_window")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValueError("payload.channel_order sequence is required")
        if tuple(map(str, order)) != self.selected_channels:
            raise ValueError("runtime channel order differs from frozen training contract")
        array = np.asarray(values, dtype=np.float32)
        expected_window = int(self.manifest.input_contract["window_length"])
        if array.shape != (expected_window, len(self.selected_channels)):
            raise ValueError(
                f"telemetry_window shape must equal ({expected_window}, {len(self.selected_channels)})"
            )
        return self.preprocessor.transform(array[None, ...])

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        return self.predict_raw_batch((request,))[0]

    def predict_raw_batch(
        self, requests: Sequence[ModelInferenceRequest]
    ) -> list[RawModelOutput]:
        """Vectorized inference with one output per already-admitted request."""

        batch = list(requests)
        if not batch:
            return []
        import torch

        windows = np.concatenate([self._window(request) for request in batch], axis=0)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            detector_logits, component_logits, class_logits = self.model(
                torch.from_numpy(windows).to(device)
            )
        detector_probability = self._softmax_batch(detector_logits)
        direct_component = self._softmax_batch(component_logits)
        class_probability = self._softmax_batch(class_logits)

        summary = _summary_features(windows)
        logits = summary @ self.logistic_coefficient + float(self.logistic_intercept)
        logistic_detection = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        deep_detection = detector_probability[:, 1]
        blend = float(self.detection_selection["detector_deep_blend_alpha"])
        detection_probability = blend * deep_detection + (1.0 - blend) * logistic_detection

        component_index = {name: index for index, name in enumerate(self.component_names)}
        derived_component = np.zeros(
            (len(batch), len(self.component_names)), dtype=np.float64
        )
        for class_index, name in enumerate(self.class_names):
            derived_component[:, component_index[self.fault_to_component[name]]] += (
                class_probability[:, class_index]
            )
        alpha = float(self.diagnosis_selection["hierarchy_blend_alpha"])
        component_probability = alpha * direct_component + (1.0 - alpha) * derived_component
        threshold = float(self.detection_selection["detection_threshold"])

        outputs: list[RawModelOutput] = []
        for row_index, request in enumerate(batch):
            predicted_component = self.component_names[int(component_probability[row_index].argmax())]
            allowed = np.asarray([
                self.fault_to_component[name] == predicted_component for name in self.class_names
            ])
            policy = self.diagnosis_selection.get("classification_policy", "hard_component")
            if policy == "hard_component":
                conditioned = np.where(allowed, class_probability[row_index], 0.0)
            elif policy == "soft_component":
                conditioned = class_probability[row_index] * np.asarray([
                    component_probability[row_index, component_index[self.fault_to_component[name]]]
                    for name in self.class_names
                ])
            else:
                conditioned = class_probability[row_index].copy()
            conditioned /= max(float(conditioned.sum()), 1e-12)
            dataset_probabilities = dict(
                zip(self.class_names, map(float, conditioned), strict=True)
            )
            try:
                mapping = map_runtime_fault_attribution(
                    dataset_probabilities, dict(self.class_map), set(self.runtime_fault_ids)
                )
                attribution = mapping.mapped_attribution
                unsupported = mapping.unsupported_probabilities
                unsupported_mass = mapping.unsupported_mass
                anomaly_score = float(detection_probability[row_index])
                abstained = False
            except NoMappedFaultMassError:
                attribution = {name: 0.0 for name in sorted(self.runtime_fault_ids)}
                unsupported = dataset_probabilities
                unsupported_mass = 1.0
                anomaly_score = 0.0
                abstained = True
            outputs.append(
                RawModelOutput(
                    episode_id=request.episode_id,
                    model_id=self.manifest.model_id,
                    model_version=self.manifest.model_version,
                    output_semantics=self.manifest.output_semantics,
                    anomaly_score=anomaly_score,
                    fault_attribution=attribution,
                    out_of_distribution_score=unsupported_mass,
                    metadata={
                        "provider": type(self).__name__,
                        "detected": float(detection_probability[row_index]) >= threshold,
                        "fusion_abstained": abstained,
                        "detection_probability": float(detection_probability[row_index]),
                        "deep_detection_probability": float(deep_detection[row_index]),
                        "logistic_detection_probability": float(logistic_detection[row_index]),
                        "detection_threshold": threshold,
                        "predicted_component": predicted_component,
                        "predicted_dataset_class": self.class_names[int(conditioned.argmax())],
                        "component_probabilities": dict(zip(
                            self.component_names,
                            map(float, component_probability[row_index]),
                            strict=True,
                        )),
                        "dataset_class_probabilities": dataset_probabilities,
                        "head_trace": {
                            "classification_policy": policy,
                            "raw_class_probabilities": dict(zip(self.class_names, map(float, class_probability[row_index]), strict=True)),
                            "direct_component_probabilities": dict(zip(self.component_names, map(float, direct_component[row_index]), strict=True)),
                            "derived_component_probabilities": dict(zip(self.component_names, map(float, derived_component[row_index]), strict=True)),
                            "raw_top_class": self.class_names[int(class_probability[row_index].argmax())],
                            "discarded_class_mass": float(np.clip(class_probability[row_index, ~allowed].sum(), 0.0, 1.0)) if policy == "hard_component" else 0.0,
                            "class_to_runtime_fault": dict(self.class_map),
                            "score_semantics": "UNCALIBRATED_MODEL_OUTPUT_SHARED_EVIDENCE",
                        },
                        "out_of_distribution_score_semantics": "UNMAPPED_CLASS_MASS_NOT_INPUT_NOVELTY",
                        "unsupported_dataset_class_mass": unsupported,
                        "specialist_contract": "p2_residual_tcn_plus_logistic_detector",
                        "batch_size": len(batch),
                    },
                )
            )
        return outputs


def build_registry_from_adcs_three_task_provider(
    provider: ADCSThreeTaskTorchProvider,
    *,
    require_admission: bool | None = None,
    admission_signer: AdmissionSigner | None = None,
) -> DataDrivenModelRegistry:
    strict = provider.model_artifact_sha256 is not None if require_admission is None else bool(
        require_admission
    )
    if strict:
        if provider.model_artifact_sha256 is None:
            raise ValueError(
                "strict ADCS registry requires a hash-pinned model package artifact"
            )
        contract_registry = build_temporal_window_contract_registry(
            provider=provider,
            model_artifact_sha256=provider.model_artifact_sha256,
            selected_channels=provider.selected_channels,
            window_length=int(provider.manifest.input_contract["window_length"]),
            allow_missing_values=True,
        )
        registry = DataDrivenModelRegistry(
            admission_registry=contract_registry,
            admission_signer=admission_signer or AdmissionSigner.from_environment_or_ephemeral(),
            require_admission=True,
        )
    else:
        registry = DataDrivenModelRegistry.for_research()
    registry.register(provider)
    return registry


def save_adcs_three_task_torch_package(
    provider: ADCSThreeTaskTorchProvider, path: str | Path
) -> str:
    """Persist a weights-only P2 package with all inference-time statistics."""
    import torch

    path = Path(path)
    state = provider.preprocessor
    bundle = {
        "schema_version": ADCS_PACKAGE_SCHEMA_V2,
        "model_family": "shared_residual_tcn",
        "manifest": provider.manifest.model_dump(mode="json"),
        "selected_channels": list(provider.selected_channels),
        "channel_order_contract": dict(provider.channel_order_contract or {}),
        "class_names": list(provider.class_names),
        "component_names": list(provider.component_names),
        "fault_to_component": dict(provider.fault_to_component),
        "class_map": dict(provider.class_map),
        "runtime_fault_ids": sorted(provider.runtime_fault_ids),
        "detection_selection": dict(provider.detection_selection),
        "diagnosis_selection": dict(provider.diagnosis_selection),
        "preprocessor": {
            "method": state.method,
            "impute": torch.from_numpy(state.impute),
            "center": torch.from_numpy(state.center),
            "scale": torch.from_numpy(state.scale),
            "lower": torch.from_numpy(state.lower),
            "upper": torch.from_numpy(state.upper),
            "normalized_clip": state.normalized_clip,
        },
        "logistic_coefficient": torch.from_numpy(provider.logistic_coefficient),
        "logistic_intercept": float(provider.logistic_intercept),
        "model_state_dict": provider.model.state_dict(),
    }
    source_root = Path(__file__).resolve().parents[3]
    artifacts = ArtifactRoot(path.parent, source_root=source_root)
    target = artifacts.write_generated(path.name, lambda temporary: torch.save(bundle, temporary))
    return _sha256(target)


def load_adcs_three_task_torch_package(
    path: str | Path, *, expected_sha256: str, device: str = "cpu"
) -> ADCSThreeTaskTorchProvider:
    """Load a hash-pinned weights-only P2 package."""
    import torch
    from astrograph.data_driven.three_task_tcn import make_shared_residual_tcn

    path = Path(path)
    if _sha256(path) != expected_sha256:
        raise ValueError("ADCS model package SHA-256 mismatch")
    bundle = torch.load(path, map_location=device, weights_only=True)
    schema_version = bundle.get("schema_version")
    if schema_version == ADCS_PACKAGE_SCHEMA_V1:
        raise ValueError(
            "legacy ADCS package v1 has no trustworthy positional channel-order contract; "
            "migrate it with scripts/data_driven/migrate_adcs_p2_channel_order.py"
        )
    if schema_version != ADCS_PACKAGE_SCHEMA_V2:
        raise ValueError("unsupported ADCS model package schema")
    if bundle.get("model_family") != "shared_residual_tcn":
        raise ValueError("unsupported ADCS model family")
    selected = tuple(bundle["selected_channels"])
    classes = tuple(bundle["class_names"])
    components = tuple(bundle["component_names"])
    model = make_shared_residual_tcn(len(selected), len(components), len(classes)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    raw = bundle["preprocessor"]
    preprocessor = ChannelPreprocessor(
        method=raw["method"], impute=raw["impute"].cpu().numpy(),
        center=raw["center"].cpu().numpy(), scale=raw["scale"].cpu().numpy(),
        lower=raw["lower"].cpu().numpy(), upper=raw["upper"].cpu().numpy(),
        normalized_clip=raw["normalized_clip"],
    )
    return ADCSThreeTaskTorchProvider(
        manifest=DataDrivenModelManifest.model_validate(bundle["manifest"]),
        model=model, preprocessor=preprocessor, selected_channels=selected,
        class_names=classes, component_names=components,
        fault_to_component=bundle["fault_to_component"], class_map=bundle["class_map"],
        runtime_fault_ids=frozenset(bundle["runtime_fault_ids"]),
        detection_selection=bundle["detection_selection"],
        logistic_coefficient=bundle["logistic_coefficient"].cpu().numpy(),
        logistic_intercept=float(bundle["logistic_intercept"]),
        diagnosis_selection=bundle["diagnosis_selection"],
        model_artifact_sha256=expected_sha256,
        channel_order_contract=bundle["channel_order_contract"],
    )


__all__ = [
    "ADCSThreeTaskTorchProvider", "build_registry_from_adcs_three_task_provider",
    "save_adcs_three_task_torch_package", "load_adcs_three_task_torch_package",
    "declared_channel_order_contract", "ADCS_PACKAGE_SCHEMA_V1",
    "ADCS_PACKAGE_SCHEMA_V2", "CHANNEL_ORDER_CONTRACT_SCHEMA",
]
