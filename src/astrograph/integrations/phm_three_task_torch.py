"""Runtime boundary for the PHM causal detector and hierarchical diagnoser.

The provider is deliberately dataset-adapter code rather than workflow code:
it validates the frozen channel order, applies Train-only preprocessing and
translates dataset classes only through the reviewed runtime candidate map.
"""
from __future__ import annotations

import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PHMThreeTaskTorchProvider:
    """Serve independently selected detection and diagnosis specialists."""

    manifest: DataDrivenModelManifest
    detector_model: Any
    diagnosis_model: Any
    preprocessor: ChannelPreprocessor
    selected_channels: tuple[str, ...]
    class_names: tuple[str, ...]
    component_names: tuple[str, ...]
    fault_to_component: Mapping[str, str]
    class_map: Mapping[str, str]
    runtime_fault_ids: frozenset[str]
    detection_selection: Mapping[str, Any]
    diagnosis_selection: Mapping[str, Any]
    model_artifact_sha256: str | None = None

    @property
    def lifecycle_profile(self):
        return BUILTIN_PROVIDER_QUALIFICATIONS.get("inprocess-model-runtime.v1")

    @property
    def requires_admission(self) -> bool:
        return self.model_artifact_sha256 is not None

    def __post_init__(self) -> None:
        if tuple(self.manifest.supported_fault_ids) != tuple(sorted(self.runtime_fault_ids)):
            raise ValueError("manifest/runtime candidate universes differ")
        if not self.selected_channels:
            raise ValueError("selected channel order cannot be empty")
        if set(self.class_names) != set(self.fault_to_component):
            raise ValueError("fault-to-component contract does not cover the class universe")
        if self.detection_selection.get("detector") != "deep":
            raise ValueError("this runtime package supports only a selected deep detector")
        if self.detection_selection.get("detection_score_normalization", "none") != "none":
            raise ValueError("stateful detection-score normalization is not runtime-supported")
        if int(self.detection_selection.get("alarm_minimum_consecutive", 1)) != 1:
            raise ValueError("stateful consecutive-window alarm policy is not runtime-supported")
        self.detector_model.eval()
        self.diagnosis_model.eval()

    @staticmethod
    def _softmax(logits):
        import torch

        return torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    def _window(self, request: ModelInferenceRequest) -> np.ndarray:
        payload = request.payload
        order = payload.get("channel_order")
        values = payload.get("telemetry_window")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValueError("payload.channel_order sequence is required")
        if tuple(str(name) for name in order) != self.selected_channels:
            raise ValueError("runtime channel order differs from the frozen training contract")
        array = np.asarray(values, dtype=np.float32)
        expected_window = int(self.manifest.input_contract.get("window_length", 60))
        if array.shape != (expected_window, len(self.selected_channels)):
            raise ValueError(
                "telemetry_window shape must equal "
                f"({expected_window}, {len(self.selected_channels)})"
            )
        return self.preprocessor.transform(array[None, ...])

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        import torch

        window = self._window(request)
        detector_device = next(self.detector_model.parameters()).device
        diagnosis_device = next(self.diagnosis_model.parameters()).device
        with torch.inference_mode():
            detector_logits, _, _ = self.detector_model(
                torch.from_numpy(window).to(detector_device)
            )
            _, component_logits, class_logits = self.diagnosis_model(
                torch.from_numpy(window).to(diagnosis_device)
            )
        detection_probability = float(self._softmax(detector_logits)[1])
        direct_component = self._softmax(component_logits)
        class_probability = self._softmax(class_logits)
        component_index = {name: index for index, name in enumerate(self.component_names)}
        derived_component = np.zeros(len(self.component_names), dtype=np.float64)
        for index, name in enumerate(self.class_names):
            derived_component[
                component_index[self.fault_to_component[name]]
            ] += class_probability[index]
        alpha = float(self.diagnosis_selection["hierarchy_blend_alpha"])
        component_probability = alpha * direct_component + (1.0 - alpha) * derived_component
        predicted_component = self.component_names[int(component_probability.argmax())]
        allowed = np.asarray([
            self.fault_to_component[name] == predicted_component for name in self.class_names
        ])
        conditioned = np.where(allowed, class_probability, 0.0)
        conditioned /= max(float(conditioned.sum()), 1e-12)
        dataset_probabilities = {
            name: float(value) for name, value in zip(self.class_names, conditioned, strict=True)
        }
        try:
            mapping = map_runtime_fault_attribution(
                dataset_probabilities, dict(self.class_map), set(self.runtime_fault_ids)
            )
            mapped_attribution = mapping.mapped_attribution
            unsupported_probabilities = mapping.unsupported_probabilities
            unsupported_mass = mapping.unsupported_mass
            runtime_anomaly_score = detection_probability
            fusion_abstained = False
        except NoMappedFaultMassError:
            mapped_attribution = {name: 0.0 for name in sorted(self.runtime_fault_ids)}
            unsupported_probabilities = dataset_probabilities
            unsupported_mass = 1.0
            runtime_anomaly_score = 0.0
            fusion_abstained = True
        threshold = float(self.detection_selection["detection_threshold"])
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=self.manifest.model_id,
            model_version=self.manifest.model_version,
            output_semantics=self.manifest.output_semantics,
            anomaly_score=runtime_anomaly_score,
            fault_attribution=mapped_attribution,
            out_of_distribution_score=unsupported_mass,
            metadata={
                "provider": type(self).__name__,
                "detected": detection_probability >= threshold,
                "fusion_abstained": fusion_abstained,
                "detection_probability": detection_probability,
                "detection_threshold": threshold,
                "predicted_component": predicted_component,
                "predicted_dataset_class": self.class_names[int(conditioned.argmax())],
                "component_probabilities": dict(zip(
                    self.component_names, map(float, component_probability), strict=True
                )),
                "dataset_class_probabilities": dataset_probabilities,
                "unsupported_dataset_class_mass": unsupported_probabilities,
                "specialist_contract": "causal_detector_plus_hierarchical_diagnoser",
            },
        )


def build_registry_from_phm_three_task_provider(
    provider: PHMThreeTaskTorchProvider,
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
                "strict PHM registry requires a hash-pinned model package artifact"
            )
        contract_registry = build_temporal_window_contract_registry(
            provider=provider,
            model_artifact_sha256=provider.model_artifact_sha256,
            selected_channels=provider.selected_channels,
            window_length=int(provider.manifest.input_contract["window_length"]),
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


def save_phm_three_task_torch_package(
    provider: PHMThreeTaskTorchProvider,
    path: str | Path,
    *,
    hierarchical_contract: Mapping[str, Any],
) -> str:
    """Persist tensors plus explicit contracts; no pickled model object is stored."""
    import torch

    path = Path(path)
    state = provider.preprocessor
    bundle = {
        "schema_version": "astrograph-phm-three-task-torch-package.v1",
        "manifest": provider.manifest.model_dump(mode="json"),
        "selected_channels": list(provider.selected_channels),
        "class_names": list(provider.class_names),
        "component_names": list(provider.component_names),
        "fault_to_component": dict(provider.fault_to_component),
        "class_map": dict(provider.class_map),
        "runtime_fault_ids": sorted(provider.runtime_fault_ids),
        "detection_selection": dict(provider.detection_selection),
        "diagnosis_selection": dict(provider.diagnosis_selection),
        "hierarchical_contract": dict(hierarchical_contract),
        "preprocessor": {
            "method": state.method,
            "impute": torch.from_numpy(state.impute),
            "center": torch.from_numpy(state.center),
            "scale": torch.from_numpy(state.scale),
            "lower": torch.from_numpy(state.lower),
            "upper": torch.from_numpy(state.upper),
            "normalized_clip": state.normalized_clip,
        },
        "detector_state_dict": provider.detector_model.state_dict(),
        "diagnosis_state_dict": provider.diagnosis_model.state_dict(),
    }
    source_root = Path(__file__).resolve().parents[3]
    artifacts = ArtifactRoot(path.parent, source_root=source_root)
    target = artifacts.write_generated(path.name, lambda temporary: torch.save(bundle, temporary))
    return _sha256(target)


def load_phm_three_task_torch_package(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str = "cpu",
) -> PHMThreeTaskTorchProvider:
    """Load a hash-pinned, weights-only package into the common registry boundary."""
    import torch
    from astrograph.data_driven.three_task_tcn import (
        make_causal_multiview_detector_tcn,
        make_hierarchical_mechanism_gat_tcn,
    )

    path = Path(path)
    if _sha256(path) != expected_sha256:
        raise ValueError("PHM model package SHA-256 mismatch")
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get("schema_version") != "astrograph-phm-three-task-torch-package.v1":
        raise ValueError("unsupported PHM model package schema")
    selected = tuple(bundle["selected_channels"])
    classes = tuple(bundle["class_names"])
    components = tuple(bundle["component_names"])
    contract = bundle["hierarchical_contract"]
    component_index = {name: index for index, name in enumerate(components)}
    class_component = [
        component_index[bundle["fault_to_component"][name]] for name in classes
    ]
    detector = make_causal_multiview_detector_tcn(
        len(selected), len(components), len(classes),
        window_size=int(bundle["manifest"]["input_contract"]["window_length"]),
    ).to(device)
    diagnosis = make_hierarchical_mechanism_gat_tcn(
        len(selected), len(components), len(classes),
        component_groups=contract["component_groups"],
        subsystem_groups=contract["subsystem_groups"],
        component_subsystem=contract["component_subsystem"],
        class_component=class_component,
        subsystem_edges=contract["subsystem_edges"],
    ).to(device)
    detector.load_state_dict(bundle["detector_state_dict"])
    diagnosis.load_state_dict(bundle["diagnosis_state_dict"])
    raw_state = bundle["preprocessor"]
    preprocessor = ChannelPreprocessor(
        method=raw_state["method"],
        impute=raw_state["impute"].cpu().numpy(),
        center=raw_state["center"].cpu().numpy(),
        scale=raw_state["scale"].cpu().numpy(),
        lower=raw_state["lower"].cpu().numpy(),
        upper=raw_state["upper"].cpu().numpy(),
        normalized_clip=raw_state["normalized_clip"],
    )
    return PHMThreeTaskTorchProvider(
        manifest=DataDrivenModelManifest.model_validate(bundle["manifest"]),
        detector_model=detector, diagnosis_model=diagnosis, preprocessor=preprocessor,
        selected_channels=selected, class_names=classes, component_names=components,
        fault_to_component=bundle["fault_to_component"], class_map=bundle["class_map"],
        runtime_fault_ids=frozenset(bundle["runtime_fault_ids"]),
        detection_selection=bundle["detection_selection"],
        diagnosis_selection=bundle["diagnosis_selection"],
        model_artifact_sha256=expected_sha256,
    )


__all__ = [
    "PHMThreeTaskTorchProvider", "build_registry_from_phm_three_task_provider",
    "save_phm_three_task_torch_package", "load_phm_three_task_torch_package",
]
