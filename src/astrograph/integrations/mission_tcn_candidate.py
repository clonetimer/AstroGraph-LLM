"""Strict provider wrapper for Mission three-head TCN packages.

This module is intentionally a provider adapter, not a promotion mechanism.  A
package is executable here only when a host-owned execution binding points to it
and the package itself has ``production_admitted=true``.  Training never flips
that bit automatically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from astrograph.assurance import AdmissionSigner
from astrograph.assurance.model_admission_helpers import build_temporal_window_contract_registry
from astrograph.data_driven import (
    DataDrivenModelManifest, DataDrivenModelRegistry, DataDrivenTaskType,
    ModelCapability, ModelOutputSemantics, ModelRuntime, RawModelOutput,
)
from astrograph.data_driven.three_task_tcn import make_shared_residual_tcn
from astrograph.agents.mission.storage import file_sha


@dataclass
class MissionThreeTaskTCNProvider:
    manifest: DataDrivenModelManifest
    model: Any
    selected_channels: tuple[str, ...]
    channel_units: tuple[str, ...]
    fault_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    model_artifact_sha256: str
    detection_threshold: float = 0.5
    requires_admission: bool = True

    @property
    def provider_id(self) -> str:
        return "astrograph:mission-three-task-tcn-provider.v1"

    def predict_raw(self, request):
        import torch
        order=request.payload.get("channel_order")
        values=request.payload.get("telemetry_window")
        if order != list(self.selected_channels):
            raise ValueError("TCN provider channel order mismatch")
        x=np.asarray(values,dtype="float32")
        expected=(int(self.manifest.input_contract["window_length"]),len(self.selected_channels))
        if x.shape!=expected or not np.isfinite(x).all():
            raise ValueError("TCN provider requires one finite exact-shape telemetry window")
        self.model.eval()
        with torch.no_grad():
            det,loc,cls=self.model(torch.tensor(((x-self.mean)/self.scale)[None,:,:],dtype=torch.float32))
            detp=det.softmax(-1)[0].cpu().numpy();locp=loc.softmax(-1)[0].cpu().numpy();clsp=cls.softmax(-1)[0].cpu().numpy()
        anomaly=float(detp[1])
        class_values={"NORMAL":float(1.0-anomaly)}
        for fault_id,p in zip(self.fault_ids,clsp,strict=True):
            class_values[fault_id]=float(anomaly*float(p))
        total=sum(class_values.values())
        if total<=0: raise ValueError("TCN provider produced zero probability mass")
        class_values={k:float(v/total) for k,v in class_values.items()}
        predicted_component=self.component_ids[int(locp.argmax())]
        predicted_fault=self.fault_ids[int(clsp.argmax())]
        return RawModelOutput(
            episode_id=request.episode_id,model_id=self.manifest.model_id,model_version=self.manifest.model_version,
            output_semantics=ModelOutputSemantics.PROBABILITIES,class_values=class_values,
            metadata={
                "provider":"MissionThreeTaskTCNProvider",
                "head_trace":{
                    "detection":{"classes":["NORMAL","ANOMALY"],"probabilities":detp.tolist(),"anomaly_probability":anomaly,"threshold":self.detection_threshold,"detected":anomaly>=self.detection_threshold},
                    "localization":{"classes":list(self.component_ids),"probabilities":locp.tolist(),"predicted_component":predicted_component,"root_cause_supported":False},
                    "classification":{"classes":list(self.fault_ids),"probabilities":clsp.tolist(),"predicted_fault":predicted_fault},
                },
                "score_semantics":"CALIBRATED_PROBABILITY" if ModelCapability.CALIBRATED_PROBABILITY in self.manifest.capabilities else "UNCALIBRATED_SOFTMAX",
            },
        )


def load_mission_three_task_tcn_provider(
    package_manifest: str | Path, *, model_id: str, model_version: str,
    expected_manifest_sha256: str | None = None,
) -> MissionThreeTaskTCNProvider:
    import torch
    path=Path(package_manifest).resolve()
    if expected_manifest_sha256 and file_sha(path)!=expected_manifest_sha256:
        raise ValueError("execution binding package manifest hash mismatch")
    package=json.loads(path.read_text(encoding="utf-8"))
    if package.get("schema_version")!="astrograph-agent-tcn-candidate.v1":
        raise ValueError("unrecognized Mission TCN package schema")
    if package.get("production_admitted") is not True:
        raise PermissionError("TCN package is not production-admitted")
    weights=path.parent/"weights.pt"
    digest=file_sha(weights)
    if digest!=package.get("weights_sha256"):
        raise ValueError("TCN package weights hash mismatch")
    channels=tuple(str(x) for x in package["channel_names"])
    units=tuple(str(x) for x in package["channel_units"])
    faults=tuple(str(x) for x in package["fault_ids"])
    components=tuple(str(x) for x in package["component_ids"])
    mean=np.asarray(package["mean"],dtype="float32");scale=np.asarray(package["scale"],dtype="float32")
    if len(channels)!=len(units) or mean.shape!=(len(channels),) or scale.shape!=(len(channels),) or np.any(scale<=0):
        raise ValueError("TCN package channel/normalization contract is invalid")
    model=make_shared_residual_tcn(len(channels),len(components),len(faults),width=int(package["width"]))
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True));model.eval()
    capabilities=[ModelCapability.FAULT_RANKING,ModelCapability.NORMAL_CLASS,ModelCapability.TEMPORAL_CONTEXT]
    calibrated=bool(package.get("calibrated"))
    calibration_id=package.get("calibration_id") if calibrated else None
    if calibrated:
        if not calibration_id: raise ValueError("calibrated package requires calibration_id")
        capabilities.append(ModelCapability.CALIBRATED_PROBABILITY)
    manifest=DataDrivenModelManifest(
        model_id=model_id,model_version=model_version,display_name=f"Mission three-task TCN {model_id}",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.PROBABILITIES,capabilities=capabilities,
        supported_fault_ids=list(faults),feature_schema_ids=[f"astrograph.generic-window.{model_id}.{model_version}.v1"],
        input_contract={"window_length":int(package["window_size"]),"channel_order":list(channels),"channel_units":list(units)},
        calibration_id=calibration_id,training_dataset_ids=[str(package.get("training_data_sha256","unknown"))],
        claim_boundary="Strict admitted three-head TCN execution. Localization denotes affected-component output unless independent root-cause evidence is present.",
        metadata={"candidate_manifest_sha256":file_sha(path),"training_manifest_sha256":package.get("training_manifest_sha256")},
    )
    return MissionThreeTaskTCNProvider(
        manifest=manifest,model=model,selected_channels=channels,channel_units=units,fault_ids=faults,component_ids=components,
        mean=mean,scale=scale,model_artifact_sha256=digest,
        detection_threshold=float(package.get("detection_threshold",0.5)),
    )


def build_strict_registry(provider: MissionThreeTaskTCNProvider) -> DataDrivenModelRegistry:
    contracts=build_temporal_window_contract_registry(
        provider=provider,model_artifact_sha256=provider.model_artifact_sha256,
        selected_channels=provider.selected_channels,window_length=int(provider.manifest.input_contract["window_length"]),
        allow_missing_values=False,
    )
    registry=DataDrivenModelRegistry(
        admission_registry=contracts,admission_signer=AdmissionSigner.from_environment_or_ephemeral(),require_admission=True,
    )
    registry.register(provider)
    return registry


__all__=["MissionThreeTaskTCNProvider","load_mission_three_task_tcn_provider","build_strict_registry"]
