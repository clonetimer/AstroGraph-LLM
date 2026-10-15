"""Host-owned bridge from capability discovery to strictly admitted model execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.assurance import canonical_sha256
from astrograph.data_driven import ModelInferenceRequest, ObservationScope
from astrograph.integrations.mission_tcn_candidate import load_mission_three_task_tcn_provider, build_strict_registry
from astrograph.integrations.adcs_three_task_torch import (
    load_adcs_three_task_torch_package, build_registry_from_adcs_three_task_provider,
)

from .generic_series import GenericSeriesManifest
from .generic_series_contract import build_generic_series_input_contract, materialize_model_window
from .storage import file_sha


class _Strict(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)


class ModelExecutionBinding(_Strict):
    binding_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    provider_kind: Literal["MISSION_THREE_TASK_TCN_PACKAGE", "ADCS_NATIVE_THREE_TASK_PACKAGE"]
    package_manifest: str = Field(min_length=1)
    package_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_channels: tuple[str, ...] = ()
    channel_units: tuple[str, ...] = ()
    enabled: bool = True
    online_allowed: bool = True
    claim_boundary: str = Field(default="Host-bound strict model execution; capability compatibility alone never authorizes this binding.",min_length=1)

    @model_validator(mode="after")
    def provider_contract(self):
        if self.provider_kind == "ADCS_NATIVE_THREE_TASK_PACKAGE":
            if not self.required_channels:
                raise ValueError("ADCS native binding requires frozen required_channels")
            if len(self.required_channels) != len(set(self.required_channels)):
                raise ValueError("ADCS native required_channels must be unique")
            if len(self.channel_units) != len(self.required_channels):
                raise ValueError("ADCS native channel_units must align with required_channels")
        return self


class ModelExecutionBindingCatalog(_Strict):
    schema_version: Literal["astrograph-model-execution-bindings.v1"] = "astrograph-model-execution-bindings.v1"
    bindings: tuple[ModelExecutionBinding, ...] = ()

    @field_validator("bindings")
    @classmethod
    def unique(cls,value):
        ids=[x.binding_id for x in value];caps=[x.capability_id for x in value]
        if len(ids)!=len(set(ids)): raise ValueError("execution binding IDs must be unique")
        if len(caps)!=len(set(caps)): raise ValueError("only one active execution binding per capability is allowed")
        return value


class ModelExecutionResolver:
    def __init__(self,catalog:ModelExecutionBindingCatalog,*,project_root:str|Path,read_path):
        self.catalog=catalog;self.project_root=Path(project_root).resolve();self.read_path=read_path
        self.by_capability={x.capability_id:x for x in catalog.bindings if x.enabled}

    @classmethod
    def from_json(cls,path:str|Path,*,project_root:str|Path,read_path):
        p=Path(path)
        if not p.is_absolute(): p=Path(project_root)/p
        p=read_path(str(p))
        return cls(ModelExecutionBindingCatalog.model_validate_json(p.read_text(encoding="utf-8")),project_root=project_root,read_path=read_path)

    @staticmethod
    def _read_table(path:Path)->pd.DataFrame:
        suffix=path.suffix.lower()
        if suffix==".csv": return pd.read_csv(path,low_memory=False)
        if suffix in {".jsonl",".ndjson"}: return pd.read_json(path,lines=True)
        if suffix==".json": return pd.read_json(path)
        if suffix==".parquet": return pd.read_parquet(path)
        if suffix==".feather": return pd.read_feather(path)
        raise ValueError(f"unsupported generic series format: {suffix}")

    def _resolve_package(self,binding:ModelExecutionBinding)->Path:
        p=Path(binding.package_manifest)
        if not p.is_absolute(): p=self.project_root/p
        return self.read_path(str(p))

    def profile_hint(self, manifest: GenericSeriesManifest) -> dict:
        """Return host-bound domain hints only for an exact execution contract match."""
        names=tuple(x.name for x in manifest.channels)
        units=tuple((x.unit or "") for x in manifest.channels)
        for binding in self.by_capability.values():
            if binding.provider_kind != "ADCS_NATIVE_THREE_TASK_PACKAGE":
                continue
            if names == binding.required_channels and units == binding.channel_units:
                return {"domain":"adcs","bound_artifact_contract":True,"binding_id":binding.binding_id}
        return {"domain":"generic","bound_artifact_contract":False,"binding_id":None}

    def _generic_frame_contract(self, *, series_manifest_path:Path, binding:ModelExecutionBinding):
        series_manifest=GenericSeriesManifest.model_validate_json(series_manifest_path.read_text(encoding="utf-8"))
        data_path=Path(series_manifest.data_path)
        if not data_path.is_absolute(): data_path=series_manifest_path.parent/data_path
        data_path=self.read_path(str(data_path));frame=self._read_table(data_path)
        wanted=[x.name for x in series_manifest.channels]
        missing=sorted(set(wanted)-set(map(str,frame.columns)))
        if missing: raise ValueError(f"generic series missing model-bound channels: {missing}")
        for name in wanted: frame[name]=pd.to_numeric(frame[name],errors="coerce")
        keep=([series_manifest.timestamp_column] if series_manifest.timestamp_column else [])+wanted;frame=frame[keep].copy()
        contract=build_generic_series_input_contract(frame,timestamp_column=series_manifest.timestamp_column,
            channel_order=wanted,channel_units=[x.unit for x in series_manifest.channels],source_data_sha256=file_sha(data_path))
        return series_manifest,data_path,frame,wanted,contract

    def _execute_adcs_native(self, *, binding:ModelExecutionBinding, series_manifest_path:Path, episode_id:str)->dict:
        package_path=self._resolve_package(binding)
        if file_sha(package_path)!=binding.package_manifest_sha256:
            raise ValueError("execution binding package manifest changed")
        provider=load_adcs_three_task_torch_package(package_path,expected_sha256=binding.package_manifest_sha256,device="cpu")
        if provider.manifest.model_id != binding.model_id or provider.manifest.model_version != binding.model_version:
            raise ValueError("ADCS native binding model identity mismatch")
        series_manifest,data_path,frame,wanted,contract=self._generic_frame_contract(series_manifest_path=series_manifest_path,binding=binding)
        units=tuple((x.unit or "") for x in series_manifest.channels)
        if tuple(wanted)!=binding.required_channels or tuple(wanted)!=provider.selected_channels:
            raise ValueError("ADCS native execution requires exact frozen provider channel order")
        if units!=binding.channel_units:
            raise ValueError("ADCS native execution requires exact frozen channel units")
        window=materialize_model_window(frame,contract=contract,window_size=int(provider.manifest.input_contract["window_length"]))
        payload={"channel_order":list(window.channel_order),"telemetry_window":[list(row) for row in window.values]}
        payload_hash=canonical_sha256(payload)
        if payload_hash!=window.input_data_sha256:
            raise ValueError("generic ADCS model window hash diverged from admission payload hash")
        registry=build_registry_from_adcs_three_task_provider(provider)
        request=ModelInferenceRequest(
            episode_id=episode_id,model_id=provider.manifest.model_id,model_version=provider.manifest.model_version,
            feature_schema_id=provider.manifest.feature_schema_ids[0],input_data_sha256=payload_hash,
            registered_fault_ids=list(provider.manifest.supported_fault_ids),payload=payload,
            observation_scope=ObservationScope(scope_id="GS-"+payload_hash[:20],source_id="generic-series-adcs",
                window_start_s=float(window.start_row),window_end_s=float(window.end_row_exclusive-1),source_data_sha256=contract.source_data_sha256),
        )
        prediction=registry.predict(request);raw=prediction.metadata.get("raw_output_metadata",{}) or {}
        class_rows=list(prediction.classes)
        heads={
            "detection":{"detected":bool(raw.get("detected",False)),"anomaly_probability":raw.get("detection_probability"),
                         "threshold":raw.get("detection_threshold")},
            "localization":{"predicted_component":raw.get("predicted_component"),
                            "classes":list((raw.get("component_probabilities") or {}).keys()),
                            "probabilities":list((raw.get("component_probabilities") or {}).values()),
                            "root_cause_supported":False},
            "classification":{"predicted_fault":class_rows[0].class_id if class_rows else None,
                              "classes":[x.class_id for x in class_rows],
                              "probabilities":[x.probability for x in class_rows]},
        }
        return {
            "status":"SUCCEEDED","execution_authorized":True,"provider_id":"ADCS_NATIVE_THREE_TASK_V2",
            "capability_id":binding.capability_id,"binding_id":binding.binding_id,
            "model_id":prediction.model_id,"model_version":prediction.model_version,
            "input_contract":contract.model_dump(mode="json"),"model_window":window.model_dump(mode="json"),
            "prediction":prediction.model_dump(mode="json"),"heads":heads,
            "admission":prediction.metadata.get("model_admission"),
            "claim_boundary":binding.claim_boundary,
        }

    def execute_generic(self,*,series_manifest_path:Path,capability_profile:dict,episode_id:str)->dict:
        compatible=[]
        for head,result in capability_profile.get("tasks",{}).items():
            for row in result.get("compatible",[]):
                compatible.append((head,row))
        by_cap={row["capability_id"]:row for _,row in compatible}
        candidates=[]
        for capability_id,row in by_cap.items():
            binding=self.by_capability.get(capability_id)
            if binding is None: continue
            if binding.model_id!=row["model_id"] or binding.model_version!=row["model_version"]:
                candidates.append({"capability_id":capability_id,"binding_id":binding.binding_id,"status":"REJECTED","reason":"CAPABILITY_BINDING_MODEL_IDENTITY_MISMATCH"});continue
            if not binding.online_allowed:
                candidates.append({"capability_id":capability_id,"binding_id":binding.binding_id,"status":"REJECTED","reason":"BINDING_NOT_ONLINE_ALLOWED"});continue
            candidates.append({"capability_id":capability_id,"binding_id":binding.binding_id,"status":"ELIGIBLE","priority":row.get("priority",0)})
        eligible=sorted((x for x in candidates if x["status"]=="ELIGIBLE"),key=lambda x:(-x["priority"],x["binding_id"]))
        if not eligible:
            return {"status":"NOT_AVAILABLE","execution_authorized":False,"candidates":candidates,
                    "claim_boundary":"No compatible capability has an enabled strict execution binding."}
        chosen=eligible[0];binding=self.by_capability[chosen["capability_id"]]
        if binding.provider_kind == "ADCS_NATIVE_THREE_TASK_PACKAGE":
            result=self._execute_adcs_native(binding=binding,series_manifest_path=series_manifest_path,episode_id=episode_id)
            result["candidates"]=candidates
            return result
        package_path=self._resolve_package(binding)
        if file_sha(package_path)!=binding.package_manifest_sha256:
            raise ValueError("execution binding package manifest changed")
        provider=load_mission_three_task_tcn_provider(package_path,model_id=binding.model_id,model_version=binding.model_version,
            expected_manifest_sha256=binding.package_manifest_sha256)
        series_manifest=GenericSeriesManifest.model_validate_json(series_manifest_path.read_text(encoding="utf-8"))
        data_path=Path(series_manifest.data_path)
        if not data_path.is_absolute(): data_path=series_manifest_path.parent/data_path
        data_path=self.read_path(str(data_path));frame=self._read_table(data_path)
        wanted=[x.name for x in series_manifest.channels]
        missing=sorted(set(wanted)-set(map(str,frame.columns)))
        if missing: raise ValueError(f"generic series missing model-bound channels: {missing}")
        for name in wanted: frame[name]=pd.to_numeric(frame[name],errors="coerce")
        keep=([series_manifest.timestamp_column] if series_manifest.timestamp_column else [])+wanted;frame=frame[keep].copy()
        contract=build_generic_series_input_contract(frame,timestamp_column=series_manifest.timestamp_column,
            channel_order=wanted,channel_units=[x.unit for x in series_manifest.channels],source_data_sha256=file_sha(data_path))
        if tuple(wanted)!=provider.selected_channels:
            raise ValueError("execution binding requires exact provider channel order")
        if tuple((x.unit or "") for x in series_manifest.channels)!=provider.channel_units:
            raise ValueError("execution binding requires exact provider channel units")
        window=materialize_model_window(frame,contract=contract,window_size=int(provider.manifest.input_contract["window_length"]))
        payload={"channel_order":list(window.channel_order),"telemetry_window":[list(row) for row in window.values]}
        payload_hash=canonical_sha256(payload)
        if payload_hash!=window.input_data_sha256:
            raise ValueError("generic model window hash diverged from admission payload hash")
        registry=build_strict_registry(provider)
        request=ModelInferenceRequest(
            episode_id=episode_id,model_id=provider.manifest.model_id,model_version=provider.manifest.model_version,
            feature_schema_id=provider.manifest.feature_schema_ids[0],input_data_sha256=payload_hash,
            registered_fault_ids=list(provider.fault_ids),payload=payload,
            observation_scope=ObservationScope(scope_id="GS-"+payload_hash[:20],source_id="generic-series",
                window_start_s=float(window.start_row),window_end_s=float(window.end_row_exclusive-1),source_data_sha256=contract.source_data_sha256),
        )
        prediction=registry.predict(request);raw=prediction.metadata.get("raw_output_metadata",{})
        return {
            "status":"SUCCEEDED","execution_authorized":True,"capability_id":chosen["capability_id"],
            "binding_id":binding.binding_id,"model_id":prediction.model_id,"model_version":prediction.model_version,
            "input_contract":contract.model_dump(mode="json"),"model_window":window.model_dump(mode="json"),
            "prediction":prediction.model_dump(mode="json"),"heads":raw.get("head_trace",{}),
            "admission":prediction.metadata.get("model_admission"),"candidates":candidates,
            "claim_boundary":binding.claim_boundary,
        }


__all__=["ModelExecutionBinding","ModelExecutionBindingCatalog","ModelExecutionResolver"]
