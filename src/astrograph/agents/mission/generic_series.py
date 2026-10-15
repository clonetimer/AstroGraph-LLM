"""Host-owned generic time-series adapter for Mission diagnostics.

The adapter deliberately reuses DatasetDiagnosisRunner.  It does not invent a
classifier for arbitrary channels: detection is always available for numeric
series, while physical causes require an explicit/registered signal mapping and
knowledge edges.  Classification/localization models are handled separately by
the model capability registry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.data.adapter import SIGNAL_TO_ANOMALY, SIGNAL_TO_UNIT, engineering_unit_matches
from astrograph.data_sources import InMemoryDataSource
from astrograph.diagnosis import DatasetDiagnosisRequest, DatasetDiagnosisRunner, JsonDiagnosisRunStateStore
from astrograph.knowledge.repository import CsvKnowledgeRepository
from .generic_series_contract import build_generic_series_input_contract
from .identifiability import screen_single_signal_identifiability
from .storage import file_sha


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeriesChannel(_Strict):
    name: str = Field(min_length=1)
    unit: str | None = Field(default=None, max_length=64)
    anomaly_id: str | None = Field(default=None, min_length=1)
    anomaly_name: str | None = Field(default=None, min_length=1)
    component: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def anomaly_pair(self):
        if (self.anomaly_id is None) != (self.anomaly_name is None):
            raise ValueError("anomaly_id and anomaly_name must be supplied together")
        return self


class GenericSeriesManifest(_Strict):
    schema_version: Literal["astrograph-generic-series-manifest.v1"] = "astrograph-generic-series-manifest.v1"
    data_path: str = Field(min_length=1)
    entity_id: str = Field(default="series", min_length=1, max_length=200)
    timestamp_column: str | None = None
    channels: list[SeriesChannel] = Field(min_length=1, max_length=512)
    start: str | float | int | None = None
    end: str | float | int | None = None
    baseline_fraction: float = Field(default=0.2, gt=0.0, lt=0.8)
    z_threshold: float = Field(default=6.0, gt=0.0)
    minimum_baseline_points: int = Field(default=10, ge=3)
    knowledge_edges_path: str | None = None
    operating_condition: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("channels")
    @classmethod
    def unique_names(cls, value: list[SeriesChannel]) -> list[SeriesChannel]:
        names=[x.name for x in value]
        if len(names)!=len(set(names)):
            raise ValueError("generic series channel names must be unique")
        return value


class GenericSeriesAdapter:
    """Run generic series detection and source-bound graph reasoning."""

    def __init__(self, *, project_root: str | Path, workspace_root: str | Path, read_path):
        self.project_root=Path(project_root).resolve()
        self.workspace_root=Path(workspace_root).resolve()
        self.read_path=read_path

    def _resolve_relative(self, manifest_path: Path, value: str, *, directory: bool=False) -> Path:
        candidate=Path(value)
        if not candidate.is_absolute():
            candidate=manifest_path.parent/candidate
        return self.read_path(str(candidate), directory=directory)

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        suffix=path.suffix.lower()
        if suffix==".csv": return pd.read_csv(path, low_memory=False)
        if suffix in {".jsonl", ".ndjson"}: return pd.read_json(path, lines=True)
        if suffix==".json": return pd.read_json(path)
        if suffix==".parquet": return pd.read_parquet(path)
        if suffix==".feather": return pd.read_feather(path)
        raise ValueError(f"unsupported generic series format: {suffix}")

    @staticmethod
    def _mapping(manifest: GenericSeriesManifest) -> tuple[dict[str, dict[str,str]], list[dict[str,Any]]]:
        # The generic adapter owns semantic admission, including unit checks.
        # Disable DatasetDiagnosisRunner's historical name-only fallback so an
        # incompatible or missing unit cannot silently regain a physical meaning.
        mapping={"__policy__":{"allow_builtin_fallback":"false"}};receipts=[]
        for channel in manifest.channels:
            source="UNMAPPED"
            pair=None
            expected_unit=None
            if channel.anomaly_id is not None:
                pair=(channel.anomaly_id, channel.anomaly_name);source="MANIFEST_EXPLICIT"
            elif channel.name in SIGNAL_TO_ANOMALY:
                expected_unit=SIGNAL_TO_UNIT.get(channel.name)
                if expected_unit is None:
                    source="UNIT_CONTRACT_NOT_REGISTERED"
                elif channel.unit is None:
                    source="UNIT_REQUIRED"
                elif not engineering_unit_matches(expected_unit, channel.unit):
                    source="UNIT_MISMATCH"
                else:
                    pair=SIGNAL_TO_ANOMALY[channel.name];source="BUILTIN_REGISTERED_SIGNAL"
            else:
                expected_unit=None
            if pair is not None:
                mapping[channel.name]={"anomaly_id":str(pair[0]),"anomaly_name":str(pair[1])}
            receipts.append({"channel":channel.name,"unit":channel.unit,"expected_unit":expected_unit,
                             "mapping_source":source,"semantic_mapping_allowed":pair is not None,
                             "anomaly_id":None if pair is None else pair[0],"component":channel.component})
        return mapping,receipts

    def run(self, manifest_path: str | Path, *, request_id: str) -> dict[str, Any]:
        manifest_path=self.read_path(str(manifest_path))
        manifest=GenericSeriesManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        data_path=self._resolve_relative(manifest_path,manifest.data_path)
        frame=self._read_table(data_path)
        wanted=[x.name for x in manifest.channels]
        required=wanted+([manifest.timestamp_column] if manifest.timestamp_column else [])
        missing=sorted(set(required)-set(str(c) for c in frame.columns))
        if missing: raise ValueError(f"generic series is missing declared columns: {missing}")
        for name in wanted:
            converted=pd.to_numeric(frame[name],errors="coerce")
            if converted.notna().sum()<manifest.minimum_baseline_points:
                raise ValueError(f"channel {name} has too few finite numeric samples")
            frame[name]=converted
        keep=([manifest.timestamp_column] if manifest.timestamp_column else [])+wanted
        frame=frame[keep].copy()
        input_contract=build_generic_series_input_contract(
            frame,timestamp_column=manifest.timestamp_column,channel_order=wanted,
            channel_units=[x.unit for x in manifest.channels],source_data_sha256=file_sha(data_path))
        mapping,mapping_receipt=self._mapping(manifest)
        identifiability=screen_single_signal_identifiability(
            wanted, fault_catalog_path=self.project_root/"configs/fault_modes/lab_fault_catalog.json"
        )
        edge_value=manifest.knowledge_edges_path or "configs/knowledge/generic_series_fault_catalog_edges.csv"
        edge_path=self._resolve_relative(manifest_path,edge_value) if manifest.knowledge_edges_path else self.read_path(str(self.project_root/edge_value))
        repository=CsvKnowledgeRepository(edge_path,knowledge_version="astrograph-generic-series-e0.2")
        source=InMemoryDataSource({manifest.entity_id:frame},timestamp_column=manifest.timestamp_column)
        state_root=self.workspace_root/"generic-series-state"
        runner=DatasetDiagnosisRunner(source,repository,state_store=JsonDiagnosisRunStateStore(state_root))
        request=DatasetDiagnosisRequest(request_id=request_id,entity_id=manifest.entity_id,start=manifest.start,end=manifest.end,
            channels=wanted,baseline_fraction=manifest.baseline_fraction,z_threshold=manifest.z_threshold,
            minimum_baseline_points=manifest.minimum_baseline_points,signal_mapping=mapping,
            operating_condition=manifest.operating_condition,
            metadata={**manifest.metadata,"generic_series_manifest":str(manifest_path.name),"labels_not_used":True})
        result=runner.run(request,output_dir=self.workspace_root/"generic-series-output")
        mapped=sum(1 for x in result.detections if x.mapped)
        return {
            "status":"SUCCEEDED",
            "mode":"GENERIC_SERIES_DATASET_RUNNER",
            "series_manifest":manifest.model_dump(mode="json"),
            "input_contract":input_contract.model_dump(mode="json"),
            "mapping_receipt":mapping_receipt,
            "identifiability":identifiability.model_dump(mode="json"),
            "channel_count":len(wanted),
            "mapped_detection_count":mapped,
            "unmapped_detection_count":len(result.detections)-mapped,
            "diagnosis":result.model_dump(mode="json"),
            "knowledge_contract":{
                "knowledge_version":repository.knowledge_version,
                "edges_path":str(edge_path),
                "edges_sha256":repository.content_hash,
                "review_status":"PROVISIONAL_E0_SOURCE_DERIVED",
                "source_derivation":"Exact channel matches only; CONTEXT-only and non-exact channel names are excluded.",
            },
            "capability_boundary":{
                "detection":"AVAILABLE_GENERIC_ROBUST_BASELINE",
                "physical_cause":"AVAILABLE_ONLY_FOR_MAPPED_ANOMALIES_WITH_KNOWLEDGE",
                "classification":"NOT_EXECUTED_WITHOUT_COMPATIBLE_MODEL",
                "localization":"NOT_EXECUTED_WITHOUT_COMPATIBLE_MODEL",
                "truth_or_labels_used":False,
            },
        }
