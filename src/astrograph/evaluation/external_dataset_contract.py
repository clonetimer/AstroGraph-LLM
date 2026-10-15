"""R20 offline content admission. Never grants runtime fault or flight authority."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA = r"^[0-9a-f]{64}$"
PARTITIONS = {"MODEL_TRAIN", "CALIBRATION_FIT", "DEVELOPMENT_SELECTION", "CONFIRMATORY_EVALUATION", "EXTERNAL_VALIDATION"}
TASK_FIELDS = ("evaluation_task_id", "label_semantics_sha256", "target_universe_sha256")

def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_identity(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}

def seal(payload: dict, key="receipt_sha256") -> dict:
    return {**payload, key: digest({k:v for k,v in payload.items() if k != key})}

def verify_seal(payload: dict, key="receipt_sha256") -> None:
    if payload.get(key) != digest({k:v for k,v in payload.items() if k != key}):
        raise ValueError("CONTENT_IDENTITY_DRIFT")

def task_identity(payload: dict) -> dict:
    result = {k:payload.get(k) for k in TASK_FIELDS}
    if not all(isinstance(v,str) and v for v in result.values()):
        raise ValueError("TASK_IDENTITY_REQUIRED")
    for name in TASK_FIELDS[1:]:
        value=result[name]
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError("TASK_HASH_INVALID")
    return result

def require_same_task(calibration: dict, external: dict) -> None:
    try: equal = task_identity(calibration) == task_identity(external)
    except ValueError: equal = False
    if not equal: raise ValueError("CROSS_TASK_CALIBRATION_REUSE_BLOCKED")

class ExternalDatasetReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-external-evaluation-dataset-receipt.v1"] = "astrograph-external-evaluation-dataset-receipt.v1"
    dataset_id: str = Field(min_length=1)
    dataset_family: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=SHA)
    source_artifact_size_bytes: int = Field(ge=1)
    evaluation_task_id: str = Field(min_length=1)
    task_kind: str = Field(min_length=1)
    label_semantics: dict[str, Any]
    label_semantics_sha256: str = Field(pattern=SHA)
    target_universe: list[str] = Field(min_length=1)
    target_universe_sha256: str = Field(pattern=SHA)
    group_policy: dict[str, Any]
    truth_provenance: dict[str, Any]
    payload_manifest_sha256: str = Field(pattern=SHA)
    allowed_claims: list[str]
    forbidden_claims: list[str]
    receipt_sha256: str = Field(pattern=SHA)

    @model_validator(mode='after')
    def verify(self):
        if len(set(self.target_universe)) != len(self.target_universe): raise ValueError("TARGET_UNIVERSE_DUPLICATE")
        if digest(self.label_semantics)!=self.label_semantics_sha256 or digest(self.target_universe)!=self.target_universe_sha256:
            raise ValueError("TASK_SEMANTICS_DRIFT")
        verify_seal(self.model_dump(mode='json'))
        return self

class ExternalValidationReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal["astrograph-external-validation-receipt.v1"] = "astrograph-external-validation-receipt.v1"
    evaluation_task_id: str = Field(min_length=1)
    label_semantics_sha256: str = Field(pattern=SHA)
    target_universe_sha256: str = Field(pattern=SHA)
    dataset_receipt_sha256: str = Field(pattern=SHA)
    snapshot_receipt_sha256: str = Field(pattern=SHA)
    predictions_sha256: str = Field(pattern=SHA)
    evaluation_report_sha256: str = Field(pattern=SHA)
    track_id: str
    score_semantics: str = "UNCALIBRATED_ANOMALY_SCORE"
    status: str
    qualification: str = "NOT_FLIGHT_QUALIFIED"
    receipt_sha256: str = Field(pattern=SHA)

    @model_validator(mode='after')
    def verify(self):
        task_identity(self.model_dump())
        verify_seal(self.model_dump(mode='json'))
        if self.qualification!='NOT_FLIGHT_QUALIFIED':raise ValueError("QUALIFICATION_PROMOTION_FORBIDDEN")
        return self

def build_snapshot(data_root: Path, selected: list[Path], *, dataset_receipt: dict, truth_paths: list[Path], partitions: dict, groups: list[str]) -> dict:
    ExternalDatasetReceipt.model_validate(dataset_receipt)
    root=data_root.resolve()
    def row(p):
        p=p.resolve()
        if not p.is_relative_to(root):raise ValueError("SNAPSHOT_PATH_ESCAPES_DATA_ROOT")
        return {"relative_path":p.relative_to(root).as_posix(),**file_identity(p)}
    files=sorted([row(p) for p in selected],key=lambda r:r['relative_path'])
    if len(set(r['relative_path'] for r in files))!=len(files):raise ValueError("SNAPSHOT_DUPLICATE_FILE")
    return seal({"schema_version":"astrograph-external-dataset-snapshot-receipt.v1", "dataset_receipt_sha256":dataset_receipt['receipt_sha256'],
        "files":files,"truth_artifacts":sorted([row(p) for p in truth_paths],key=lambda r:r['relative_path']),
        "partition_manifest_sha256":digest(partitions),"group_identity_sha256":digest(sorted(groups)),"task_identity_sha256":digest(task_identity(dataset_receipt))})

def verify_snapshot(data_root: Path, receipt: dict, *, dataset_receipt=None, partitions=None, groups=None) -> None:
    verify_seal(receipt)
    if dataset_receipt is not None:
        ExternalDatasetReceipt.model_validate(dataset_receipt)
        if receipt['dataset_receipt_sha256']!=dataset_receipt['receipt_sha256'] or receipt['task_identity_sha256']!=digest(task_identity(dataset_receipt)):
            raise ValueError('DATASET_TASK_BINDING_DRIFT')
    if partitions is not None and receipt['partition_manifest_sha256']!=digest(partitions):raise ValueError('PARTITION_MANIFEST_DRIFT')
    if groups is not None and receipt['group_identity_sha256']!=digest(sorted(groups)):raise ValueError('GROUP_IDENTITY_DRIFT')
    root=data_root.resolve()
    for row in receipt['files']+receipt['truth_artifacts']:
        path=(root/row['relative_path']).resolve()
        if not path.is_relative_to(root) or file_identity(path)!={k:row[k] for k in ('size','sha256')}:
            raise ValueError("DATASET_OR_TRUTH_BYTES_DRIFT")
