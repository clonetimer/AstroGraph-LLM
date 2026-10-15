from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.assurance.artifact_paths import ArtifactRoot
from astrograph.audit.store import canonical_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentStatus(str, Enum):
    REGISTERED = "REGISTERED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VersionCombination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_version: str
    dataset_manifest_sha256: str = ""
    knowledge_release_id: str
    knowledge_version: str
    knowledge_edges_sha256: str = ""
    policy_version: str
    policy_sha256: str
    anomaly_rule_set_version: str
    anomaly_rule_sha256: str
    condition_catalog_version: str
    condition_catalog_sha256: str
    code_version: str


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-experiment-record.v1"
    experiment_id: str = Field(min_length=16)
    purpose: str = Field(min_length=1)
    combination: VersionCombination
    status: ExperimentStatus = ExperimentStatus.REGISTERED
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Version-combination experiment registry only. Registration does not establish model accuracy, "
        "flight validity or statistical significance."
    )

    @classmethod
    def create_id(cls, purpose: str, combination: VersionCombination) -> str:
        payload = canonical_json({"purpose": purpose, "combination": combination.model_dump(mode="json")})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class JsonExperimentRegistry:
    schema_version = "astrograph-experiment-registry.v1"

    def __init__(
        self,
        path: str | Path,
        *,
        source_root: str | Path | None = None,
    ):
        requested = Path(path).expanduser()
        self._artifact_root = ArtifactRoot(requested.parent, source_root=source_root)
        self._relative_name = requested.name
        if not self._relative_name:
            raise ValueError("experiment registry path must include a filename")
        self.path = self._artifact_root.resolve(self._relative_name)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "experiments": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported experiment registry schema")
        payload.setdefault("experiments", [])
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path = self._artifact_root.write_json(self._relative_name, payload)

    def register(self, record: ExperimentRecord) -> ExperimentRecord:
        payload = self._load()
        serialized = record.model_dump(mode="json")
        for item in payload["experiments"]:
            if item["experiment_id"] != record.experiment_id:
                continue
            existing = ExperimentRecord.model_validate(item)
            if canonical_json(existing.combination.model_dump(mode="json")) != canonical_json(record.combination.model_dump(mode="json")):
                raise ValueError("experiment_id collision with a different version combination")
            return existing
        payload["experiments"].append(serialized)
        payload["experiments"].sort(key=lambda item: (item["created_at"], item["experiment_id"]))
        self._save(payload)
        return record

    def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        result_summary: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        payload = self._load()
        for index, item in enumerate(payload["experiments"]):
            if item["experiment_id"] != experiment_id:
                continue
            record = ExperimentRecord.model_validate(item)
            record.status = status
            record.updated_at = _now()
            if result_summary is not None:
                record.result_summary = result_summary
            payload["experiments"][index] = record.model_dump(mode="json")
            self._save(payload)
            return record
        raise KeyError(experiment_id)

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        for item in self._load()["experiments"]:
            if item["experiment_id"] == experiment_id:
                return ExperimentRecord.model_validate(item)
        return None

    def list_records(self) -> list[ExperimentRecord]:
        return [ExperimentRecord.model_validate(item) for item in self._load()["experiments"]]
