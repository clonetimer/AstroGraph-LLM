from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.audit.store import canonical_json
from astrograph.assurance.artifact_paths import ArtifactRoot


class DiagnosticCaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-diagnostic-case-record.v2"
    case_record_id: str = Field(min_length=16)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_type: Literal["anomaly_list", "anomaly_episode", "paired_residual_run"] = "anomaly_list"
    dataset_id: str = ""
    run_id: str = ""
    paired_nominal_run_id: str | None = None
    split: str = ""
    synthetic_case_id: str = ""
    request_id: str
    detector_id: str = ""
    rule_set_version: str = ""
    anomaly_count: int = Field(ge=0)
    anomaly_hash: str
    episode_count: int = Field(default=0, ge=0)
    episode_ids: list[str] = Field(default_factory=list)
    episode_hash: str = ""
    diagnosis_status: str
    conclusion_fault_id: str | None = None
    conclusion_fault_name: str | None = None
    top_score: float = 0.0
    score_margin: float = 0.0
    policy_version: str
    knowledge_version: str
    audit_id: str
    diagnosis_path: str = ""
    report_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = "Synthetic engineering diagnostic case; not formal accuracy evidence."

    @classmethod
    def create_id(
        cls,
        *,
        dataset_id: str,
        run_id: str,
        detector_id: str,
        rule_set_version: str,
        policy_version: str,
        knowledge_version: str,
        source_type: str = "paired_residual_run",
        source_fingerprint: str = "",
    ) -> str:
        payload = "|".join(
            [
                dataset_id,
                run_id,
                detector_id,
                rule_set_version,
                policy_version,
                knowledge_version,
                source_type,
                source_fingerprint,
            ]
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]


class DiagnosticCaseRegistry:
    schema_version = "astrograph-diagnostic-case-registry.v2"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _migrate_record(item: dict[str, Any]) -> dict[str, Any]:
        if item.get("schema_version") == "astrograph-diagnostic-case-record.v2":
            return item
        migrated = dict(item)
        migrated["schema_version"] = "astrograph-diagnostic-case-record.v2"
        migrated.setdefault("source_type", "paired_residual_run")
        migrated.setdefault("episode_count", 0)
        migrated.setdefault("episode_ids", [])
        migrated.setdefault("episode_hash", "")
        return migrated

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "updated_at": "", "cases": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") not in {self.schema_version, "astrograph-diagnostic-case-registry.v1"}:
            raise ValueError("unsupported diagnostic case registry schema")
        cases = [self._migrate_record(item) for item in payload.get("cases", [])]
        return {"schema_version": self.schema_version, "updated_at": payload.get("updated_at", ""), "cases": cases}


    def _write_payload(self, payload: dict[str, Any]) -> None:
        source_root = Path(__file__).resolve().parents[3]
        artifacts = ArtifactRoot(self.path.parent, source_root=source_root)
        artifacts.write_text(
            self.path.name,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def upsert(self, record: DiagnosticCaseRecord) -> None:
        payload = self._load()
        cases = payload["cases"]
        serialized = record.model_dump(mode="json")
        for index, current in enumerate(cases):
            if current["case_record_id"] != record.case_record_id:
                continue
            volatile = {"created_at", "audit_id"}
            current_stable = {key: value for key, value in current.items() if key not in volatile}
            new_stable = {key: value for key, value in serialized.items() if key not in volatile}
            if canonical_json(current_stable) != canonical_json(new_stable):
                raise ValueError(f"case_record_id collision with different content: {record.case_record_id}")
            cases[index] = serialized
            break
        else:
            cases.append(serialized)
        cases.sort(key=lambda item: (item.get("dataset_id", ""), item.get("run_id", ""), item["case_record_id"]))
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_payload(payload)

    def get(self, case_record_id: str) -> DiagnosticCaseRecord | None:
        for record in self.list_records():
            if record.case_record_id == case_record_id:
                return record
        return None

    def delete(self, case_record_id: str) -> bool:
        payload = self._load()
        remaining = [item for item in payload["cases"] if item["case_record_id"] != case_record_id]
        if len(remaining) == len(payload["cases"]):
            return False
        payload["cases"] = remaining
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_payload(payload)
        return True

    def list_records(self) -> list[DiagnosticCaseRecord]:
        return [DiagnosticCaseRecord.model_validate(item) for item in self._load()["cases"]]
