from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.audit.store import sha256_json
from astrograph.errors import IdempotencyConflictError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestLifecycleStatus(str, Enum):
    RECEIVED = "RECEIVED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RequestTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RequestLifecycleStatus
    at: str = Field(default_factory=_now)
    details: dict[str, Any] = Field(default_factory=dict)


class RequestLifecycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-request-lifecycle-record.v1"
    request_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RequestLifecycleStatus
    received_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    response_hash: str = ""
    audit_id: str = ""
    diagnosis_status: str = ""
    error_code: str = ""
    error_message: str = ""
    replay_count: int = Field(default=0, ge=0)
    conflict_count: int = Field(default=0, ge=0)
    transitions: list[RequestTransition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JsonRequestLifecycleStore:
    schema_version = "astrograph-request-lifecycle-store.v1"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "records": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported request lifecycle store schema")
        payload.setdefault("records", {})
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def receive(self, request_id: str, request_payload: Any) -> RequestLifecycleRecord:
        request_hash = sha256_json(request_payload)
        with self._lock:
            payload = self._load()
            current_raw = payload["records"].get(request_id)
            if current_raw:
                current = RequestLifecycleRecord.model_validate(current_raw)
                if current.request_hash != request_hash:
                    current.conflict_count += 1
                    current.updated_at = _now()
                    payload["records"][request_id] = current.model_dump(mode="json")
                    self._save(payload)
                    raise IdempotencyConflictError(
                        "The request_id was already used with a different request payload.",
                        details={"request_id": request_id, "conflict_count": current.conflict_count},
                    )
                return current
            record = RequestLifecycleRecord(
                request_id=request_id,
                request_hash=request_hash,
                status=RequestLifecycleStatus.RECEIVED,
                transitions=[RequestTransition(status=RequestLifecycleStatus.RECEIVED)],
            )
            payload["records"][request_id] = record.model_dump(mode="json")
            self._save(payload)
            return record

    def _transition(self, request_id: str, status: RequestLifecycleStatus, **details: Any) -> RequestLifecycleRecord:
        with self._lock:
            payload = self._load()
            raw = payload["records"].get(request_id)
            if raw is None:
                raise KeyError(request_id)
            record = RequestLifecycleRecord.model_validate(raw)
            if record.status == status and status in {RequestLifecycleStatus.RUNNING, RequestLifecycleStatus.COMPLETED}:
                return record
            record.status = status
            record.updated_at = _now()
            record.transitions.append(RequestTransition(status=status, details=details))
            for key in ["response_hash", "audit_id", "diagnosis_status", "error_code", "error_message"]:
                if key in details:
                    setattr(record, key, str(details[key]))
            payload["records"][request_id] = record.model_dump(mode="json")
            self._save(payload)
            return record

    def mark_running(self, request_id: str) -> RequestLifecycleRecord:
        return self._transition(request_id, RequestLifecycleStatus.RUNNING)

    def mark_completed(self, request_id: str, response: dict[str, Any]) -> RequestLifecycleRecord:
        audit = response.get("audit", {})
        diagnosis = response.get("diagnosis", {})
        return self._transition(
            request_id,
            RequestLifecycleStatus.COMPLETED,
            response_hash=sha256_json(response),
            audit_id=audit.get("audit_id", ""),
            diagnosis_status=diagnosis.get("status", ""),
        )

    def mark_failed(self, request_id: str, exc: Exception) -> RequestLifecycleRecord:
        return self._transition(
            request_id,
            RequestLifecycleStatus.FAILED,
            error_code=getattr(exc, "code", "ASTROGRAPH_INTERNAL_ERROR"),
            error_message=str(exc),
        )

    def mark_replay(self, request_id: str) -> RequestLifecycleRecord:
        with self._lock:
            payload = self._load()
            raw = payload["records"].get(request_id)
            if raw is None:
                raise KeyError(request_id)
            record = RequestLifecycleRecord.model_validate(raw)
            record.replay_count += 1
            record.updated_at = _now()
            payload["records"][request_id] = record.model_dump(mode="json")
            self._save(payload)
            return record

    def recover_interrupted(self, *, include_received: bool = True) -> dict[str, Any]:
        """Mark non-terminal records as FAILED after a service restart.

        This does not re-run diagnosis. It closes ambiguous in-flight state so callers
        can safely resubmit with the same request_id or a new request_id.
        """
        recovered: list[str] = []
        with self._lock:
            payload = self._load()
            for request_id, raw in list(payload["records"].items()):
                record = RequestLifecycleRecord.model_validate(raw)
                if record.status == RequestLifecycleStatus.RUNNING or (
                    include_received and record.status == RequestLifecycleStatus.RECEIVED
                ):
                    previous = record.status.value
                    record.status = RequestLifecycleStatus.FAILED
                    record.updated_at = _now()
                    record.error_code = "ASTROGRAPH_RECOVERED_AFTER_RESTART"
                    record.error_message = f"Recovered non-terminal request after restart; previous_status={previous}."
                    record.transitions.append(
                        RequestTransition(
                            status=RequestLifecycleStatus.FAILED,
                            details={
                                "recovery": True,
                                "previous_status": previous,
                                "error_code": record.error_code,
                            },
                        )
                    )
                    payload["records"][request_id] = record.model_dump(mode="json")
                    recovered.append(request_id)
            self._save(payload)
        return {
            "schema_version": "astrograph-request-recovery-report.v1",
            "recovered_count": len(recovered),
            "recovered_request_ids": sorted(recovered),
        }

    def replay_timeline(self, request_id: str) -> dict[str, Any]:
        record = self.get(request_id)
        if record is None:
            raise KeyError(request_id)
        terminal = record.status in {RequestLifecycleStatus.COMPLETED, RequestLifecycleStatus.FAILED}
        transition_statuses = [item.status.value for item in record.transitions]
        return {
            "schema_version": "astrograph-request-timeline.v1",
            "request_id": request_id,
            "current_status": record.status.value,
            "terminal": terminal,
            "transition_count": len(record.transitions),
            "transitions": [item.model_dump(mode="json") for item in record.transitions],
            "transition_statuses": transition_statuses,
            "response_hash": record.response_hash,
            "audit_id": record.audit_id,
            "diagnosis_status": record.diagnosis_status,
            "error_code": record.error_code,
            "replay_count": record.replay_count,
            "conflict_count": record.conflict_count,
        }

    def get(self, request_id: str) -> RequestLifecycleRecord | None:
        raw = self._load()["records"].get(request_id)
        return None if raw is None else RequestLifecycleRecord.model_validate(raw)

    def list_records(self) -> list[RequestLifecycleRecord]:
        return sorted(
            [RequestLifecycleRecord.model_validate(item) for item in self._load()["records"].values()],
            key=lambda item: (item.received_at, item.request_id),
        )
