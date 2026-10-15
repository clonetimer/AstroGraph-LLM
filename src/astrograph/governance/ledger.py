from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.audit.store import sha256_json
from astrograph.governance.models import utc_now


class LedgerEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-append-only-ledger-envelope.v1"
    sequence: int = Field(ge=1)
    recorded_at: str
    record_type: str = Field(min_length=1)
    previous_hash: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    payload: dict[str, Any]
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class LedgerAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-ledger-audit-result.v1"
    status: str
    record_count: int = Field(ge=0)
    last_hash: str = ""
    errors: list[str] = Field(default_factory=list)


class AppendOnlyJsonlLedger:
    """Append-only hash-chained JSONL ledger.

    The ledger detects line deletion, insertion, reordering and payload edits. It is
    not a substitute for an external signature or immutable storage, but it gives
    the release gate deterministic tamper evidence inside a source distribution.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    @staticmethod
    def _hashable_payload(
        *, sequence: int, recorded_at: str, record_type: str, previous_hash: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": "astrograph-append-only-ledger-envelope.v1",
            "sequence": sequence,
            "recorded_at": recorded_at,
            "record_type": record_type,
            "previous_hash": previous_hash,
            "payload": payload,
        }

    def append(self, record_type: str, record: BaseModel | dict[str, Any]) -> LedgerEnvelope:
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else dict(record)
        with self._lock:
            audit = self.audit()
            if audit.status != "PASS":
                raise ValueError("cannot append to an invalid ledger")
            sequence = audit.record_count + 1
            previous_hash = audit.last_hash
            recorded_at = utc_now()
            hashable = self._hashable_payload(
                sequence=sequence,
                recorded_at=recorded_at,
                record_type=record_type,
                previous_hash=previous_hash,
                payload=payload,
            )
            envelope = LedgerEnvelope(
                **hashable,
                record_hash=sha256_json(hashable),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(envelope.model_dump_json() + "\n")
            return envelope

    def audit(self) -> LedgerAuditResult:
        if not self.path.exists():
            return LedgerAuditResult(status="PASS", record_count=0)
        errors: list[str] = []
        previous_hash = ""
        record_count = 0
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                errors.append(f"EMPTY_LINE:{line_number}")
                continue
            try:
                envelope = LedgerEnvelope.model_validate(json.loads(line))
            except Exception as exc:  # pydantic supplies exact field errors to the caller log
                errors.append(f"INVALID_RECORD:{line_number}:{exc.__class__.__name__}")
                continue
            expected_sequence = record_count + 1
            if envelope.sequence != expected_sequence:
                errors.append(f"SEQUENCE_MISMATCH:{line_number}:{envelope.sequence}!={expected_sequence}")
            if envelope.previous_hash != previous_hash:
                errors.append(f"PREVIOUS_HASH_MISMATCH:{line_number}")
            hashable = self._hashable_payload(
                sequence=envelope.sequence,
                recorded_at=envelope.recorded_at,
                record_type=envelope.record_type,
                previous_hash=envelope.previous_hash,
                payload=envelope.payload,
            )
            expected_hash = sha256_json(hashable)
            if envelope.record_hash != expected_hash:
                errors.append(f"RECORD_HASH_MISMATCH:{line_number}")
            previous_hash = envelope.record_hash
            record_count += 1
        return LedgerAuditResult(
            status="PASS" if not errors else "FAIL",
            record_count=record_count,
            last_hash=previous_hash,
            errors=errors,
        )
