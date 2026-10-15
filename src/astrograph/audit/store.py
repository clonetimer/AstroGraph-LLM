from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from astrograph.schemas import DiagnosticAuditRecord


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class AuditStore(Protocol):
    def append(self, record: DiagnosticAuditRecord) -> None: ...


class NullAuditStore:
    def append(self, record: DiagnosticAuditRecord) -> None:
        return None


class JsonlAuditStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: DiagnosticAuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json(exclude_none=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
