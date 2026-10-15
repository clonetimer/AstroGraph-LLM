from __future__ import annotations

import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SOURCE_URI_RE = re.compile(r"^source://([a-z0-9][a-z0-9._-]*)$")


class ExternalSourceIntegrityStatus(str, Enum):
    HASH_FROZEN = "HASH_FROZEN"
    UNVERIFIED_EXTERNAL_SOURCE = "UNVERIFIED_EXTERNAL_SOURCE"


class ExternalSourceResolutionError(RuntimeError):
    """Raised when a logical external source cannot be resolved fail-closed."""


class ExternalSourceResolver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_env: str = Field(pattern=r"^ASTROGRAPH_[A-Z0-9_]+$")
    relative_path: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("external source relative_path must remain below its configured root")
        return path.as_posix()


class ExternalSourceIntegrity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: ExternalSourceIntegrityStatus

    @model_validator(mode="after")
    def validate_status_and_hash(self) -> "ExternalSourceIntegrity":
        if self.status == ExternalSourceIntegrityStatus.HASH_FROZEN and self.sha256 is None:
            raise ValueError("HASH_FROZEN external source requires sha256")
        if self.status == ExternalSourceIntegrityStatus.UNVERIFIED_EXTERNAL_SOURCE and self.sha256 is not None:
            raise ValueError("UNVERIFIED_EXTERNAL_SOURCE must not present an unverified hash as frozen")
        return self


class ExternalSourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    logical_uri: str
    resolver: ExternalSourceResolver
    integrity: ExternalSourceIntegrity
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_matching_logical_uri(self) -> "ExternalSourceRecord":
        if self.logical_uri != f"source://{self.source_id}":
            raise ValueError("logical_uri must be source://<source_id>")
        return self


class ResolvedExternalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    logical_uri: str
    path: Path
    integrity_status: ExternalSourceIntegrityStatus
    hash_verified: bool
    sha256: str | None = None


class ExternalSourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = "external-source-registry-v1"
    dataset_id: str = Field(min_length=1)
    sources: list[ExternalSourceRecord]

    @field_validator("sources")
    @classmethod
    def require_unique_source_ids(cls, value: list[ExternalSourceRecord]) -> list[ExternalSourceRecord]:
        ids = [item.source_id for item in value]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("external source IDs must be nonempty and unique")
        return value

    @classmethod
    def from_json(cls, path: str | Path) -> "ExternalSourceRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)

    def get(self, logical_uri: str) -> ExternalSourceRecord:
        match = _SOURCE_URI_RE.fullmatch(logical_uri)
        if match is None:
            raise ExternalSourceResolutionError(f"invalid external source URI: {logical_uri}")
        source_id = match.group(1)
        record = next((item for item in self.sources if item.source_id == source_id), None)
        if record is None:
            raise ExternalSourceResolutionError(f"external source is not registered: {logical_uri}")
        return record

    def resolve(
        self,
        logical_uri: str,
        *,
        environ: Mapping[str, str] | None = None,
        require_hash_frozen: bool = False,
        require_exists: bool = True,
    ) -> ResolvedExternalSource:
        record = self.get(logical_uri)
        environment = os.environ if environ is None else environ
        raw_root = str(environment.get(record.resolver.root_env, "")).strip()
        if not raw_root:
            raise ExternalSourceResolutionError(
                f"{record.resolver.root_env} is required to resolve {logical_uri}"
            )
        root = Path(raw_root).expanduser().resolve()
        path = (root / record.resolver.relative_path).resolve()
        if path != root and root not in path.parents:
            raise ExternalSourceResolutionError(
                f"resolved source escapes configured root: {logical_uri}"
            )
        if require_exists and not path.is_file():
            raise ExternalSourceResolutionError(f"external source file is missing: {path}")
        if require_hash_frozen and record.integrity.status != ExternalSourceIntegrityStatus.HASH_FROZEN:
            raise ExternalSourceResolutionError(
                f"external source lacks a frozen integrity receipt: {logical_uri}"
            )

        actual_sha256: str | None = None
        verified = False
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            actual_sha256 = digest
            expected = record.integrity.sha256
            if expected is not None:
                if digest != expected:
                    raise ExternalSourceResolutionError(
                        f"external source hash mismatch for {logical_uri}: expected {expected}, got {digest}"
                    )
                verified = True

        return ResolvedExternalSource(
            source_id=record.source_id,
            logical_uri=record.logical_uri,
            path=path,
            integrity_status=record.integrity.status,
            hash_verified=verified,
            sha256=actual_sha256,
        )


__all__ = [
    "ExternalSourceIntegrity",
    "ExternalSourceIntegrityStatus",
    "ExternalSourceRecord",
    "ExternalSourceRegistry",
    "ExternalSourceResolutionError",
    "ExternalSourceResolver",
    "ResolvedExternalSource",
]
