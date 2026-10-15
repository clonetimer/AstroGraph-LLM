from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, re
from typing import Any, Mapping

PROTOCOL_VERSION = "astrograph-capability-protocol.v1"
PROJECT_STATUS = "B014_TEMPORAL_STACK_MERGED_CONFIRMATION_NOT_EXECUTED"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

class CapabilityProtocolError(ValueError):
    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, dict(details or {})

@dataclass(frozen=True)
class CapabilityRequest:
    request_id: str
    method: str
    profile: str
    params: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CapabilityRequest":
        version = str(raw.get("protocol_version", ""))
        if version != PROTOCOL_VERSION:
            raise CapabilityProtocolError("PROTOCOL_VERSION_MISMATCH", f"expected {PROTOCOL_VERSION}, got {version!r}")
        request_id = str(raw.get("request_id", ""))
        method = str(raw.get("method", ""))
        profile = str(raw.get("profile", ""))
        if not _ID.fullmatch(request_id):
            raise CapabilityProtocolError("INVALID_REQUEST_ID", "request_id is missing or invalid")
        if method not in {"health", "capabilities.list", "capabilities.invoke", "jobs.status", "shutdown"}:
            raise CapabilityProtocolError("INVALID_METHOD", f"unsupported method {method!r}")
        if not profile:
            raise CapabilityProtocolError("PROFILE_REQUIRED", "profile is required")
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise CapabilityProtocolError("INVALID_PARAMS", "params must be an object")
        return cls(request_id=request_id, method=method, profile=profile, params=dict(params), protocol_version=version)

    def digest(self) -> str:
        return sha256_json({"protocol_version": self.protocol_version, "request_id": self.request_id, "method": self.method, "profile": self.profile, "params": self.params})

@dataclass(frozen=True)
class CapabilityResponse:
    request_id: str
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    protocol_version: str = PROTOCOL_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "request_id": self.request_id, "status": self.status, "result": self.result, "error": self.error, "receipt": self.receipt}
