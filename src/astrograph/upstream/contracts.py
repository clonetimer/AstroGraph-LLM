from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION = "astrograph-upstream-recompute-request.v1"
UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION = "astrograph-upstream-recompute-response.v1"
UPSTREAM_PROVIDER_CAPABILITIES_SCHEMA_VERSION = "astrograph-upstream-provider-capabilities.v1"
MANUAL_PROVIDER_ID = "manual-import"
SAT_AGENT_PROVIDER_ID = "sat-agent-platform"
MATLAB_PROVIDER_ID = "matlab-simulink"
PIPELINE_RUNTIME_SCHEMA_VERSION = "sat-sim.diagnostic-pipeline-runtime.v1"
PIPELINE_MAPPING_SCHEMA_VERSION = "sat-sim.astrograph-diagnostic-mapping-bundle.v1"


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class UpstreamProviderCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str
    display_name: str
    status: Literal["AVAILABLE", "MANUAL", "UNAVAILABLE"]
    execution_mode: Literal["MANUAL_EXCHANGE", "LOCAL_OPTIONAL", "REMOTE_OPTIONAL"]
    zero_core_dependency: bool = True
    configured: bool = False
    reason: str
    claim_boundary: str


class UpstreamRecomputeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION] = UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=160)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_id: str = Field(min_length=1, max_length=120)
    provider_mode: str = Field(min_length=1, max_length=120)
    runtime_filename: str = Field(min_length=1, max_length=180)
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_filename: str = Field(min_length=1, max_length=180)
    mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_raw_input_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1, max_length=4000)


class UpstreamRecomputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION] = UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION
    request_id: str
    source_job_id: str
    source_artifact_kind: Literal["PIPELINE_RUNTIME"] = "PIPELINE_RUNTIME"
    source_runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_mapping_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_runtime_schema_version: Literal[PIPELINE_RUNTIME_SCHEMA_VERSION] = PIPELINE_RUNTIME_SCHEMA_VERSION
    required_output_runtime_schema_version: Literal[PIPELINE_RUNTIME_SCHEMA_VERSION] = PIPELINE_RUNTIME_SCHEMA_VERSION
    required_output_mapping_schema_version: Literal[PIPELINE_MAPPING_SCHEMA_VERSION] = PIPELINE_MAPPING_SCHEMA_VERSION
    maintenance_input_contract_ids: list[str] = Field(default_factory=list)
    derived_runtime_patch_allowed: Literal[False] = False
    required_output_files: list[str] = Field(default_factory=lambda: ["runtime.json", "mapping.json"])
    claim_boundary: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_identity(self) -> "UpstreamRecomputeRequest":
        core = self.model_dump(mode="json", exclude={"request_sha256"})
        if canonical_sha256(core) != self.request_sha256:
            raise ValueError("upstream recompute request_sha256 mismatch")
        stable_id = canonical_sha256({key: value for key, value in core.items() if key != "request_id"})
        if self.request_id != "URR-" + stable_id[:32].upper():
            raise ValueError("upstream recompute request_id does not match immutable request fields")
        return self


def build_pipeline_runtime_recompute_request(
    *,
    source_job_id: str,
    source_runtime_sha256: str,
    source_mapping_bundle_sha256: str,
    maintenance_input_contract_ids: list[str] | None = None,
) -> UpstreamRecomputeRequest:
    body_without_id = {
        "schema_version": UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION,
        "source_job_id": source_job_id,
        "source_artifact_kind": "PIPELINE_RUNTIME",
        "source_runtime_sha256": source_runtime_sha256,
        "source_mapping_bundle_sha256": source_mapping_bundle_sha256,
        "input_runtime_schema_version": PIPELINE_RUNTIME_SCHEMA_VERSION,
        "required_output_runtime_schema_version": PIPELINE_RUNTIME_SCHEMA_VERSION,
        "required_output_mapping_schema_version": PIPELINE_MAPPING_SCHEMA_VERSION,
        "maintenance_input_contract_ids": sorted(set(maintenance_input_contract_ids or [])),
        "derived_runtime_patch_allowed": False,
        "required_output_files": ["runtime.json", "mapping.json"],
        "claim_boundary": (
            "The upstream producer must recompute a complete PIPELINE_RUNTIME runtime and mapping bundle from governed raw telemetry or an equivalent upstream source. "
            "AstroGraph will not patch derived feature vectors, stage decisions or fusion inputs in place. Optional automatic providers are not core dependencies; manual request/response exchange remains available when none is configured."
        ),
    }
    stable_id = canonical_sha256(body_without_id)
    core = {
        "schema_version": body_without_id["schema_version"],
        "request_id": "URR-" + stable_id[:32].upper(),
        **{key: value for key, value in body_without_id.items() if key != "schema_version"},
    }
    payload = {**core, "request_sha256": canonical_sha256(core)}
    return UpstreamRecomputeRequest.model_validate(payload)


def _validate_request_with_stable_id(payload: dict[str, Any]) -> UpstreamRecomputeRequest:
    # Construct then independently verify using the stable non-ID identity. This avoids a self-referential hash fixed point.
    core = {key: value for key, value in payload.items() if key != "request_sha256"}
    request_sha = canonical_sha256(core)
    if request_sha != payload.get("request_sha256"):
        raise ValueError("upstream recompute request_sha256 mismatch")
    stable_id = canonical_sha256({key: value for key, value in core.items() if key != "request_id"})
    expected_id = "URR-" + stable_id[:32].upper()
    if core.get("request_id") != expected_id:
        raise ValueError("upstream recompute request_id does not match immutable request fields")
    return UpstreamRecomputeRequest.model_validate(payload)


def validate_upstream_request(payload: dict[str, Any]) -> UpstreamRecomputeRequest:
    if payload.get("schema_version") != UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported upstream recompute request schema")
    return _validate_request_with_stable_id(dict(payload))


def provider_capabilities() -> dict[str, Any]:
    from .registry import build_default_registry
    return build_default_registry().capabilities()
