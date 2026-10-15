from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrograph.integrations.sat_sim_pipeline import SatSimPipelineClosureRequest
from .contracts import (
    MANUAL_PROVIDER_ID,
    PIPELINE_MAPPING_SCHEMA_VERSION,
    PIPELINE_RUNTIME_SCHEMA_VERSION,
    UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION,
    UpstreamRecomputeRequest,
    UpstreamRecomputeResponse,
    canonical_sha256,
    validate_upstream_request,
)

_MAX_BUNDLE_BYTES = 96 * 1024 * 1024
_MAX_MEMBER_BYTES = 48 * 1024 * 1024
_MAX_MEMBERS = 12
_RESPONSE_MANIFEST = "response.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _response_template(request: UpstreamRecomputeRequest) -> dict[str, Any]:
    return {
        "schema_version": UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "provider_id": MANUAL_PROVIDER_ID,
        "provider_mode": "MANUAL_EXCHANGE",
        "runtime_filename": "runtime.json",
        "runtime_sha256": "<fill sha256>",
        "mapping_filename": "mapping.json",
        "mapping_sha256": "<fill sha256>",
        "source_raw_input_receipt_sha256": None,
        "claim_boundary": "The response must contain complete recomputed runtime/mapping artifacts. Derived runtime patching is forbidden.",
    }


def build_manual_request_bundle(request: UpstreamRecomputeRequest | dict[str, Any]) -> bytes:
    if not isinstance(request, UpstreamRecomputeRequest):
        request = validate_upstream_request(dict(request))
    readme = f"""AstroGraph upstream recompute request\n\nRequest ID: {request.request_id}\nRequest SHA256: {request.request_sha256}\n\nThis bundle does not require AstroGraph, MATLAB, Simulink or sat_agent_platform on the same machine.\nAn external producer must recompute complete runtime.json and mapping.json artifacts from governed raw telemetry (or an equivalent upstream source), then fill response_template.json, rename it to response.json, and return a ZIP containing exactly response.json, runtime.json and mapping.json.\n\nAstroGraph will reject in-place feature/runtime patches, request identity mismatches, unsafe ZIPs, hash mismatches and cross-project runtime/mapping identity failures.\n"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("request.json", _json_bytes(request.model_dump(mode="json")))
        archive.writestr("response_template.json", _json_bytes(_response_template(request)))
        archive.writestr("README.txt", readme.encode("utf-8"))
    return buf.getvalue()


def build_manual_response_bundle(
    *, request: UpstreamRecomputeRequest | dict[str, Any], runtime_bytes: bytes, mapping_bytes: bytes,
    source_raw_input_receipt_sha256: str | None = None,
) -> bytes:
    if not isinstance(request, UpstreamRecomputeRequest):
        request = validate_upstream_request(dict(request))
    try:
        runtime = json.loads(runtime_bytes.decode("utf-8"))
        mapping = json.loads(mapping_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manual response runtime/mapping must be UTF-8 JSON") from exc
    if runtime.get("schema_version") != PIPELINE_RUNTIME_SCHEMA_VERSION:
        raise ValueError("manual response runtime schema mismatch")
    if mapping.get("schema_version") != PIPELINE_MAPPING_SCHEMA_VERSION:
        raise ValueError("manual response mapping schema mismatch")
    SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping, persist_dir=None)
    if source_raw_input_receipt_sha256 is not None:
        value = str(source_raw_input_receipt_sha256)
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("source_raw_input_receipt_sha256 must be lowercase SHA256 when provided")
    response = {
        "schema_version": UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "provider_id": MANUAL_PROVIDER_ID,
        "provider_mode": "MANUAL_EXCHANGE",
        "runtime_filename": "runtime.json",
        "runtime_sha256": sha256_bytes(runtime_bytes),
        "mapping_filename": "mapping.json",
        "mapping_sha256": sha256_bytes(mapping_bytes),
        "source_raw_input_receipt_sha256": source_raw_input_receipt_sha256,
        "claim_boundary": (
            "This package contains complete upstream-produced runtime/mapping artifacts bound to one AstroGraph recompute request. "
            "Packaging does not prove the upstream raw telemetry provenance; include a producer receipt hash when available."
        ),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("response.json", _json_bytes(response))
        archive.writestr("runtime.json", runtime_bytes)
        archive.writestr("mapping.json", mapping_bytes)
    return buf.getvalue()


def _read_safe_zip(data: bytes) -> dict[str, bytes]:
    if len(data) > _MAX_BUNDLE_BYTES:
        raise ValueError("upstream response bundle exceeds size limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("upstream response bundle must be a valid ZIP") from exc
    members: dict[str, bytes] = {}
    total_uncompressed = 0
    with archive:
        infos = archive.infolist()
        if len(infos) > _MAX_MEMBERS:
            raise ValueError("upstream response bundle contains too many members")
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            path = Path(name)
            mode = (info.external_attr >> 16) & 0o170000
            if path.is_absolute() or len(path.parts) != 1 or name in {"", ".", ".."} or mode == 0o120000:
                raise ValueError("upstream response bundle contains an unsafe member")
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError("upstream response bundle member exceeds size limit")
            total_uncompressed += int(info.file_size)
            if total_uncompressed > _MAX_BUNDLE_BYTES:
                raise ValueError("upstream response bundle expanded content exceeds size limit")
            if name in members:
                raise ValueError("upstream response bundle contains duplicate member names")
            members[name] = archive.read(info)
    return members


@dataclass(frozen=True)
class ManualRecomputeAdmission:
    response: dict[str, Any]
    runtime: dict[str, Any]
    mapping: dict[str, Any]
    runtime_bytes: bytes
    mapping_bytes: bytes
    admission_receipt: dict[str, Any]


def verify_manual_response_bundle(
    *, request: UpstreamRecomputeRequest | dict[str, Any], bundle_bytes: bytes, admitted_at: str, actor: str
) -> ManualRecomputeAdmission:
    if not isinstance(request, UpstreamRecomputeRequest):
        request = validate_upstream_request(dict(request))
    members = _read_safe_zip(bundle_bytes)
    if _RESPONSE_MANIFEST not in members:
        raise ValueError("upstream response bundle is missing response.json")
    try:
        response = json.loads(members[_RESPONSE_MANIFEST].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("upstream response.json must be UTF-8 JSON") from exc
    try:
        response = UpstreamRecomputeResponse.model_validate(response).model_dump(mode="json")
    except Exception as exc:
        raise ValueError(f"invalid upstream recompute response contract: {exc}") from exc
    if response.get("request_id") != request.request_id or response.get("request_sha256") != request.request_sha256:
        raise ValueError("upstream response is not bound to the current recompute request")
    if response.get("provider_id") != MANUAL_PROVIDER_ID or response.get("provider_mode") != "MANUAL_EXCHANGE":
        raise ValueError("manual import response must identify the manual provider")
    runtime_name = str(response.get("runtime_filename") or "")
    mapping_name = str(response.get("mapping_filename") or "")
    if Path(runtime_name).name != runtime_name or Path(mapping_name).name != mapping_name:
        raise ValueError("upstream response filenames must be basenames")
    if runtime_name != "runtime.json" or mapping_name != "mapping.json":
        raise ValueError("manual upstream response must use runtime.json and mapping.json")
    expected_names = {_RESPONSE_MANIFEST, runtime_name, mapping_name}
    if set(members) != expected_names:
        raise ValueError("upstream response ZIP must contain exactly response.json and the declared runtime/mapping files")
    runtime_bytes = members[runtime_name]
    mapping_bytes = members[mapping_name]
    if sha256_bytes(runtime_bytes) != str(response.get("runtime_sha256") or ""):
        raise ValueError("upstream runtime SHA256 mismatch")
    if sha256_bytes(mapping_bytes) != str(response.get("mapping_sha256") or ""):
        raise ValueError("upstream mapping SHA256 mismatch")
    try:
        runtime = json.loads(runtime_bytes.decode("utf-8"))
        mapping = json.loads(mapping_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("upstream runtime/mapping must be UTF-8 JSON") from exc
    if runtime.get("schema_version") != PIPELINE_RUNTIME_SCHEMA_VERSION:
        raise ValueError("upstream response runtime schema mismatch")
    if mapping.get("schema_version") != PIPELINE_MAPPING_SCHEMA_VERSION:
        raise ValueError("upstream response mapping schema mismatch")
    SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping, persist_dir=None)
    response_identity = canonical_sha256(response)
    receipt_core = {
        "schema_version": "astrograph-upstream-recompute-admission-receipt.v1",
        "status": "ADMITTED",
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "provider_id": MANUAL_PROVIDER_ID,
        "actor": actor,
        "admitted_at": admitted_at,
        "response_sha256": response_identity,
        "response_bundle_sha256": sha256_bytes(bundle_bytes),
        "runtime_sha256": sha256_bytes(runtime_bytes),
        "mapping_sha256": sha256_bytes(mapping_bytes),
        "source_raw_input_receipt_sha256": response.get("source_raw_input_receipt_sha256"),
        "runtime_schema_version": runtime.get("schema_version"),
        "mapping_schema_version": mapping.get("schema_version"),
        "cross_project_identity_verified": True,
        "derived_runtime_patch_used": False,
        "formal_release_eligible": False,
        "claim_boundary": "Admission proves request identity, file integrity, supported schemas and runtime/mapping cross-project identity. It does not prove the upstream producer actually used flight-grade raw telemetry and does not grant formal release authority.",
    }
    receipt = {**receipt_core, "receipt_sha256": canonical_sha256(receipt_core)}
    return ManualRecomputeAdmission(
        response=response,
        runtime=runtime,
        mapping=mapping,
        runtime_bytes=runtime_bytes,
        mapping_bytes=mapping_bytes,
        admission_receipt=receipt,
    )


def build_manual_child_prepared_receipt(
    *, source_job_id: str, child_job_id: str, actor: str, created_at: str,
    request: UpstreamRecomputeRequest | dict[str, Any], admission_receipt: dict[str, Any],
    source_report_sha256: str, child_preflight: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(request, UpstreamRecomputeRequest):
        request = validate_upstream_request(dict(request))
    if admission_receipt.get("status") != "ADMITTED" or admission_receipt.get("request_sha256") != request.request_sha256:
        raise ValueError("manual upstream child requires an admission receipt bound to the current request")
    core = {
        "schema_version": "astrograph-upstream-recompute-child-prepared-receipt.v1",
        "status": "PREPARED",
        "source_job_id": source_job_id,
        "child_job_id": child_job_id,
        "provider_id": MANUAL_PROVIDER_ID,
        "actor": actor,
        "created_at": created_at,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "source_runtime_sha256": request.source_runtime_sha256,
        "source_mapping_bundle_sha256": request.source_mapping_bundle_sha256,
        "admission_receipt_sha256": admission_receipt.get("receipt_sha256"),
        "source_raw_input_receipt_sha256": admission_receipt.get("source_raw_input_receipt_sha256"),
        "child_runtime_sha256": admission_receipt.get("runtime_sha256"),
        "child_mapping_bundle_sha256": admission_receipt.get("mapping_sha256"),
        "source_report_sha256": source_report_sha256,
        "child_preflight_sha256": canonical_sha256(child_preflight),
        "upstream_recompute_received": True,
        "raw_telemetry_recompute_verified": False,
        "formal_release_eligible": False,
        "claim_boundary": (
            "This receipt proves that a manually returned complete runtime/mapping bundle passed AstroGraph request binding, hash, schema and cross-project identity checks before child creation. "
            "AstroGraph cannot independently prove how the external producer obtained those artifacts; raw-telemetry recomputation provenance remains the upstream producer's responsibility."
        ),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def verify_manual_child_prepared_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "astrograph-upstream-recompute-child-prepared-receipt.v1" or payload.get("status") != "PREPARED":
        raise ValueError("invalid manual upstream child prepared receipt")
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(core) != payload.get("receipt_sha256"):
        raise ValueError("manual upstream child prepared receipt_sha256 mismatch")
    return payload


def build_manual_child_completion_receipt(
    *, prepared: dict[str, Any], completed_at: str, current_source_report_sha256: str,
    current_source_runtime_sha256: str, current_source_mapping_sha256: str,
    actual_child_runtime_sha256: str, actual_child_mapping_sha256: str,
    child_report: dict[str, Any],
) -> dict[str, Any]:
    prepared = verify_manual_child_prepared_receipt(prepared)
    if current_source_report_sha256 != prepared.get("source_report_sha256"):
        raise ValueError("manual upstream rerun source report changed during child execution")
    if current_source_runtime_sha256 != prepared.get("source_runtime_sha256"):
        raise ValueError("manual upstream rerun source runtime changed during child execution")
    if current_source_mapping_sha256 != prepared.get("source_mapping_bundle_sha256"):
        raise ValueError("manual upstream rerun source mapping changed during child execution")
    if actual_child_runtime_sha256 != prepared.get("child_runtime_sha256"):
        raise ValueError("manual upstream rerun child runtime does not match admitted response")
    if actual_child_mapping_sha256 != prepared.get("child_mapping_bundle_sha256"):
        raise ValueError("manual upstream rerun child mapping does not match admitted response")
    if child_report.get("artifact_kind") != "PIPELINE_RUNTIME" or child_report.get("status") != "SUCCEEDED":
        raise ValueError("manual upstream rerun child did not complete the PIPELINE_RUNTIME path")
    diagnosis = child_report.get("frozen_diagnosis") or {}
    core = {
        "schema_version": "astrograph-upstream-recompute-child-completion-receipt.v1",
        "status": "COMPLETED",
        "source_job_id": prepared.get("source_job_id"),
        "child_job_id": prepared.get("child_job_id"),
        "provider_id": prepared.get("provider_id"),
        "completed_at": completed_at,
        "prepared_receipt_sha256": prepared.get("receipt_sha256"),
        "request_id": prepared.get("request_id"),
        "request_sha256": prepared.get("request_sha256"),
        "source_raw_input_receipt_sha256": prepared.get("source_raw_input_receipt_sha256"),
        "child_report_sha256": canonical_sha256(child_report),
        "child_diagnosis_status": diagnosis.get("status"),
        "child_candidate_fault_id": diagnosis.get("candidate_fault_id"),
        "upstream_recompute_received": True,
        "raw_telemetry_recompute_verified": False,
        "formal_release_eligible": False,
        "claim_boundary": prepared.get("claim_boundary"),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def verify_manual_child_completion_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "astrograph-upstream-recompute-child-completion-receipt.v1" or payload.get("status") != "COMPLETED":
        raise ValueError("invalid manual upstream child completion receipt")
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(core) != payload.get("receipt_sha256"):
        raise ValueError("manual upstream child completion receipt_sha256 mismatch")
    return payload
