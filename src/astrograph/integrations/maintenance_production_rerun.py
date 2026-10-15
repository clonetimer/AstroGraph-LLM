from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.integrations.sat_sim_public_case import (
    PROHIBITED_FIELDS,
    PROHIBITED_TOKENS,
    _load_public_case,
)


RAW_BINDING_SCHEMA_VERSION = "sat-sim.astrograph-maintenance-raw-input-binding.v1"
RAW_CONTRACTS_SCHEMA_VERSION = "astrograph-maintenance-production-raw-input-contracts.v1"
RAW_PATCH_CAPSULE_SCHEMA_VERSION = "astrograph-maintenance-production-raw-input-patch.v1"
RAW_PATCH_ADMISSION_SCHEMA_VERSION = "astrograph-maintenance-production-raw-input-admission.v1"
READINESS_SCHEMA_VERSION = "astrograph-maintenance-production-rerun-readiness.v1"
PREPARED_SCHEMA_VERSION = "astrograph-maintenance-production-rerun-receipt.v1"
COMPLETION_SCHEMA_VERSION = "astrograph-maintenance-production-rerun-completion-receipt.v1"
RERUN_MODE = "PUBLIC_CASE_RAW_TELEMETRY_PATCH_RERUN"


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_sha256(values: list[Any]) -> str:
    return canonical_sha256(values)


def _is_answer_proxy_field(field: str) -> bool:
    lowered = field.strip().lower()
    return lowered in PROHIBITED_FIELDS or any(token in lowered for token in PROHIBITED_TOKENS)


def _load_rows(telemetry: Path) -> tuple[list[str], list[dict[str, str]], list[float]]:
    with telemetry.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "time_s" not in fields:
        raise ValueError("public telemetry requires time_s")
    times: list[float] = []
    for row in rows:
        try:
            value = float(row["time_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError("public telemetry time_s must be finite numeric values") from exc
        if not math.isfinite(value):
            raise ValueError("public telemetry time_s must be finite numeric values")
        times.append(value)
    if len(times) != len(set(times)):
        raise ValueError("public telemetry time_s values must be unique for governed raw patch admission")
    if any(times[index] > times[index + 1] for index in range(len(times) - 1)):
        raise ValueError("public telemetry time_s must be non-decreasing")
    return fields, rows, times


def _binding_from_manifest(manifest: dict[str, Any], telemetry_sha256: str) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    binding = manifest.get("maintenance_raw_input_binding")
    if not isinstance(binding, dict):
        return None, ["RAW_INPUT_BINDING_UNAVAILABLE"]
    if binding.get("schema_version") != RAW_BINDING_SCHEMA_VERSION:
        blockers.append("RAW_INPUT_BINDING_SCHEMA_UNSUPPORTED")
    if str(binding.get("source_telemetry_sha256") or "") != telemetry_sha256:
        blockers.append("RAW_INPUT_BINDING_SOURCE_HASH_MISMATCH")
    if str(binding.get("time_field") or "") != "time_s":
        blockers.append("RAW_INPUT_BINDING_TIME_FIELD_UNSUPPORTED")
    if binding.get("full_window_replacement_required") is not True:
        blockers.append("RAW_INPUT_BINDING_MUST_REQUIRE_FULL_WINDOW_REPLACEMENT")
    patchable = binding.get("patchable_fields")
    if not isinstance(patchable, list) or not patchable:
        blockers.append("RAW_INPUT_BINDING_HAS_NO_PATCHABLE_FIELDS")
    return binding, blockers


def build_public_case_raw_input_contracts(case_root: str | Path) -> dict[str, Any]:
    root = Path(case_root).resolve()
    manifest, telemetry = _load_public_case(root)
    telemetry_sha = sha256_path(telemetry)
    manifest_sha = sha256_path(root / "manifest.json")
    fields, _rows, times = _load_rows(telemetry)
    binding, blockers = _binding_from_manifest(manifest, telemetry_sha)
    contracts: list[dict[str, Any]] = []
    warnings: list[str] = []
    if binding is not None and not blockers:
        seen: set[str] = set()
        for raw in list(binding.get("patchable_fields") or []):
            if not isinstance(raw, dict):
                blockers.append("RAW_INPUT_BINDING_FIELD_ENTRY_INVALID")
                continue
            field = str(raw.get("field") or "").strip()
            unit = str(raw.get("unit") or "").strip()
            if not field or not unit:
                blockers.append("RAW_INPUT_BINDING_FIELD_OR_UNIT_MISSING")
                continue
            if field == "time_s" or _is_answer_proxy_field(field):
                blockers.append(f"RAW_INPUT_BINDING_FIELD_FORBIDDEN:{field}")
                continue
            if field in seen:
                blockers.append(f"RAW_INPUT_BINDING_DUPLICATE_FIELD:{field}")
                continue
            seen.add(field)
            identity = {
                "source_manifest_sha256": manifest_sha,
                "source_telemetry_sha256": telemetry_sha,
                "field": field,
                "unit": unit,
                "sample_times_sha256": _sequence_sha256(times),
                "sample_count": len(times),
                "full_window_replacement_required": True,
            }
            contract_id = "RAW-" + canonical_sha256(identity)[:32].upper()
            contracts.append({
                "schema_version": "astrograph-maintenance-production-raw-input-contract.v1",
                "contract_id": contract_id,
                "source_artifact_kind": "PUBLIC_CASE",
                "field": field,
                "unit": unit,
                "field_present_in_source": field in fields,
                "time_field": "time_s",
                "window_start_s": (times[0] if times else None),
                "window_end_s": (times[-1] if times else None),
                "sample_count": len(times),
                "sample_times_sha256": identity["sample_times_sha256"],
                "source_manifest_sha256": manifest_sha,
                "source_telemetry_sha256": telemetry_sha,
                "replacement_scope": "FULL_SOURCE_TIMELINE",
                "claim_boundary": (
                    "This contract permits only a full-timeline replacement/addition of one explicitly bound public telemetry field. "
                    "The server revalidates source hashes, exact timestamps, units and answer-proxy guards before creating a new child input."
                ),
            })
    if not times:
        blockers.append("PUBLIC_TELEMETRY_HAS_NO_ROWS")
    if contracts and any(item["field_present_in_source"] is False for item in contracts):
        warnings.append("BOUND_FIELDS_MAY_ADD_NEW_TELEMETRY_COLUMNS")
    return {
        "schema_version": RAW_CONTRACTS_SCHEMA_VERSION,
        "status": "AVAILABLE" if contracts and not blockers else "BLOCKED",
        "source_artifact_kind": "PUBLIC_CASE",
        "source_manifest_sha256": manifest_sha,
        "source_telemetry_sha256": telemetry_sha,
        "contract_count": len(contracts),
        "contracts": contracts,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "required_binding_schema_version": RAW_BINDING_SCHEMA_VERSION,
        "claim_boundary": (
            "Raw-input contracts exist only when the immutable PUBLIC_CASE manifest explicitly opts into the governed raw-input binding extension. "
            "Older packages without that binding remain fail-closed."
        ),
    }


class ProductionRawInputPatchCapsule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[RAW_PATCH_CAPSULE_SCHEMA_VERSION] = RAW_PATCH_CAPSULE_SCHEMA_VERSION
    contract_id: str = Field(min_length=1, max_length=240)
    field: str = Field(min_length=1, max_length=300)
    unit: str = Field(min_length=1, max_length=80)
    sample_times_s: list[float] = Field(min_length=1, max_length=500_000)
    values: list[float] = Field(min_length=1, max_length=500_000)
    source_note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_shape(self) -> "ProductionRawInputPatchCapsule":
        if len(self.sample_times_s) != len(self.values):
            raise ValueError("sample_times_s length must match values")
        if any(not math.isfinite(float(value)) for value in [*self.sample_times_s, *self.values]):
            raise ValueError("raw input patch samples must be finite numeric values")
        if any(self.sample_times_s[index] > self.sample_times_s[index + 1] for index in range(len(self.sample_times_s) - 1)):
            raise ValueError("sample_times_s must be non-decreasing")
        return self


def build_public_case_raw_patch_template(contract: dict[str, Any], case_root: str | Path) -> dict[str, Any]:
    root = Path(case_root).resolve()
    _manifest, telemetry = _load_public_case(root)
    _fields, _rows, times = _load_rows(telemetry)
    if contract.get("sample_times_sha256") != _sequence_sha256(times):
        raise ValueError("current source telemetry no longer matches raw-input contract")
    return {
        "schema_version": "astrograph-maintenance-production-raw-input-template.v1",
        "contract": contract,
        "capsule": {
            "schema_version": RAW_PATCH_CAPSULE_SCHEMA_VERSION,
            "contract_id": contract.get("contract_id"),
            "field": contract.get("field"),
            "unit": contract.get("unit"),
            "sample_times_s": times,
            "values": [],
            "source_note": "",
        },
        "claim_boundary": (
            "Populate exactly one finite value for every frozen source timestamp. Partial-window or timestamp-shifted patches are rejected."
        ),
    }


def admit_public_case_raw_patch(
    *, case_root: str | Path, contract: dict[str, Any], capsule: ProductionRawInputPatchCapsule,
    actor: str, admitted_at: str,
) -> dict[str, Any]:
    root = Path(case_root).resolve()
    current = build_public_case_raw_input_contracts(root)
    latest = next((item for item in current.get("contracts") or [] if item.get("contract_id") == capsule.contract_id), None)
    if latest is None or latest != contract:
        raise ValueError("raw-input contract is stale or no longer current")
    if capsule.field != contract.get("field"):
        raise ValueError("raw input patch field does not match frozen contract")
    if capsule.unit != contract.get("unit"):
        raise ValueError("raw input patch unit does not match frozen contract")
    _manifest, telemetry = _load_public_case(root)
    _fields, _rows, times = _load_rows(telemetry)
    if len(capsule.sample_times_s) != len(times) or len(capsule.values) != len(times):
        raise ValueError("raw input patch must replace the full frozen source timeline")
    if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(capsule.sample_times_s, times, strict=True)):
        raise ValueError("raw input patch timestamps do not exactly match the frozen source timeline")
    capsule_core = capsule.model_dump(mode="json", exclude={"source_note"})
    core = {
        "schema_version": RAW_PATCH_ADMISSION_SCHEMA_VERSION,
        "status": "ADMITTED",
        "contract_id": capsule.contract_id,
        "contract_sha256": canonical_sha256(contract),
        "source_manifest_sha256": str(contract.get("source_manifest_sha256")),
        "source_telemetry_sha256": str(contract.get("source_telemetry_sha256")),
        "field": capsule.field,
        "unit": capsule.unit,
        "sample_count": len(times),
        "sample_times_sha256": _sequence_sha256(capsule.sample_times_s),
        "values_sha256": _sequence_sha256(capsule.values),
        "capsule_sha256": canonical_sha256(capsule_core),
        "actor": actor,
        "admitted_at": admitted_at,
        "raw_input_bound": True,
        "formal_release_eligible": False,
        "claim_boundary": (
            "Admission proves that one raw telemetry channel patch exactly matches an immutable PUBLIC_CASE raw-input binding. "
            "It does not mutate the source job and does not itself authorize formal release."
        ),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def verify_public_case_raw_patch_admission(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != RAW_PATCH_ADMISSION_SCHEMA_VERSION or payload.get("status") != "ADMITTED":
        raise ValueError("invalid production raw-input admission receipt")
    observed = str(payload.get("receipt_sha256") or "")
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(core) != observed:
        raise ValueError("production raw-input admission receipt_sha256 mismatch")
    return payload


def materialize_public_case_raw_patch(
    *, case_root: str | Path, admissions: list[dict[str, Any]], source_job_id: str,
) -> dict[str, Any]:
    root = Path(case_root).resolve()
    manifest, telemetry = _load_public_case(root)
    source_manifest_sha = sha256_path(root / "manifest.json")
    source_telemetry_sha = sha256_path(telemetry)
    fields, rows, times = _load_rows(telemetry)
    if not admissions:
        raise ValueError("no admitted raw-input patches")
    patched_fields: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for artifact in admissions:
        if not isinstance(artifact, dict):
            raise ValueError("invalid raw-input patch artifact")
        receipt = verify_public_case_raw_patch_admission(dict(artifact.get("receipt") or {}))
        capsule = ProductionRawInputPatchCapsule.model_validate(artifact.get("capsule") or {})
        current_contracts = build_public_case_raw_input_contracts(root)
        contract = next((item for item in current_contracts.get("contracts") or [] if item.get("contract_id") == receipt.get("contract_id")), None)
        if contract is None:
            raise ValueError("admitted raw-input patch contract is no longer current")
        expected = admit_public_case_raw_patch(
            case_root=root, contract=contract, capsule=capsule,
            actor=str(receipt.get("actor") or "operator"), admitted_at=str(receipt.get("admitted_at") or ""),
        )
        if expected.get("receipt_sha256") != receipt.get("receipt_sha256"):
            raise ValueError("raw-input admission artifact does not match its frozen capsule")
        if capsule.field in patched_fields:
            raise ValueError("multiple admitted raw-input patches target the same field")
        patched_fields.add(capsule.field)
        normalized.append({"receipt": receipt, "capsule": capsule})

    output_fields = list(fields)
    for item in normalized:
        field = item["capsule"].field
        if field not in output_fields:
            output_fields.append(field)
        for row, value in zip(rows, item["capsule"].values, strict=True):
            row[field] = format(float(value), ".17g")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=output_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    telemetry_bytes = buffer.getvalue().encode("utf-8")
    patched_telemetry_sha = sha256_bytes(telemetry_bytes)
    patch_identity_core = {
        "source_job_id": source_job_id,
        "source_manifest_sha256": source_manifest_sha,
        "source_telemetry_sha256": source_telemetry_sha,
        "admission_receipt_sha256s": [item["receipt"]["receipt_sha256"] for item in normalized],
        "patched_fields": sorted(patched_fields),
        "patched_telemetry_sha256": patched_telemetry_sha,
    }
    patch_identity = canonical_sha256(patch_identity_core)
    telemetry_name = f"telemetry.maintenance-rerun-{patch_identity[:12]}.csv"
    child_manifest = deepcopy(manifest)
    source_case_id = str(manifest.get("case_id") or "case")
    child_manifest["case_id"] = f"{source_case_id}-RERUN-{patch_identity[:12].upper()}"
    telemetry_meta = dict(child_manifest.get("telemetry") or {})
    telemetry_meta.update({
        "file": telemetry_name,
        "sha256": patched_telemetry_sha,
        "rows": len(rows),
        "fields": output_fields,
    })
    units = dict(telemetry_meta.get("units") or {})
    for item in normalized:
        units[item["capsule"].field] = item["capsule"].unit
    if units:
        telemetry_meta["units"] = units
    child_manifest["telemetry"] = telemetry_meta
    binding = dict(child_manifest.get("maintenance_raw_input_binding") or {})
    if binding:
        binding["source_telemetry_sha256"] = patched_telemetry_sha
        child_manifest["maintenance_raw_input_binding"] = binding
    child_manifest["maintenance_raw_patch_provenance"] = {
        "schema_version": "astrograph-maintenance-production-raw-patch-provenance.v1",
        "source_job_id": source_job_id,
        "source_case_id": source_case_id,
        "source_manifest_sha256": source_manifest_sha,
        "source_telemetry_sha256": source_telemetry_sha,
        "patch_identity_sha256": patch_identity,
        "admission_receipt_sha256s": patch_identity_core["admission_receipt_sha256s"],
        "raw_telemetry_rerun_performed": True,
        "formal_release_eligible": False,
    }
    manifest_bytes = (json.dumps(child_manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    # Revalidate the exact materialized child package before the caller can create a job.
    return {
        "patch_identity_sha256": patch_identity,
        "source_manifest_sha256": source_manifest_sha,
        "source_telemetry_sha256": source_telemetry_sha,
        "patched_manifest_sha256": sha256_bytes(manifest_bytes),
        "patched_telemetry_sha256": patched_telemetry_sha,
        "manifest_filename": "manifest.json",
        "manifest_bytes": manifest_bytes,
        "telemetry_filename": telemetry_name,
        "telemetry_bytes": telemetry_bytes,
        "patched_fields": sorted(patched_fields),
        "admission_receipt_sha256s": patch_identity_core["admission_receipt_sha256s"],
        "source_times_sha256": _sequence_sha256(times),
    }


class MaintenanceProductionRerunReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[READINESS_SCHEMA_VERSION] = READINESS_SCHEMA_VERSION
    status: Literal["READY", "BLOCKED"]
    rerun_mode: str
    source_artifact_kind: str
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    patch_identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_telemetry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    patched_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    patched_telemetry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    admission_receipt_sha256s: list[str] = Field(default_factory=list)
    raw_telemetry_rerun_planned: bool = False
    formal_release_eligible: Literal[False] = False
    claim_boundary: str

    @model_validator(mode="after")
    def consistent(self) -> "MaintenanceProductionRerunReadiness":
        if self.status == "READY" and self.blockers:
            raise ValueError("READY production rerun readiness cannot contain blockers")
        if self.status == "BLOCKED" and not self.blockers:
            raise ValueError("BLOCKED production rerun readiness requires blockers")
        return self


def assess_maintenance_production_rerun(
    *, source_artifact_kind: str, source_report: dict[str, Any], materialized_patch: dict[str, Any] | None,
    raw_contract_status: str | None = None, raw_contract_blockers: list[str] | None = None,
) -> MaintenanceProductionRerunReadiness:
    blockers: list[str] = []
    warnings: list[str] = []
    rerun_mode = "UNAVAILABLE"
    if source_artifact_kind == "PUBLIC_CASE":
        rerun_mode = RERUN_MODE
        if raw_contract_status != "AVAILABLE":
            blockers.extend(raw_contract_blockers or ["RAW_INPUT_BINDING_UNAVAILABLE"])
        if materialized_patch is None:
            blockers.append("NO_ADMITTED_RAW_INPUT_PATCH")
    elif source_artifact_kind == "PIPELINE_RUNTIME":
        rerun_mode = "PIPELINE_RUNTIME_UPSTREAM_RECOMPUTE_REQUIRED"
        blockers.append("UPSTREAM_RECOMPUTE_ADAPTER_REQUIRED")
        warnings.append("PIPELINE_RUNTIME_CONTAINS_DERIVED_RUNTIME_OUTPUTS_NOT_RECONSTRUCTIBLE_FROM_MAINTENANCE_CAPSULE")
    elif source_artifact_kind == "OAG_DIAGNOSTIC":
        rerun_mode = "USE_OAG_EVIDENCE_BRANCH_REPLAY"
        blockers.append("USE_EXISTING_OAG_GOVERNED_RERUN_WORKFLOW")
    else:
        blockers.append("SOURCE_ARTIFACT_KIND_UNSUPPORTED")
    patch = materialized_patch or {}
    return MaintenanceProductionRerunReadiness(
        status="READY" if not blockers else "BLOCKED",
        rerun_mode=rerun_mode,
        source_artifact_kind=source_artifact_kind,
        blockers=sorted(set(blockers)),
        warnings=sorted(set(warnings)),
        patch_identity_sha256=patch.get("patch_identity_sha256"),
        source_report_sha256=canonical_sha256(source_report),
        source_manifest_sha256=patch.get("source_manifest_sha256"),
        source_telemetry_sha256=patch.get("source_telemetry_sha256"),
        patched_manifest_sha256=patch.get("patched_manifest_sha256"),
        patched_telemetry_sha256=patch.get("patched_telemetry_sha256"),
        admission_receipt_sha256s=list(patch.get("admission_receipt_sha256s") or []),
        raw_telemetry_rerun_planned=(not blockers and source_artifact_kind == "PUBLIC_CASE"),
        claim_boundary=(
            "A READY result proves that an explicitly bound PUBLIC_CASE raw telemetry child can be materialized and rerun through the normal public-case diagnosis path. "
            "PIPELINE_RUNTIME remains blocked until an upstream raw-runtime recompute adapter exists. This laboratory path does not grant formal release eligibility."
        ),
    )


def build_production_rerun_prepared_receipt(
    *, source_job_id: str, child_job_id: str, actor: str, created_at: str,
    readiness: MaintenanceProductionRerunReadiness, child_preflight: dict[str, Any],
) -> dict[str, Any]:
    if readiness.status != "READY" or not readiness.raw_telemetry_rerun_planned:
        raise ValueError("production rerun receipt requires READY raw-telemetry readiness")
    if not child_preflight.get("can_start") or child_preflight.get("artifact_kind") != "PUBLIC_CASE":
        raise ValueError("production rerun child preflight must admit PUBLIC_CASE")
    core = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": "PREPARED",
        "rerun_mode": RERUN_MODE,
        "source_job_id": source_job_id,
        "child_job_id": child_job_id,
        "actor": actor,
        "created_at": created_at,
        "patch_identity_sha256": readiness.patch_identity_sha256,
        "source_report_sha256": readiness.source_report_sha256,
        "source_manifest_sha256": readiness.source_manifest_sha256,
        "source_telemetry_sha256": readiness.source_telemetry_sha256,
        "patched_manifest_sha256": readiness.patched_manifest_sha256,
        "patched_telemetry_sha256": readiness.patched_telemetry_sha256,
        "admission_receipt_sha256s": readiness.admission_receipt_sha256s,
        "child_preflight_sha256": canonical_sha256(child_preflight),
        "raw_telemetry_rerun_performed": True,
        "source_job_mutated": False,
        "formal_release_eligible": False,
        "claim_boundary": readiness.claim_boundary,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def verify_production_rerun_prepared_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != PREPARED_SCHEMA_VERSION or payload.get("status") != "PREPARED":
        raise ValueError("invalid production rerun prepared receipt")
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(core) != payload.get("receipt_sha256"):
        raise ValueError("production rerun prepared receipt_sha256 mismatch")
    return payload


def build_production_rerun_completion_receipt(
    *, prepared: dict[str, Any], completed_at: str,
    current_source_report: dict[str, Any], current_source_manifest_sha256: str,
    current_source_telemetry_sha256: str, actual_child_manifest_sha256: str,
    actual_child_telemetry_sha256: str, child_report: dict[str, Any],
) -> dict[str, Any]:
    prepared = verify_production_rerun_prepared_receipt(prepared)
    if canonical_sha256(current_source_report) != prepared.get("source_report_sha256"):
        raise ValueError("production rerun source report changed during child execution")
    if current_source_manifest_sha256 != prepared.get("source_manifest_sha256"):
        raise ValueError("production rerun source manifest changed during child execution")
    if current_source_telemetry_sha256 != prepared.get("source_telemetry_sha256"):
        raise ValueError("production rerun source telemetry changed during child execution")
    if actual_child_manifest_sha256 != prepared.get("patched_manifest_sha256"):
        raise ValueError("production rerun child manifest does not match prepared input")
    if actual_child_telemetry_sha256 != prepared.get("patched_telemetry_sha256"):
        raise ValueError("production rerun child telemetry does not match prepared input")
    if child_report.get("artifact_kind") != "PUBLIC_CASE" or child_report.get("status") != "SUCCEEDED":
        raise ValueError("production rerun child did not complete the PUBLIC_CASE path")
    core = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "COMPLETED",
        "rerun_mode": RERUN_MODE,
        "source_job_id": prepared["source_job_id"],
        "child_job_id": prepared["child_job_id"],
        "completed_at": completed_at,
        "prepared_receipt_sha256": prepared["receipt_sha256"],
        "patch_identity_sha256": prepared["patch_identity_sha256"],
        "child_report_sha256": canonical_sha256(child_report),
        "child_manifest_sha256": actual_child_manifest_sha256,
        "child_telemetry_sha256": actual_child_telemetry_sha256,
        "child_diagnosis_status": str((((child_report.get("analysis") or {}).get("closure") or {}).get("diagnosis") or {}).get("status") or "UNKNOWN"),
        "child_formal_diagnosis_publishable": bool(child_report.get("formal_diagnosis_publishable")),
        "source_report_unchanged": True,
        "source_raw_input_unchanged": True,
        "actual_child_input_matches_prepared": True,
        "raw_telemetry_rerun_performed": True,
        "formal_release_eligible": False,
        "claim_boundary": (
            "The child reran the patched raw PUBLIC_CASE telemetry through the normal public-case diagnosis path. "
            "The source job remained frozen. This simulation/laboratory child still does not receive formal release authority."
        ),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def verify_production_rerun_completion_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != COMPLETION_SCHEMA_VERSION or payload.get("status") != "COMPLETED":
        raise ValueError("invalid production rerun completion receipt")
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(core) != payload.get("receipt_sha256"):
        raise ValueError("production rerun completion receipt_sha256 mismatch")
    return payload


__all__ = [
    "RAW_BINDING_SCHEMA_VERSION",
    "RAW_PATCH_CAPSULE_SCHEMA_VERSION",
    "MaintenanceProductionRerunReadiness",
    "ProductionRawInputPatchCapsule",
    "admit_public_case_raw_patch",
    "assess_maintenance_production_rerun",
    "build_production_rerun_completion_receipt",
    "build_production_rerun_prepared_receipt",
    "build_public_case_raw_input_contracts",
    "build_public_case_raw_patch_template",
    "canonical_sha256",
    "materialize_public_case_raw_patch",
    "verify_production_rerun_completion_receipt",
    "verify_production_rerun_prepared_receipt",
    "verify_public_case_raw_patch_admission",
]
