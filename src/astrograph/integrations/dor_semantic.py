from __future__ import annotations

"""Fail-closed bridge from AstroGraph evidence facts to DOR localization.

The bridge deliberately sits after deterministic feature/evidence construction
and before formal diagnosis release.  DOR is an independent semantic audit,
not an additional probability ranking and not a root-cause authority.
"""

from .process_control import run_managed_subprocess
import csv
import io
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus, EvidenceRole
from astrograph.evidence.executable import stable_id

from .dor_runtime import DORRuntimeIdentity, canonical_json_bytes, inspect_dor_runtime, sha256_bytes


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _sha256(value: bytes) -> str:
    return sha256_bytes(value)


class DORSemanticAuditStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    ALTERNATIVES = "ALTERNATIVES"
    UNKNOWN = "UNKNOWN"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


class DORSemanticFaultBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str = Field(min_length=1)
    target_component_id: str = Field(min_length=1)
    component_kind: str = Field(default="PHYSICAL", pattern=r"^(PHYSICAL|SENSOR|ACTUATOR|SOFTWARE|VIRTUAL)$")
    component_type: str = Field(min_length=1)
    subsystem: str = Field(min_length=1)
    fault_mode: str = Field(min_length=1)
    category: str = Field(pattern=r"^(COMPONENT|SENSOR|COMMAND|PARAMETRIC|DISCRETE|SOFTWARE)$")
    exclusion_group: str = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)


class DORSemanticPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^astrograph-dor-semantic-package\.v1$")
    package_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    faults: list[DORSemanticFaultBinding] = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_fault_ids(self) -> "DORSemanticPackage":
        fault_ids = [item.fault_id for item in self.faults]
        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("DOR semantic package contains duplicate fault_id values")
        return self


class DORSemanticAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_package_path: str = Field(min_length=1)
    executable_path: str = Field(default="dor", min_length=1)
    timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    min_support: float = Field(default=1.0, ge=0.0, le=1.0)
    retain_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    max_contradiction_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    release_gate: bool = True


class DORSemanticAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-semantic-audit.v1"
    audit_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    package_id: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    status: DORSemanticAuditStatus
    reason: str = Field(min_length=1)
    release_gate_enabled: bool
    eligible_for_formal_fault_release: bool
    supported_fault_ids: list[str] = Field(default_factory=list)
    candidate_fault_sets: list[list[str]] = Field(default_factory=list)
    input_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_ontology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_mechanism_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_evidence_csv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable: str = Field(min_length=1)
    executable_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime: DORRuntimeIdentity | None = None
    bridge_version: str = Field(min_length=1)
    dor_stack_info: dict[str, Any] | None = None
    dor_contract: str | None = None
    dor_result: dict[str, Any] | None = None
    safety: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Independent E0 semantic-localization audit only. It is not a BiRRF vote, "
        "does not recalibrate model probabilities, and makes no root-cause or propagation claim."
    )


class DORSemanticAuditService:
    """Compile AstroGraph evidence semantics and execute DOR in a subprocess."""

    bridge_version = "astrograph-dor-semantic-bridge@1.0.0"

    @staticmethod
    def load_package(path: str | Path) -> tuple[DORSemanticPackage, bytes]:
        raw = Path(path).read_bytes()
        return DORSemanticPackage.model_validate_json(raw), raw

    @staticmethod
    def compile_inputs(
        bundle: EpisodeEvidenceBundle,
        package: DORSemanticPackage,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        definitions = {item.evidence_definition_id: item for item in bundle.evidence_definitions}
        facts = {item.evidence_definition_id: item for item in bundle.evidence_facts}
        if len(definitions) != len(bundle.evidence_definitions):
            raise ValueError("runtime evidence bundle contains duplicate evidence_definition_id values")
        if len(facts) != len(bundle.evidence_facts):
            raise ValueError("runtime evidence bundle contains duplicate fact bindings")
        package_faults = {item.fault_id: item for item in package.faults}
        registered_faults = {item.target_fault_id for item in bundle.evidence_definitions}
        missing = sorted(registered_faults - set(package_faults))
        extra = sorted(set(package_faults) - registered_faults)
        if missing or extra:
            raise ValueError(
                "semantic package and runtime evidence fault sets differ: "
                f"missing={missing}, extra={extra}"
            )
        missing_facts = sorted(set(definitions) - set(facts))
        if missing_facts:
            raise ValueError(f"runtime evidence facts missing definitions: {missing_facts}")

        ontology_faults: list[dict[str, Any]] = []
        for fault_id in sorted(package_faults):
            binding = package_faults[fault_id]
            expected: list[dict[str, Any]] = []
            contradictions: list[dict[str, Any]] = []
            for definition in sorted(
                (item for item in bundle.evidence_definitions if item.target_fault_id == fault_id),
                key=lambda item: item.evidence_definition_id,
            ):
                signature = {
                    "scope_type": "MECHANISM",
                    "scope_id": definition.evidence_definition_id,
                    "weight": 1.0 if definition.role == EvidenceRole.REQUIRED else 0.5,
                }
                if definition.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING}:
                    expected.append({**signature, "expected_state": "ABNORMAL"})
                else:
                    contradictions.append({**signature, "state": "ABNORMAL"})
            if not expected:
                raise ValueError(f"fault {fault_id} has no positive AstroGraph evidence definition")
            ontology_faults.append(
                {
                    "fault_id": fault_id,
                    "target_component_id": binding.target_component_id,
                    "fault_mode": binding.fault_mode,
                    "category": binding.category,
                    "exclusion_group": binding.exclusion_group,
                    "expected_evidence": expected,
                    "contradictions": contradictions,
                }
            )

        components: dict[str, dict[str, str]] = {}
        for item in package.faults:
            components.setdefault(
                item.target_component_id,
                {
                    "component_id": item.target_component_id,
                    "component_type": item.component_type,
                    "kind": item.component_kind,
                    "subsystem": item.subsystem,
                },
            )
        provenance_sources = list(package.source_refs)
        for item in package.faults:
            provenance_sources.extend(item.source_refs)
        ontology = {
            "contract": "DOR-FAULT-ONTOLOGY/0.5",
            "ontology_id": f"{package.package_id}@{package.version}",
            "domain": package.domain,
            "components": [components[key] for key in sorted(components)],
            "sensors": [],
            "commands": [],
            "topology_edges": [],
            "faults": ontology_faults,
            "provenance": {"sources": list(dict.fromkeys(provenance_sources))},
        }
        mechanism_map = {
            "contract": "DOR-MECHANISM-COMPONENT-MAP/0.5",
            "map_id": f"{package.package_id}-astrograph-evidence-bindings@{package.version}",
            "bindings": [
                {
                    "mechanism_id": definition.evidence_definition_id,
                    "component_ids": [package_faults[definition.target_fault_id].target_component_id],
                    "sensor_ids": [],
                    "provenance": "; ".join(definition.source_refs) or package.package_id,
                }
                for definition in sorted(bundle.evidence_definitions, key=lambda item: item.evidence_definition_id)
            ],
        }

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=("evidence_id", "scope_type", "scope_id", "state", "strength", "validity"),
            lineterminator="\n",
        )
        writer.writeheader()
        invalid_statuses = {
            EvidenceFactStatus.MISSING_DATA,
            EvidenceFactStatus.NOT_APPLICABLE,
            EvidenceFactStatus.INVALID_DATA,
            EvidenceFactStatus.UNKNOWN,
        }
        for definition_id in sorted(definitions):
            fact = facts[definition_id]
            if fact.status in invalid_statuses:
                state, validity = "MISSING", "INVALID"
            elif fact.status in {EvidenceFactStatus.SATISFIED, EvidenceFactStatus.CONTRADICTED}:
                state, validity = "ABNORMAL", "VALID"
            else:
                state, validity = "NORMAL", "VALID"
            writer.writerow(
                {
                    "evidence_id": fact.evidence_fact_id,
                    "scope_type": "MECHANISM",
                    "scope_id": definition_id,
                    "state": state,
                    # AstroGraph facts are crisp threshold outcomes, not calibrated probabilities.
                    "strength": "1.0",
                    "validity": validity,
                }
            )
        return ontology, mechanism_map, buffer.getvalue()

    def audit(self, bundle: EpisodeEvidenceBundle, request: DORSemanticAuditRequest) -> DORSemanticAudit:
        bundle_bytes = _canonical_bytes(bundle.model_dump(mode="json"))
        package_id = "UNAVAILABLE"
        package_version = "UNAVAILABLE"
        package_hash = _sha256(b"")
        ontology_hash = _sha256(b"")
        mechanism_hash = _sha256(b"")
        evidence_hash = _sha256(b"")
        stack_info: dict[str, Any] | None = None
        executable = request.executable_path
        executable_sha256: str | None = None
        runtime: DORRuntimeIdentity | None = None
        try:
            package, package_raw = self.load_package(request.semantic_package_path)
            package_id, package_version = package.package_id, package.version
            package_hash = _sha256(package_raw)
            ontology, mechanism_map, evidence_csv = self.compile_inputs(bundle, package)
            ontology_bytes = _canonical_bytes(ontology)
            mechanism_bytes = _canonical_bytes(mechanism_map)
            evidence_bytes = evidence_csv.encode("utf-8")
            ontology_hash = _sha256(ontology_bytes)
            mechanism_hash = _sha256(mechanism_bytes)
            evidence_hash = _sha256(evidence_bytes)
            runtime = inspect_dor_runtime(request.executable_path, timeout_s=request.timeout_s)
            executable = runtime.executable
            executable_sha256 = runtime.executable_sha256
            stack_info = runtime.stack_info
            with tempfile.TemporaryDirectory(prefix="astrograph-dor-") as temp_dir:
                root = Path(temp_dir)
                ontology_path = root / "ontology.json"
                mechanism_path = root / "mechanism_map.json"
                evidence_path = root / "evidence.csv"
                output_path = root / "localization.json"
                ontology_path.write_bytes(ontology_bytes)
                mechanism_path.write_bytes(mechanism_bytes)
                evidence_path.write_bytes(evidence_bytes)
                command = [
                    executable,
                    "localize",
                    "--ontology",
                    str(ontology_path),
                    "--mechanism-map",
                    str(mechanism_path),
                    "--evidence",
                    str(evidence_path),
                    "--output",
                    str(output_path),
                    "--min-support",
                    str(request.min_support),
                    "--retain-margin",
                    str(request.retain_margin),
                    "--max-contradiction-fraction",
                    str(request.max_contradiction_fraction),
                ]
                completed = run_managed_subprocess(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "no subprocess output").strip()[-2000:]
                    raise RuntimeError(f"DOR localize exited {completed.returncode}: {detail}")
                result = json.loads(output_path.read_text(encoding="utf-8"))
            status = DORSemanticAuditStatus(str(result.get("status")))
            safety = dict(result.get("safety") or {})
            if result.get("contract") != "DOR-LOCALIZATION-RESULT/0.5":
                raise ValueError(f"unexpected DOR localization contract: {result.get('contract')}")
            if safety.get("root_cause_claimed") is not False or safety.get("propagation_inferred") is not False:
                raise ValueError("DOR localization violated the no-root-cause/no-propagation claim boundary")
            supported = [str(item) for item in result.get("supported_fault_set", [])]
            eligible = status == DORSemanticAuditStatus.SUPPORTED and bool(supported)
            return self._result(
                bundle=bundle,
                request=request,
                package_id=package_id,
                package_version=package_version,
                status=status,
                reason=str(result.get("reason") or "DOR returned no reason"),
                eligible=eligible,
                supported=supported,
                candidate_sets=[[str(item) for item in row] for row in result.get("candidate_fault_sets", [])],
                bundle_hash=_sha256(bundle_bytes),
                package_hash=package_hash,
                ontology_hash=ontology_hash,
                mechanism_hash=mechanism_hash,
                evidence_hash=evidence_hash,
                result=result,
                safety=safety,
                stack_info=stack_info,
                executable=executable,
                executable_sha256=executable_sha256,
                runtime=runtime,
            )
        except Exception as exc:
            return self._result(
                bundle=bundle,
                request=request,
                package_id=package_id,
                package_version=package_version,
                status=DORSemanticAuditStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
                eligible=False,
                supported=[],
                candidate_sets=[],
                bundle_hash=_sha256(bundle_bytes),
                package_hash=package_hash,
                ontology_hash=ontology_hash,
                mechanism_hash=mechanism_hash,
                evidence_hash=evidence_hash,
                result=None,
                safety={"root_cause_claimed": False, "propagation_inferred": False},
                stack_info=stack_info,
                executable=executable,
                executable_sha256=executable_sha256,
                runtime=runtime,
            )

    @staticmethod
    def _result(
        *,
        bundle: EpisodeEvidenceBundle,
        request: DORSemanticAuditRequest,
        package_id: str,
        package_version: str,
        status: DORSemanticAuditStatus,
        reason: str,
        eligible: bool,
        supported: list[str],
        candidate_sets: list[list[str]],
        bundle_hash: str,
        package_hash: str,
        ontology_hash: str,
        mechanism_hash: str,
        evidence_hash: str,
        result: dict[str, Any] | None,
        safety: dict[str, Any],
        stack_info: dict[str, Any] | None,
        executable: str,
        executable_sha256: str | None,
        runtime: DORRuntimeIdentity | None,
    ) -> DORSemanticAudit:
        identity = {
            "episode_id": bundle.episode.episode_id,
            "package_sha256": package_hash,
            "bundle_sha256": bundle_hash,
            "executable_sha256": executable_sha256,
            "runtime_source_sha256": runtime.distribution_source_sha256 if runtime else None,
            "dor_version": runtime.dor_version if runtime else None,
            "status": status.value,
            "reason": reason,
            "result": result,
        }
        return DORSemanticAudit(
            audit_id=stable_id("DOR-SA", identity),
            episode_id=bundle.episode.episode_id,
            package_id=package_id,
            package_version=package_version,
            status=status,
            reason=reason,
            release_gate_enabled=request.release_gate,
            eligible_for_formal_fault_release=eligible,
            supported_fault_ids=supported,
            candidate_fault_sets=candidate_sets,
            input_bundle_sha256=bundle_hash,
            semantic_package_sha256=package_hash,
            generated_ontology_sha256=ontology_hash,
            generated_mechanism_map_sha256=mechanism_hash,
            generated_evidence_csv_sha256=evidence_hash,
            executable=executable,
            executable_sha256=executable_sha256,
            runtime=runtime,
            bridge_version=DORSemanticAuditService.bridge_version,
            dor_stack_info=stack_info,
            dor_contract=(str(result.get("contract")) if result else None),
            dor_result=result,
            safety=safety,
        )


def semantic_release_blockers(
    audit: DORSemanticAudit | None,
    accepted_fault_id: str | None,
) -> list[str]:
    """Return explicit gate blockers only for a proposed concrete fault release."""

    if audit is None or not audit.release_gate_enabled or accepted_fault_id is None:
        return []
    if not audit.eligible_for_formal_fault_release:
        return [f"DOR_SEMANTIC_AUDIT_{audit.status.value}"]
    if accepted_fault_id not in audit.supported_fault_ids:
        return [
            "DOR_SEMANTIC_AUDIT_FAULT_MISMATCH:"
            f"governor={accepted_fault_id},dor={','.join(audit.supported_fault_ids) or 'NONE'}"
        ]
    return []
