from __future__ import annotations

"""Configuration-driven routing of AstroGraph evidence through an ISGN Atlas."""

from .process_control import run_managed_subprocess
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus
from astrograph.evidence.executable import stable_id
from astrograph.integrations.external_runtime import (
    PythonCLIRuntimeIdentity,
    inspect_python_cli_runtime,
    sha256_file,
)


class ISGNEvidenceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_definition_id: str = Field(min_length=1)
    evidence_kind: str = Field(min_length=1)
    variables: list[str] = Field(min_length=1)
    source_scope: str | None = None
    phenomenon: str | None = None
    active_statuses: list[EvidenceFactStatus] = Field(
        default_factory=lambda: [EvidenceFactStatus.SATISFIED]
    )

    @field_validator("variables", "active_statuses")
    @classmethod
    def unique_values(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("ISGN evidence binding values must be unique")
        return value


class ISGNRoutingPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-isgn-routing-package.v1"
    package_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    atlas_path: str = Field(min_length=1)
    atlas_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bindings: list[ISGNEvidenceBinding] = Field(min_length=1)
    require_all_active_evidence_mapped: bool = True
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_bindings(self) -> "ISGNRoutingPackage":
        ids = [item.evidence_definition_id for item in self.bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("ISGN routing package contains duplicate evidence bindings")
        return self


class ISGNRoutingStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class ISGNRoutingAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-isgn-routing-audit-request.v1"
    package_path: str = Field(min_length=1)
    expected_package_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    executable_path: str = "isgn"
    timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    release_gate_enabled: bool = True


class ISGNRoutingAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-isgn-routing-audit.v1"
    audit_id: str
    status: ISGNRoutingStatus
    episode_id: str
    package_id: str | None = None
    package_path: str
    package_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    atlas_path: str | None = None
    atlas_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    system_id: str | None = None
    runtime: PythonCLIRuntimeIdentity | None = None
    route_decisions: list[dict[str, Any]] = Field(default_factory=list)
    active_evidence_count: int = Field(default=0, ge=0)
    mapped_evidence_count: int = Field(default=0, ge=0)
    unmapped_active_evidence_ids: list[str] = Field(default_factory=list)
    eligible_for_mechanism_evidence: bool = False
    release_gate_enabled: bool = True
    errors: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "ISGN routes existing health evidence to certified or explicitly uncovered mechanism scopes. "
        "It does not create fault labels, certify the supplied Atlas, or infer root cause."
    )


class ISGNRoutingAuditService:
    @staticmethod
    def _load_package(request: ISGNRoutingAuditRequest) -> tuple[Path, str, ISGNRoutingPackage]:
        path = Path(request.package_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ISGN routing package not found: {path}")
        digest = sha256_file(path)
        if digest != request.expected_package_sha256.lower():
            raise ValueError(
                f"ISGN routing package SHA-256 mismatch: expected "
                f"{request.expected_package_sha256.lower()}, got {digest}"
            )
        package = ISGNRoutingPackage.model_validate_json(path.read_text(encoding="utf-8"))
        return path, digest, package

    @staticmethod
    def _compile_events(
        bundle: EpisodeEvidenceBundle,
        package: ISGNRoutingPackage,
    ) -> tuple[list[dict[str, Any]], list[str], int]:
        definitions = {item.evidence_definition_id for item in bundle.evidence_definitions}
        bindings = {item.evidence_definition_id: item for item in package.bindings}
        unknown = sorted(set(bindings) - definitions)
        if unknown:
            raise ValueError(f"ISGN routing bindings reference unknown evidence definitions: {unknown}")
        events: list[dict[str, Any]] = []
        unmapped: list[str] = []
        active_count = 0
        for fact in bundle.evidence_facts:
            binding = bindings.get(fact.evidence_definition_id)
            if binding is None:
                if fact.status in {EvidenceFactStatus.SATISFIED, EvidenceFactStatus.CONTRADICTED}:
                    active_count += 1
                    unmapped.append(fact.evidence_definition_id)
                continue
            if fact.status not in binding.active_statuses:
                continue
            active_count += 1
            events.append(
                {
                    "event_id": fact.evidence_fact_id,
                    "evidence_kind": binding.evidence_kind,
                    "variables": binding.variables,
                    "source_scope": binding.source_scope,
                    "phenomenon": binding.phenomenon,
                    "score": fact.observed_value,
                    "metadata": {
                        "evidence_definition_id": fact.evidence_definition_id,
                        "evidence_status": fact.status.value,
                    },
                }
            )
        return events, sorted(set(unmapped)), active_count

    def audit(
        self,
        bundle: EpisodeEvidenceBundle,
        request: ISGNRoutingAuditRequest,
    ) -> ISGNRoutingAudit:
        base = {
            "audit_id": stable_id(
                "ISGN-ROUTE-AUDIT",
                {
                    "episode_id": bundle.episode.episode_id,
                    "package_path": str(Path(request.package_path).resolve()),
                    "expected_package_sha256": request.expected_package_sha256.lower(),
                },
            ),
            "episode_id": bundle.episode.episode_id,
            "package_path": str(Path(request.package_path).resolve()),
            "release_gate_enabled": request.release_gate_enabled,
        }
        try:
            package_path, package_sha, package = self._load_package(request)
            atlas_path = (package_path.parent / package.atlas_path).resolve()
            if not atlas_path.is_file():
                raise FileNotFoundError(f"ISGN Mechanism Atlas not found: {atlas_path}")
            atlas_sha = sha256_file(atlas_path)
            if atlas_sha != package.atlas_sha256:
                raise ValueError(
                    f"ISGN Mechanism Atlas SHA-256 mismatch: expected {package.atlas_sha256}, got {atlas_sha}"
                )
            atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
            if atlas.get("format") != "isgn-mechanism-atlas-v1":
                raise ValueError("unsupported ISGN Mechanism Atlas format")
            if atlas.get("system_id") != package.system_id:
                raise ValueError("ISGN routing package system_id does not match Atlas")
            events, unmapped, active_count = self._compile_events(bundle, package)
            runtime = inspect_python_cli_runtime(
                request.executable_path,
                distribution="isgn-toolkit",
                import_name="isgn",
                timeout_s=request.timeout_s,
            )
            routes: list[dict[str, Any]] = []
            if events:
                with tempfile.TemporaryDirectory(prefix="astrograph-isgn-") as directory:
                    root = Path(directory)
                    events_path = root / "events.json"
                    output_path = root / "routes.json"
                    events_path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
                    completed = run_managed_subprocess(
                        [
                            runtime.executable,
                            "route-evidence",
                            "--atlas",
                            str(atlas_path),
                            "--events",
                            str(events_path),
                            "--out",
                            str(output_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_s,
                        check=False,
                    )
                    if completed.returncode != 0:
                        detail = (completed.stderr or completed.stdout or "no ISGN output").strip()[-2000:]
                        raise RuntimeError(f"ISGN route-evidence exited {completed.returncode}: {detail}")
                    output = json.loads(output_path.read_text(encoding="utf-8"))
                if output.get("format") != "isgn-evidence-routing-v1":
                    raise ValueError("unsupported ISGN evidence routing output format")
                if output.get("system_id") != package.system_id:
                    raise ValueError("ISGN routing output system_id mismatch")
                routes = list(output.get("routes") or [])
                if len(routes) != len(events):
                    raise ValueError("ISGN routing output count does not match input events")
            all_certified = bool(routes) and all(
                item.get("decision") == "ROUTED_CERTIFIED_SCOPE" for item in routes
            )
            mapping_ok = not (package.require_all_active_evidence_mapped and unmapped)
            eligible = all_certified and mapping_ok
            status = (
                ISGNRoutingStatus.NOT_APPLICABLE
                if active_count == 0
                else ISGNRoutingStatus.QUALIFIED
                if eligible
                else ISGNRoutingStatus.BLOCKED
            )
            return ISGNRoutingAudit(
                **base,
                status=status,
                package_id=package.package_id,
                package_sha256=package_sha,
                atlas_path=str(atlas_path),
                atlas_sha256=atlas_sha,
                system_id=package.system_id,
                runtime=runtime,
                route_decisions=routes,
                active_evidence_count=active_count,
                mapped_evidence_count=len(events),
                unmapped_active_evidence_ids=unmapped,
                eligible_for_mechanism_evidence=eligible,
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as exc:
            return ISGNRoutingAudit(
                **base,
                status=ISGNRoutingStatus.ERROR,
                errors=[f"{type(exc).__name__}: {exc}"],
            )


def isgn_routing_release_blockers(
    audit: ISGNRoutingAudit | None,
    accepted_fault_id: str | None,
) -> list[str]:
    if audit is None or not audit.release_gate_enabled or accepted_fault_id is None:
        return []
    if audit.status == ISGNRoutingStatus.ERROR:
        return ["ISGN_MECHANISM_ROUTING_ERROR"]
    if not audit.eligible_for_mechanism_evidence:
        return ["ISGN_MECHANISM_SCOPE_UNSUPPORTED"]
    return []
