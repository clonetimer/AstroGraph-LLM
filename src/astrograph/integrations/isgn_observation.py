from __future__ import annotations

"""Fail-closed ISGN Observation Gate bridge for frozen mechanism datasets."""

from .process_control import run_managed_subprocess
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.evidence.executable import stable_id
from astrograph.integrations.external_runtime import (
    PythonCLIRuntimeIdentity,
    inspect_python_cli_runtime,
    sha256_file,
)
from astrograph.semantics import is_prohibited_field


class ISGNObservationInputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-isgn-observation-input.v2"
    dataset_id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_role: Literal["NORMAL_MECHANISM_DISCOVERY"]
    production_mechanism_eligible: bool
    state_names: list[str] = Field(min_length=1)
    input_names: list[str]
    require_measurement_chain: bool = True
    require_observed_timebase: bool = False
    require_temporal_sufficiency: bool = True
    sample_time_source_ref: str | None = None
    source_refs: list[str] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @field_validator("state_names", "input_names", "source_refs")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("ISGN observation contract lists must be unique")
        return value

    @model_validator(mode="after")
    def bind_observed_timebase_provenance(self) -> "ISGNObservationInputContract":
        if self.require_observed_timebase and not self.sample_time_source_ref:
            raise ValueError(
                "require_observed_timebase requires sample_time_source_ref provenance"
            )
        if (
            self.sample_time_source_ref is not None
            and self.sample_time_source_ref not in self.source_refs
        ):
            raise ValueError("sample_time_source_ref must be present in source_refs")
        return self


class ISGNObservationStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    CONDITIONALLY_QUALIFIED = "CONDITIONALLY_QUALIFIED"
    OBSERVATION_INSUFFICIENT = "OBSERVATION_INSUFFICIENT"
    CONTRACT_INSUFFICIENT = "CONTRACT_INSUFFICIENT"
    ERROR = "ERROR"


class ISGNObservationAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-isgn-observation-audit-request.v1"
    contract_path: str = Field(min_length=1)
    expected_contract_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    executable_path: str = "isgn"
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    allow_conditionally_qualified: bool = False
    release_gate_enabled: bool = True


class ISGNObservationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-isgn-observation-audit.v2"
    audit_id: str
    status: ISGNObservationStatus
    dataset_id: str | None = None
    system_id: str | None = None
    contract_path: str
    contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dataset_path: str | None = None
    dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime: PythonCLIRuntimeIdentity | None = None
    source_role: str | None = None
    production_mechanism_eligible: bool = False
    state_names: list[str] = Field(default_factory=list)
    input_names: list[str] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    sample_time_present: bool = False
    sample_time_source_ref: str | None = None
    timebase_status: str | None = None
    temporal_sufficiency_status: str | None = None
    eligible_for_mechanism_certification: bool = False
    release_gate_enabled: bool = True
    errors: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "Observation qualification only establishes whether the frozen normal dataset is adequate to attempt "
        "mechanism discovery under ISGN defaults. It does not discover, complete, or certify a mechanism."
    )


class ISGNObservationAuditService:
    @staticmethod
    def _load_contract(
        request: ISGNObservationAuditRequest,
    ) -> tuple[Path, str, ISGNObservationInputContract]:
        path = Path(request.contract_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ISGN observation contract not found: {path}")
        digest = sha256_file(path)
        if digest != request.expected_contract_sha256.lower():
            raise ValueError(
                f"ISGN observation contract SHA-256 mismatch: expected "
                f"{request.expected_contract_sha256.lower()}, got {digest}"
            )
        contract = ISGNObservationInputContract.model_validate_json(path.read_text(encoding="utf-8"))
        return path, digest, contract

    @staticmethod
    def _validate_dataset(
        contract_path: Path,
        contract: ISGNObservationInputContract,
    ) -> tuple[Path, str, dict[str, Any], bool]:
        dataset_path = (contract_path.parent / contract.dataset_path).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"ISGN observation dataset not found: {dataset_path}")
        digest = sha256_file(dataset_path)
        if digest != contract.dataset_sha256:
            raise ValueError(
                f"ISGN observation dataset SHA-256 mismatch: expected {contract.dataset_sha256}, got {digest}"
            )
        with np.load(dataset_path, allow_pickle=False) as archive:
            required = {"X", "U", "trajectory_id", "schema_json", "dt", "regime", "variant"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"ISGN PublicDataset NPZ missing members: {missing}")
            x = np.asarray(archive["X"])
            u = np.asarray(archive["U"])
            schema = json.loads(str(archive["schema_json"]))
            dt = float(archive["dt"])
            sample_time = (
                np.asarray(archive["sample_time_s"], dtype=np.float64)
                if "sample_time_s" in archive.files
                else None
            )
        states = [str(item.get("name")) for item in schema.get("states", [])]
        inputs = [str(item.get("name")) for item in schema.get("inputs", [])]
        if states != contract.state_names or inputs != contract.input_names:
            raise ValueError("ISGN dataset state/input names do not match observation contract")
        prohibited = sorted(name for name in [*states, *inputs] if is_prohibited_field(name))
        if prohibited:
            raise ValueError(f"ISGN mechanism dataset exposes prohibited label/truth fields: {prohibited}")
        if x.ndim != 3 or u.ndim != 3 or x.shape[0] != u.shape[0] or x.shape[1] != u.shape[1] + 1:
            raise ValueError("ISGN PublicDataset shapes are inconsistent")
        if len(states) != x.shape[2] or len(inputs) != u.shape[2]:
            raise ValueError("ISGN PublicDataset schema dimensions do not match arrays")
        if not np.isfinite(x).all() or not np.isfinite(u).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("ISGN PublicDataset contains non-finite values or invalid dt")
        if sample_time is not None:
            if not contract.sample_time_source_ref:
                raise ValueError(
                    "ISGN sample_time_s is present but the observation contract lacks provenance"
                )
            if sample_time.shape != x.shape[:2]:
                raise ValueError("ISGN sample_time_s shape must match X trajectory/time axes")
            if not np.isfinite(sample_time).all():
                raise ValueError("ISGN sample_time_s contains non-finite values")
            if np.any(np.diff(sample_time, axis=1) <= 0.0):
                raise ValueError("ISGN sample_time_s must be strictly increasing per trajectory")
        return dataset_path, digest, schema, sample_time is not None

    def audit(self, request: ISGNObservationAuditRequest) -> ISGNObservationAudit:
        base = {
            "audit_id": stable_id(
                "ISGN-OBS-AUDIT",
                {
                    "contract_path": str(Path(request.contract_path).resolve()),
                    "expected_contract_sha256": request.expected_contract_sha256.lower(),
                },
            ),
            "contract_path": str(Path(request.contract_path).resolve()),
            "release_gate_enabled": request.release_gate_enabled,
        }
        try:
            contract_path, contract_sha, contract = self._load_contract(request)
            if not contract.production_mechanism_eligible:
                raise ValueError("observation contract is not production-mechanism eligible")
            dataset_path, dataset_sha, _, sample_time_present = self._validate_dataset(
                contract_path, contract
            )
            runtime = inspect_python_cli_runtime(
                request.executable_path,
                distribution="isgn-toolkit",
                import_name="isgn",
                timeout_s=request.timeout_s,
            )
            with tempfile.TemporaryDirectory(prefix="astrograph-isgn-observe-") as directory:
                output_path = Path(directory) / "observation.json"
                completed = run_managed_subprocess(
                    [
                        runtime.executable,
                        "observe",
                        "--data",
                        str(dataset_path),
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
                    raise RuntimeError(f"ISGN observe exited {completed.returncode}: {detail}")
                report = json.loads(output_path.read_text(encoding="utf-8"))
            raw_status = str(report.get("status"))
            if raw_status not in {
                "QUALIFIED",
                "CONDITIONALLY_QUALIFIED",
                "OBSERVATION_INSUFFICIENT",
            }:
                raise ValueError(f"unsupported ISGN observation status: {raw_status!r}")
            checks = {str(item.get("name")): item for item in report.get("checks", [])}
            timebase = checks.get("timebase_integrity")
            temporal = checks.get("temporal_sufficiency")
            if timebase is None or temporal is None:
                raise ValueError(
                    "ISGN 2.3 observation report must contain timebase_integrity and "
                    "temporal_sufficiency checks"
                )
            timebase_status = str(timebase.get("status") or "")
            temporal_status = str(temporal.get("status") or "")
            if timebase_status not in {"PASS", "WARN", "FAIL", "N/A"}:
                raise ValueError(f"unsupported ISGN timebase status: {timebase_status!r}")
            if temporal_status not in {"PASS", "WARN", "FAIL", "N/A"}:
                raise ValueError(
                    f"unsupported ISGN temporal sufficiency status: {temporal_status!r}"
                )
            measurement = checks.get("measurement_chain")
            measurement_ok = not contract.require_measurement_chain or (
                measurement is not None and measurement.get("status") in {"PASS", "WARN"}
            )
            contract_checks_sufficient = (
                measurement_ok
                and (not contract.require_observed_timebase or timebase_status != "N/A")
                and (not contract.require_temporal_sufficiency or temporal_status != "N/A")
            )
            if not contract_checks_sufficient:
                status = ISGNObservationStatus.CONTRACT_INSUFFICIENT
            elif "FAIL" in {timebase_status, temporal_status}:
                status = ISGNObservationStatus.OBSERVATION_INSUFFICIENT
            else:
                status = ISGNObservationStatus(raw_status)
            eligible = status == ISGNObservationStatus.QUALIFIED or (
                status == ISGNObservationStatus.CONDITIONALLY_QUALIFIED
                and request.allow_conditionally_qualified
            )
            return ISGNObservationAudit(
                **base,
                status=status,
                dataset_id=contract.dataset_id,
                system_id=contract.system_id,
                contract_sha256=contract_sha,
                dataset_path=str(dataset_path),
                dataset_sha256=dataset_sha,
                runtime=runtime,
                source_role=contract.source_role,
                production_mechanism_eligible=contract.production_mechanism_eligible,
                state_names=contract.state_names,
                input_names=contract.input_names,
                report=report,
                sample_time_present=sample_time_present,
                sample_time_source_ref=contract.sample_time_source_ref,
                timebase_status=timebase_status,
                temporal_sufficiency_status=temporal_status,
                eligible_for_mechanism_certification=eligible,
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
            return ISGNObservationAudit(
                **base,
                status=ISGNObservationStatus.ERROR,
                errors=[f"{type(exc).__name__}: {exc}"],
            )


def isgn_observation_release_blockers(
    audit: ISGNObservationAudit | None,
    accepted_fault_id: str | None,
) -> list[str]:
    if audit is None or not audit.release_gate_enabled or accepted_fault_id is None:
        return []
    if audit.status == ISGNObservationStatus.ERROR:
        return ["ISGN_OBSERVATION_AUDIT_ERROR"]
    if not audit.eligible_for_mechanism_certification:
        return ["ISGN_OBSERVATION_NOT_QUALIFIED"]
    return []
