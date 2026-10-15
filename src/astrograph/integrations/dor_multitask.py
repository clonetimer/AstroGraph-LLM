from __future__ import annotations

"""Fail-closed DOR cross-task arbitration and optional fault identification.

The bridge consumes already-produced task artifacts.  It does not derive a
detection, localization, classification, root cause, or parameter change from
AstroGraph scores, and its result is a release qualification rather than a
fourth BiRRF ranking branch.
"""

from .process_control import run_managed_subprocess
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence.executable import stable_id

from .dor_runtime import DORRuntimeIdentity, inspect_dor_runtime, sha256_file


class DORArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class DORIdentificationRunRequest(BaseModel):
    """Inputs for DOR ``identify``; localization/root come from the parent request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology: DORArtifactBinding
    evidence_csv: DORArtifactBinding
    isgn_identification: DORArtifactBinding | None = None


class DORMultitaskAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-multitask-audit-request.v1"
    detection: DORArtifactBinding | None = None
    classification: DORArtifactBinding | None = None
    localization: DORArtifactBinding | None = None
    root_cause: DORArtifactBinding | None = None
    identification: DORArtifactBinding | None = None
    identification_run: DORIdentificationRunRequest | None = None
    case_id: str | None = None
    executable_path: str = Field(default="dor", min_length=1)
    timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    release_gate: bool = True

    @model_validator(mode="after")
    def validate_inputs(self) -> "DORMultitaskAuditRequest":
        if self.identification is not None and self.identification_run is not None:
            raise ValueError("provide a frozen identification or identification_run, not both")
        if self.identification_run is not None and self.localization is None:
            raise ValueError("identification_run requires a localization artifact")
        supplied = sum(
            item is not None
            for item in (
                self.detection,
                self.classification,
                self.localization,
                self.root_cause,
                self.identification,
                self.identification_run,
            )
        )
        if supplied < 2:
            raise ValueError("multitask arbitration requires at least two task artifacts")
        return self


class DORMultitaskStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    ALTERNATIVES = "ALTERNATIVES"
    UNKNOWN = "UNKNOWN"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


class DORMultitaskAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-multitask-audit.v1"
    audit_id: str = Field(min_length=1)
    status: DORMultitaskStatus
    reason: str = Field(min_length=1)
    diagnostic_depth: str
    release_gate_enabled: bool
    eligible_for_formal_fault_release: bool
    strongest_fault_ids: list[str] = Field(default_factory=list)
    physical_change_claimed: bool = False
    contradictions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    input_artifact_sha256: dict[str, str] = Field(default_factory=dict)
    generated_identification_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    runtime: DORRuntimeIdentity | None = None
    dor_contract: str | None = None
    dor_result: dict[str, Any] | None = None
    safety: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Cross-task consistency and physical-change qualification only. This audit does not "
        "create a BiRRF vote, choose a fault among contradictions, or imply operational authority."
    )


class DORMultitaskAuditService:
    bridge_version = "astrograph-dor-multitask-bridge@1.0.0"
    _CONTRACTS = {
        "localization": "DOR-LOCALIZATION-RESULT/0.5",
        "root_cause": "DOR-ROOT-CAUSE-RESULT/0.6",
        "identification": "DOR-FAULT-IDENTIFICATION-RESULT/0.7",
    }

    @staticmethod
    def _verify_binding(name: str, binding: DORArtifactBinding) -> tuple[Path, dict[str, Any] | None]:
        path = Path(binding.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DOR {name} artifact not found: {path}")
        actual = sha256_file(path)
        if actual != binding.sha256.lower():
            raise ValueError(f"DOR {name} artifact SHA-256 mismatch")
        if name == "evidence_csv":
            return path, None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"DOR {name} artifact must contain a JSON object")
        expected_contract = DORMultitaskAuditService._CONTRACTS.get(name)
        if expected_contract is not None and payload.get("contract") != expected_contract:
            raise ValueError(
                f"DOR {name} contract {payload.get('contract')!r} does not match {expected_contract!r}"
            )
        return path, payload

    @staticmethod
    def _strongest_fault_ids(result: dict[str, Any]) -> list[str]:
        depth = str(result.get("diagnostic_depth") or "NONE")
        claims = dict(result.get("claims") or {})
        if depth == "CLASSIFICATION":
            return sorted(map(str, (claims.get("classification") or {}).get("prediction_set") or []))
        key = {
            "LOCALIZATION": "localization",
            "ROOT_CAUSE": "root_cause",
            "IDENTIFICATION": "identification",
        }.get(depth)
        return sorted(map(str, (claims.get(key) or {}).get("fault_ids") or [])) if key else []

    @staticmethod
    def _validate_identification_safety(payload: dict[str, Any]) -> None:
        safety = dict(payload.get("safety") or {})
        required_true = (
            "localization_is_not_identification",
            "residual_is_not_parameter_drift",
            "model_disagreement_is_not_fault_mode",
            "isgn_required_for_parametric_or_structure_claims",
        )
        for key in required_true:
            if safety.get(key) is not True:
                raise ValueError(f"DOR identification safety invariant violated: {key}")
        if safety.get("root_cause_implied") is not False:
            raise ValueError("DOR identification safety invariant violated: root_cause_implied")

    def audit(self, request: DORMultitaskAuditRequest) -> DORMultitaskAudit:
        runtime: DORRuntimeIdentity | None = None
        hashes: dict[str, str] = {}
        generated_identification_sha: str | None = None
        result: dict[str, Any] | None = None
        try:
            bindings = {
                "detection": request.detection,
                "classification": request.classification,
                "localization": request.localization,
                "root_cause": request.root_cause,
                "identification": request.identification,
            }
            paths: dict[str, Path] = {}
            payloads: dict[str, dict[str, Any]] = {}
            for name, binding in bindings.items():
                if binding is not None:
                    path, payload = self._verify_binding(name, binding)
                    paths[name] = path
                    hashes[name] = binding.sha256.lower()
                    if payload is not None:
                        payloads[name] = payload
                        artifact_case_id = payload.get("case_id")
                        if (
                            request.case_id is not None
                            and artifact_case_id not in (None, "", request.case_id)
                        ):
                            raise ValueError(f"DOR {name} artifact case_id mismatch")
            if "identification" in payloads:
                self._validate_identification_safety(payloads["identification"])
            runtime = inspect_dor_runtime(request.executable_path, timeout_s=request.timeout_s)
            with tempfile.TemporaryDirectory(prefix="astrograph-dor-multitask-") as directory:
                root = Path(directory)
                if request.identification_run is not None:
                    ontology_path, _ = self._verify_binding(
                        "ontology", request.identification_run.ontology
                    )
                    evidence_path, _ = self._verify_binding(
                        "evidence_csv", request.identification_run.evidence_csv
                    )
                    hashes["ontology"] = request.identification_run.ontology.sha256.lower()
                    hashes["evidence_csv"] = (
                        request.identification_run.evidence_csv.sha256.lower()
                    )
                    identification_path = root / "identification.json"
                    command = [
                        runtime.executable,
                        "identify",
                        "--ontology",
                        str(ontology_path),
                        "--localization",
                        str(paths["localization"]),
                        "--evidence",
                        str(evidence_path),
                        "--output",
                        str(identification_path),
                    ]
                    if "root_cause" in paths:
                        command.extend(["--root-cause", str(paths["root_cause"])])
                    if request.identification_run.isgn_identification is not None:
                        isgn_path, _ = self._verify_binding(
                            "isgn_identification",
                            request.identification_run.isgn_identification,
                        )
                        hashes["isgn_identification"] = (
                            request.identification_run.isgn_identification.sha256.lower()
                        )
                        command.extend(["--isgn-identification", str(isgn_path)])
                    completed = run_managed_subprocess(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_s,
                        check=False,
                    )
                    if completed.returncode != 0:
                        detail = (completed.stderr or completed.stdout or "no DOR output").strip()[-2000:]
                        raise RuntimeError(
                            f"DOR identification exited {completed.returncode}: {detail}"
                        )
                    identification_payload = json.loads(
                        identification_path.read_text(encoding="utf-8")
                    )
                    if identification_payload.get("contract") != self._CONTRACTS["identification"]:
                        raise ValueError("DOR generated an unexpected identification contract")
                    self._validate_identification_safety(identification_payload)
                    generated_identification_sha = sha256_file(identification_path)
                    paths["identification"] = identification_path
                    payloads["identification"] = identification_payload

                output_path = root / "multitask.json"
                command = [runtime.executable, "multi-task-arbitrate"]
                for name in (
                    "detection",
                    "classification",
                    "localization",
                    "root_cause",
                    "identification",
                ):
                    if name in paths:
                        command.extend([f"--{name.replace('_', '-')}", str(paths[name])])
                if request.case_id is not None:
                    command.extend(["--case-id", request.case_id])
                command.extend(["--output", str(output_path)])
                completed = run_managed_subprocess(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode not in {0, 2}:
                    detail = (completed.stderr or completed.stdout or "no DOR output").strip()[-2000:]
                    raise RuntimeError(f"DOR multitask exited {completed.returncode}: {detail}")
                result = json.loads(output_path.read_text(encoding="utf-8"))

            if result.get("contract") != "DOR-MULTITASK-DIAGNOSIS/0.8":
                raise ValueError(f"unexpected DOR multitask contract: {result.get('contract')}")
            safety = dict(result.get("safety") or {})
            for key in (
                "detection_is_not_diagnosis",
                "classification_is_not_root_cause",
                "localization_is_not_root_cause",
                "identification_does_not_imply_root_cause",
                "missing_optional_task_does_not_create_evidence",
            ):
                if safety.get(key) is not True:
                    raise ValueError(f"DOR multitask safety invariant violated: {key}")
            if safety.get("test_truth_allowed") is not False:
                raise ValueError("DOR multitask safety invariant violated: test_truth_allowed")
            status = DORMultitaskStatus(str(result.get("status")))
            depth = str(result.get("diagnostic_depth") or "NONE")
            strongest = self._strongest_fault_ids(result)
            physical_change_claimed = bool(
                depth == "IDENTIFICATION"
                and (payloads.get("identification", {}).get("safety") or {}).get(
                    "physical_change_claimed"
                )
                is True
            )
            eligible = status == DORMultitaskStatus.SUPPORTED and (
                depth == "DETECTION" or bool(strongest)
            )
            identity = {
                "case_id": request.case_id,
                "input_artifact_sha256": hashes,
                "generated_identification_sha256": generated_identification_sha,
                "runtime_fingerprint": runtime.stack_fingerprint_sha256,
                "result": result,
            }
            return DORMultitaskAudit(
                audit_id=stable_id("DOR-MT", identity),
                status=status,
                reason=str(result.get("reason") or "DOR returned no reason"),
                diagnostic_depth=depth,
                release_gate_enabled=request.release_gate,
                eligible_for_formal_fault_release=eligible,
                strongest_fault_ids=strongest,
                physical_change_claimed=physical_change_claimed,
                contradictions=list(map(str, result.get("contradictions") or [])),
                warnings=list(map(str, result.get("warnings") or [])),
                input_artifact_sha256=hashes,
                generated_identification_sha256=generated_identification_sha,
                runtime=runtime,
                dor_contract=str(result["contract"]),
                dor_result=result,
                safety=safety,
            )
        except Exception as exc:
            identity = {
                "case_id": request.case_id,
                "input_artifact_sha256": hashes,
                "generated_identification_sha256": generated_identification_sha,
                "runtime_fingerprint": runtime.stack_fingerprint_sha256 if runtime else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return DORMultitaskAudit(
                audit_id=stable_id("DOR-MT", identity),
                status=DORMultitaskStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
                diagnostic_depth="NONE",
                release_gate_enabled=request.release_gate,
                eligible_for_formal_fault_release=False,
                input_artifact_sha256=hashes,
                generated_identification_sha256=generated_identification_sha,
                runtime=runtime,
            )


def multitask_release_blockers(
    audit: DORMultitaskAudit | None,
    accepted_fault_id: str | None,
) -> list[str]:
    if audit is None or not audit.release_gate_enabled:
        return []
    if audit.status == DORMultitaskStatus.ERROR:
        return ["DOR_MULTITASK_ERROR"]
    if audit.status == DORMultitaskStatus.ABSTAIN:
        return ["DOR_MULTITASK_CONTRADICTION"]
    if audit.status == DORMultitaskStatus.ALTERNATIVES:
        return ["DOR_MULTITASK_UNRESOLVED_ALTERNATIVES"]
    if audit.status == DORMultitaskStatus.UNKNOWN:
        return ["DOR_MULTITASK_UNKNOWN"]
    if accepted_fault_id is None:
        return []
    if audit.diagnostic_depth == "DETECTION":
        return ["DOR_MULTITASK_INSUFFICIENT_FAULT_DEPTH"]
    if accepted_fault_id not in audit.strongest_fault_ids:
        return [
            f"DOR_MULTITASK_FAULT_MISMATCH:governor={accepted_fault_id},"
            f"dor={','.join(audit.strongest_fault_ids)}"
        ]
    return []
