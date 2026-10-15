from __future__ import annotations
from .process_control import run_managed_subprocess

import math
import re
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from astrograph.evidence.executable import stable_id

from .dor_runtime import DORRuntimeIdentity, inspect_dor_runtime, sha256_bytes, sha256_file


def _release_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"unsupported runtime version: {version!r}")
    return tuple(int(value) for value in match.groups())


def require_dor_r5_active_discovery_runtime(identity: DORRuntimeIdentity) -> None:
    required = {"DOR": (1, 0, 3), "ISGN": (2, 4, 0)}
    for name, minimum in required.items():
        component = identity.components.get(name)
        if component is None:
            raise ValueError(f"DOR runtime is missing {name}")
        versions = {
            component.stack_version,
            component.metadata_version,
            component.module_version,
        }
        if len(versions) != 1:
            raise ValueError(f"DOR runtime identity diverges for {name}")
        if _release_tuple(component.stack_version) < minimum:
            expected = ".".join(map(str, minimum))
            raise ValueError(
                f"{name} {component.stack_version} is below Active Discovery minimum {expected}"
            )


class R5TruthQuarantine(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    truth_channels_allowed: StrictBool = False
    held_out_truth_loaded: StrictBool = False
    true_structure_loaded: StrictBool = False

    @model_validator(mode="after")
    def all_false(self) -> "R5TruthQuarantine":
        if any(
            (
                self.truth_channels_allowed,
                self.held_out_truth_loaded,
                self.true_structure_loaded,
            )
        ):
            raise ValueError("Active Discovery truth quarantine fields must all be false")
        return self


class R5ExperimentRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    execution_mode: Literal["ADVISORY_ONLY", "OFFLINE_REPLAY", "SIMULATION"] = "SIMULATION"
    repeats: StrictInt = Field(default=1, ge=1)
    cost: StrictFloat | StrictInt = Field(default=1.0, gt=0.0)
    safety_eligible: StrictBool = True

    @model_validator(mode="after")
    def require_safe_offline_candidate(self) -> "R5ExperimentRecommendation":
        if self.safety_eligible is not True:
            raise ValueError("recommended experiment must be explicitly safety eligible")
        return self


class R5PredictiveAdequacy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ADEQUATE", "HYPOTHESIS_CLASS_INADEQUATE"]
    truth_channels_allowed: StrictBool = False
    min_mahalanobis: StrictFloat | StrictInt | None = Field(default=None, ge=0.0)
    threshold: StrictFloat | StrictInt | None = Field(default=None, ge=0.0)
    nearest_structure: str | None = None
    nearest_particle: str | None = None

    @model_validator(mode="after")
    def truth_stays_quarantined(self) -> "R5PredictiveAdequacy":
        if self.truth_channels_allowed is not False:
            raise ValueError("predictive adequacy cannot use truth channels")
        return self


class R5EvidenceStability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["STABLE", "UNSTABLE", "INSUFFICIENT"]
    current_top_structure: str | None = None
    robust_top_structure: str | None = None
    n_observations: StrictInt | None = Field(default=None, ge=0)
    loo_same_top_fraction: StrictFloat | StrictInt | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    loo_top_structures: list[str] = Field(default_factory=list)
    reason: str | None = None
    truth_channels_allowed: StrictBool = False

    @field_validator("loo_top_structures")
    @classmethod
    def nonempty_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("stability structure ids must be nonempty")
        return value


class R5Identifiability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "UNIQUE_SUPPORTED",
        "ALTERNATIVES",
        "UNIDENTIFIABLE",
        "HYPOTHESIS_CLASS_INADEQUATE",
    ]
    top_structure: str | None = None
    top_posterior_mass: StrictFloat | StrictInt | None = Field(default=None, ge=0.0, le=1.0)
    alternative_structures: list[str] = Field(default_factory=list)
    credible_structures: list[str] = Field(default_factory=list)
    remaining_discriminating_experiments: list[str] = Field(default_factory=list)
    max_pairwise_separation: dict[str, StrictFloat | StrictInt] = Field(default_factory=dict)
    reason: str | None = None
    certification_allowed: StrictBool = False
    truth_channels_allowed: StrictBool = False

    @field_validator(
        "alternative_structures",
        "credible_structures",
        "remaining_discriminating_experiments",
    )
    @classmethod
    def unique_nonempty_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("identifiability ids must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def no_certification_or_truth(self) -> "R5Identifiability":
        if self.certification_allowed is not False or self.truth_channels_allowed is not False:
            raise ValueError("Active Discovery cannot certify a mechanism or use truth channels")
        return self


class R5ActiveDiscoveryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dor-r5-active-discovery-input.v1"] = (
        "astrograph-dor-r5-active-discovery-input.v1"
    )
    structure_posterior: dict[str, StrictFloat | StrictInt]
    truth_quarantine: R5TruthQuarantine
    truth_channels_allowed: StrictBool = False
    next_experiment: R5ExperimentRecommendation | None = None
    predictive_adequacy: R5PredictiveAdequacy | None = None
    identifiability: R5Identifiability | None = None
    evidence_stability: R5EvidenceStability | None = None
    hypothesis_space_coverage_complete: StrictBool = False

    @model_validator(mode="after")
    def validate_posterior_and_unique_claim(self) -> "R5ActiveDiscoveryPayload":
        if self.truth_channels_allowed is not False:
            raise ValueError("top-level truth_channels_allowed must be false")
        if not self.structure_posterior:
            raise ValueError("structure_posterior must be nonempty")
        for structure_id, mass in self.structure_posterior.items():
            if not structure_id.strip() or isinstance(mass, bool):
                raise ValueError("structure posterior ids and masses must be strict")
            if not math.isfinite(float(mass)) or not 0.0 <= float(mass) <= 1.0:
                raise ValueError("structure posterior masses must be finite and in [0, 1]")
        if abs(sum(map(float, self.structure_posterior.values())) - 1.0) > 1e-6:
            raise ValueError("structure posterior masses must sum to one")
        if self.identifiability and self.identifiability.status == "UNIQUE_SUPPORTED":
            if not self.evidence_stability or self.evidence_stability.status != "STABLE":
                raise ValueError("UNIQUE_SUPPORTED requires STABLE evidence")
            if self.hypothesis_space_coverage_complete is not True:
                raise ValueError("UNIQUE_SUPPORTED requires complete hypothesis-space coverage")
        return self


class R5AdvisoryCandidateStructure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    structure_id: str = Field(min_length=1)
    posterior_mass: StrictFloat | StrictInt = Field(ge=0.0, le=1.0)


class R5AdvisoryExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    execution_mode: Literal["ADVISORY_ONLY", "OFFLINE_REPLAY", "SIMULATION"]
    repeats: StrictInt = Field(ge=1)
    cost: StrictFloat | StrictInt | None = Field(default=None, gt=0.0)
    safety_eligible: StrictBool
    execution_allowed: StrictBool = False

    @model_validator(mode="after")
    def never_executable(self) -> "R5AdvisoryExperiment":
        if self.execution_allowed is not False:
            raise ValueError("DOR Active Discovery advisory cannot allow execution")
        return self


class R5AdvisorySafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    truth_channels_allowed: StrictBool = False
    truth_quarantine_verified: StrictBool
    diagnostic_claim_allowed: StrictBool = False
    fault_identification_allowed: StrictBool = False
    root_cause_claim_allowed: StrictBool = False
    mechanism_identity_certified: StrictBool = False
    proposal_is_evidence: StrictBool = False
    experiment_execution_allowed: StrictBool = False
    raw_experiment_payload_forwarded: StrictBool = False

    @model_validator(mode="after")
    def privileges_stay_false(self) -> "R5AdvisorySafety":
        forbidden = (
            self.truth_channels_allowed,
            self.diagnostic_claim_allowed,
            self.fault_identification_allowed,
            self.root_cause_claim_allowed,
            self.mechanism_identity_certified,
            self.proposal_is_evidence,
            self.experiment_execution_allowed,
            self.raw_experiment_payload_forwarded,
        )
        if any(forbidden):
            raise ValueError("DOR Active Discovery advisory attempted privilege escalation")
        return self


class R5ActiveDiscoveryAdvisory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["DOR-ISGN-ACTIVE-DISCOVERY-ADVISORY/1.0"]
    status: Literal["ADVISORY_ONLY", "BLOCKED"]
    candidate_structures: list[R5AdvisoryCandidateStructure]
    posterior_mass_sum: StrictFloat | StrictInt = Field(ge=0.0, le=1.000001)
    next_experiment: R5AdvisoryExperiment | None = None
    predictive_adequacy: R5PredictiveAdequacy | None = None
    identifiability_status: Literal[
        "UNIQUE_SUPPORTED",
        "ALTERNATIVES",
        "UNIDENTIFIABLE",
        "HYPOTHESIS_CLASS_INADEQUATE",
    ] | None = None
    evidence_stability_status: Literal["STABLE", "UNSTABLE", "INSUFFICIENT"] | None = None
    hypothesis_space_coverage_complete: StrictBool
    remaining_discriminating_experiments: list[str]
    admission_issues: list[str]
    safety: R5AdvisorySafety
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_advisory(self) -> "R5ActiveDiscoveryAdvisory":
        structure_ids = [item.structure_id for item in self.candidate_structures]
        if len(structure_ids) != len(set(structure_ids)):
            raise ValueError("DOR advisory candidate structure ids must be unique")
        if len(self.remaining_discriminating_experiments) != len(
            set(self.remaining_discriminating_experiments)
        ):
            raise ValueError("DOR advisory remaining experiment ids must be unique")
        if self.status == "ADVISORY_ONLY":
            if self.admission_issues:
                raise ValueError("ADVISORY_ONLY cannot contain admission issues")
            if not self.safety.truth_quarantine_verified:
                raise ValueError("ADVISORY_ONLY requires verified truth quarantine")
            if abs(float(self.posterior_mass_sum) - 1.0) > 1e-6:
                raise ValueError("ADVISORY_ONLY posterior mass must sum to one")
        return self


class DORActiveDiscoveryArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class DORActiveDiscoveryAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dor-active-discovery-audit-request.v1"] = (
        "astrograph-dor-active-discovery-audit-request.v1"
    )
    input_artifact: DORActiveDiscoveryArtifactBinding
    executable_path: str = Field(default="dor", min_length=1)
    timeout_s: float = Field(default=60.0, gt=0.0, le=300.0)
    release_gate_enabled: StrictBool = True


class DORActiveDiscoveryStatus(str, Enum):
    ADVISORY_ONLY = "ADVISORY_ONLY"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class DORActiveDiscoveryAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dor-active-discovery-audit.v1"] = (
        "astrograph-dor-active-discovery-audit.v1"
    )
    audit_id: str = Field(min_length=1)
    status: DORActiveDiscoveryStatus
    input_path: str
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sanitized_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    advisory_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime: DORRuntimeIdentity | None = None
    advisory: R5ActiveDiscoveryAdvisory | None = None
    eligible_for_offline_planning: bool = False
    release_gate_enabled: bool = True
    publication_allowed: Literal[False] = False
    execution_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    birrf_branch_count_delta: Literal[0] = 0
    files_written: list[str] = Field(default_factory=list, max_length=0)
    errors: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "Offline experiment-planning qualification only. It is not a diagnosis, fault identification, "
        "root-cause proof, knowledge publication, BiRRF vote or authority to execute an experiment."
    )


class DORActiveDiscoveryAuditService:
    bridge_version = "astrograph-dor-r5-active-discovery-bridge@1.0.0"

    def audit(self, request: DORActiveDiscoveryAuditRequest) -> DORActiveDiscoveryAudit:
        input_path = Path(request.input_artifact.path).resolve()
        base = {
            "audit_id": stable_id(
                "DOR-ACTIVE-DISCOVERY-AUDIT",
                {
                    "path": str(input_path),
                    "expected_sha256": request.input_artifact.sha256.lower(),
                    "bridge_version": self.bridge_version,
                },
            ),
            "input_path": str(input_path),
            "release_gate_enabled": request.release_gate_enabled,
        }
        actual_sha: str | None = None
        runtime: DORRuntimeIdentity | None = None
        try:
            if not input_path.is_file():
                raise FileNotFoundError(f"Active Discovery input not found: {input_path}")
            actual_sha = sha256_file(input_path)
            if actual_sha != request.input_artifact.sha256.lower():
                raise ValueError("Active Discovery input SHA-256 mismatch")
            payload = R5ActiveDiscoveryPayload.model_validate_json(
                input_path.read_text(encoding="utf-8")
            )
            runtime = inspect_dor_runtime(request.executable_path, timeout_s=request.timeout_s)
            require_dor_r5_active_discovery_runtime(runtime)
            sanitized = payload.model_dump_json(exclude_none=True).encode("utf-8")
            sanitized_sha = sha256_bytes(sanitized)
            with tempfile.TemporaryDirectory(prefix="astrograph-dor-active-discovery-") as directory:
                root = Path(directory)
                request_path = root / "request.json"
                output_path = root / "advisory.json"
                request_path.write_bytes(sanitized)
                completed = run_managed_subprocess(
                    [
                        runtime.executable,
                        "active-discovery-advisory",
                        "--input",
                        str(request_path),
                        "--output",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode not in {0, 2} or not output_path.is_file():
                    detail = (completed.stderr or completed.stdout or "no DOR output").strip()[-2000:]
                    raise RuntimeError(
                        f"DOR active-discovery-advisory exited {completed.returncode}: {detail}"
                    )
                advisory_bytes = output_path.read_bytes()
            advisory = R5ActiveDiscoveryAdvisory.model_validate_json(advisory_bytes)
            if (completed.returncode == 0) != (advisory.status == "ADVISORY_ONLY"):
                raise ValueError("DOR Active Discovery exit status and advisory status disagree")
            status = DORActiveDiscoveryStatus(advisory.status)
            return DORActiveDiscoveryAudit(
                **base,
                status=status,
                input_sha256=actual_sha,
                sanitized_input_sha256=sanitized_sha,
                advisory_sha256=sha256_bytes(advisory_bytes),
                runtime=runtime,
                advisory=advisory,
                eligible_for_offline_planning=status == DORActiveDiscoveryStatus.ADVISORY_ONLY,
            )
        except Exception as exc:
            return DORActiveDiscoveryAudit(
                **base,
                status=DORActiveDiscoveryStatus.ERROR,
                input_sha256=actual_sha,
                runtime=runtime,
                errors=[str(exc)],
            )


__all__ = [
    "R5ActiveDiscoveryPayload",
    "R5ActiveDiscoveryAdvisory",
    "R5AdvisoryCandidateStructure",
    "R5AdvisoryExperiment",
    "R5AdvisorySafety",
    "DORActiveDiscoveryArtifactBinding",
    "DORActiveDiscoveryAudit",
    "DORActiveDiscoveryAuditRequest",
    "DORActiveDiscoveryAuditService",
    "DORActiveDiscoveryStatus",
    "R5EvidenceStability",
    "R5ExperimentRecommendation",
    "R5Identifiability",
    "R5PredictiveAdequacy",
    "R5TruthQuarantine",
    "require_dor_r5_active_discovery_runtime",
]
