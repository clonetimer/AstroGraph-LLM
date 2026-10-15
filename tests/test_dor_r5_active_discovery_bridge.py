from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.integrations.dor_active_discovery import (
    DORActiveDiscoveryArtifactBinding,
    DORActiveDiscoveryAuditRequest,
    DORActiveDiscoveryAuditService,
    DORActiveDiscoveryStatus,
    R5ActiveDiscoveryAdvisory,
)


DOR = Path(os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", "__optional_dor_not_found__"))


def _payload() -> dict:
    return {
        "schema_version": "astrograph-dor-r5-active-discovery-input.v1",
        "structure_posterior": {"friction": 0.55, "authority-loss": 0.35, "jam": 0.10},
        "truth_quarantine": {
            "truth_channels_allowed": False,
            "held_out_truth_loaded": False,
            "true_structure_loaded": False,
        },
        "truth_channels_allowed": False,
        "next_experiment": {
            "experiment_id": "torque-step",
            "execution_mode": "SIMULATION",
            "repeats": 2,
            "cost": 1.0,
            "safety_eligible": True,
        },
        "identifiability": {
            "status": "ALTERNATIVES",
            "remaining_discriminating_experiments": ["torque-step", "coast-down"],
            "certification_allowed": False,
            "truth_channels_allowed": False,
        },
        "hypothesis_space_coverage_complete": False,
    }


def _write_payload(tmp_path: Path, payload: dict | None = None) -> tuple[Path, str]:
    path = tmp_path / "active-discovery-input.json"
    path.write_text(json.dumps(payload or _payload(), indent=2) + "\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not DOR.is_file(), reason="optional DOR R5 runtime is not installed")
def test_real_dor_r5_advisory_is_hash_bound_read_only_and_non_executable(tmp_path: Path) -> None:
    path, digest = _write_payload(tmp_path)
    audit = DORActiveDiscoveryAuditService().audit(
        DORActiveDiscoveryAuditRequest(
            input_artifact=DORActiveDiscoveryArtifactBinding(path=str(path), sha256=digest),
            executable_path=str(DOR),
        )
    )

    assert audit.status == DORActiveDiscoveryStatus.ADVISORY_ONLY, audit.errors
    assert audit.eligible_for_offline_planning is True
    assert audit.input_sha256 == digest
    assert audit.sanitized_input_sha256 and audit.advisory_sha256
    assert audit.runtime is not None
    assert audit.runtime.components["DOR"].stack_version == "1.0.3+timer.1"
    assert audit.runtime.components["ISGN"].stack_version == "2.4.0+timer.1"
    assert audit.advisory is not None
    assert audit.advisory.next_experiment is not None
    assert audit.advisory.next_experiment.execution_allowed is False
    assert audit.advisory.safety.experiment_execution_allowed is False
    assert audit.publication_allowed is False
    assert audit.execution_allowed is False
    assert audit.active_graph_modified is False
    assert audit.birrf_branch_count_delta == 0
    assert audit.files_written == []


def test_input_hash_mismatch_fails_before_runtime_execution(tmp_path: Path) -> None:
    path, _ = _write_payload(tmp_path)
    audit = DORActiveDiscoveryAuditService().audit(
        DORActiveDiscoveryAuditRequest(
            input_artifact=DORActiveDiscoveryArtifactBinding(path=str(path), sha256="0" * 64),
            executable_path=str(tmp_path / "missing-dor"),
        )
    )
    assert audit.status == DORActiveDiscoveryStatus.ERROR
    assert "SHA-256 mismatch" in audit.errors[0]
    assert audit.runtime is None


def test_unsafe_string_boolean_fails_before_runtime_execution(tmp_path: Path) -> None:
    payload = _payload()
    payload["next_experiment"]["safety_eligible"] = "false"
    path, digest = _write_payload(tmp_path, payload)
    audit = DORActiveDiscoveryAuditService().audit(
        DORActiveDiscoveryAuditRequest(
            input_artifact=DORActiveDiscoveryArtifactBinding(path=str(path), sha256=digest),
            executable_path=str(tmp_path / "missing-dor"),
        )
    )
    assert audit.status == DORActiveDiscoveryStatus.ERROR
    assert "safety_eligible" in audit.errors[0]
    assert audit.runtime is None


def _advisory() -> dict:
    return {
        "contract": "DOR-ISGN-ACTIVE-DISCOVERY-ADVISORY/1.0",
        "status": "ADVISORY_ONLY",
        "candidate_structures": [
            {"structure_id": "friction", "posterior_mass": 0.6},
            {"structure_id": "jam", "posterior_mass": 0.4},
        ],
        "posterior_mass_sum": 1.0,
        "next_experiment": None,
        "predictive_adequacy": None,
        "identifiability_status": "ALTERNATIVES",
        "evidence_stability_status": None,
        "hypothesis_space_coverage_complete": False,
        "remaining_discriminating_experiments": ["coast-down"],
        "admission_issues": [],
        "safety": {
            "truth_channels_allowed": False,
            "truth_quarantine_verified": True,
            "diagnostic_claim_allowed": False,
            "fault_identification_allowed": False,
            "root_cause_claim_allowed": False,
            "mechanism_identity_certified": False,
            "proposal_is_evidence": False,
            "experiment_execution_allowed": False,
            "raw_experiment_payload_forwarded": False,
        },
        "reason": "planning only",
    }


def test_output_contract_rejects_duplicate_experiment_ids() -> None:
    payload = _advisory()
    payload["remaining_discriminating_experiments"] = ["probe", "probe"]
    with pytest.raises(ValueError, match="unique"):
        R5ActiveDiscoveryAdvisory.model_validate(payload)


def test_output_contract_rejects_privilege_escalation() -> None:
    payload = _advisory()
    payload["safety"]["experiment_execution_allowed"] = True
    with pytest.raises(ValueError, match="privilege escalation"):
        R5ActiveDiscoveryAdvisory.model_validate(payload)
