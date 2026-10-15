from __future__ import annotations

import pytest
from pydantic import ValidationError

from astrograph.diagnosis import (
    CapabilityAxesEvidence,
    DiagnosisEvidenceBundle,
    DiagnosisPolicyGate,
    DiagnosisReleaseStatus,
    KGEvidenceReference,
    ModelEvidenceReference,
    SafetyReplayReference,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _bundle(**updates: object) -> DiagnosisEvidenceBundle:
    data = {
        "bundle_id": "BUNDLE-1",
        "sample_id": "SAMPLE-1",
        "episode_id": "EP-1",
        "split": "test",
        "telemetry_sha256": HASH_A,
        "model_evidence": ModelEvidenceReference(
            model_id="MODEL-RF",
            model_version="0.5.17",
            model_sha256=HASH_B,
            prediction_id="PRED-1",
            prediction_sha256=HASH_C,
            probability_ranking_id="RANK-1",
            top_fault_id="THERMAL_HEATER_STUCK_ON",
            calibrated=True,
        ),
        "kg_evidence": KGEvidenceReference(
            release_id="KG-GOLD-ONLINE-V058",
            knowledge_version="astrograph-kg-gold-online-v058",
            nodes_sha256=HASH_A,
            edges_sha256=HASH_B,
            query_ids=["GQ-FORWARD", "GQ-BACKWARD"],
            query_sha256_values=[HASH_A, HASH_B],
            read_user="astrograph_reader",
            row_count=42,
        ),
        "safety_replay": SafetyReplayReference(
            replay_id="REPLAY-1",
            replay_sha256=HASH_C,
            status="PASS",
            wrong_automatic_accept_rate=0.0,
        ),
        "capability_axes": CapabilityAxesEvidence(
            supported_axes=["seed"],
            unsupported_axes=["severity", "noise"],
            requested_axes=["seed"],
            capability_manifest_id="SIM-CAP-V1",
            capability_manifest_sha256=HASH_A,
        ),
    }
    data.update(updates)
    return DiagnosisEvidenceBundle.model_validate(data)


def test_policy_gate_allows_determinate_only_with_complete_evidence() -> None:
    decision = DiagnosisPolicyGate().decide(_bundle())

    assert decision.status == DiagnosisReleaseStatus.DETERMINATE_DIAGNOSIS
    assert decision.blocking_reasons == []
    assert "TEST split evidence" in decision.warnings[0]


def test_policy_gate_missing_kg_downgrades_to_conditional() -> None:
    decision = DiagnosisPolicyGate().decide(_bundle(kg_evidence=None))

    assert decision.status == DiagnosisReleaseStatus.CONDITIONAL_DIAGNOSIS
    assert decision.blocking_reasons == ["MISSING_KG_EVIDENCE"]


def test_policy_gate_requested_unsupported_axis_returns_unknown() -> None:
    bundle = _bundle(
        capability_axes=CapabilityAxesEvidence(
            supported_axes=["seed"],
            unsupported_axes=["severity", "noise"],
            requested_axes=["seed", "severity"],
            capability_manifest_id="SIM-CAP-V1",
            capability_manifest_sha256=HASH_A,
        )
    )
    decision = DiagnosisPolicyGate().decide(bundle)

    assert decision.status == DiagnosisReleaseStatus.UNKNOWN
    assert "REQUESTED_UNSUPPORTED_CAPABILITY_AXIS" in decision.blocking_reasons
    assert "unsupported_axes=severity" in decision.warnings


def test_policy_gate_blocks_when_upstream_formal_release_is_false() -> None:
    decision = DiagnosisPolicyGate().decide(_bundle(upstream_formal_release_allowed=False))

    assert decision.status == DiagnosisReleaseStatus.CONDITIONAL_DIAGNOSIS
    assert decision.blocking_reasons == ["UPSTREAM_FORMAL_RELEASE_NOT_ALLOWED"]


def test_policy_gate_semantic_qualification_failure_returns_unknown() -> None:
    decision = DiagnosisPolicyGate().decide(
        _bundle(),
        semantic_gate_blockers=["DOR_SEMANTIC_AUDIT_FAULT_MISMATCH:governor=F-A,dor=F-B"],
    )

    assert decision.status == DiagnosisReleaseStatus.UNKNOWN
    assert decision.blocking_reasons == ["SEMANTIC_QUALIFICATION_NOT_SUPPORTED"]
    assert "EVIDENCE_CONFLICT_DETECTED" not in decision.blocking_reasons


def test_policy_gate_calibration_qualification_failure_returns_unknown() -> None:
    decision = DiagnosisPolicyGate().decide(
        _bundle(),
        calibration_gate_blockers=["FEC_MODEL_EVIDENCE_UNSUPPORTED"],
    )

    assert decision.status == DiagnosisReleaseStatus.UNKNOWN
    assert decision.blocking_reasons == ["MODEL_EVIDENCE_CALIBRATION_NOT_SUPPORTED"]
    assert "FEC_MODEL_EVIDENCE_UNSUPPORTED" in decision.warnings


def test_policy_gate_mechanism_scope_failure_returns_unknown() -> None:
    decision = DiagnosisPolicyGate().decide(
        _bundle(),
        mechanism_gate_blockers=["ISGN_MECHANISM_SCOPE_UNSUPPORTED"],
    )

    assert decision.status == DiagnosisReleaseStatus.UNKNOWN
    assert decision.blocking_reasons == ["MECHANISM_SCOPE_QUALIFICATION_NOT_SUPPORTED"]


def test_kg_evidence_rejects_invalid_query_hash() -> None:
    with pytest.raises(ValidationError, match="query_sha256_values"):
        KGEvidenceReference(
            release_id="KG-GOLD-ONLINE-V058",
            knowledge_version="astrograph-kg-gold-online-v058",
            nodes_sha256=HASH_A,
            edges_sha256=HASH_B,
            query_ids=["GQ-FORWARD"],
            query_sha256_values=["not-a-hash"],
            read_user="astrograph_reader",
            row_count=1,
        )
