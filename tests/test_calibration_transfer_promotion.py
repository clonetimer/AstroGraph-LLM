from astrograph.evidence.calibration_transfer import (
    CalibrationAuditState,
    audit_nominal_calibration_transfer,
    qualify_reference_with_transfer_audit,
)
from astrograph.evidence.trusted_reference import (
    ReferenceSelectionMode,
    ReferenceTrustState,
    ReferenceUseDecision,
    TrustedReferenceRecord,
)


def _reference(trust=ReferenceTrustState.VERIFIED_NOMINAL):
    return TrustedReferenceRecord(
        reference_id="ref", domain_id="d", selection_mode=(
            ReferenceSelectionMode.VERIFIED_NOMINAL
            if trust is ReferenceTrustState.VERIFIED_NOMINAL
            else ReferenceSelectionMode.PREFIX_ASSUMED_NOMINAL
        ),
        trust_state=trust, sample_count=2048, start_time="0", end_time="1",
        provenance_sha256="b" * 64,
        label_overlap_fraction=0.0 if trust is ReferenceTrustState.VERIFIED_NOMINAL else None,
    )


def test_independent_nominal_audit_passes_without_alarm_episodes():
    audit = audit_nominal_calibration_transfer(
        [0.1] * 512, reference_id="ref", audit_id="audit",
        maximum_active_sample_fraction=0.01,
    )
    assert audit.state is CalibrationAuditState.PASSED
    assert audit.reference_updated_from_audit is False


def test_sustained_alarm_fraction_fails_even_if_episode_rate_alone_would_pass():
    scores = [0.1] * 100 + [1.0] * 1800 + [0.1] * 100
    audit = audit_nominal_calibration_transfer(
        scores, reference_id="ref", audit_id="audit", minimum_samples=400,
        maximum_false_alarm_episodes_per_1000=0.5,
        maximum_active_sample_fraction=0.01,
    )
    assert audit.alarm_episode_count == 1
    assert audit.false_alarm_episodes_per_1000_nominal_samples == 0.5
    assert audit.active_sample_fraction > 0.8
    assert audit.state is CalibrationAuditState.FAILED


def test_short_audit_is_insufficient():
    audit = audit_nominal_calibration_transfer([0.1] * 511, reference_id="ref", audit_id="audit")
    assert audit.state is CalibrationAuditState.INSUFFICIENT


def test_passed_transfer_does_not_upgrade_assumed_reference_to_formal():
    audit = audit_nominal_calibration_transfer([0.1] * 512, reference_id="ref", audit_id="audit")
    q = qualify_reference_with_transfer_audit(_reference(ReferenceTrustState.ASSUMED_UNVERIFIED), audit)
    assert q.decision is ReferenceUseDecision.RESEARCH_ONLY
    assert q.formal_calibration_eligible is False
