from astrograph.evidence.trusted_reference import (
    CalibrationTransferState,
    ReferenceSelectionMode,
    ReferenceTrustState,
    ReferenceUseDecision,
    TrustedReferenceRecord,
    qualify_reference,
)

SHA = "a" * 64


def _record(*, trust=ReferenceTrustState.VERIFIED_NOMINAL, transfer=CalibrationTransferState.NOT_ASSESSED, overlap=0.0):
    return TrustedReferenceRecord(
        reference_id="ref", domain_id="domain", selection_mode=(
            ReferenceSelectionMode.VERIFIED_NOMINAL
            if trust is ReferenceTrustState.VERIFIED_NOMINAL
            else ReferenceSelectionMode.PREFIX_ASSUMED_NOMINAL
        ),
        trust_state=trust, sample_count=2048, start_time="0", end_time="1",
        provenance_sha256=SHA, label_overlap_fraction=overlap, transfer_state=transfer,
    )


def test_verified_nominal_requires_transfer_when_requested():
    q = qualify_reference(_record(transfer=CalibrationTransferState.PASSED), require_transfer_pass=True)
    assert q.decision is ReferenceUseDecision.QUALIFIED
    assert q.formal_calibration_eligible is True


def test_assumed_reference_cannot_become_formal_from_sample_count_alone():
    q = qualify_reference(_record(trust=ReferenceTrustState.ASSUMED_UNVERIFIED, overlap=None))
    assert q.decision is ReferenceUseDecision.RESEARCH_ONLY
    assert q.formal_calibration_eligible is False


def test_contaminated_reference_is_rejected():
    q = qualify_reference(_record(trust=ReferenceTrustState.CONTAMINATED, overlap=0.01))
    assert q.decision is ReferenceUseDecision.REJECTED
    assert q.formal_calibration_eligible is False
