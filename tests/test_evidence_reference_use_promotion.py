from astrograph.evidence.reference_use import EvidenceReferenceUseClass, evidence_reference_use
from astrograph.evidence.trusted_reference import ReferenceQualification, ReferenceUseDecision


def _q(decision, formal):
    return ReferenceQualification(reference_id="ref-1", decision=decision, reasons=("reason",), formal_calibration_eligible=formal)


def test_qualified_reference_allows_formal_support():
    use = evidence_reference_use(_q(ReferenceUseDecision.QUALIFIED, True), scope_id="signal:S1")
    assert use.use_class is EvidenceReferenceUseClass.FORMAL_SUPPORT
    assert use.candidate_generation_allowed is True
    assert use.formal_support_allowed is True


def test_research_only_reference_is_not_hard_filtered_from_candidate_generation():
    use = evidence_reference_use(_q(ReferenceUseDecision.RESEARCH_ONLY, False), scope_id="signal:S1")
    assert use.use_class is EvidenceReferenceUseClass.RESEARCH_ONLY
    assert use.candidate_generation_allowed is True
    assert use.formal_support_allowed is False


def test_rejected_reference_is_not_admitted():
    use = evidence_reference_use(_q(ReferenceUseDecision.REJECTED, False), scope_id="signal:S1")
    assert use.use_class is EvidenceReferenceUseClass.REJECTED
    assert use.candidate_generation_allowed is False
    assert use.formal_support_allowed is False
