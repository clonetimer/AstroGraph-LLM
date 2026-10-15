
import pytest
from astrograph.assurance.evidence_authoring import AuthoringStage, EvidenceAuthoringReceipt, promote
H = "c"*64

def test_agent_receipt_does_not_replace_source_or_human_approval():
    r = EvidenceAuthoringReceipt("astrograph-evidence-authoring-receipt.v1", "e", "f", AuthoringStage.DRAFT, agent_transformation_receipt_sha256=H)
    with pytest.raises(ValueError):
        promote(r, AuthoringStage.SOURCE_PINNED)

def test_linear_promotion_and_freeze_requirements():
    r = EvidenceAuthoringReceipt("astrograph-evidence-authoring-receipt.v1", "e", "f", AuthoringStage.DRAFT)
    r = promote(r, AuthoringStage.SOURCE_PINNED, source_artifact_sha256s=(H,), source_independence_groups=("g",))
    r = promote(r, AuthoringStage.EXECUTABLE, executable_rule_sha256=H)
    r = promote(r, AuthoringStage.UNIT_VALIDATED, unit_test_receipt_sha256=H)
    r = promote(r, AuthoringStage.DEVELOPMENT_VALIDATED, development_validation_receipt_sha256=H)
    r = promote(r, AuthoringStage.FROZEN, human_approval_receipt_sha256=H)
    assert r.stage is AuthoringStage.FROZEN
    assert r.authoring_receipt_sha256 == r.compute_sha256()
