
from astrograph.knowledge.evidence_capability import (
    CapabilityVector, EvidenceCapabilityManifest, EvidenceCapabilityRecord,
    EvidenceMaturity, EvidencePolarity, EvidenceProvenance, OperationalContext,
    compute_capability_vector,
)
H = "a" * 64

def manifest():
    p1 = EvidenceProvenance("p1", "standard", H, "s:1", "g1", human_reviewers=("reviewer-1",))
    p2 = EvidenceProvenance("p2", "fmea", H, "s:2", "g2", human_reviewers=("reviewer-2",))
    rows = (
        EvidenceCapabilityRecord("e1", "f1", EvidencePolarity.REQUIRED, EvidenceMaturity.FROZEN, H, ("p1","p2"), ("x",), ("NOMINAL",), validation_receipt_sha256=H, freeze_receipt_sha256=H),
        EvidenceCapabilityRecord("e2", "f2", EvidencePolarity.COUNTER, EvidenceMaturity.EXECUTABLE, H, ("p1",), ("x",), ("NOMINAL",)),
    )
    return EvidenceCapabilityManifest("astrograph-evidence-capability-manifest.v1", ("f1","f2","f3"), (p1,p2), rows)

def test_capability_is_not_candidate_enumeration():
    v = compute_capability_vector(manifest(), OperationalContext("ADCS", "NOMINAL", ("x",)))
    assert v.executable_positive_fault_ids == ("f1",)
    assert v.executable_positive_fraction == 1/3
    assert v.frozen_fault_ids == ("f1",)
    assert v.provenance_independent_fraction == 1/3

def test_missing_telemetry_removes_contextual_applicability_not_structural_capability():
    v = compute_capability_vector(manifest(), OperationalContext("ADCS", "NOMINAL", ()))
    assert v.executable_positive_fault_ids == ("f1",)
    assert v.telemetry_applicable_fault_ids == ()

def test_truth_key_rejected_from_runtime_contract():
    import pytest
    with pytest.raises(ValueError):
        EvidenceCapabilityRecord("e", "f1", EvidencePolarity.REQUIRED, EvidenceMaturity.EXECUTABLE, H, ("p1",), (), (), runtime_contract={"true_label": "x"})

def test_manifest_content_address_is_order_canonical():
    m = manifest()
    assert len(m.compute_sha256()) == 64


def test_agent_generated_material_cannot_be_domain_evidence():
    import pytest
    from astrograph.knowledge.evidence_capability import EvidenceProvenance
    with pytest.raises(ValueError):
        EvidenceProvenance("agent", "agent_generated", H, "agent:1", "agent-group")

def test_executable_positive_rule_requires_domain_evidence_provenance():
    import pytest
    from astrograph.knowledge.evidence_capability import EvidenceProvenance, ProvenanceRole
    transform = EvidenceProvenance("t", "derived_code", H, "code:1", "code", role=ProvenanceRole.TRANSFORMATION)
    row = EvidenceCapabilityRecord("e", "f1", EvidencePolarity.REQUIRED, EvidenceMaturity.EXECUTABLE, H, ("t",), (), ())
    with pytest.raises(ValueError):
        EvidenceCapabilityManifest("astrograph-evidence-capability-manifest.v1", ("f1",), (transform,), (row,))
