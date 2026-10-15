
from astrograph.knowledge.evidence_interventions import EvidenceBlock, build_nested_schedule, validate_nested_schedule
H = "b"*64

def test_schedule_is_deterministic_nested_and_outcome_blind():
    blocks = [
        EvidenceBlock("b1", ("f1",), ("e1",), "g1"),
        EvidenceBlock("b2", ("f2",), ("e2",), "g2"),
        EvidenceBlock("b3", ("f3",), ("e3",), "g3"),
    ]
    a = build_nested_schedule(("f1","f2","f3"), blocks, (0,.25,.5,1), replicate_id=0, design_seed_sha256=H)
    b = build_nested_schedule(("f1","f2","f3"), blocks, (0,.25,.5,1), replicate_id=0, design_seed_sha256=H)
    assert a == b
    validate_nested_schedule(a)
    assert a.points[-1].achieved_fault_fraction == 1

def test_shared_provenance_is_not_split():
    blocks = [EvidenceBlock("b1", ("f1",), ("e1",), "g"), EvidenceBlock("b2", ("f2",), ("e2",), "g")]
    s = build_nested_schedule(("f1","f2"), blocks, (.5,), replicate_id=0, design_seed_sha256=H)
    assert set(s.points[0].selected_block_ids) == {"b1","b2"}
