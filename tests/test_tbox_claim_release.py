import json
from pathlib import Path

import pytest

from astrograph.knowledge.tbox_claim_release import build_approved_tbox_claim_release


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/candidate_graph.json"
BUNDLE = ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/construction_bundle.json"
APPROVAL = ROOT / "governance/evidence/ai_native/tbox_vocabulary_approval_20260902/approval.json"


def test_owner_approved_tbox_claims_are_source_bound_and_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = build_approved_tbox_claim_release(
        graph_path=GRAPH,
        construction_bundle_path=BUNDLE,
        approval_path=APPROVAL,
        output_path=first_path,
    )
    second = build_approved_tbox_claim_release(
        graph_path=GRAPH,
        construction_bundle_path=BUNDLE,
        approval_path=APPROVAL,
        output_path=second_path,
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["status"] == "APPROVED_TBOX_CLAIM_SET"
    assert first["claim_count"] == first["source_binding_count"] == 22
    assert {claim["epistemic_status"] for claim in first["claims"]} == {
        "ONTOLOGY_ASSOCIATION"
    }
    assert {claim["predicate"] for claim in first["claims"]} == {"MANIFESTS_AS"}
    assert all(claim["publication_eligible"] for claim in first["claims"])
    assert not any(claim["runtime_evidence_eligible"] for claim in first["claims"])
    assert not first["active_graph_modified"]


def test_tbox_claim_compiler_rejects_graph_not_bound_by_approval(tmp_path: Path) -> None:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    graph["graph_id"] += ":tampered"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(ValueError, match="approval does not bind"):
        build_approved_tbox_claim_release(
            graph_path=tampered,
            construction_bundle_path=BUNDLE,
            approval_path=APPROVAL,
            output_path=tmp_path / "claims.json",
        )
    assert not (tmp_path / "claims.json").exists()

