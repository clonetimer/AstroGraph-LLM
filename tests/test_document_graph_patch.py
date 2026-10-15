import json
import sys
from pathlib import Path

import pytest

from astrograph.cli import main
from astrograph.knowledge.document_graph_patch import (
    build_document_graph_patch_proposal,
    compile_document_graph_patch_candidate,
    replay_document_graph_patch_topology,
)
from astrograph.knowledge.schema_v3 import (
    GraphNodeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    LifecycleStatus,
)


def _canonical_sha(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _base(path: Path) -> None:
    graph = KnowledgeGraphV3(
        graph_id="graph:test:base",
        nodes=[
            GraphNodeV3(
                node_id="component:wheel",
                node_type=KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,
                source_id="source:base",
                source_sha256="a" * 64,
                lifecycle_status=LifecycleStatus.ACTIVE,
            )
        ],
        edges=[],
    )
    path.write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _freeze(path: Path, *, ready: bool = True) -> None:
    node = {
        "node_id": "candidate:mechanism:jam",
        "node_type": "FailureMechanism",
        "canonical_name": "wheel jam",
        "description": "The reviewed source defines a wheel jam mechanism.",
        "aliases": [],
        "properties": {"source_document_id": "document:test", "source_document_sha256": "b" * 64},
        "source_fragment_ids": ["fragment:test:1"],
        "confidence": 0.9,
        "ambiguous": False,
    }
    relation = {
        "edge_id": "candidate-edge:wheel-jam",
        "source_id": "component:wheel",
        "relation_type": "SUSCEPTIBLE_TO",
        "target_id": "candidate:mechanism:jam",
        "predicate_definition": "The component may exhibit this mechanism.",
        "rationale": "Reviewed design-source association.",
        "properties": {"source_document_id": "document:test", "source_document_sha256": "b" * 64},
        "source_fragment_ids": ["fragment:test:1"],
        "confidence": 0.9,
        "ambiguous": False,
    }
    approved = [
        {
            "review_item_id": "review:node",
            "item_kind": "NODE",
            "approved_canonical_id": "mechanism:wheel-jam",
            "reviewer": "domain-reviewer",
            "review_comment": "checked",
            "bundle_sha256": "c" * 64,
            "source_bundle_path": "/machine/local/bundle.json",
            "candidate": node,
        },
        {
            "review_item_id": "review:edge",
            "item_kind": "RELATION",
            "approved_canonical_id": None,
            "reviewer": "domain-reviewer",
            "review_comment": "checked",
            "bundle_sha256": "c" * 64,
            "source_bundle_path": "/machine/local/bundle.json",
            "candidate": relation,
        },
    ]
    body = {
        "schema_version": "astrograph-document-graph-review-freeze.v1",
        "status": "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL" if ready else "BLOCKED",
        "review_sheet_sha256": "d" * 64,
        "review_item_count": 2,
        "decision_counts": {"APPROVE": 2, "DEFER": 0, "REJECT": 0},
        "approved_items": approved if ready else [],
        "blockers": [] if ready else [{"code": "MISSING_OR_INVALID_DECISION", "review_item_id": "*"}],
        "knowledge_patch_proposal_allowed": ready,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": "fixture",
    }
    payload = {**body, "freeze_sha256": _canonical_sha(body)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ready_review_becomes_type_preserving_nonpublishing_v3_patch(tmp_path: Path) -> None:
    base, freeze, output = tmp_path / "base.json", tmp_path / "freeze.json", tmp_path / "patch.json"
    _base(base)
    _freeze(freeze)
    result = build_document_graph_patch_proposal(
        review_freeze_path=freeze, base_graph_path=base, output_path=output
    )
    assert result["status"] == "READY_FOR_IMPACT_REPLAY"
    assert [item["operation_kind"] for item in result["operations"]] == [
        "ADD_NODE", "ADD_EDGE"
    ]
    edge = result["operations"][1]
    assert edge["source_id"] == "component:wheel"
    assert edge["target_id"] == "mechanism:wheel-jam"
    assert result["legacy_fault_anomaly_edge_patch_compatible"] is False
    assert result["impact_replay_required"] is True
    assert result["publication_allowed"] is False
    assert result["active_graph_modified"] is False

    compiled = compile_document_graph_patch_candidate(
        patch_proposal_path=output,
        base_graph_path=base,
        output_dir=tmp_path / "compiled",
    )
    assert compiled["status"] == "READY_FOR_IMPACT_REPLAY"
    assert compiled["candidate_node_count"] == 2
    assert compiled["candidate_edge_count"] == 1
    assert compiled["shape_status"] == "PASS"
    assert compiled["active_graph_modified"] is False

    replay = replay_document_graph_patch_topology(
        base_graph_path=base,
        candidate_graph_path=tmp_path / "compiled/candidate_graph.json",
        seed_node_ids=["component:wheel"],
        target_node_types=["FailureMechanism"],
        expected_added_path_signatures=[
            "component:wheel|SUSCEPTIBLE_TO|mechanism:wheel-jam"
        ],
        output_path=tmp_path / "compiled/topology_replay.json",
    )
    assert replay["status"] == "PASS"
    assert replay["actual_added_path_signatures"] == [
        "component:wheel|SUSCEPTIBLE_TO|mechanism:wheel-jam"
    ]
    assert replay["publication_allowed"] is False


def test_topology_replay_blocks_undeclared_path_change(tmp_path: Path) -> None:
    base, freeze, proposal = tmp_path / "base.json", tmp_path / "freeze.json", tmp_path / "proposal.json"
    _base(base)
    _freeze(freeze)
    build_document_graph_patch_proposal(
        review_freeze_path=freeze, base_graph_path=base, output_path=proposal
    )
    compile_document_graph_patch_candidate(
        patch_proposal_path=proposal,
        base_graph_path=base,
        output_dir=tmp_path / "compiled",
    )
    replay = replay_document_graph_patch_topology(
        base_graph_path=base,
        candidate_graph_path=tmp_path / "compiled/candidate_graph.json",
        seed_node_ids=["component:wheel"],
        target_node_types=["FailureMechanism"],
        expected_added_path_signatures=[],
        output_path=tmp_path / "blocked_replay.json",
    )
    assert replay["status"] == "BLOCKED"
    assert replay["unexpected_added_path_signatures"] == [
        "component:wheel|SUSCEPTIBLE_TO|mechanism:wheel-jam"
    ]


def test_blocked_review_produces_blocked_empty_patch_receipt(tmp_path: Path) -> None:
    base, freeze = tmp_path / "base.json", tmp_path / "freeze.json"
    _base(base)
    _freeze(freeze, ready=False)
    result = build_document_graph_patch_proposal(
        review_freeze_path=freeze,
        base_graph_path=base,
        output_path=tmp_path / "patch.json",
    )
    assert result["status"] == "BLOCKED"
    assert result["operations"] == []
    assert {item["code"] for item in result["blockers"]} == {
        "DOCUMENT_REVIEW_FREEZE_NOT_READY",
        "DOCUMENT_REVIEW_PATCH_PERMISSION_MISSING",
    }


def test_tampered_review_freeze_is_rejected(tmp_path: Path) -> None:
    base, freeze = tmp_path / "base.json", tmp_path / "freeze.json"
    _base(base)
    _freeze(freeze)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["approved_items"][0]["reviewer"] = "spoofed"
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze_sha256 mismatch"):
        build_document_graph_patch_proposal(
            review_freeze_path=freeze,
            base_graph_path=base,
            output_path=tmp_path / "patch.json",
        )


def test_document_graph_patch_cli_runs_ready_compile_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base, freeze = tmp_path / "base.json", tmp_path / "freeze.json"
    proposal, compiled = tmp_path / "proposal.json", tmp_path / "compiled"
    replay = tmp_path / "replay.json"
    _base(base)
    _freeze(freeze)

    monkeypatch.setattr(sys, "argv", [
        "astrograph", "propose-document-graph-patch",
        "--review-freeze", str(freeze), "--base-graph", str(base),
        "--output", str(proposal),
    ])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "READY_FOR_IMPACT_REPLAY"

    monkeypatch.setattr(sys, "argv", [
        "astrograph", "compile-document-graph-patch",
        "--patch-proposal", str(proposal), "--base-graph", str(base),
        "--output", str(compiled),
    ])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["shape_status"] == "PASS"

    monkeypatch.setattr(sys, "argv", [
        "astrograph", "replay-document-graph-patch",
        "--base-graph", str(base),
        "--candidate-graph", str(compiled / "candidate_graph.json"),
        "--seed-node-id", "component:wheel",
        "--target-node-type", "FailureMechanism",
        "--expected-added-path-signature",
        "component:wheel|SUSCEPTIBLE_TO|mechanism:wheel-jam",
        "--output", str(replay),
    ])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"

    monkeypatch.setattr(sys, "argv", [
        "astrograph", "compile-document-graph-patch",
        "--patch-proposal", str(proposal), "--base-graph", str(base),
        "--output", str(compiled),
    ])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        main()
    monkeypatch.setattr(sys, "argv", [
        "astrograph", "replay-document-graph-patch",
        "--base-graph", str(base),
        "--candidate-graph", str(compiled / "candidate_graph.json"),
        "--seed-node-id", "component:wheel",
        "--target-node-type", "FailureMechanism",
        "--output", str(replay),
    ])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        main()


def test_document_graph_patch_cli_preserves_blocked_review_as_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base, freeze = tmp_path / "base.json", tmp_path / "freeze.json"
    proposal = tmp_path / "blocked-proposal.json"
    _base(base)
    _freeze(freeze, ready=False)
    monkeypatch.setattr(sys, "argv", [
        "astrograph", "propose-document-graph-patch",
        "--review-freeze", str(freeze), "--base-graph", str(base),
        "--output", str(proposal),
    ])
    assert main() == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "BLOCKED"
    assert printed["operations"] == []
    assert json.loads(proposal.read_text(encoding="utf-8")) == printed

    monkeypatch.setattr(sys, "argv", [
        "astrograph", "propose-document-graph-patch",
        "--review-freeze", str(freeze), "--base-graph", str(base),
        "--output", str(proposal),
    ])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        main()
    assert not list(tmp_path.glob(".blocked-proposal.json.*.tmp"))
