from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from astrograph.knowledge import (
    AstroGraphKnowledgePatchReplayContract,
    AstroGraphKnowledgePatchReplayOutput,
    JsonKnowledgeGovernanceStore,
    KnowledgeChangeStatus,
    KnowledgeEdgeValue,
    KnowledgePatchOperation,
    KnowledgePatchOperationKind,
    KnowledgePatchSourceSplit,
    KnowledgeRelease,
    build_astrograph_knowledge_patch_replay_output,
    build_knowledge_patch_candidate,
    compile_knowledge_patch_release,
    experience_candidate_is_compilable,
    run_knowledge_patch_impact_replay,
    run_astrograph_knowledge_patch_impact_replay,
    sha256_file,
    validate_knowledge_patch,
)
from astrograph.diagnosis import build_structured_diagnosis_output


EDGE_FIELDS = [
    "fault_id", "fault_name", "anomaly_id", "anomaly_name", "weight", "edge_role",
    "evidence_level", "evidence_source", "source_ref", "review_status", "provisional",
    "knowledge_version", "applicable_condition_ids", "excluded_condition_ids",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _base_release(root: Path) -> KnowledgeRelease:
    _write_csv(root / "seed/fault_modes.csv", ["fault_id", "name"], [
        {"fault_id": "FAULT-1", "name": "Fault one"},
    ])
    _write_csv(root / "seed/anomalies.csv", ["anomaly_id", "name"], [
        {"anomaly_id": "ANOM-1", "name": "Anomaly one"},
        {"anomaly_id": "ANOM-2", "name": "Anomaly two"},
    ])
    _write_csv(root / "seed/causes.csv", EDGE_FIELDS, [{
        "fault_id": "FAULT-1", "fault_name": "Fault one",
        "anomaly_id": "ANOM-1", "anomaly_name": "Anomaly one",
        "weight": "1.0", "edge_role": "direct", "evidence_level": "E0",
        "evidence_source": "SOURCE-BASE", "source_ref": "base-ref",
        "review_status": "reviewed", "provisional": "false",
        "knowledge_version": "kv-base", "applicable_condition_ids": "",
        "excluded_condition_ids": "",
    }])
    return KnowledgeRelease(
        release_id="KR-BASE", knowledge_version="kv-base", ontology_version="ont-v1",
        release_status="engineering_released", edges_file="seed/causes.csv",
        edges_sha256=sha256_file(root / "seed/causes.csv"),
        fault_modes_file="seed/fault_modes.csv", anomalies_file="seed/anomalies.csv",
        evidence_levels=["E0"], review_statuses=["reviewed"], edge_count=1,
        claim_boundary="test base",
    )


def _candidate(base: KnowledgeRelease, *, split: KnowledgePatchSourceSplit = KnowledgePatchSourceSplit.DEVELOPMENT):
    edge = KnowledgeEdgeValue(
        fault_id="FAULT-1", fault_name="Fault one",
        anomaly_id="ANOM-2", anomaly_name="Anomaly two",
        weight=0.7, edge_role="indirect", evidence_level="E0",
        evidence_source="EXP-REVIEWED", source_ref="experience:EXP-1",
        review_status="provisional", provisional=True,
    )
    operation = KnowledgePatchOperation(
        operation_id="OP-ADD-FAULT1-ANOM2", kind=KnowledgePatchOperationKind.ADD,
        fault_id="FAULT-1", anomaly_id="ANOM-2", edge=edge,
        reason="reviewed development experience proposes an additional observable",
    )
    return build_knowledge_patch_candidate(
        base_release=base, proposed_release_id="KR-CANDIDATE",
        proposed_knowledge_version="kv-candidate", operations=[operation],
        source_candidate_ids=["EXPKG-1"], source_refs=["experience:EXP-1"],
        provenance_independence_groups=["GROUP-1"], source_split=split,
        created_by="human-curator",
    )


def test_patch_compiles_to_a_valid_non_active_release(tmp_path: Path) -> None:
    base = _base_release(tmp_path)
    candidate = _candidate(base)
    validation = validate_knowledge_patch(candidate, base_release=base, project_root=tmp_path)
    assert validation.status == "READY_FOR_SANDBOX"
    compiled = compile_knowledge_patch_release(
        candidate, base_release=base, project_root=tmp_path, output_dir="state/candidate"
    )
    assert compiled.release.edge_count == 2
    assert compiled.active_graph_modified is False
    assert (tmp_path / compiled.release_path).is_file()
    assert not (tmp_path / "active.json").exists()


def test_test_split_unknown_entity_and_existing_add_are_blocked(tmp_path: Path) -> None:
    base = _base_release(tmp_path)
    test_candidate = _candidate(base, split=KnowledgePatchSourceSplit.TEST)
    report = validate_knowledge_patch(test_candidate, base_release=base, project_root=tmp_path)
    assert report.status == "BLOCKED"
    assert any(item.check_id == "test_split_not_promotion_basis" and item.status == "FAIL" for item in report.checks)

    edge = KnowledgeEdgeValue(
        fault_id="FAULT-X", fault_name="Unknown", anomaly_id="ANOM-1",
        anomaly_name="Anomaly one", weight=1.0, edge_role="direct", evidence_level="E0",
        evidence_source="S", source_ref="R", review_status="provisional",
    )
    operation = KnowledgePatchOperation(
        operation_id="OP-X", kind="ADD", fault_id="FAULT-X", anomaly_id="ANOM-1",
        edge=edge, reason="invalid unknown entity",
    )
    unknown = build_knowledge_patch_candidate(
        base_release=base, proposed_release_id="KR-X", proposed_knowledge_version="kv-x",
        operations=[operation], source_candidate_ids=["C"], source_refs=["R"],
        provenance_independence_groups=["G"], source_split="DEVELOPMENT", created_by="H",
    )
    assert validate_knowledge_patch(unknown, base_release=base, project_root=tmp_path).status == "BLOCKED"


def test_sandbox_impact_blocks_undeclared_changes(tmp_path: Path) -> None:
    base = _base_release(tmp_path)
    compiled = compile_knowledge_patch_release(
        _candidate(base), base_release=base, project_root=tmp_path,
        output_dir="state/candidate",
    )

    def runner(release: KnowledgeRelease) -> dict:
        return {
            "kg_forward": {"edge_count": release.edge_count},
            "kg_backward": {"top_fault": "FAULT-1"},
            "birrf": {"top_fault": "FAULT-1"},
            "decision": "ACCEPT",
        }

    passed = run_knowledge_patch_impact_replay(
        compiled, base_release=base, replay_runner=runner,
        declared_expected_change_paths=["$.kg_forward.edge_count"],
    )
    assert passed.status == "PASS"
    assert passed.changed_paths == ["$.kg_forward.edge_count"]
    assert passed.active_graph_modified is False

    blocked = run_knowledge_patch_impact_replay(
        compiled, base_release=base, replay_runner=runner,
        declared_expected_change_paths=[],
    )
    assert blocked.status == "BLOCKED"
    assert blocked.unexpected_change_paths == ["$.kg_forward.edge_count"]


def test_official_astrograph_replay_contract_fixes_scenarios_and_dimensions(tmp_path: Path) -> None:
    base = _base_release(tmp_path)
    compiled = compile_knowledge_patch_release(
        _candidate(base), base_release=base, project_root=tmp_path,
        output_dir="state/candidate",
    )
    provisional = AstroGraphKnowledgePatchReplayContract(
        contract_id="REPLAY-1", scenario_set_id="SCENARIOS-1",
        scenario_set_sha256="a" * 64, required_episode_ids=["EP-1"],
        allowed_change_dimensions=["kg_forward_fault_ids"],
    )
    contract = provisional.model_copy(
        update={"contract_sha256": provisional.compute_sha256()}
    )

    def runner(release: KnowledgeRelease) -> AstroGraphKnowledgePatchReplayOutput:
        forward = ["FAULT-1"] if release.release_id == "KR-BASE" else ["FAULT-1", "FAULT-2"]
        return AstroGraphKnowledgePatchReplayOutput(
            scenario_set_id="SCENARIOS-1", scenario_set_sha256="a" * 64,
            episodes=[{
                "episode_id": "EP-1", "kg_forward_fault_ids": forward,
                "kg_backward_fault_ids": ["FAULT-1"], "birrf_top_fault_id": "FAULT-1",
                "governor_disposition": "ACCEPT", "policy_status": "WITHHELD",
                "structured_output_sha256": "b" * 64,
            }],
        )

    passed = run_astrograph_knowledge_patch_impact_replay(
        compiled, base_release=base, replay_runner=runner, contract=contract,
    )
    assert passed.status == "PASS"
    assert passed.changed_paths == ["episode:EP-1.kg_forward_fault_ids"]

    strict_provisional = contract.model_copy(update={
        "allowed_change_dimensions": [], "contract_sha256": None,
    })
    strict = strict_provisional.model_copy(
        update={"contract_sha256": strict_provisional.compute_sha256()}
    )
    blocked = run_astrograph_knowledge_patch_impact_replay(
        compiled, base_release=base, replay_runner=runner, contract=strict,
    )
    assert blocked.status == "BLOCKED"
    assert blocked.unexpected_change_paths == ["episode:EP-1.kg_forward_fault_ids"]

    with pytest.raises(ValueError, match="scenario identity"):
        run_astrograph_knowledge_patch_impact_replay(
            compiled, base_release=base,
            replay_runner=lambda release: runner(release).model_copy(
                update={"scenario_set_id": "WRONG"}
            ),
            contract=contract,
        )


def test_replay_adapter_extracts_typed_real_output_and_rejects_tampering() -> None:
    def ranking(source: str, ranking_id: str, fault_ids: list[str]) -> dict:
        return {
            "ranking_id": ranking_id, "episode_id": "EP-1", "source": source,
            "producer_id": "test", "producer_version": "1",
            "candidates": [
                {"fault_id": fault_id, "rank": index, "score": 1.0 / index,
                 "source": source}
                for index, fault_id in enumerate(fault_ids, start=1)
            ],
        }

    fusion = {
        "episode_id": "EP-1",
        "data_ranking": ranking("DATA_DRIVEN", "R-DATA", ["F-1", "F-2"]),
        "forward_ranking": ranking("KG_FORWARD", "R-FWD", ["F-2", "F-1"]),
        "backward_ranking": ranking("KG_BACKWARD", "R-BWD", ["F-1"]),
        "birrf_result": {
            "fusion_id": "FUSION-1", "episode_id": "EP-1", "config": {},
            "candidates": [], "source_ranking_ids": {
                "DATA_DRIVEN": "R-DATA", "KG_FORWARD": "R-FWD",
                "KG_BACKWARD": "R-BWD",
            },
            "top_fault_id": "F-2",
        },
        "registered_fault_ids": ["F-1", "F-2"],
    }
    payload = {
        "simulation_event": {"run_id": "RUN-1"},
        "data_gate_report": {"passed": True},
        "episode_evidence_bundle": {
            "episode": {"episode_id": "EP-1"}, "evidence_facts": [],
        },
        "candidate_fusion": fusion,
        "hypothesis_set": {"hypotheses": []},
        "decision_record": {"episode_id": "EP-1", "disposition": "ABSTAIN"},
        "diagnosis_policy_decision": {
            "status": "WITHHELD", "blocking_reasons": [], "warnings": [],
        },
        "verification_plan": {"tasks": []},
    }
    payload["structured_diagnosis_output"] = build_structured_diagnosis_output(
        payload
    ).model_dump(mode="json")

    replay = build_astrograph_knowledge_patch_replay_output(
        scenario_set_id="SCENARIOS-1", scenario_set_sha256="a" * 64,
        episode_payloads=[payload],
    )
    assert replay.episodes[0].kg_forward_fault_ids == ["F-2", "F-1"]
    assert replay.episodes[0].kg_backward_fault_ids == ["F-1"]
    assert replay.episodes[0].birrf_top_fault_id == "F-2"
    assert replay.episodes[0].governor_disposition == "ABSTAIN"
    assert replay.episodes[0].structured_output_sha256 == payload[
        "structured_diagnosis_output"
    ]["output_sha256"]

    tampered = json.loads(json.dumps(payload))
    tampered["structured_diagnosis_output"]["final_decision"]["disposition"] = "ACCEPTED"
    with pytest.raises(ValueError, match="output_sha256 mismatch"):
        build_astrograph_knowledge_patch_replay_output(
            scenario_set_id="SCENARIOS-1", scenario_set_sha256="a" * 64,
            episode_payloads=[tampered],
        )


def test_patch_must_pass_preflight_and_sandbox_before_governed_publish_and_rollback(tmp_path: Path) -> None:
    base = _base_release(tmp_path)
    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps(base.model_dump(mode="json")), encoding="utf-8")
    candidate = _candidate(base)
    compiled = compile_knowledge_patch_release(
        candidate, base_release=base, project_root=tmp_path, output_dir="state/candidate"
    )
    impact = run_knowledge_patch_impact_replay(
        compiled, base_release=base,
        replay_runner=lambda release: {"edge_count": release.edge_count},
        declared_expected_change_paths=["$.edge_count"],
    )
    store = JsonKnowledgeGovernanceStore(
        tmp_path / "governance.json", project_root=tmp_path,
        active_release_path=active_path, history_dir=tmp_path / "history",
    )
    draft = store.create_draft(
        title="reviewed patch", description="sandboxed", candidate_release_path=compiled.release_path,
        created_by="curator", patch_validation=compiled.validation.model_dump(mode="json"),
        impact_report=impact.model_dump(mode="json"),
    )
    assert draft.status is KnowledgeChangeStatus.DRAFT
    assert draft.validation["sandbox_impact"]["status"] == "PASS"
    submitted = store.submit(draft.change_id, actor="curator")
    assert submitted.status is KnowledgeChangeStatus.SUBMITTED
    assert "patch_preflight" in submitted.validation
    approved = store.review(draft.change_id, reviewer="independent-reviewer", approve=True, comment="approved")
    assert approved.status is KnowledgeChangeStatus.APPROVED
    published = store.publish(draft.change_id, actor="release-manager")
    assert published.status is KnowledgeChangeStatus.PUBLISHED
    assert KnowledgeRelease.from_json(active_path).release_id == "KR-CANDIDATE"
    rolled = store.rollback(draft.change_id, actor="release-manager", comment="drill rollback")
    assert rolled.status is KnowledgeChangeStatus.ROLLED_BACK
    assert KnowledgeRelease.from_json(active_path).release_id == "KR-BASE"


def test_current_experience_candidate_requires_human_edge_mapping() -> None:
    eligible, reasons = experience_candidate_is_compilable({
        "review_status": "HUMAN_REVIEW_REQUIRED",
        "active_graph_modified": False,
        "fault_ids": ["FAULT-1"],
    })
    assert eligible is False
    assert reasons == ["HUMAN_GRAPH_EDGE_MAPPING_REQUIRED"]

    class ValidatedCandidate(BaseModel):
        review_status: str
        active_graph_modified: bool
        fault_ids: list[str]

    eligible, reasons = experience_candidate_is_compilable(ValidatedCandidate(
        review_status="HUMAN_REVIEW_REQUIRED",
        active_graph_modified=False,
        fault_ids=["FAULT-1"],
    ))
    assert eligible is False
    assert reasons == ["HUMAN_GRAPH_EDGE_MAPPING_REQUIRED"]

    assert experience_candidate_is_compilable(object()) == (
        False, ["EXPERIENCE_CANDIDATE_PAYLOAD_INVALID"]
    )


def test_governance_rejects_missing_or_failed_patch_impact(tmp_path: Path) -> None:
    base = _base_release(tmp_path)
    (tmp_path / "active.json").write_text(json.dumps(base.model_dump(mode="json")), encoding="utf-8")
    compiled = compile_knowledge_patch_release(
        _candidate(base), base_release=base, project_root=tmp_path, output_dir="state/candidate"
    )
    store = JsonKnowledgeGovernanceStore(
        tmp_path / "gov.json", project_root=tmp_path, active_release_path="active.json",
        history_dir="history",
    )
    with pytest.raises(ValueError, match="compiled knowledge patch release requires"):
        store.create_draft(
            title="bypass", description="", candidate_release_path=compiled.release_path,
            created_by="curator",
        )
    with pytest.raises(ValueError, match="provided together"):
        store.create_draft(
            title="bad", description="", candidate_release_path=compiled.release_path,
            created_by="curator", patch_validation=compiled.validation.model_dump(mode="json"),
        )

    impact = run_knowledge_patch_impact_replay(
        compiled, base_release=base,
        replay_runner=lambda release: {"edge_count": release.edge_count},
        declared_expected_change_paths=["$.edge_count"],
    )
    mismatched = compiled.validation.model_dump(mode="json")
    mismatched["candidate_id"] = "KPATCH-DIFFERENT"
    with pytest.raises(ValueError, match="candidate IDs differ"):
        store.create_draft(
            title="mismatch", description="", candidate_release_path=compiled.release_path,
            created_by="curator", patch_validation=mismatched,
            impact_report=impact.model_dump(mode="json"),
        )
