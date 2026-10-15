from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.knowledge import (
    KnowledgeMappingCandidateSet,
    ReplayScenarioSetCandidate,
    SourceAdjudicationPackage,
    validate_dataset_kg_review_package,
)
from astrograph.knowledge.coverage_review import (
    build_source_adjudication_package,
    extract_adcs_generator_conditions,
    extract_phm_generator_conditions,
    freeze_supplemental_human_review,
    refreeze_supplemental_human_review,
    unfreeze_supplemental_human_review,
    validate_source_adjudication_package,
    validate_supplemental_review_freeze,
    validate_supplemental_review_refreeze,
    validate_supplemental_review_unfreeze,
    write_source_adjudication_review_sheet,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "governance/evidence/ai_native/kg_coverage_review_20260831"
SUPPLEMENTAL = PACKAGE / "human_review_decisions_with_supplemental_review_zh.csv"
FREEZE = PACKAGE / "supplemental_review_temporary_freeze.json"
SOURCE_ADJUDICATION = PACKAGE / "source_adjudication_20260901.json"
UNFREEZE = PACKAGE / "supplemental_review_unfreeze_20260901.json"
SOTA_AUDIT = (
    ROOT / "governance/evidence/research/data_driven_runs/"
    "domain_sota_group_cv_20260901/fairness_audit.json"
)
SOURCE_HANDOFF = PACKAGE / "human_review_decisions_source_adjudicated_20260901.csv"
REFREEZE = PACKAGE / "supplemental_review_refreeze_20260901.json"


def test_generated_review_package_closes_the_declared_31_class_inventory() -> None:
    manifest = validate_dataset_kg_review_package(PACKAGE)
    assert manifest.dataset_fault_class_count == 31
    assert manifest.semantic_tbox_coverage_count == 31
    assert manifest.formal_runtime_id_count == 19
    assert manifest.executable_kg_evidence_count == 9
    assert manifest.uncovered_candidate_count == 22
    assert manifest.replay_episode_count == 62
    assert manifest.publication_allowed is False
    assert manifest.active_graph_modified is False
    generator_groups = {
        item.independence_group
        for item in manifest.source_anchors
        if item.role in {"GENERATOR_SOURCE", "DATASET_INDEX"}
    }
    assert generator_groups == {"SAT-SIM-WHOLE-SPACECRAFT-GENERATOR"}


def test_mapping_candidates_and_replay_set_are_content_addressed() -> None:
    candidates = KnowledgeMappingCandidateSet.model_validate_json(
        (PACKAGE / "knowledge_mapping_candidates.json").read_text(encoding="utf-8")
    )
    assert candidates.candidate_count == 22
    assert sum(item.requires_new_runtime_fault_id for item in candidates.candidates) == 12
    assert all(item.human_decision is None for item in candidates.candidates)
    replay = ReplayScenarioSetCandidate.model_validate_json(
        (PACKAGE / "replay_scenario_set_candidate.json").read_text(encoding="utf-8")
    )
    assert replay.status == "CANDIDATE_NOT_FROZEN"
    assert replay.allowed_change_dimensions == [
        "kg_forward_fault_ids", "kg_backward_fault_ids", "structured_output_sha256"
    ]
    assert {item.development_group_id for item in replay.episodes} == {"3", "8"}
    assert not ({"4", "9"} & {item.development_group_id for item in replay.episodes})


def test_review_package_rejects_artifact_and_embedded_content_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "package"
    shutil.copytree(PACKAGE, copied)
    payload = json.loads((copied / "knowledge_mapping_candidates.json").read_text(encoding="utf-8"))
    payload["candidates"][0]["component_id"] = "tampered"
    (copied / "knowledge_mapping_candidates.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_dataset_kg_review_package(copied)
    with pytest.raises(ValidationError, match="candidate_set_sha256 mismatch"):
        KnowledgeMappingCandidateSet.model_validate(payload)


def test_generator_ast_extractors_do_not_execute_source(tmp_path: Path) -> None:
    adcs = tmp_path / "adcs.py"
    adcs.write_text(
        """
raise RuntimeError('must not execute')
CONDITIONS = (
    Condition('nominal', 'nominal', None, {}, 'base', 'nominal', '', (), 'high', 'nominal'),
    Condition('fault-a', 'fault', 'effect-a', {'x': (0, 1)}, 'component.a', 'desc', '', ('S1',), 'medium', 'severe'),
)
DIAGNOSTIC_CLASS_OVERRIDES = {'fault-a': 'class-a'}
""",
        encoding="utf-8",
    )
    phm = tmp_path / "phm.py"
    phm.write_text(
        """
raise RuntimeError('must not execute')
CONDITIONS = (
    FaultCondition('fault-b', 'eps', 'effect-b', {'ratio': (0.2, 0.4)}, 'battery', 'desc'),
)
""",
        encoding="utf-8",
    )
    conditions, overrides = extract_adcs_generator_conditions(adcs)
    assert conditions["fault-a"]["component_source"] == "component.a"
    assert conditions["fault-a"]["source_ids"] == ["S1"]
    assert overrides == {"fault-a": "class-a"}
    assert extract_phm_generator_conditions(phm)["fault-b"] == {
        "subsystem": "eps", "effect": "effect-b", "parameter_names": ["ratio"],
        "component_source": "battery", "description": "desc", "source_ids": [],
        "calibration_confidence": None, "severity_band": None,
    }


def test_returned_supplemental_review_is_frozen_without_manufactured_approval(
    tmp_path: Path,
) -> None:
    output = tmp_path / "freeze.json"
    receipt = freeze_supplemental_human_review(
        package_dir=PACKAGE, supplemental_review_path=SUPPLEMENTAL, output_path=output,
    )
    assert receipt.reviewed_candidate_count == 22
    assert receipt.formal_decision_count == 0
    assert receipt.reviewer_identity_present is False
    assert receipt.temporary_frozen is True
    assert receipt.publication_allowed is False
    assert receipt.active_graph_modified is False
    assert validate_supplemental_review_freeze(
        package_dir=PACKAGE, supplemental_review_path=SUPPLEMENTAL, receipt_path=output,
    ) == receipt


def test_governed_supplemental_review_freeze_is_current() -> None:
    receipt = validate_supplemental_review_freeze(
        package_dir=PACKAGE, supplemental_review_path=SUPPLEMENTAL, receipt_path=FREEZE,
    )
    assert sum(receipt.advisory_status_counts.values()) == 22


def test_review_unfreeze_rejects_unverified_comparison_audit(tmp_path: Path) -> None:
    generators = {}
    for name in ("adcs_v522_generator", "adcs_imported_base_generator", "phm_v6_generator"):
        source = tmp_path / f"{name}.py"
        source.write_text(f"# {name}\n", encoding="utf-8")
        generators[name] = source
    adjudication_path = tmp_path / "source_adjudication.json"
    package = build_source_adjudication_package(
        package_dir=PACKAGE,
        supplemental_review_path=SUPPLEMENTAL,
        adjudication_draft_path=PACKAGE / "source_adjudication_draft_20260901.json",
        generator_source_paths=generators,
        output_path=adjudication_path,
    )
    assert validate_source_adjudication_package(
        package_dir=PACKAGE,
        supplemental_review_path=SUPPLEMENTAL,
        generator_source_paths=generators,
        adjudication_path=adjudication_path,
    ) == package
    handoff = write_source_adjudication_review_sheet(
        supplemental_review_path=SUPPLEMENTAL,
        source_adjudication_path=adjudication_path,
        output_path=tmp_path / "human_handoff.csv",
    )
    with handoff.open(encoding="utf-8-sig", newline="") as stream:
        handoff_rows = list(csv.DictReader(stream))
    assert len(handoff_rows) == 22
    assert all(row["source_status"] for row in handoff_rows)
    assert all(row["decision"] == row["reviewer"] == "" for row in handoff_rows)
    generators["phm_v6_generator"].write_text("# drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generator source hash mismatch"):
        validate_source_adjudication_package(
            package_dir=PACKAGE,
            supplemental_review_path=SUPPLEMENTAL,
            generator_source_paths=generators,
            adjudication_path=adjudication_path,
        )
    fake_audit = tmp_path / "fake_sota_audit.json"
    fake_audit.write_text(json.dumps({
        "status": "PASS_DOMAIN_SOTA_FAIR_COMPARISON",
        "anti_suppression_checks": {"all": True},
        "audit_sha256": "0" * 64,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="must pass before review unfreeze"):
        unfreeze_supplemental_human_review(
            package_dir=PACKAGE,
            supplemental_review_path=SUPPLEMENTAL,
            freeze_receipt_path=FREEZE,
            sota_comparison_audit_path=fake_audit,
            source_adjudication_path=adjudication_path,
            output_path=tmp_path / "unfreeze.json",
        )


def test_governed_source_adjudication_and_unfreeze_are_current() -> None:
    source = SourceAdjudicationPackage.model_validate_json(
        SOURCE_ADJUDICATION.read_text(encoding="utf-8")
    )
    assert len(source.records) == 22
    assert source.status_counts == {
        "SOURCE_CONFIRMED": 13,
        "SOURCE_CONFIRMED_WITH_REQUIRED_CANDIDATE_MODIFICATION": 4,
        "SOURCE_SCOPE_LIMIT_REQUIRED": 5,
    }
    assert source.formal_decision_count == 0
    receipt = validate_supplemental_review_unfreeze(
        package_dir=PACKAGE,
        supplemental_review_path=SUPPLEMENTAL,
        freeze_receipt_path=FREEZE,
        sota_comparison_audit_path=SOTA_AUDIT,
        source_adjudication_path=SOURCE_ADJUDICATION,
        unfreeze_receipt_path=UNFREEZE,
    )
    assert receipt.temporary_frozen is False
    assert receipt.formal_decision_count == 0
    assert receipt.publication_allowed is receipt.active_graph_modified is False
    with SOURCE_HANDOFF.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 22
    assert all(row["source_finding"] for row in rows)
    assert all(
        row["decision"] == row["approved_runtime_fault_id"]
        == row["edge_role_decision"] == row["reviewer"] == ""
        for row in rows
    )


def test_source_adjudicated_branch_is_refrozen_before_graph_refactor(tmp_path: Path) -> None:
    output = tmp_path / "refreeze.json"
    kwargs = {
        "package_dir": PACKAGE,
        "supplemental_review_path": SUPPLEMENTAL,
        "freeze_receipt_path": FREEZE,
        "sota_comparison_audit_path": SOTA_AUDIT,
        "source_adjudication_path": SOURCE_ADJUDICATION,
        "unfreeze_receipt_path": UNFREEZE,
        "human_handoff_path": SOURCE_HANDOFF,
    }
    receipt = refreeze_supplemental_human_review(
        **kwargs,
        freeze_reason="Pause formal decisions while the reusable KG construction pipeline is refactored.",
        output_path=output,
    )
    assert receipt.candidate_count == 22
    assert receipt.temporary_frozen is True
    assert receipt.formal_decision_count == 0
    assert receipt.publication_allowed is receipt.active_graph_modified is False
    assert validate_supplemental_review_refreeze(
        **kwargs, refreeze_receipt_path=output,
    ) == receipt
    governed = validate_supplemental_review_refreeze(
        **kwargs, refreeze_receipt_path=REFREEZE,
    )
    assert governed.status == "REFROZEN_PENDING_FORMAL_DECISION_AFTER_SOURCE_ADJUDICATION"
