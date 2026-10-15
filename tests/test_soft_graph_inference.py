from __future__ import annotations

import pytest
import hashlib
import json
from pydantic import ValidationError

from astrograph.evidence import (
    EvidenceFactStatus,
    SoftEvidenceBundle,
    SoftEvidenceFact,
    SoftQuantificationMode,
)
from astrograph.knowledge import (
    SoftGraphProvenancePolicy,
    SoftGraphSuggestedAction,
    build_soft_graph_inference,
    load_soft_graph_provenance_policy,
)


def _fact(
    definition_id: str,
    fault_id: str,
    *,
    role: str = "REQUIRED",
    point: float | None = 0.8,
    lower: float | None = 0.7,
    upper: float | None = 0.9,
    calibrated: bool = True,
) -> SoftEvidenceFact:
    return SoftEvidenceFact(
        soft_evidence_fact_id=f"SEF-{definition_id}",
        evidence_fact_id=f"EF-{definition_id}",
        evidence_definition_id=definition_id,
        target_fault_id=fault_id,
        role=role,
        hard_status=EvidenceFactStatus.SATISFIED,
        predicate_truth_degree=point,
        predicate_truth_lower=lower,
        predicate_truth_upper=upper,
        quantification_mode=(
            SoftQuantificationMode.LOGISTIC_INTERVAL
            if calibrated
            else SoftQuantificationMode.BINARY_COMPATIBILITY
        ),
        calibration_policy_id="POLICY-1" if calibrated else None,
        calibration_policy_sha256="b" * 64 if calibrated else None,
        uncertainty_quantified=calibrated,
    )


def _bundle(facts: list[SoftEvidenceFact]) -> SoftEvidenceBundle:
    provisional = SoftEvidenceBundle(
        episode_id="EP-1",
        source_evidence_sha256="a" * 64,
        facts=facts,
        calibration_policy_id=("POLICY-1" if any(x.uncertainty_quantified for x in facts) else None),
        calibration_policy_sha256=("b" * 64 if any(x.uncertainty_quantified for x in facts) else None),
        calibrated_fact_count=sum(x.uncertainty_quantified for x in facts),
        unavailable_fact_count=sum(x.predicate_truth_degree is None for x in facts),
    )
    return provisional.model_copy(update={"bundle_sha256": provisional.compute_sha256()})


def test_uncalibrated_soft_graph_is_a_non_voting_no_op() -> None:
    result = build_soft_graph_inference(
        _bundle([_fact("ED-1", "FAULT-1", point=1.0, lower=1.0, upper=1.0, calibrated=False)]),
        knowledge_release_id="KR-1",
    )
    assert result.suggested_action is SoftGraphSuggestedAction.NO_OP
    assert result.reason_codes == ["NO_CALIBRATED_SOFT_EVIDENCE"]
    assert result.runtime_effect_applied is False
    assert result.result_sha256 == result.compute_sha256()


def test_interval_separation_requires_two_declared_independence_groups() -> None:
    facts = [
        _fact("ED-A1", "FAULT-A", lower=0.85, point=0.9, upper=0.95),
        _fact("ED-A2", "FAULT-A", role="SUPPORTING", lower=0.8, point=0.85, upper=0.9),
        _fact("ED-B1", "FAULT-B", lower=0.1, point=0.15, upper=0.2),
    ]
    result = build_soft_graph_inference(
        _bundle(facts),
        candidate_fault_ids=["FAULT-A", "FAULT-B"],
        independence_groups_by_definition={
            "ED-A1": "SOURCE-GROUP-1",
            "ED-A2": "SOURCE-GROUP-2",
            "ED-B1": "SOURCE-GROUP-3",
        },
    )
    assert result.suggested_action is SoftGraphSuggestedAction.SINGLE_CANDIDATE
    assert result.candidate_fault_ids == ["FAULT-A"]
    assert result.fault_assessments[0].declared_independent_group_count == 2
    assert result.runtime_effect_applied is False


def test_uncalibrated_compatibility_facts_cannot_influence_a_mixed_bundle() -> None:
    result = build_soft_graph_inference(
        _bundle([
            _fact("ED-A1", "FAULT-A", lower=0.8, point=0.85, upper=0.9),
            _fact("ED-A2", "FAULT-A", role="SUPPORTING", lower=0.75, point=0.8, upper=0.85),
            _fact(
                "ED-BINARY", "FAULT-B", point=1.0, lower=1.0, upper=1.0,
                calibrated=False,
            ),
        ]),
        independence_groups_by_definition={"ED-A1": "A-1", "ED-A2": "A-2"},
    )
    assert result.suggested_action is SoftGraphSuggestedAction.SINGLE_CANDIDATE
    assert result.candidate_fault_ids == ["FAULT-A"]
    by_fault = {item.fault_id: item for item in result.fault_assessments}
    assert by_fault["FAULT-B"].available_observation_count == 1
    assert by_fault["FAULT-B"].calibrated_observation_count == 0
    assert by_fault["FAULT-B"].support_point == 0.0


def test_overlapping_intervals_return_candidate_set() -> None:
    result = build_soft_graph_inference(
        _bundle([
            _fact("ED-A1", "FAULT-A", lower=0.6, point=0.7, upper=0.8),
            _fact("ED-A2", "FAULT-A", role="SUPPORTING", lower=0.6, point=0.7, upper=0.8),
            _fact("ED-B1", "FAULT-B", lower=0.55, point=0.65, upper=0.75),
            _fact("ED-B2", "FAULT-B", role="SUPPORTING", lower=0.55, point=0.65, upper=0.75),
        ]),
        independence_groups_by_definition={
            "ED-A1": "A-1", "ED-A2": "A-2", "ED-B1": "B-1", "ED-B2": "B-2",
        },
    )
    assert result.suggested_action is SoftGraphSuggestedAction.CANDIDATE_SET
    assert result.candidate_fault_ids == ["FAULT-A", "FAULT-B"]


def test_counter_evidence_and_undeclared_provenance_fail_to_abstain() -> None:
    counter = build_soft_graph_inference(
        _bundle([
            _fact("ED-P", "FAULT-A", lower=0.8, point=0.9, upper=0.95),
            _fact("ED-C", "FAULT-A", role="COUNTER", lower=0.7, point=0.8, upper=0.9),
        ]),
        independence_groups_by_definition={"ED-P": "P", "ED-C": "C"},
    )
    assert counter.suggested_action is SoftGraphSuggestedAction.ABSTAIN
    assert "SOFT_GRAPH_COUNTER_EVIDENCE_OR_CONFLICT" in counter.reason_codes

    undeclared = build_soft_graph_inference(_bundle([_fact("ED-P", "FAULT-A")]))
    assert undeclared.suggested_action is SoftGraphSuggestedAction.ABSTAIN
    assert undeclared.fault_assessments[0].declared_independent_group_count == 0


def test_soft_graph_result_hash_tampering_is_rejected() -> None:
    result = build_soft_graph_inference(_bundle([_fact("ED-P", "FAULT-A")]))
    raw = result.model_dump(mode="json")
    raw["reason_codes"] = ["TAMPERED"]
    with pytest.raises(ValidationError, match="result_sha256 mismatch"):
        type(result).model_validate(raw)


def test_frozen_provenance_policy_is_hash_bound_and_drives_declared_groups(tmp_path) -> None:
    provisional = SoftGraphProvenancePolicy(
        policy_id="SGP-1", version="1", declarations=[
            {
                "group_id": "INDEPENDENT-SOURCE-A", "evidence_definition_ids": ["ED-A1"],
                "basis_refs": ["review:A"], "rationale": "separate reviewed acquisition",
            },
            {
                "group_id": "INDEPENDENT-SOURCE-B", "evidence_definition_ids": ["ED-A2"],
                "basis_refs": ["review:B"], "rationale": "separate reviewed acquisition",
            },
        ],
    )
    policy = provisional.model_copy(update={"policy_sha256": provisional.compute_sha256()})
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.model_dump(mode="json")), encoding="utf-8")
    asset_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_soft_graph_provenance_policy(path, expected_sha256=asset_sha)
    result = build_soft_graph_inference(
        _bundle([
            _fact("ED-A1", "FAULT-A", lower=0.8, point=0.9, upper=0.95),
            _fact("ED-A2", "FAULT-A", role="SUPPORTING", lower=0.75, point=0.85, upper=0.9),
        ]),
        independence_groups_by_definition=loaded.definition_groups(),
        provenance_policy_id=loaded.policy_id,
        provenance_policy_sha256=loaded.policy_sha256,
    )
    assert result.suggested_action is SoftGraphSuggestedAction.SINGLE_CANDIDATE
    assert result.provenance_policy_id == "SGP-1"
    with pytest.raises(ValueError, match="asset SHA-256 mismatch"):
        load_soft_graph_provenance_policy(path, expected_sha256="0" * 64)
