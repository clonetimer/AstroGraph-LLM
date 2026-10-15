from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.agentization.lineage_resolution import ArtifactState, LineageArtifact, Step6AssessmentInput, Step6Route, assess_step6_route
from astrograph.agentization.stable_release import CheckStatus, ReleaseCheck, ReleaseScope, StabilityInput, assess_stability


def art(role: str, state: ArtifactState=ArtifactState.VERIFIED) -> LineageArtifact:
    return LineageArtifact(artifact_id=role,role=role,state=state)


def inp(*arts: LineageArtifact, caps=()):
    return Step6AssessmentInput(project_status="B006_WIP_NOT_RELEASED",artifacts=arts,available_capabilities=tuple(caps))


def test_authentic_route_requires_complete_attested_chain():
    roles=("original_model_package","state_dict_identity","feature_contract","preprocessor_receipt","training_receipt","label_free_dev_traces","temporal_v2_receipt","development_split_receipt")
    d=assess_step6_route(inp(*(art(r) for r in roles)))
    assert d.selected_route is Step6Route.RESTORE_AUTHENTIC_CHAIN
    assert d.authentic_migration_claim_allowed


def test_missing_original_selects_clean_room_when_recipe_is_verified():
    roles=("training_dataset_manifest","training_labels","feature_contract","preprocessor_spec","training_recipe","development_split_receipt")
    d=assess_step6_route(inp(*(art(r) for r in roles), art("original_model_package",ArtifactState.MISSING)))
    assert d.selected_route is Step6Route.CLEAN_ROOM_RETRAIN_NEW_LINEAGE
    assert not d.authentic_migration_claim_allowed
    assert not d.historical_metrics_reusable


def test_reference_baseline_is_distinct_from_reconstruction():
    roles=("training_dataset_manifest","training_labels","feature_contract","preprocessor_spec","development_split_receipt")
    d=assess_step6_route(inp(*(art(r) for r in roles)))
    assert d.selected_route is Step6Route.REFERENCE_BASELINE_NEW_LINEAGE
    assert any("not a reconstruction" in s for s in next(x for x in d.route_eligibility if x.route is d.selected_route).claim_boundary)


def test_model_independent_route_requires_label_free_generator():
    roles=("telemetry_dataset_manifest","development_split_receipt")
    d=assess_step6_route(inp(*(art(r) for r in roles),caps=("label_free_temporal_score_generator",)))
    assert d.selected_route is Step6Route.MODEL_INDEPENDENT_TEMPORAL_ROUTE
    assert not d.historical_metrics_reusable


def test_scope_freeze_when_no_route_is_evidence_feasible():
    d=assess_step6_route(inp())
    assert d.selected_route is Step6Route.SCOPE_FREEZE


def test_groups_4_9_cannot_be_selection_groups():
    with pytest.raises(ValidationError):
        Step6AssessmentInput(project_status="B006_WIP_NOT_RELEASED",selection_groups=(4,9),artifacts=())


def test_decision_is_outcome_blind_and_deterministic():
    roles=("telemetry_dataset_manifest","development_split_receipt")
    i=inp(*(art(r) for r in roles),caps=("label_free_temporal_score_generator",))
    a=assess_step6_route(i); b=assess_step6_route(i)
    assert a.decision_sha256==b.decision_sha256
    assert not a.outcome_data_read and a.groups_used_for_decision==()


def test_software_stable_does_not_promote_research():
    checks=(ReleaseCheck(check_id="full_software_suite",required_for_software=True,required_for_research=True,status=CheckStatus.PASS,reason="pass"),ReleaseCheck(check_id="confirmatory",required_for_software=False,required_for_research=True,status=CheckStatus.BLOCKED,reason="none"))
    r=assess_stability(StabilityInput(candidate_id="B007",scope=ReleaseScope.SOFTWARE_STABLE_RESEARCH_BLOCKED,project_status_before="B006_WIP_NOT_RELEASED",checks=checks,disclosed_research_blockers=("Step 6 lineage unavailable",),disabled_research_capabilities=("adcs_historical_p2",)))
    assert r.software_stable
    assert not r.research_release_allowed
    assert r.status=="STABLE_SOFTWARE_BASELINE_RESEARCH_BLOCKED"


def test_required_software_skip_blocks_stable_status():
    c=(ReleaseCheck(check_id="required",required_for_software=True,required_for_research=False,status=CheckStatus.SKIP,reason="missing"),)
    r=assess_stability(StabilityInput(candidate_id="B007",scope=ReleaseScope.SOFTWARE_STABLE_RESEARCH_BLOCKED,project_status_before="B006",checks=c,disclosed_research_blockers=("blocked",)))
    assert not r.software_stable


def test_qualification_promotion_rejected():
    with pytest.raises(ValidationError):
        StabilityInput(candidate_id="B007",scope=ReleaseScope.SOFTWARE_STABLE_RESEARCH_BLOCKED,project_status_before="B006",checks=(),disclosed_research_blockers=("x",),qualification_change_requested=True)
