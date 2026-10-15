from __future__ import annotations

from datetime import datetime, timezone

import pytest

from astrograph.agents.mission.joint_evidence import build_joint_evidence_package
from astrograph.agents.mission.literature_context import build_literature_context_assessment
from astrograph.agents.mission.storage import sha
from astrograph.lifecycle import ExpertFeedback, bind_feedback_to_joint_evidence, attribute_feedback_error_layers
from astrograph.lifecycle.feedback_regression_bridge import bind_feedback_to_failure_regression


def _core():
    return {"status":"SUCCEEDED","diagnosis":{"status":"INSUFFICIENT_EVIDENCE","selected_fault_id":None,
        "candidate_fault_ids":["FAULT_A"],"working_hypothesis_fault_id":"FAULT_A","reason_codes":["WORKING_ONLY"],
        "candidate_explanations":[]}}


def _package(core=None):
    core=core or _core()
    source={"citation_id":"DOC:1","source_sha256":"a"*64,"source_locator":"page:1",
            "reviewed_claim_polarity":"SUPPORTS","reviewed_target_fault_id":"FAULT_A","reviewed_claim_id":"claim:a"}
    literature=build_literature_context_assessment(core,[source])
    return build_joint_evidence_package(core_result=core,
        model_execution={"status":"SUCCEEDED","three_task_contract":{"classification":{"status":"EXECUTED","fault_id":"FAULT_A","formal_diagnostic_vote_allowed":False}}},
        generic_result={"diagnosis":{"fault_candidates":[{"fault_id":"FAULT_A","score":0.7}]}},
        literature_context=literature,sources=[source])


def _feedback(core, refs, layers):
    return ExpertFeedback(feedback_id="feedback:dev49",task_id="episode:dev49",result_sha256=sha(core),
        feedback_type="DIAGNOSIS_CORRECTION",reviewer="expert:dev49",reviewed_at=datetime.now(timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=tuple(layers),
        diagnosis_error_summary="working hypothesis incorrect",recommended_change="evaluate governed challenger",evidence_refs=tuple(refs))


def test_model_reference_suggests_model_layer_without_overriding_expert() -> None:
    core=_core();package=_package(core)
    node=next(x for x in package.nodes if x.branch=="MODEL")
    feedback=_feedback(core,[node.evidence_id],["model"])
    bound=bind_feedback_to_joint_evidence(feedback=feedback,joint_evidence_package=package,original_result=core)
    attribution=attribute_feedback_error_layers(feedback=feedback,binding=bound,joint_evidence_package=package)
    assert attribution.status=="ALIGNED"
    assert attribution.suggested_target_layers==["model"]
    assert attribution.expert_target_layers==["model"]
    assert attribution.expert_decision_overridden is False


def test_literature_and_core_references_generate_review_only_layer_suggestions() -> None:
    core=_core();package=_package(core)
    refs=[next(x for x in package.nodes if x.branch=="LITERATURE").evidence_id,
          next(x for x in package.nodes if x.branch=="CORE_DIAGNOSIS").evidence_id]
    feedback=_feedback(core,refs,["knowledge","planning"])
    bound=bind_feedback_to_joint_evidence(feedback=feedback,joint_evidence_package=package,original_result=core)
    attribution=attribute_feedback_error_layers(feedback=feedback,binding=bound,joint_evidence_package=package)
    assert attribution.status=="PARTIALLY_ALIGNED"
    assert attribution.suggested_target_layers==["execution","knowledge","memory","planning"]
    assert attribution.overlap_layers==["knowledge","planning"]
    assert attribution.suggestion_only_layers==["execution","memory"]
    assert attribution.expert_target_layers==["knowledge","planning"]


def test_unbound_legacy_feedback_gets_no_structured_attribution() -> None:
    core=_core();package=_package(core)
    feedback=_feedback(core,["legacy:free-text"],["tool"])
    bound=bind_feedback_to_joint_evidence(feedback=feedback,joint_evidence_package=package,original_result=core)
    attribution=attribute_feedback_error_layers(feedback=feedback,binding=bound,joint_evidence_package=package)
    assert attribution.status=="NO_STRUCTURED_EVIDENCE_BINDING"
    assert attribution.suggested_target_layers==[]
    assert attribution.expert_target_layers==["tool"]


def test_mismatched_binding_or_package_is_rejected() -> None:
    core=_core();package=_package(core)
    node=next(x for x in package.nodes if x.branch=="MODEL")
    feedback=_feedback(core,[node.evidence_id],["model"])
    bound=bind_feedback_to_joint_evidence(feedback=feedback,joint_evidence_package=package,original_result=core)
    changed=_package(core).model_copy(update={"package_id":"JEP-DIFFERENT"})
    with pytest.raises(ValueError,match="does not match"):
        attribute_feedback_error_layers(feedback=feedback,binding=bound,joint_evidence_package=changed)


def test_failure_regression_persists_error_attribution_without_changing_target_layers(tmp_path) -> None:
    core=_core();package=_package(core)
    model_node=next(x for x in package.nodes if x.branch=="MODEL")
    literature_node=next(x for x in package.nodes if x.branch=="LITERATURE")
    feedback=_feedback(core,[model_node.evidence_id,literature_node.evidence_id],["model","knowledge"])
    result=bind_feedback_to_failure_regression(feedback=feedback,original_result=core,regression_root=tmp_path,
        joint_evidence_package=package.model_dump(mode="json"))
    assert result["evidence_error_attribution"]["status"]=="PARTIALLY_ALIGNED"
    assert result["failure_case"]["metadata"]["target_layers"]==["model","knowledge"]
    assert result["failure_case"]["metadata"]["evidence_error_attribution"]["suggested_target_layers"]==["knowledge","memory","model"]
