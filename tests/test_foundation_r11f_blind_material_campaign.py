"""R11-F blinded four-arm assessment scaffolding; scripted service, no model inference."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrograph.evaluation.material_llm_blind_campaign import (
    ARM_ORDER, BlindMaterialCase, run_blind_material_campaign,
    score_blind_material_reviews,
)


def make_case(tmp_path, case_id="CASE-A"):
    doc = tmp_path / (case_id + "-reference.md")
    doc.write_text("Synthetic test stand: increased torque may raise housing temperature.\n")
    snapshot = {
        "candidate_fault_id": "FAULT_A",
        "candidate_fault_ids": ["FAULT_A", "FAULT_B"],
        "material_source_group_id": case_id,
        "evidence": [{"evidence_id": "EV-1", "interpretation": "A hypothetical observation."}],
    }
    review = {
        "source_group_id": case_id,
        "candidates": [{"fault_id": "FAULT_A"}, {"fault_id": "FAULT_B"}],
        "competing_hypothesis_review": {"status": "COMPARE_ONLY"},
    }
    return BlindMaterialCase(
        case_id=case_id, group_id="BENCH-FAMILY",
        frozen_diagnosis=snapshot, reviewed_material=review,
        document_paths=(doc,),
    )


class ScriptedService:
    def __init__(self, arm, calls):
        self.arm = arm
        self.calls = calls
        self.config = SimpleNamespace(
            material_tool_calls_enabled=(arm == ARM_ORDER[-1]), a2o_enabled=False,
        )
        self.paperseek_client = None

    def enrich(self, **kwargs):
        self.calls.append((self.arm, deepcopy(kwargs)))
        assert "ground_truth" not in str(kwargs)
        return {
            "status": "SUCCEEDED",
            "generation": {
                "status": "SUCCEEDED",
                "explanation": {
                    "abstained": False,
                    "candidate_fault_id": kwargs["diagnosis_snapshot"]["candidate_fault_id"],
                    "claims": [{"kind": "HYPOTHESIS", "text": "Reference is inconclusive."}],
                },
                "citation_reference_validation": "PASS",
                "semantic_hallucination_assessment": "NOT_EVALUATED",
                "material_tool_investigation": {
                    "status": "DISABLED" if self.arm != ARM_ORDER[-1] else "COMPLETED",
                    "review_coverage_audit": {"status": "MODEL_DECLINED_AVAILABLE_REVIEW"},
                },
            },
            "timings_s": {"total": 0.01},
        }


def factory(calls):
    return lambda arm: ScriptedService(arm, calls)


def test_four_arms_keep_frozen_inputs_and_never_get_truth(tmp_path):
    case = make_case(tmp_path)
    original = deepcopy(case)
    calls = []
    root = tmp_path / "run"
    outcomes = run_blind_material_campaign(
        [case], service_factory=factory(calls), output_root=root,
    )
    assert [row["arm"] for row in outcomes["cells"]] == list(ARM_ORDER)
    assert len(calls) == 4
    assert [len(call["document_paths"]) for _, call in calls] == [0, 1, 1, 1]
    assert [call["material_review_context"] is not None for _, call in calls] == [
        False, False, True, True,
    ]
    assert all(call["diagnosis_snapshot"]["candidate_fault_id"] == "FAULT_A"
               for _, call in calls)
    assert case == original
    assert outcomes["gold_labels_loaded"] is False
    assert outcomes["diagnostic_accuracy_gain_identifiable"] is False
    assert outcomes["model_real_inference_verified"] is False
    assert (root / "manifest.json").is_file()
    assert (root / "outcomes.json").is_file()
    assert "gold" not in (root / "manifest.json").read_text().lower()


@pytest.mark.parametrize("field", ["ground_truth", "true_fault_id", "evaluation_metadata", "hidden_answer"])
def test_private_truth_keys_rejected_before_any_model_call(tmp_path, field):
    case = make_case(tmp_path)
    case.frozen_diagnosis[field] = "FOREIGN"
    calls = []
    with pytest.raises(ValueError, match="FORBIDDEN_TRUTH_FIELD"):
        run_blind_material_campaign([case], service_factory=factory(calls),
                                    output_root=tmp_path / "reject")
    assert not calls
    assert not (tmp_path / "reject").exists()


def test_review_scope_and_candidate_mismatches_rejected(tmp_path):
    case = make_case(tmp_path)
    case.reviewed_material["source_group_id"] = "FOREIGN-CASE"
    with pytest.raises(ValueError, match="BLIND_REVIEW_SOURCE_MISMATCH"):
        case.verify()
    case.reviewed_material["source_group_id"] = case.case_id
    case.reviewed_material["candidates"][1]["fault_id"] = "FOREIGN-FAULT"
    with pytest.raises(ValueError, match="BLIND_REVIEW_CANDIDATES_MISMATCH"):
        case.verify()


def test_arm_tool_flag_misconfiguration_fails_closed(tmp_path):
    case = make_case(tmp_path)
    calls = []

    def wrong(arm):
        service = ScriptedService(arm, calls)
        service.config.material_tool_calls_enabled = False
        return service

    with pytest.raises(ValueError, match="BLIND_ARM_TOOL_FLAG_MISMATCH"):
        run_blind_material_campaign([case], service_factory=wrong,
                                    output_root=tmp_path / "run")


def test_reusing_campaign_root_cannot_overwrite_old_evaluation(tmp_path):
    case = make_case(tmp_path)
    root = tmp_path / "sealed"
    run_blind_material_campaign([case], service_factory=factory([]), output_root=root)
    with pytest.raises(FileExistsError, match="NEW_BLIND_CAMPAIGN_OUTPUT_REQUIRED"):
        run_blind_material_campaign([case], service_factory=factory([]), output_root=root)


def test_review_scoring_only_after_run_and_never_reports_candidate_accuracy_gain(tmp_path):
    case = make_case(tmp_path)
    outcomes = run_blind_material_campaign([case], service_factory=factory([]),
                                           output_root=tmp_path / "run")
    reviews = [
        {"case_id": case.case_id, "arm": arm,
         "evidence_entailment": 2, "counterevidence_recognition": 3,
         "abstention_appropriateness": 4, "review_actionability": 1}
        for arm in ARM_ORDER
    ]
    scores = score_blind_material_reviews(outcomes, reviews)
    assert scores["rated_cells"] == 4
    assert scores["arms"]["LLM_ONLY"]["evidence_entailment"] == 2.0
    assert scores["diagnostic_accuracy_gain"] is None
    assert scores["statistical_superiority_proven"] is False
    with pytest.raises(ValueError, match="INCOMPLETE"):
        score_blind_material_reviews(outcomes, reviews[:2])
    with pytest.raises(ValueError, match="INVALID_BLIND_REVIEW_RATING"):
        score_blind_material_reviews(outcomes, [dict(reviews[0], evidence_entailment=True), *reviews[1:]])


def test_unavailable_review_does_not_become_approved_graph(tmp_path):
    case = make_case(tmp_path)
    no_review = BlindMaterialCase(case_id=case.case_id, group_id=case.group_id,
        frozen_diagnosis=case.frozen_diagnosis, reviewed_material=None,
        document_paths=case.document_paths)
    calls = []
    result = run_blind_material_campaign([no_review], service_factory=factory(calls),
                                         output_root=tmp_path / "run")
    assert [x["material_binding_status"] for x in result["cells"]] == ["NOT_AVAILABLE"] * 4
    assert all(call["material_review_context"] is None for _, call in calls)
