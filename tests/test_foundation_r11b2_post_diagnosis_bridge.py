"""R11-B2 synthetic tests for the shared post-diagnosis LLM bridge."""
from __future__ import annotations

from copy import deepcopy

import pytest
from astrograph.integrations.post_diagnosis_explanation import run_frozen_post_diagnosis_explanation


class CaptureService:
    def __init__(self):
        self.calls = []

    def enrich(self, **kwargs):
        self.calls.append(kwargs)
        return {"generation": {"status": "NOT_CONFIGURED"},
                "diagnostic_candidate_can_be_overridden": False}


def _snapshot(group="run-a"):
    return {"candidate_fault_id": "A", "candidate_fault_ids": ["A", "B"],
            "material_source_group_id": group, "evidence": []}


def _review(group="run-a"):
    return {"source_group_id": group, "candidates": [{"fault_id": "A"}, {"fault_id": "B"}]}


def _invoke(tmp_path, service, kind="RESULT_LOG_REVIEW", snapshot=None,
            material=None, source_id="run-a"):
    return run_frozen_post_diagnosis_explanation(
        source_kind=kind, source_id=source_id,
        task="Explain a frozen synthetic pump diagnosis",
        diagnosis_snapshot=_snapshot() if snapshot is None else snapshot,
        document_paths=[], output_root=tmp_path / "knowledge",
        knowledge_service=service, material_review_context=material,
    )


def test_same_task_review_reaches_tool_context_without_mutating_inputs(tmp_path):
    service = CaptureService()
    original = _review()
    frozen = _snapshot()
    result = _invoke(tmp_path, service, snapshot=frozen, material=original)
    assert result["post_diagnosis_binding"]["status"] == "BOUND_FROZEN_DIAGNOSIS"
    assert service.calls[0]["material_review_context"] == original
    assert service.calls[0]["material_review_context"] is not original
    service.calls[0]["material_review_context"]["candidates"].clear()
    assert len(original["candidates"]) == 2
    assert frozen["candidate_fault_ids"] == ["A", "B"]


@pytest.mark.parametrize("snapshot,review,reason", [
    ({"candidate_fault_id": "A", "candidate_fault_ids": ["A", "B"]},
     {"candidates": [{"fault_id": "A"}, {"fault_id": "B"}]}, "SOURCE_IDENTITY_NOT_PROVEN"),
    (_snapshot("other-run"), _review(), "SOURCE_IDENTITY_NOT_PROVEN"),
    (_snapshot(), _review("other-run"), "SOURCE_IDENTITY_NOT_PROVEN"),
    (_snapshot(), {"source_group_id": "run-a", "candidates": [
        {"fault_id": "A"}, {"fault_id": "C"}]}, "FROZEN_CANDIDATES_MISMATCH"),
    ({"candidate_fault_id": "A", "material_source_group_id": "run-a"},
     _review(), "FROZEN_CANDIDATES_MISSING"),
    ({"candidate_fault_id": "C", "candidate_fault_ids": ["A", "B"],
      "material_source_group_id": "run-a"}, _review(), "FROZEN_TOP_CANDIDATE_MISMATCH"),
])
def test_cross_run_and_unbound_material_never_reaches_model(tmp_path, snapshot, review, reason):
    service = CaptureService()
    result = _invoke(tmp_path, service, snapshot=snapshot, material=review)
    assert result["post_diagnosis_binding"]["status"] == "DENIED_TASK_BINDING"
    assert result["post_diagnosis_binding"]["reason_code"] == reason
    assert service.calls[0]["material_review_context"] is None
    assert "other-run" not in str(result)


@pytest.mark.parametrize("kind", ["PUBLIC_CASE", "SAT_SIM_PIPELINE", "OAG_DIAGNOSTIC", "ONLINE_RESULT"])
def test_post_diagnosis_entrypoints_without_review_remain_usable(tmp_path, kind):
    service = CaptureService()
    result = _invoke(tmp_path, service, kind=kind)
    assert service.calls[0]["material_review_context"] is None
    assert result["post_diagnosis_binding"]["status"] == "NOT_AVAILABLE"
    assert result["diagnostic_candidate_can_be_overridden"] is False


def test_invalid_source_is_rejected_before_llm_invocation(tmp_path):
    service = CaptureService()
    with pytest.raises(ValueError, match="UNSUPPORTED_POST_DIAGNOSIS_SOURCE"):
        _invoke(tmp_path, service, kind="DIAGNOSIS_CORE")
    assert not service.calls
