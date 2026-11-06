"""Shared post-diagnosis LLM explanation boundary for frozen upstream results.

This adapter never invokes fault detection or adds candidates. Independently
reviewed material must match the run ID on both the frozen diagnosis and review.
If identity is unproven, the LLM can still use ordinary retrieval, but not the
reviewed GraphRAG tool context.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

_ALLOWED_SOURCES = frozenset({
    "RESULT_LOG_REVIEW", "PUBLIC_CASE", "SAT_SIM_PIPELINE",
    "OAG_DIAGNOSTIC", "ONLINE_RESULT",
})


def _checked_material(
    source_id: str,
    diagnosis_snapshot: dict[str, Any],
    material_review_context: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if material_review_context is None:
        return None, {"status": "NOT_AVAILABLE", "diagnostic_vote_allowed": False}

    def deny(reason: str) -> tuple[None, dict[str, Any]]:
        return None, {"status": "DENIED_TASK_BINDING", "reason_code": reason,
                      "diagnostic_vote_allowed": False}

    if not isinstance(material_review_context, dict):
        return deny("INVALID_REVIEW")
    group = diagnosis_snapshot.get("material_source_group_id")
    review_group = material_review_context.get("source_group_id")
    if not (isinstance(group, str) and group == source_id and
            isinstance(review_group, str) and review_group == source_id):
        return deny("SOURCE_IDENTITY_NOT_PROVEN")
    frozen = diagnosis_snapshot.get("candidate_fault_ids")
    candidates = material_review_context.get("candidates")
    if not isinstance(frozen, list) or not (1 <= len(frozen) <= 12) or not isinstance(candidates, list):
        return deny("FROZEN_CANDIDATES_MISSING")
    if not all(isinstance(fid, str) and 0 < len(fid) <= 160 for fid in frozen):
        return deny("FROZEN_CANDIDATES_INVALID")
    reviewed = [r.get("fault_id") if isinstance(r, dict) else None for r in candidates]
    if (len(set(frozen)) != len(frozen) or len(reviewed) != len(frozen) or
            not all(isinstance(fid, str) and fid for fid in reviewed) or
            len(set(reviewed)) != len(reviewed) or set(frozen) != set(reviewed)):
        return deny("FROZEN_CANDIDATES_MISMATCH")
    top = diagnosis_snapshot.get("candidate_fault_id")
    if top is not None and top not in frozen:
        return deny("FROZEN_TOP_CANDIDATE_MISMATCH")
    return deepcopy(material_review_context), {
        "status": "BOUND_FROZEN_DIAGNOSIS", "candidate_count": len(frozen),
        "source_group_checked": True, "diagnostic_vote_allowed": False,
    }


def run_frozen_post_diagnosis_explanation(
    *,
    source_kind: str,
    source_id: str,
    task: str,
    diagnosis_snapshot: dict[str, Any],
    document_paths: list[Path],
    output_root: Path,
    knowledge_service: Any,
    material_review_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the existing LLM service only after explicit task binding.

    ONLINE_RESULT is an opt-in callable interface, not an automatic hook into
    the online detection DAG; this module does not modify upstream diagnosis.
    """
    if source_kind not in _ALLOWED_SOURCES:
        raise ValueError("UNSUPPORTED_POST_DIAGNOSIS_SOURCE")
    if not isinstance(source_id, str) or not (1 <= len(source_id) <= 160):
        raise ValueError("POST_DIAGNOSIS_SOURCE_ID_INVALID")
    if not isinstance(diagnosis_snapshot, dict):
        raise TypeError("POST_DIAGNOSIS_SNAPSHOT_INVALID")
    trusted_review, binding = _checked_material(source_id, diagnosis_snapshot, material_review_context)
    result = knowledge_service.enrich(
        task=task,
        diagnosis_snapshot=deepcopy(diagnosis_snapshot),
        document_paths=list(document_paths),
        output_root=output_root,
        material_review_context=trusted_review,
    )
    if not isinstance(result, dict):
        raise TypeError("POST_DIAGNOSIS_EXPLANATION_INVALID")
    result["post_diagnosis_binding"] = {
        "schema_version": "astrograph-post-diagnosis-binding.v1",
        "source_kind": source_kind,
        **binding,
    }
    return result
