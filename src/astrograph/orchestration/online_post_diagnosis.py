"""Opt-in, after-persistence read-only LLM explanation of a frozen online result.

This is not an online DAG stage, diagnostic voter, or release gate. All model
and retrieval failures are isolated from the completed online diagnosis.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from astrograph.diagnosis.structured_output import StructuredDiagnosisOutput
from astrograph.orchestration.models import ExecutionResult, ExecutionStatus


class OnlinePostDiagnosisBindingError(ValueError):
    """The supplied result is not an independently verified online final result."""


def _sha_json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def freeze_online_diagnosis(result: ExecutionResult) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Check host-authored diagnosis and its immutable lineage before LLM access."""
    if not isinstance(result, ExecutionResult) or result.status != ExecutionStatus.SUCCEEDED:
        raise OnlinePostDiagnosisBindingError("ONLINE_RESULT_NOT_SUCCEEDED")
    if not result.workflow_id.startswith("simulation.whole_spacecraft.online_diagnosis."):
        raise OnlinePostDiagnosisBindingError("ONLINE_WORKFLOW_ID_MISMATCH")
    payload = result.output_payload
    try:
        frozen = StructuredDiagnosisOutput.model_validate(payload["structured_diagnosis_output"])
        decision = payload["decision_record"]
        episode = payload["episode_evidence_bundle"]["episode"]
        event = payload["simulation_event"]
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlinePostDiagnosisBindingError("ONLINE_RESULT_INTEGRITY_INVALID") from exc
    # online.persist.v1 intentionally omits the large runtime stage artifacts
    # from the public ExecutionResult. Validate every retained lineage digest
    # without pretending the removed raw artifact map is still available.
    if any(key not in payload or _sha_json(payload[key]) != digest
           for key, digest in frozen.lineage_sha256.items()):
        raise OnlinePostDiagnosisBindingError("ONLINE_RESULT_LINEAGE_MISMATCH")
    if (not result.task_id or not frozen.run_id or
            result.metadata.get("run_id") != frozen.run_id or
            event.get("run_id") != frozen.run_id or
            episode.get("episode_id") != frozen.episode_id or
            decision.get("episode_id") != frozen.episode_id or
            decision.get("accepted_fault_id") != frozen.final_decision["accepted_fault_id"] or
            decision.get("disposition") != frozen.final_decision["disposition"]):
        raise OnlinePostDiagnosisBindingError("ONLINE_RESULT_IDENTITY_MISMATCH")
    accepted = frozen.final_decision.get("accepted_fault_id")
    ranked = frozen.candidate_summary.get("candidates") or []
    if not isinstance(ranked, list):
        raise OnlinePostDiagnosisBindingError("ONLINE_CANDIDATE_LIST_INVALID")
    candidate_ids: list[str] = []
    if accepted is not None:
        if not isinstance(accepted, str) or not accepted:
            raise OnlinePostDiagnosisBindingError("ONLINE_TOP_CANDIDATE_INVALID")
        candidate_ids.append(accepted)
    for row in ranked:
        if not isinstance(row, dict):
            raise OnlinePostDiagnosisBindingError("ONLINE_CANDIDATE_LIST_INVALID")
        fid = row.get("fault_id")
        if not isinstance(fid, str) or not fid:
            raise OnlinePostDiagnosisBindingError("ONLINE_CANDIDATE_LIST_INVALID")
        if fid not in candidate_ids:
            candidate_ids.append(fid)
    if len(candidate_ids) > 12:
        # Do not silently truncate frozen candidates to make graph tools pass.
        raise OnlinePostDiagnosisBindingError("ONLINE_CANDIDATE_COUNT_EXCEEDS_REVIEW_LIMIT")
    evidence = payload.get("episode_evidence_bundle", {}).get("evidence_facts") or []
    observation_receipts = [{
        "evidence_id": item["evidence_fact_id"],
        "field": "upstream_evidence_fact",
        "interpretation": str(item.get("status", "UNKNOWN")),
    } for item in evidence[:20] if isinstance(item, dict) and item.get("evidence_fact_id")]
    snapshot = {
        "schema_version": "astrograph-platform-frozen-diagnosis.v1",
        "diagnosis_mode": "ONLINE_POST_RESULT_EXPLANATION",
        "status": str(decision.get("disposition", "UNKNOWN")),
        "candidate_fault_id": accepted,
        "candidate_fault_ids": candidate_ids,
        "material_source_group_id": frozen.run_id,
        "evidence": observation_receipts,
        "formal_diagnosis_publishable": False,
        "labels_used": False,
        "truth_used": False,
    }
    receipt = {
        "source_kind": "ONLINE_RESULT", "source_run_id": frozen.run_id,
        "task_id_sha256": hashlib.sha256(result.task_id.encode("utf-8")).hexdigest(),
        "structured_output_sha256": frozen.output_sha256,
        "original_decision_sha256": _sha_json(decision),
        "diagnostic_vote_allowed": False,
        "candidate_count": len(candidate_ids),
    }
    return frozen.run_id, snapshot, receipt


def _verify_persisted_artifacts(result: ExecutionResult, output_root: Path) -> None:
    """Match the returned host decision to committed on-disk artifact bytes."""
    try:
        expected_dir = Path(result.metadata["artifact_dir"]).resolve(strict=True)
        actual_dir = output_root.resolve(strict=True)
        if actual_dir != expected_dir:
            raise OnlinePostDiagnosisBindingError("ONLINE_ARTIFACT_DIRECTORY_MISMATCH")
        manifest = json.loads((actual_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("run_id") != result.metadata.get("run_id"):
            raise OnlinePostDiagnosisBindingError("ONLINE_ARTIFACT_RUN_MISMATCH")
        for key in ("structured_diagnosis_output", "decision_record"):
            entry = manifest["files"][key]
            path = (actual_dir / entry["path"]).resolve(strict=True)
            if not path.is_relative_to(actual_dir):
                raise OnlinePostDiagnosisBindingError("ONLINE_ARTIFACT_PATH_ESCAPE")
            content = path.read_bytes()
            if (len(content) != entry["size_bytes"] or
                    hashlib.sha256(content).hexdigest() != entry["sha256"] or
                    json.loads(content) != result.output_payload[key]):
                raise OnlinePostDiagnosisBindingError("ONLINE_ARTIFACT_CONTENT_MISMATCH")
    except OnlinePostDiagnosisBindingError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OnlinePostDiagnosisBindingError("ONLINE_ARTIFACT_INTEGRITY_INVALID") from exc


def run_online_post_diagnosis_explanation(
    *, result: ExecutionResult, knowledge_service: Any,
    output_root: Path, document_paths: list[Path] | None = None,
    material_review_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bounded sidecar receipt; never mutate original ExecutionResult.

    The source output is already persisted by the canonical online DAG. This
    function also serves artifact-driven runtimes via an explicit opt-in call.
    """
    output_root = Path(output_root)
    root = output_root / "post_diagnosis"
    receipt: dict[str, Any] = {"schema_version": "astrograph-online-post-explanation.v1",
                              "status": "FAILED_CLOSED", "diagnostic_vote_allowed": False,
                              "core_result_modified": False}
    original = deepcopy(result)
    try:
        run_id, snapshot, identity = freeze_online_diagnosis(result)
        _verify_persisted_artifacts(result, output_root)
        receipt["identity"] = identity
        from astrograph.integrations.post_diagnosis_explanation import run_frozen_post_diagnosis_explanation
        explanation = run_frozen_post_diagnosis_explanation(
            source_kind="ONLINE_RESULT", source_id=run_id,
            task="Explain this frozen online diagnosis using evidence only; never revise the governor decision.",
            diagnosis_snapshot=snapshot, document_paths=list(document_paths or []),
            output_root=root / "knowledge", knowledge_service=knowledge_service,
            material_review_context=material_review_context,
        )
        receipt["status"] = "SUCCEEDED"
        receipt["knowledge_status"] = str(explanation.get("status", "UNKNOWN"))
        receipt["generation_status"] = str((explanation.get("generation") or {}).get("status", "UNKNOWN"))
        receipt["review_binding"] = deepcopy(explanation.get("post_diagnosis_binding") or {})
    except Exception as exc:
        receipt["status"] = "FAILED_CLOSED"
        receipt["reason_code"] = (
            str(exc) if isinstance(exc, OnlinePostDiagnosisBindingError)
            else "OPTIONAL_EXPLANATION_FAILED"
        )
        receipt["error_type"] = type(exc).__name__
    if result != original:
        receipt = {"schema_version": receipt["schema_version"],
                   "status": "FAILED_CLOSED", "reason_code": "ONLINE_SOURCE_MUTATED",
                   "diagnostic_vote_allowed": False, "core_result_modified": True}
        result.__dict__.update(original.__dict__)
    try:
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / "online_explanation_receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(root / "online_explanation_receipt.json")
    except OSError:
        receipt["receipt_persisted"] = False
    return receipt
