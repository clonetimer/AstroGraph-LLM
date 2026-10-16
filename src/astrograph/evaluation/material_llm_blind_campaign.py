"""R11-F: leakage-resistant, paired evaluation of post-diagnosis LLM analysis.

Scores *explanations*, not frozen upstream fault classification. Truth and human
ratings must be applied after the model calls, never passed to this runner.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from astrograph.integrations.post_diagnosis_explanation import (
    run_frozen_post_diagnosis_explanation,
)

ARM_ORDER = ("LLM_ONLY", "LLM_RAG", "LLM_RAG_GRAPHRAG", "LLM_FULL_TOOLS")
_BAD_KEYS = frozenset({
    "ground_truth", "true_fault_id", "truth_fault_id", "correct_fault_id",
    "expected_fault_id", "evaluation_metadata", "hidden_answer", "gold_fault_id",
    "oracle_fault_id", "reference_answer", "labels", "target_label",
})
_CASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _scan_for_labels(obj: Any, *, location: str = "input") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str) or key.lower() in _BAD_KEYS:
                raise ValueError("BLINDED_INPUT_HAS_FORBIDDEN_TRUTH_FIELD")
            _scan_for_labels(value, location=location)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _scan_for_labels(value, location=location)


@dataclass(frozen=True)
class BlindMaterialCase:
    case_id: str
    group_id: str
    frozen_diagnosis: dict[str, Any]
    reviewed_material: dict[str, Any] | None
    document_paths: tuple[Path, ...] = ()
    task: str = "Explain this frozen diagnosis; cite evidence and uncertainty."

    def verify(self) -> dict[str, Any]:
        if not _CASE_RE.fullmatch(self.case_id) or not _CASE_RE.fullmatch(self.group_id):
            raise ValueError("INVALID_BLIND_CASE_OR_GROUP_ID")
        if not isinstance(self.frozen_diagnosis, dict):
            raise TypeError("FROZEN_DIAGNOSIS_REQUIRED")
        _scan_for_labels(self.frozen_diagnosis)
        _scan_for_labels(self.reviewed_material)
        if not isinstance(self.task, str) or not self.task.strip() or len(self.task) > 2000:
            raise ValueError("INVALID_BLIND_CASE_TASK")
        if self.frozen_diagnosis.get("material_source_group_id") != self.case_id:
            raise ValueError("BLIND_CASE_RUN_ID_MISMATCH")
        ids = self.frozen_diagnosis.get("candidate_fault_ids")
        if (not isinstance(ids, list) or not (1 <= len(ids) <= 12) or
                any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids))):
            raise ValueError("BLIND_FROZEN_CANDIDATES_INVALID")
        top = self.frozen_diagnosis.get("candidate_fault_id")
        if top is not None and top not in ids:
            raise ValueError("BLIND_TOP_CANDIDATE_INVALID")
        if self.reviewed_material is not None:
            if not isinstance(self.reviewed_material, dict):
                raise TypeError("BLIND_REVIEW_MUST_BE_DICT")
            if self.reviewed_material.get("source_group_id") != self.case_id:
                raise ValueError("BLIND_REVIEW_SOURCE_MISMATCH")
            reviewed = [row.get("fault_id") for row in self.reviewed_material.get("candidates", [])
                        if isinstance(row, dict)]
            if len(reviewed) != len(ids) or len(set(reviewed)) != len(ids) or set(reviewed) != set(ids):
                raise ValueError("BLIND_REVIEW_CANDIDATES_MISMATCH")
        pins = []
        for path in self.document_paths:
            item = Path(path).resolve(strict=True)
            if not item.is_file() or item.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
                raise ValueError("BLIND_DOCUMENT_NOT_SUPPORTED")
            if any(token in item.name.lower() for token in ("ground_truth", "answer_key", "labels")):
                raise ValueError("BLIND_DOCUMENT_NAME_SUGGESTS_TRUTH")
            pins.append({"sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                         "filename": item.name, "size_bytes": item.stat().st_size})
        return {"case_id": self.case_id, "group_id": self.group_id,
                "snapshot_sha256": _digest(self.frozen_diagnosis),
                "review_sha256": _digest(self.reviewed_material) if self.reviewed_material is not None else None,
                "document_pins": pins}


def run_blind_material_campaign(
    cases: list[BlindMaterialCase],
    *, service_factory: Callable[[str], Any], output_root: Path,
) -> dict[str, Any]:
    """Run paired arms. service_factory is trusted host wiring, NEVER model input.

    Live connectivity/credentials are owned by the caller. CI supplies a scripted
    backend. Neither labels nor evaluator ratings are accepted by this function.
    """
    if not cases or len(cases) > 100:
        raise ValueError("BLIND_CASE_COUNT_INVALID")
    pins = [case.verify() for case in cases]
    if len({p["case_id"] for p in pins}) != len(pins):
        raise ValueError("BLIND_CASE_IDS_DUPLICATED")
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError("NEW_BLIND_CAMPAIGN_OUTPUT_REQUIRED")
    frozen_cases = deepcopy(cases)
    manifest = {
        "schema_version": "astrograph-material-llm-blind-campaign.v1",
        "arms": list(ARM_ORDER), "cases": pins,
        "truth_exposed_to_runner": False,
        "candidate_classification_accuracy_comparable_between_arms": False,
        "claim_boundary": "Only post-diagnosis explanation differs across arms; frozen diagnosis is identical.",
    }
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "manifest.json", manifest)
    cells = []
    for case, pin in zip(cases, pins):
        for arm in ARM_ORDER:
            service = service_factory(arm)
            config = getattr(service, "config", None)
            if config is None or bool(config.material_tool_calls_enabled) != (arm == ARM_ORDER[-1]):
                raise ValueError("BLIND_ARM_TOOL_FLAG_MISMATCH")
            if bool(config.a2o_enabled) or getattr(service, "paperseek_client", None) is not None:
                raise ValueError("BLIND_ARM_EXTERNAL_KNOWLEDGE_NOT_FROZEN")
            if case.verify() != pin:
                raise ValueError("BLIND_INPUT_CHANGED_DURING_CAMPAIGN")
            use_docs = arm != ARM_ORDER[0]
            use_graph = arm in ARM_ORDER[2:]
            try:
                result = run_frozen_post_diagnosis_explanation(
                    source_kind="RESULT_LOG_REVIEW", source_id=case.case_id,
                    task=case.task, diagnosis_snapshot=deepcopy(case.frozen_diagnosis),
                    document_paths=list(case.document_paths if use_docs else ()),
                    material_review_context=deepcopy(case.reviewed_material) if use_graph else None,
                    output_root=root / "trials" / case.case_id / arm,
                    knowledge_service=service,
                )
                generation = result.get("generation") or {}
                explanation = generation.get("explanation") or {}
                investigation = generation.get("material_tool_investigation") or {}
                # Count reference-valid IDs separately from semantic entailment.
                cell = {
                    "case_id": case.case_id, "group_id": case.group_id, "arm": arm,
                    "status": result.get("status"), "model_status": generation.get("status"),
                    "abstained": explanation.get("abstained"),
                    "claim_count": len(explanation.get("claims") or []),
                    "citation_reference_validation": generation.get("citation_reference_validation"),
                    "semantic_hallucination_assessment": generation.get("semantic_hallucination_assessment"),
                    "tool_investigation_status": investigation.get("status"),
                    "tool_coverage_audit": investigation.get("review_coverage_audit"),
                    "material_binding_status": (result.get("post_diagnosis_binding") or {}).get("status"),
                    "latency_s": (result.get("timings_s") or {}).get("total"),
                    "backend_identity_status": "UNVERIFIED_BY_EVALUATOR",
                }
            except Exception as exc:
                cell = {"case_id": case.case_id, "group_id": case.group_id, "arm": arm,
                        "status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                        "model_status": "FAILED", "backend_identity_status": "UNVERIFIED_BY_EVALUATOR"}
            if case.verify() != pin or cases != frozen_cases:
                raise ValueError("BLIND_INPUT_MUTATED_BY_BACKEND")
            cells.append(cell)
            _write_json(root / "trials" / case.case_id / arm / "evaluation_receipt.json", cell)
    output = {"schema_version": "astrograph-material-llm-blind-outcomes.v1",
              "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
              "cells": cells, "gold_labels_loaded": False,
              "diagnostic_accuracy_gain_identifiable": False,
              "human_semantic_ratings": "NOT_COLLECTED",
              "model_real_inference_verified": False,
              "claim_boundary": "These are tool/protocol receipts, not blinded model-quality conclusions."}
    _write_json(root / "outcomes.json", output)
    return output


def score_blind_material_reviews(
    outcomes: dict[str, Any], reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate, after-run evaluator-only scoring; ratings never enter prompts.

    Ratings [0..4] cover semantic entailment, contradiction handling, reasonable
    abstention and actionability. Scores are descriptive, not blinded causal proof.
    """
    cells = outcomes.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("BLIND_OUTCOMES_MISSING")
    keys = {(c["case_id"], c["arm"]) for c in cells}
    if len(keys) != len(cells) or not keys <= {(c["case_id"], arm) for c in cells for arm in ARM_ORDER}:
        raise ValueError("BLIND_OUTCOMES_INVALID")
    measures = ("evidence_entailment", "counterevidence_recognition",
                "abstention_appropriateness", "review_actionability")
    scores = {}
    for review in reviews:
        if set(review) != {"case_id", "arm", *measures}:
            raise ValueError("INVALID_BLINDED_REVIEW_FIELDS")
        key = (review["case_id"], review["arm"])
        if key not in keys or key in scores:
            raise ValueError("DUPLICATE_OR_FOREIGN_BLIND_REVIEW")
        if any(type(review[name]) is not int or not 0 <= review[name] <= 4 for name in measures):
            raise ValueError("INVALID_BLIND_REVIEW_RATING")
        scores[key] = {name: review[name] for name in measures}
    if set(scores) != keys:
        raise ValueError("BLIND_REVIEWS_INCOMPLETE")
    by_arm = {}
    for arm in ARM_ORDER:
        subset = [score for (case_id, name), score in scores.items() if name == arm]
        by_arm[arm] = {metric: round(sum(r[metric] for r in subset) / len(subset), 4)
                       for metric in measures}
    return {"schema_version": "astrograph-material-llm-blind-human-scoring.v1",
            "arms": by_arm, "rated_cells": len(scores),
            "diagnostic_accuracy_gain": None,
            "diagnostic_accuracy_gain_reason": "FROZEN_FAULT_CANDIDATES_IDENTICAL_ACROSS_ARMS",
            "rating_origin": "EXTERNAL_AFTER_RUN_HUMAN_ADJUDICATION_REQUIRED",
            "statistical_superiority_proven": False}
