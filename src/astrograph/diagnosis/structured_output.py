"""Final, deterministic synthesis of all online diagnosis stages."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _retrieval_trace(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Retain IDs/hashes/topology without copying untrusted retrieved prose."""

    hit_refs = [
        {
            key: item.get(key)
            for key in (
                "chunk_id", "source_id", "source_uri", "source_sha256", "chunk_sha256",
                "knowledge_version", "rank", "lane", "section_id", "section_path",
                "modalities", "source_locators", "structure_sha256",
            )
            if item.get(key) is not None
        }
        for item in retrieval.get("local_hits", [])
    ]
    path_refs: list[dict[str, Any]] = []
    for path in retrieval.get("ontology_paths", []):
        current = path.get("seed_concept_id")
        hops: list[dict[str, Any]] = []
        for hop in path.get("hops", []):
            if hop.get("source_concept_id") != current:
                raise ValueError("ontology retrieval path is not contiguous")
            hops.append({
                key: hop.get(key)
                for key in (
                    "relation_id", "source_concept_id", "relation_type",
                    "target_concept_id", "source_id", "source_sha256", "knowledge_version",
                    "source_fragment_ids", "epistemic_status", "association_semantics",
                    "retrieval_only", "diagnostic_evidence_allowed", "runtime_vote_allowed",
                )
                if hop.get(key) is not None
            })
            current = hop.get("target_concept_id")
        if hops and current != path.get("target_concept_id"):
            raise ValueError("ontology retrieval path target does not match its final hop")
        path_refs.append({
            "seed_concept_id": path.get("seed_concept_id"),
            "target_concept_id": path.get("target_concept_id"),
            "hops": hops,
            "lane": path.get("lane", "ONTOLOGY"),
        })
    return {
        "local_hit_references": hit_refs,
        "ontology_path_references": path_refs,
        "ontology_seed_concept_ids": retrieval.get("ontology_seed_concept_ids", []),
        "ontology_asset_audits": retrieval.get("ontology_asset_audits", []),
        "document_graph_trace_links": retrieval.get("document_graph_trace_links", []),
        "document_evidence_claims": retrieval.get("document_evidence_claims", []),
        "document_claim_admissions": retrieval.get("document_claim_admissions", []),
        "access_filtered_chunk_count": retrieval.get("access_filtered_chunk_count", 0),
        "governance_filtered_chunk_count": retrieval.get(
            "governance_filtered_chunk_count", 0
        ),
        "document_access_reason_counts": retrieval.get(
            "document_access_reason_counts", {}
        ),
        "access_filtered_ontology_item_count": retrieval.get(
            "access_filtered_ontology_item_count", 0
        ),
        "ontology_paths_truncated": retrieval.get("ontology_paths_truncated", False),
        "retrieval_reason_codes": retrieval.get("reason_codes", []),
        "retrieval_claim_boundary": retrieval.get("claim_boundary"),
        "path_use_attribution": "CONTEXT_AVAILABLE_NOT_CAUSAL_ATTRIBUTION",
        "diagnostic_effect_claimed": False,
    }


def _document_claim_shadow_trace(shadow: dict[str, Any]) -> dict[str, Any]:
    """Retain governed document-graph outcome without copying source prose or paths."""

    if not shadow:
        return {
            "status": "NOT_CONFIGURED",
            "execution_mode": "SHADOW_ONLY",
            "candidate_fusion_modified": False,
            "reason_codes": [],
        }
    if shadow.get("execution_mode", "SHADOW_ONLY") != "SHADOW_ONLY":
        raise ValueError("document claim online result must remain SHADOW_ONLY")
    if shadow.get("candidate_fusion_modified", False):
        raise ValueError("document claim shadow must not modify candidate fusion")
    build = shadow.get("ranking_build") or {}
    ranking = build.get("ranking") or {}
    candidates = ranking.get("candidates") or []
    contributing_fault_ids = sorted(
        str(item["fault_id"])
        for item in candidates
        if item.get("contributes_to_fusion", False)
    )
    return {
        "status": shadow.get("status", "NOT_CONFIGURED"),
        "execution_mode": "SHADOW_ONLY",
        "claim_trace_sha256": shadow.get("claim_trace_sha256"),
        "mapping_bundle_sha256": shadow.get("mapping_bundle_sha256"),
        "candidate_fusion_modified": False,
        "reason_codes": shadow.get("reason_codes", []),
        "ranking_id": ranking.get("ranking_id"),
        "ranking_source": ranking.get("source"),
        "consumed_claim_ids": build.get("consumed_claim_ids", []),
        "consumed_mapping_ids": build.get("consumed_mapping_ids", []),
        "ignored_ineligible_claim_count": len(
            build.get("ignored_ineligible_claim_ids", [])
        ),
        "contributing_fault_ids": contributing_fault_ids,
        "kg_fault_coverage": ranking.get("kg_fault_coverage"),
        "diagnostic_effect_claimed": False,
    }


class StructuredDiagnosisOutput(BaseModel):
    """Machine-readable final answer produced after every diagnostic gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-structured-diagnosis-output.v1"
    output_id: str = Field(min_length=1)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    final_decision: dict[str, Any]
    evidence_summary: dict[str, Any]
    candidate_summary: dict[str, Any]
    uncertainty_and_soft_graph_summary: dict[str, Any]
    knowledge_and_retrieval_summary: dict[str, Any]
    hypothesis_summary: dict[str, Any]
    safety_and_release_summary: dict[str, Any]
    verification_summary: dict[str, Any]
    extension_qualification_summary: dict[str, Any]
    execution_context_summary: dict[str, Any]
    spatiotemporal_summary: dict[str, Any]
    trace_and_artifact_summary: dict[str, Any]
    lineage_sha256: dict[str, str]
    claim_boundary: str = (
        "Deterministic shadow-mode synthesis of upstream artifacts. Retrieved text and experience "
        "remain context, not episode evidence; only the governor and release gate determine disposition."
    )

    @model_validator(mode="after")
    def validate_identity(self) -> "StructuredDiagnosisOutput":
        expected = self.compute_sha256()
        if self.output_sha256 != expected:
            raise ValueError("structured diagnosis output_sha256 mismatch")
        if self.output_id != "STRUCTURED-" + expected[:20].upper():
            raise ValueError("structured diagnosis output_id mismatch")
        return self

    def compute_sha256(self) -> str:
        return _hash(self.model_dump(mode="json", exclude={"output_id", "output_sha256"}))


def build_structured_diagnosis_output(payload: dict[str, Any]) -> StructuredDiagnosisOutput:
    evidence = payload["episode_evidence_bundle"]
    facts = evidence.get("evidence_facts", [])
    hypotheses = payload["hypothesis_set"].get("hypotheses", [])
    fusion = payload.get("candidate_fusion") or {}
    birrf = fusion.get("birrf_result") or {}
    soft_evidence = payload.get("soft_evidence") or {}
    soft_graph = payload.get("soft_graph_shadow") or {}
    retrieval = payload.get("pre_reasoning_retrieval") or {}
    document_claim_shadow = payload.get("document_claim_shadow") or {}
    decision = payload["decision_record"]
    policy = payload.get("diagnosis_policy_decision") or {}
    critique = payload.get("critique_result") or {}
    core = payload.get("deterministic_core_result") or {}
    verification = payload.get("verification_plan") or {}
    simulation = payload["simulation_event"]
    episode = evidence.get("episode") or {}

    lineage_keys = [
        "dataset_profile_receipt", "diagnosis_execution_plan", "simulation_event",
        "data_gate_report", "episode_evidence_bundle", "soft_evidence",
        "data_driven_prediction", "candidate_fusion", "pre_reasoning_retrieval",
        "document_claim_shadow",
        "soft_graph_shadow", "component_spatiotemporal_fault_graph",
        "spatiotemporal_template_matches", "paperseek_fallback_decision",
        "hypothesis_set", "critique_result", "deterministic_core_result",
        "decision_record", "diagnosis_evidence_bundle", "diagnosis_policy_decision",
        "episode_evidence_graph", "verification_plan", "dor_semantic_audit",
        "dor_control_plane_audit", "dor_multitask_audit", "dor_vertical_freeze_audit",
        "fec_evidence_audit", "isgn_routing_audit", "isgn_observation_audit",
        "agent_trace", "llm_runtime_snapshot", "reasoning_activation",
        "llm_diagnostic_trajectory",
    ]
    lineage = {key: _hash(payload[key]) for key in lineage_keys if payload.get(key) is not None}
    status_counts: dict[str, int] = {}
    for fact in facts:
        status = str(fact.get("status", "UNKNOWN"))
        status_counts[status] = status_counts.get(status, 0) + 1
    hypothesis_by_status: dict[str, list[str]] = {}
    for item in hypotheses:
        hypothesis_by_status.setdefault(str(item.get("status", "UNKNOWN")), []).append(
            str(item.get("target_fault_id") or item.get("hypothesis_id"))
        )

    body = {
        "run_id": str(simulation["run_id"]),
        "episode_id": str(episode.get("episode_id") or decision["episode_id"]),
        "final_decision": {
            "disposition": decision["disposition"],
            "accepted_fault_id": decision.get("accepted_fault_id"),
            "accepted_hypothesis_id": decision.get("accepted_hypothesis_id"),
            "formal_release_allowed": decision.get("formal_release_allowed", False),
            "requires_human_review": decision.get("requires_human_review", True),
            "comparison_status": decision.get("comparison_status", "NOT_COMPARED"),
            "reasons": decision.get("reasons", []),
            "unresolved_gaps": decision.get("unresolved_gaps", []),
        },
        "evidence_summary": {
            "data_gate_passed": payload["data_gate_report"].get("passed", False),
            "evidence_fact_count": len(facts),
            "fact_status_counts": status_counts,
            "satisfied_fact_ids": [item.get("evidence_fact_id") for item in facts if item.get("status") == "SATISFIED"],
            "contradicted_fact_ids": [item.get("evidence_fact_id") for item in facts if item.get("status") == "CONTRADICTED"],
        },
        "candidate_summary": {
            "fusion_enabled": bool(fusion),
            "top_fault_id": birrf.get("top_fault_id"),
            "candidates": birrf.get("candidates", []),
            "conflict_flags": birrf.get("conflict_flags", []),
            "open_set_assessment": (birrf.get("metadata") or {}).get("open_set_assessment"),
            "data_driven_prediction": payload.get("data_driven_prediction"),
        },
        "uncertainty_and_soft_graph_summary": {
            "soft_evidence_configured": bool(soft_evidence),
            "soft_evidence_bundle_sha256": soft_evidence.get("bundle_sha256"),
            "soft_calibration_policy_id": soft_evidence.get("calibration_policy_id"),
            "soft_calibration_policy_sha256": soft_evidence.get("calibration_policy_sha256"),
            "calibrated_fact_count": soft_evidence.get("calibrated_fact_count", 0),
            "unavailable_fact_count": soft_evidence.get("unavailable_fact_count", 0),
            "soft_graph_inference_id": soft_graph.get("inference_id"),
            "soft_graph_result_sha256": soft_graph.get("result_sha256"),
            "suggested_action": soft_graph.get("suggested_action", "NOT_CONFIGURED"),
            "candidate_fault_ids": soft_graph.get("candidate_fault_ids", []),
            "reason_codes": soft_graph.get("reason_codes", []),
            "runtime_effect_applied": soft_graph.get("runtime_effect_applied", False),
            "fault_assessments": soft_graph.get("fault_assessments", []),
        },
        "knowledge_and_retrieval_summary": {
            "kg_reference": (payload.get("diagnosis_evidence_bundle") or {}).get("kg_evidence"),
            "retrieval_status": retrieval.get("status", "NOT_CONFIGURED"),
            "retrieval_package_id": retrieval.get("package_id"),
            "local_hit_count": len(retrieval.get("local_hits", [])),
            "ontology_path_count": len(retrieval.get("ontology_paths", [])),
            "experience_hit_count": retrieval.get("experience_hit_count", 0),
            "experience_source_ids": retrieval.get("experience_source_ids", []),
            "experience_reuse_receipt": retrieval.get("experience_reuse_receipt"),
            "knowledge_update_candidates": retrieval.get("knowledge_update_candidates", []),
            "active_graph_modified": retrieval.get("active_graph_modified", False),
            "document_graph_diagnosis_shadow": _document_claim_shadow_trace(
                document_claim_shadow
            ),
            **_retrieval_trace(retrieval),
        },
        "hypothesis_summary": {
            "count": len(hypotheses),
            "by_status": hypothesis_by_status,
            "validation_metadata": payload["hypothesis_set"].get("validation_metadata", {}),
        },
        "safety_and_release_summary": {
            "critique_blocking": critique.get("blocking", False),
            "critique_findings": critique.get("findings", []),
            "deterministic_core_result": core,
            "policy_status": policy.get("status", "NOT_EVALUATED"),
            "policy_blocking_reasons": policy.get("blocking_reasons", []),
            "policy_warnings": policy.get("warnings", []),
        },
        "verification_summary": {
            "verification_plan_id": verification.get("verification_plan_id"),
            "task_count": len(verification.get("tasks", [])),
            "tasks": verification.get("tasks", []),
        },
        "extension_qualification_summary": {
            key: (payload.get(key) or {}).get("status", "NOT_CONFIGURED")
            for key in (
                "dor_semantic_audit", "dor_control_plane_audit", "dor_multitask_audit",
                "dor_vertical_freeze_audit", "fec_evidence_audit", "isgn_routing_audit",
                "isgn_observation_audit",
            )
        },
        "execution_context_summary": {
            "dataset_profile_receipt": payload.get("dataset_profile_receipt"),
            "diagnosis_execution_plan": payload.get("diagnosis_execution_plan"),
            "reasoner_mode": payload.get("reasoner_mode"),
            "model_id": payload.get("model_id"),
            "workflow_version": payload.get("workflow_version"),
            "llm_runtime_snapshot": payload.get("llm_runtime_snapshot"),
            "reasoning_activation": payload.get("reasoning_activation"),
            "paperseek_fallback_decision": payload.get("paperseek_fallback_decision"),
            "llm_diagnostic_trajectory": payload.get("llm_diagnostic_trajectory"),
        },
        "spatiotemporal_summary": {
            "component_spatiotemporal_fault_graph": payload.get(
                "component_spatiotemporal_fault_graph"
            ),
            "spatiotemporal_template_matches": payload.get(
                "spatiotemporal_template_matches", []
            ),
        },
        "trace_and_artifact_summary": {
            "agent_trace": payload.get("agent_trace"),
            "runtime_stage_artifact_sha256": {
                stage_id: {
                    "content_sha256": artifact.get("content_sha256"),
                    "semantic_sha256": artifact.get("semantic_sha256"),
                }
                for stage_id, artifact in sorted(
                    (payload.get("runtime_stage_artifacts") or {}).items()
                )
            },
        },
        "lineage_sha256": lineage,
    }
    identity_payload = {
        "schema_version": "astrograph-structured-diagnosis-output.v1",
        **body,
        "claim_boundary": StructuredDiagnosisOutput.model_fields["claim_boundary"].default,
    }
    output_sha256 = _hash(identity_payload)
    return StructuredDiagnosisOutput(
        output_id="STRUCTURED-" + output_sha256[:20].upper(),
        output_sha256=output_sha256,
        **body,
    )


__all__ = ["StructuredDiagnosisOutput", "build_structured_diagnosis_output"]
