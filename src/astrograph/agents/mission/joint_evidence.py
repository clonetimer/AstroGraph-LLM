"""Traceable, non-mutating joint evidence package for Mission diagnosis output."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .storage import sha

Branch = Literal["MODEL", "KG", "LITERATURE", "CORE_DIAGNOSIS"]


class JointEvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    branch: Branch
    role: str = Field(min_length=1)
    fault_id: str | None = None
    state: str = Field(min_length=1)
    source_sha256: str | None = None
    source_locator: str | None = None
    source_ref: str | None = None
    diagnostic_vote_allowed: bool = False
    contributes_to_committed_core: bool = False
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class JointEvidencePackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-joint-evidence-package.v1"] = "astrograph-joint-evidence-package.v1"
    package_id: str = Field(min_length=1)
    core_diagnosis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: list[JointEvidenceNode]
    branch_counts: dict[str, int]
    literature_diagnostic_vote_allowed: Literal[False] = False
    retrieval_relevance_is_stance: Literal[False] = False
    core_diagnosis_modified: Literal[False] = False
    claim_boundary: str = (
        "Traceability package only. MODEL, KG, LITERATURE, and committed CORE_DIAGNOSIS "
        "remain distinct. Retrieved literature does not become diagnostic evidence merely by relevance."
    )

    @model_validator(mode="after")
    def unique_nodes_and_counts(self) -> "JointEvidencePackage":
        ids = [item.evidence_id for item in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("joint evidence IDs must be unique")
        expected = {branch: sum(item.branch == branch for item in self.nodes) for branch in ("MODEL", "KG", "LITERATURE", "CORE_DIAGNOSIS")}
        if self.branch_counts != expected:
            raise ValueError("joint evidence branch counts do not match nodes")
        if any(item.branch == "LITERATURE" and item.diagnostic_vote_allowed for item in self.nodes):
            raise ValueError("literature nodes must remain non-voting")
        return self


def _node_id(branch: str, payload: object) -> tuple[str, str]:
    digest = sha(payload)
    return f"EVID-{branch}-{digest[:20].upper()}", digest


def _append_node(nodes: list[JointEvidenceNode], *, branch: Branch, role: str, state: str,
                 payload: object, fault_id: str | None = None, source_sha256: str | None = None,
                 source_locator: str | None = None, source_ref: str | None = None,
                 diagnostic_vote_allowed: bool = False, contributes_to_committed_core: bool = False) -> None:
    evidence_id, digest = _node_id(branch, payload)
    if any(item.evidence_id == evidence_id for item in nodes):
        return
    nodes.append(JointEvidenceNode(
        evidence_id=evidence_id,
        branch=branch,
        role=role,
        fault_id=fault_id,
        state=state,
        source_sha256=source_sha256,
        source_locator=source_locator,
        source_ref=source_ref,
        diagnostic_vote_allowed=diagnostic_vote_allowed,
        contributes_to_committed_core=contributes_to_committed_core,
        payload_sha256=digest,
    ))


def build_joint_evidence_package(
    *,
    core_result: dict[str, Any],
    model_execution: dict[str, Any] | None,
    generic_result: dict[str, Any] | None,
    literature_context: dict[str, Any] | None,
    sources: list[dict[str, Any]],
) -> JointEvidencePackage:
    nodes: list[JointEvidenceNode] = []
    diagnosis = core_result.get("diagnosis", {}) if isinstance(core_result, dict) else {}
    core_payload = {
        "status": diagnosis.get("status"),
        "selected_fault_id": diagnosis.get("selected_fault_id"),
        "candidate_fault_ids": diagnosis.get("candidate_fault_ids") or [],
        "working_hypothesis_fault_id": diagnosis.get("working_hypothesis_fault_id"),
        "reason_codes": diagnosis.get("reason_codes") or [],
    }
    _append_node(
        nodes,
        branch="CORE_DIAGNOSIS",
        role="COMMITTED_DIAGNOSIS",
        state=str(diagnosis.get("status") or core_result.get("status") or "UNKNOWN"),
        fault_id=diagnosis.get("selected_fault_id") or diagnosis.get("working_hypothesis_fault_id"),
        payload=core_payload,
        source_ref="diagnosis_result",
        contributes_to_committed_core=True,
    )

    execution = model_execution or {}
    three_task = execution.get("three_task_contract") or {}
    for key, role in (("detection", "ANOMALY_DETECTION"), ("localization", "AFFECTED_COMPONENT_LOCALIZATION"), ("classification", "FAULT_CLASSIFICATION")):
        payload = three_task.get(key)
        if not isinstance(payload, dict):
            continue
        fault_id = payload.get("fault_id") or payload.get("class_id") or payload.get("predicted_fault_id")
        _append_node(
            nodes,
            branch="MODEL",
            role=role,
            state=str(payload.get("status") or "EXECUTED"),
            fault_id=fault_id,
            payload=payload,
            source_ref=str(execution.get("provider_id") or execution.get("model_id") or "model_execution"),
            diagnostic_vote_allowed=bool(payload.get("formal_diagnostic_vote_allowed", False)),
        )
    # Preserve an explicit execution-level record even when the provider uses a
    # different three-task schema; this makes absence of normalized heads visible.
    if execution.get("status") and not any(item.branch == "MODEL" for item in nodes):
        payload = {k: execution.get(k) for k in ("status", "provider_id", "model_id", "reason", "decision_eligibility") if k in execution}
        _append_node(nodes, branch="MODEL", role="MODEL_EXECUTION", state=str(execution.get("status")), payload=payload, source_ref="model_execution")

    layered = (generic_result or {}).get("diagnosis", {}) if isinstance(generic_result, dict) else {}
    for index, item in enumerate(layered.get("fault_candidates") or []):
        fault_id = item.get("fault_id")
        if not fault_id:
            continue
        _append_node(
            nodes,
            branch="KG",
            role="PROVISIONAL_CAUSE_CANDIDATE",
            state="PROVISIONAL_E0",
            fault_id=str(fault_id),
            payload={"index": index, **item},
            source_ref="generic_knowledge_reasoning",
            diagnostic_vote_allowed=False,
        )
    for explanation in diagnosis.get("candidate_explanations") or []:
        fault_id = explanation.get("fault_id")
        paths = explanation.get("graph_paths") or []
        if not fault_id or not paths:
            continue
        _append_node(
            nodes,
            branch="KG",
            role="GRAPH_PATH_CONTEXT",
            state=str(explanation.get("knowledge_state") or "UNKNOWN"),
            fault_id=str(fault_id),
            payload={"fault_id": fault_id, "graph_paths": paths},
            source_ref="core_candidate_explanation",
            diagnostic_vote_allowed=False,
        )

    assessments = (literature_context or {}).get("assessments") or []
    sources_by_citation = {str(item.get("citation_id") or item.get("document_id") or ""): item for item in sources}
    for row in assessments:
        citation = str(row.get("citation_id") or "")
        source = sources_by_citation.get(citation, {})
        stance = str(row.get("stance") or "UNASSESSED_CONTEXT")
        _append_node(
            nodes,
            branch="LITERATURE",
            role="REVIEWED_STANCE" if row.get("reviewed_mapping") else "RETRIEVED_CONTEXT",
            state=stance,
            fault_id=row.get("target_fault_id"),
            payload={
                "citation_id": citation,
                "stance": stance,
                "reviewed_mapping": bool(row.get("reviewed_mapping")),
                "reviewed_claim_id": source.get("reviewed_claim_id"),
                "reviewed_mapping_id": source.get("reviewed_mapping_id"),
            },
            source_sha256=row.get("source_sha256"),
            source_locator=row.get("source_locator"),
            source_ref=citation or None,
            diagnostic_vote_allowed=False,
        )

    counts = {branch: sum(item.branch == branch for item in nodes) for branch in ("MODEL", "KG", "LITERATURE", "CORE_DIAGNOSIS")}
    core_sha = sha(core_result)
    package_payload = {
        "core_diagnosis_sha256": core_sha,
        "nodes": [item.model_dump(mode="json") for item in nodes],
        "branch_counts": counts,
    }
    return JointEvidencePackage(
        package_id="JEP-" + sha(package_payload)[:20].upper(),
        core_diagnosis_sha256=core_sha,
        nodes=nodes,
        branch_counts=counts,
    )


def joint_evidence_report(package: JointEvidencePackage) -> dict[str, Any]:
    reviewed = [item for item in package.nodes if item.branch == "LITERATURE" and item.role == "REVIEWED_STANCE"]
    unassessed = [item for item in package.nodes if item.branch == "LITERATURE" and item.role == "RETRIEVED_CONTEXT"]
    feedback_catalog=[{
        "evidence_id":item.evidence_id,"branch":item.branch,"role":item.role,"state":item.state,
        "fault_id":item.fault_id,"source_ref":item.source_ref
    } for item in package.nodes]
    return {
        "schema_version": "astrograph-joint-evidence-report.v1",
        "package_id": package.package_id,
        "branch_counts": dict(package.branch_counts),
        "reviewed_literature_stance_count": len(reviewed),
        "unassessed_literature_context_count": len(unassessed),
        "core_diagnosis_sha256": package.core_diagnosis_sha256,
        "core_diagnosis_modified": False,
        "literature_diagnostic_vote_allowed": False,
        "expert_feedback_reference_catalog":feedback_catalog,
        "claim_boundary": package.claim_boundary,
    }


__all__ = ["JointEvidenceNode", "JointEvidencePackage", "build_joint_evidence_package", "joint_evidence_report"]
