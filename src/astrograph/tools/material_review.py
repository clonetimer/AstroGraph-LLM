"""Task-bound, read-only LLM investigation tools for an already frozen material review.

The model chooses IDs, never paths, snapshots, edges, scopes, or arbitrary queries.
Every returned verdict is an extract of deterministic R7--R10 results calculated
against the host-approved source snapshot.  These tools cannot diagnose or vote.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.orchestration.models import PermissionScope, SideEffectLevel, WorkflowContext
from astrograph.tools.registry import ToolDefinition, ToolRegistry, ToolSpec

VERSION = "material-review-tools.r11.1"
TOOL_IDS = (
    "material.rag.compare_hypotheses",
    "material.graph.search_causal_paths",
    "material.graph.check_context",
    "material.graph.check_temporal",
)


class MaterialToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_id: Literal[
        "material.rag.compare_hypotheses",
        "material.graph.search_causal_paths",
        "material.graph.check_context",
        "material.graph.check_temporal",
    ]
    fault_id: str | None = Field(default=None, max_length=160)
    path_id: str | None = Field(default=None, max_length=160)


class MaterialToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[MaterialToolAction] = Field(default_factory=list, max_length=3)


def _safe_str(value: Any, limit: int = 180) -> str:
    return str(value or "")[:limit]


def _bounded_chains(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    review = (candidate.get("multi_hop") or {}).get("causal_chain_review") or {}
    rows = []
    for chain in (review.get("chains") or [])[:12]:
        rows.append({
            "path_id": chain.get("path_id"),
            "hop_count": chain.get("hop_count"),
            "terminal_symptom_id": chain.get("terminal_symptom_id"),
            "reviewed_context_consistency": deepcopy(chain.get("reviewed_context_consistency") or {}),
            "reviewed_temporal_consistency": deepcopy(chain.get("reviewed_temporal_consistency") or {}),
            "joint_applicability_status": chain.get("joint_applicability_status"),
            "cross_document_mechanism_splice": bool(chain.get("cross_document_mechanism_splice")),
            "hops": [{
                "edge_id": hop.get("edge_id"), "direction": hop.get("direction"),
                "source_name": _safe_str(hop.get("source_name")),
                "target_name": _safe_str(hop.get("target_name")),
                "source_sha256": hop.get("source_sha256"),
                "source_locators": list(hop.get("source_locators") or [])[:4],
                "reviewer": _safe_str(hop.get("reviewer")),
                "truth_source_ref": _safe_str(hop.get("truth_source_ref")),
                "evidence_quote": _safe_str(hop.get("evidence_quote"), 280),
            } for hop in (chain.get("hops") or [])[:4]],
        })
    return rows


def build_material_review_tool_registry(material_review: dict[str, Any]) -> ToolRegistry:
    """Create a sealed allow-list tied to the current host-frozen candidates.

    Neither the model nor a tool argument may supply new reference material,
    graph edges, equipment scope, approval snapshot, or a filesystem location.
    """
    candidates = list(material_review.get("candidates") or [])[:12]
    valid = {}
    for item in candidates:
        fid = item.get("fault_id")
        if isinstance(fid, str) and fid and fid not in valid:
            valid[fid] = {
                "fault_id": fid,
                "reference_retrieval": {
                    "status": (item.get("reference_retrieval") or {}).get("status"),
                    "hits": [{"knowledge_id": h.get("knowledge_id"),
                              "quote": _safe_str(h.get("quote"), 420),
                              "source_sha256": h.get("source_sha256"),
                              "knowledge_validation_status": h.get("knowledge_validation_status")}
                             for h in ((item.get("reference_retrieval") or {}).get("hits") or [])[:5]],
                    "reviewed_counterevidence": deepcopy((item.get("reference_retrieval") or {}).get("reviewed_counterevidence") or {}),
                },
                "bidirectional_graph_review": deepcopy(item.get("bidirectional_graph_review") or {}),
                "contrastive_reference_review": deepcopy(item.get("contrastive_reference_review") or {}),
                "chains": _bounded_chains(item),
                "blocked": deepcopy(((item.get("multi_hop") or {}).get("causal_chain_review") or {}).get("blocked") or [])[:12],
            }
    comparison = deepcopy(material_review.get("competing_hypothesis_review") or {})
    registry = ToolRegistry()

    def require_fault(payload):
        fid = payload.get("fault_id")
        if fid not in valid:
            raise ValueError("FAULT_ID_NOT_IN_FROZEN_CANDIDATES")
        return valid[fid]

    def require_path(payload):
        candidate = require_fault(payload)
        path_id = payload.get("path_id")
        paths = [item for item in candidate["chains"] if item["path_id"] == path_id]
        if not paths or not path_id:
            raise ValueError("PATH_ID_NOT_IN_FROZEN_REVIEW")
        return paths[0]

    def compare(payload):
        return {"status": "FROZEN_HOST_CANDIDATE_COMPARISON", "comparison": deepcopy(comparison),
                "candidate_ids": sorted(valid), "diagnostic_vote_allowed": False,
                "current_root_cause_proven": False}

    def search(payload):
        row = require_fault(payload)
        return {"status": "REVIEWED_REFERENCE_CHAINS_NOT_FLIGHT_CAUSATION",
                "fault_id": row["fault_id"], "chains": deepcopy(row["chains"][:6]),
                "blocked": deepcopy(row["blocked"][:8]),
                "reference_retrieval": deepcopy(row["reference_retrieval"]),
                "bidirectional_graph_review": deepcopy(row["bidirectional_graph_review"]),
                "contrastive_reference_review": deepcopy(row["contrastive_reference_review"]),
                "diagnostic_vote_allowed": False, "current_root_cause_proven": False}

    def context(payload):
        row = require_path(payload)
        return {"path_id": row["path_id"], "fault_id": payload["fault_id"],
                "reviewed_context_consistency": deepcopy(row["reviewed_context_consistency"]),
                "joint_applicability_status": row["joint_applicability_status"],
                "observed_flight_conditions_verified": False, "current_root_cause_proven": False}

    def temporal(payload):
        row = require_path(payload)
        return {"path_id": row["path_id"], "fault_id": payload["fault_id"],
                "reviewed_temporal_consistency": deepcopy(row["reviewed_temporal_consistency"]),
                "actual_telemetry_event_alignment_verified": False, "current_root_cause_proven": False}

    handlers = (compare, search, context, temporal)
    for tool_id, handler in zip(TOOL_IDS, handlers):
        properties = {} if tool_id == TOOL_IDS[0] else {"fault_id": {"type": "string", "minLength": 1}}
        required = [] if tool_id == TOOL_IDS[0] else ["fault_id"]
        if tool_id in TOOL_IDS[2:]:
            properties["path_id"] = {"type": "string", "minLength": 1}
            required.append("path_id")
        registry.register(ToolDefinition(
            spec=ToolSpec(tool_id=tool_id, tool_version=VERSION,
                          input_schema={"type":"object", "properties":properties,
                                        "required":required, "additionalProperties":False},
                          output_schema={"type":"object"}, deterministic=True,
                          side_effect_level=SideEffectLevel.READ_ONLY,
                          required_permissions=[PermissionScope.READ_KNOWLEDGE],
                          timeout_s=5.0, max_output_bytes=750_000),
            handler=handler))
    registry.seal()
    return registry


def summarize_material_causal_review(material_review: dict[str, Any], *, max_candidates: int = 5) -> dict[str, Any]:
    """Bounded and already-validated causal proof ledger for the first LLM pass."""
    return {"status": "REVIEWED_CAUSAL_REFERENCE_ONLY_NOT_OBSERVED_ROOT_CAUSE",
            "candidate_ids": [row.get("fault_id") for row in (material_review.get("candidates") or [])[:max_candidates]],
            "candidates": [{
                "fault_id": candidate.get("fault_id"),
                "chains": [{"path_id": row.get("path_id"),
                            "terminal_symptom_id": row.get("terminal_symptom_id"),
                            "hop_edge_ids": [hop.get("edge_id") for hop in (row.get("hops") or [])[:4]],
                            "reviewed_context_consistency": row.get("reviewed_context_consistency"),
                            "reviewed_temporal_consistency": row.get("reviewed_temporal_consistency"),
                            "joint_applicability_status": row.get("joint_applicability_status"),
                            "joint_applicability_verified": False,
                            "current_root_cause_proven": False}
                           for row in ((candidate.get("multi_hop") or {}).get("causal_chain_review") or {}).get("chains", [])[:4]],
                "blocked": [{"edge_id": b.get("edge_id"),
                             "path_edge_ids": b.get("path_edge_ids"),
                             "reasons": (b.get("reasons") or [])[:8]}
                            for b in ((candidate.get("multi_hop") or {}).get("causal_chain_review") or {}).get("blocked", [])[:6]],
            } for candidate in (material_review.get("candidates") or [])[:max_candidates]],
            "diagnostic_vote_allowed": False, "current_root_cause_proven": False}


def execute_material_tool_plan(*, backend: Any, material_review: dict[str, Any],
                               run_id: str = "material-review") -> dict[str, Any]:
    """Optional two-pass LLM-selected tool protocol, with host-side validation.

    No raw model-selected path or document is ever used. Invalid selections are
    denied rather than supplying fallback files or broadening scope.
    """
    registry = build_material_review_tool_registry(material_review)
    ids = [str(r.get("fault_id")) for r in material_review.get("candidates", [])[:12] if r.get("fault_id")]
    paths = {str(r.get("fault_id")): [str(x.get("path_id")) for x in
             ((r.get("multi_hop") or {}).get("causal_chain_review") or {}).get("chains", [])[:12]]
             for r in material_review.get("candidates", [])[:12] if r.get("fault_id")}
    plan = backend.generate_structured(
        agent_id="platform-knowledge-material-tool-selector", output_model=MaterialToolPlan,
        payload={"task": "Select up to three read-only evidence review tools. Use only given candidate and path IDs. "
                         "Never request diagnosis, graph editing, external files or spacecraft controls. "
                         "When no useful investigation is available, return an empty actions list.",
                 "available_tools": [spec.model_dump(mode="json") for spec in registry.list_specs()],
                 "frozen_candidate_ids": ids, "reviewed_path_ids_by_candidate": paths,
                 "causal_review": summarize_material_causal_review(material_review)})
    plan = MaterialToolPlan.model_validate(plan)
    context = WorkflowContext(workflow_id="material.llm.investigation", run_id=run_id,
                              permissions=[PermissionScope.READ_KNOWLEDGE])
    outputs = []
    for action in plan.actions:
        payload = {}
        if action.tool_id != TOOL_IDS[0]:
            payload["fault_id"] = action.fault_id
        if action.tool_id in TOOL_IDS[2:]:
            payload["path_id"] = action.path_id
        try:
            result = registry.invoke(action.tool_id, payload, context)
            outputs.append({"tool_id": action.tool_id, "status": "SUCCEEDED", "output": result})
        except (ValueError, KeyError, TypeError, PermissionError) as exc:
            outputs.append({"tool_id": action.tool_id, "status": "DENIED",
                            "reason": "ARGUMENT_NOT_IN_FROZEN_REVIEW"})
    return {"status": "COMPLETED", "tool_version": VERSION,
            "calls": outputs, "receipts": registry.executor.receipts(),
            "diagnostic_vote_allowed": False, "current_root_cause_proven": False}
