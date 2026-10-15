"""Policy overlays on one diagnosis kernel, never independent verdict engines.

Normal is strictly the legacy host policy. Advanced-mode budgets can only be
changed by the host, not by requests or model responses. There is deliberately
no automatic mode escalation or arbitrary execution of generated proposals.
"""
from __future__ import annotations
from .models import AgentSettings, AgentRequest, ModeBudget
from .storage import sha

MODE_OBJECTIVES = {
    "normal": "Run the established evidence-fusion diagnosis with bounded eligible checks.",
    "thinking": "Investigate competing hypotheses, applicability and counterevidence; stop without new evidence.",
    "creative": "Preserve the validated diagnosis; separately propose falsifiable hypotheses and offline experiments.",
}

TASK_TOOLS = {
    "diagnose": {
        "model.infer", "model.capabilities", "model.execute_compatible", "graph.reason", "diagnosis.fuse",
        "knowledge.graph.mechanism", "knowledge.graph.match_observations",
    },
    "research": set(),
    "build_knowledge": {"graph.build"},
    "train_model": {"model.train"},
    "infer_candidate": {"model.infer_candidate"},
    "replay": {"dataset.replay", "model.capabilities"},
}
COMMON_TOOLS = {"document.convert", "document.clean", "rag.retrieve", "literature.search", "output.structured"}

def allowed_tools(task: str, work_mode: str) -> frozenset[str]:
    tools = COMMON_TOOLS | TASK_TOOLS[task]
    if work_mode == "creative":
        tools = tools | {"exploration.propose"}
    return frozenset(tools)

def resolve_mode(settings: AgentSettings, request: AgentRequest, *, backend_available: bool):
    if request.work_mode not in settings.allowed_work_modes:
        raise PermissionError("WORK_MODE_NOT_AUTHORIZED")
    budget = settings.work_mode_budgets.get(request.work_mode, ModeBudget())
    updates = {k: v for k, v in budget.model_dump().items() if k != "max_proposals" and v is not None}
    # validated reconstruction rather than unvalidated model_copy overrides
    effective = AgentSettings.model_validate({**settings.model_dump(), **updates})
    planner_actual = "llm_assisted" if effective.planner == "llm_assisted" and backend_available and effective.max_llm_calls else "deterministic"
    degradations = []
    if planner_actual != effective.planner:
        degradations.append("LLM_UNAVAILABLE_OR_ZERO_BUDGET: deterministic investigation; no live LLM reasoning claim")
    body = {
        "schema_version": "astrograph-work-mode.v1",
        "requested_mode": request.work_mode, "effective_mode": request.work_mode,
        "display_name": {"normal": "标准诊断", "thinking": "深入复核", "creative": "扩展排查"}[request.work_mode],
        "objective": MODE_OBJECTIVES[request.work_mode],
        "planner_requested": effective.planner, "planner_effective": planner_actual,
        "budgets": {k: getattr(effective, k) for k in ("max_checks", "max_optional_actions", "max_llm_calls")},
        "max_proposals": budget.max_proposals,
        "allowed_task_tools": sorted(allowed_tools(request.task, request.work_mode)),
        "capability_degradations": degradations,
        "shared_diagnostic_kernel": True, "mode_changes_diagnostic_thresholds": False,
        "mode_grants_permissions": False, "automatic_mode_escalation": False,
        "exploration_memory_policy": "TASK_LOCAL_UNVERIFIED_NEVER_CONFIRM_AUTOMATICALLY",
    }
    return effective, {**body, "policy_sha256": sha(body)}

def investigation_brief(diagnosis_result: dict) -> dict:
    d = diagnosis_result["diagnosis"]
    return {
        "status": "EVIDENCE_GROUNDED_INVESTIGATION", "not_a_new_diagnosis": True,
        "competing_fault_ids": d.get("candidate_fault_ids", []),
        "working_hypothesis": d.get("working_hypothesis_fault_id"),
        "candidate_evidence": d.get("candidate_explanations", []),
        "checks": diagnosis_result.get("planning_trace", []),
        "stopping_reason": diagnosis_result.get("stopping_reason"),
        "investigation_policy": "Evidence gaps are not contradictions; repeated sources are not new evidence.",
    }
