"""Deterministic activation boundary between diagnosis algorithms and an LLM advisor."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import EpisodeEvidenceBundle, EvidenceFactStatus


class RealtimeReasoningActivationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-realtime-reasoning-activation-policy.v1"
    mode: Literal["ALGORITHM_EVENT_GATED", "ALWAYS", "NEVER"] = "ALGORITHM_EVENT_GATED"
    activate_on_supported_evidence: bool = True
    activate_on_incomplete_evidence: bool = True
    activate_on_candidate_fusion: bool = True
    activate_on_non_nominal_core: bool = True
    deterministic_fallback_on_llm_error: bool = True


class ReasoningActivationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reasoning-activation-decision.v1"
    decision_id: str
    requested_reasoner_mode: Literal["rule", "llm", "hybrid"]
    activation_policy_mode: Literal["ALGORITHM_EVENT_GATED", "ALWAYS", "NEVER"]
    llm_invocation_allowed: bool
    llm_invoked: bool
    operator_requested: bool
    trigger_codes: list[str] = Field(default_factory=list)
    deterministic_path_completed_first: bool = True
    hard_realtime_path_blocked_by_llm: Literal[False] = False
    llm_has_direct_algorithm_control: Literal[False] = False
    claim_boundary: str = (
        "This is an advisory soft-real-time reasoning branch. The deterministic diagnosis path remains "
        "authoritative and does not wait for an LLM to execute control or recovery actions."
    )


def decide_reasoning_activation(
    *, request_mode: Literal["rule", "llm", "hybrid"],
    policy: RealtimeReasoningActivationPolicy,
    bundle: EpisodeEvidenceBundle,
    deterministic_core_result: dict[str, Any],
    candidate_fusion_available: bool,
    operator_requested: bool,
) -> ReasoningActivationDecision:
    triggers: list[str] = []
    statuses = {fact.status for fact in bundle.evidence_facts}
    if operator_requested:
        triggers.append("OPERATOR_REQUEST")
    if policy.activate_on_supported_evidence and EvidenceFactStatus.SATISFIED in statuses:
        triggers.append("SUPPORTED_EVIDENCE_EVENT")
    if policy.activate_on_incomplete_evidence and statuses.intersection(
        {EvidenceFactStatus.MISSING_DATA, EvidenceFactStatus.INVALID_DATA, EvidenceFactStatus.UNKNOWN}
    ):
        triggers.append("EVIDENCE_INCOMPLETE")
    if policy.activate_on_candidate_fusion and candidate_fusion_available:
        triggers.append("CANDIDATE_FUSION_AVAILABLE")
    core_status = str(deterministic_core_result.get("diagnosis_status", "UNKNOWN")).upper()
    if policy.activate_on_non_nominal_core and core_status not in {"NO_FAULT", "NOMINAL", "HEALTHY"}:
        triggers.append(f"CORE_{core_status}")
    allowed = request_mode in {"llm", "hybrid"} and policy.mode != "NEVER"
    invoked = allowed and (policy.mode == "ALWAYS" or bool(triggers))
    stable = {
        "episode_id": bundle.episode.episode_id, "request_mode": request_mode,
        "policy": policy.model_dump(mode="json"), "triggers": sorted(set(triggers)), "invoked": invoked,
    }
    decision_id = "RAD-" + hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20].upper()
    return ReasoningActivationDecision(
        decision_id=decision_id, requested_reasoner_mode=request_mode,
        activation_policy_mode=policy.mode, llm_invocation_allowed=allowed, llm_invoked=invoked,
        operator_requested=operator_requested, trigger_codes=sorted(set(triggers)),
    )


__all__ = ["RealtimeReasoningActivationPolicy", "ReasoningActivationDecision", "decide_reasoning_activation"]
