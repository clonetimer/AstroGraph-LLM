"""Validated, non-voting candidate revisions from an LLM investigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.investigation.diagnostic_probes import DiagnosticProbeKind
if TYPE_CHECKING:
    from astrograph.orchestration.llm_diagnostic_loop import LLMControlledDiagnosticTrajectory


class ShadowCandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_fault_ids: list[str] = Field(default_factory=list, max_length=16)
    abstain_recommended: bool
    cited_observation_ids: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def unique_ids(self) -> "ShadowCandidateProposal":
        if len(self.candidate_fault_ids) != len(set(self.candidate_fault_ids)):
            raise ValueError("shadow candidate fault IDs must be unique")
        if len(self.cited_observation_ids) != len(set(self.cited_observation_ids)):
            raise ValueError("shadow observation IDs must be unique")
        return self


class ShadowCandidateRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-shadow-candidate-revision.v1"
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    reason_code: str
    baseline_candidate_fault_ids: list[str] = Field(default_factory=list)
    revised_candidate_fault_ids: list[str] = Field(default_factory=list)
    abstain_recommended: bool | None = None
    cited_observation_ids: list[str] = Field(default_factory=list)
    active_telemetry_observation_ids: list[str] = Field(default_factory=list)
    candidate_relevant_telemetry_observation_ids: list[str] = Field(default_factory=list)
    candidate_set_changed: bool = False
    active_telemetry_attributable: bool = False
    claim_boundary: str = (
        "An exploratory shadow candidate revision only. It is not an EvidenceFact, "
        "formal hypothesis status, release decision, or causal benefit claim."
    )


def validate_shadow_candidate_revision(
    *, raw_proposal: Any, trajectory: LLMControlledDiagnosticTrajectory | None,
    registered_fault_ids: set[str], baseline_candidate_fault_ids: list[str] | None,
) -> ShadowCandidateRevision:
    """Fail the shadow artifact closed without interrupting formal reasoning."""
    baseline = (
        list(baseline_candidate_fault_ids)
        if isinstance(baseline_candidate_fault_ids, list) else []
    )

    def unavailable(reason: str) -> ShadowCandidateRevision:
        return ShadowCandidateRevision(
            status="UNAVAILABLE", reason_code=reason, baseline_candidate_fault_ids=baseline,
        )

    if trajectory is None or not trajectory.final_synthesis_consumes_trajectory:
        return unavailable("NO_COMPLETED_DIAGNOSTIC_TRAJECTORY")
    if not baseline or len(baseline) != len(set(baseline)) or not set(baseline) <= registered_fault_ids:
        return unavailable("BASELINE_CANDIDATES_UNAVAILABLE")
    try:
        proposal = ShadowCandidateProposal.model_validate(raw_proposal)
    except (ValueError, TypeError):
        return unavailable("INVALID_SHADOW_PROPOSAL")
    if not set(proposal.candidate_fault_ids) <= registered_fault_ids:
        return unavailable("UNREGISTERED_SHADOW_FAULT_ID")
    observation_by_id = {
        step.observation.observation_id: step
        for step in trajectory.steps if step.observation is not None
    }
    if not set(proposal.cited_observation_ids) <= set(observation_by_id):
        return unavailable("UNKNOWN_SHADOW_OBSERVATION_ID")
    active_ids = sorted(
        observation_id for observation_id in proposal.cited_observation_ids
        if (
            observation_by_id[observation_id].proposal.capability_id == "execute_diagnostic_probe"
            and observation_by_id[observation_id].observation is not None
            and observation_by_id[observation_id].observation.content.get("kind")
            == DiagnosticProbeKind.PRE_POST_CHANGE.value
            and observation_by_id[observation_id].observation.content.get("creates_new_raw_observation") is True
            and observation_by_id[observation_id].observation.content.get("formal_evidence_generated") is False
        )
    )
    relevant_active_ids = sorted(
        observation_id for observation_id in active_ids
        if isinstance(
            observation_by_id[observation_id].observation.content.get("target_fault_ids"),
            (list, tuple),
        )
        and all(
            isinstance(fault_id, str)
            for fault_id in observation_by_id[observation_id].observation.content["target_fault_ids"]
        )
        and set(
            observation_by_id[observation_id].observation.content["target_fault_ids"]
        ) & set(baseline)
    )
    revised = sorted(proposal.candidate_fault_ids)
    changed = set(revised) != set(baseline)
    return ShadowCandidateRevision(
        status="AVAILABLE", reason_code=(
            "ACTIVE_PROBE_OUTSIDE_BASELINE_CANDIDATES"
            if changed and active_ids and not relevant_active_ids else
            "VALIDATED_SHADOW_REVISION"
        ),
        baseline_candidate_fault_ids=baseline,
        revised_candidate_fault_ids=revised,
        abstain_recommended=proposal.abstain_recommended,
        cited_observation_ids=proposal.cited_observation_ids,
        active_telemetry_observation_ids=active_ids,
        candidate_relevant_telemetry_observation_ids=relevant_active_ids,
        candidate_set_changed=changed,
        active_telemetry_attributable=changed and bool(relevant_active_ids),
    )
