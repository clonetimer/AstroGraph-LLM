from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from astrograph.services.reasoning import StructuredReasoningBackend
from astrograph.investigation.diagnostic_probes import DiagnosticProbeCatalog, DiagnosticProbeExecutor
from astrograph.investigation.active_telemetry_probes import (
    ActiveTelemetryProbeExecutor,
    is_active_telemetry_probe,
    is_admitted_telemetry_probe,
)
from astrograph.investigation.shadow_candidate_revision import ShadowCandidateRevision


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class DiagnosticLoopStatus(str, Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    BACKEND_UNSUPPORTED = "BACKEND_UNSUPPORTED"


class LLMMethodSubstanceStatus(str, Enum):
    SUBSTANTIVE_METHOD_PATH = "SUBSTANTIVE_METHOD_PATH"
    WRAPPER_RISK = "WRAPPER_RISK"
    NOT_EXECUTED = "NOT_EXECUTED"


class DiagnosticCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str
    description: str
    input_artifact_ids: list[str]
    output_kind: str
    permission: str = "READ_ONLY"


class DiagnosticActionProposal(BaseModel):
    """One model-selected action; free-form tool names and arguments are forbidden."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str
    objective: str = Field(min_length=1, max_length=500)
    cited_observation_ids: list[str] = Field(default_factory=list)
    probe_id: str | None = None


class DiagnosticObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    capability_id: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: dict[str, Any]


class DiagnosticDeliberationStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    proposal: DiagnosticActionProposal
    policy_status: str
    observation: DiagnosticObservation | None = None


class LLMMethodSubstanceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: LLMMethodSubstanceStatus
    llm_selected_action_count: int = Field(ge=0)
    executed_diagnostic_capability_count: int = Field(ge=0)
    observation_conditioned_replan_count: int = Field(ge=0)
    final_synthesis_consumes_trajectory: bool
    independent_governor_required: bool = True
    reason_codes: list[str]
    claim_boundary: str = (
        "This audit establishes structural LLM participation only. Diagnostic benefit requires "
        "preregistered ablation and independent holdout evidence."
    )


class DiagnosticContributionStatus(str, Enum):
    NO_NEW_INFORMATION = "NO_NEW_INFORMATION"
    DERIVED_OBSERVATION_ONLY = "DERIVED_OBSERVATION_ONLY"
    NEW_INFORMATION_NO_DIAGNOSTIC_CHANGE = "NEW_INFORMATION_NO_DIAGNOSTIC_CHANGE"
    EVIDENCE_GAP_REDUCED = "EVIDENCE_GAP_REDUCED"
    CANDIDATE_SET_REFINED = "CANDIDATE_SET_REFINED"
    DECISION_CHANGED = "DECISION_CHANGED"


class DiagnosticContributionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DiagnosticContributionStatus
    probe_execution_count: int = Field(ge=0)
    new_observation_count: int = Field(ge=0)
    new_raw_observation_count: int = Field(ge=0)
    formal_evidence_generated_count: int = Field(ge=0)
    resolved_gap_count: int = Field(default=0, ge=0)
    candidate_set_changed: bool = False
    decision_changed: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "This audit distinguishes structural tool use from diagnostic information gain. Shadow probes derived "
        "from already-admitted evidence are not counted as new raw observations or formal evidence."
    )


class LLMControlledDiagnosticTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-llm-controlled-diagnostic-trajectory.v2"
    episode_id: str
    architecture: str = "ALGORITHM_TRIGGERED_LLM_CONTROLLED_INVESTIGATION"
    status: DiagnosticLoopStatus
    steps: list[DiagnosticDeliberationStep]
    termination_reason: str
    allowed_capability_ids: list[str]
    final_synthesis_consumes_trajectory: bool
    trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    substance_audit: LLMMethodSubstanceAudit
    contribution_audit: DiagnosticContributionAudit

    def compute_sha256(self) -> str:
        return _digest({
            "episode_id": self.episode_id,
            "status": self.status.value,
            "steps": [item.model_dump(mode="json") for item in self.steps],
            "termination_reason": self.termination_reason,
            "allowed_capability_ids": self.allowed_capability_ids,
            "final_synthesis_consumes_trajectory": self.final_synthesis_consumes_trajectory,
            "substance_audit": self.substance_audit.model_dump(mode="json"),
            "contribution_audit": self.contribution_audit.model_dump(mode="json"),
        })

    @model_validator(mode="after")
    def validate_trajectory_hash(self) -> "LLMControlledDiagnosticTrajectory":
        if self.trajectory_sha256 != self.compute_sha256():
            raise ValueError("trajectory_sha256 mismatch")
        return self


class ControlledDiagnosticDeliberationService:
    """Let an LLM choose read-only diagnostic operations inside a fixed safety envelope.

    Detection and emergency handling never wait for this loop. The model cannot name a
    Python callable, mutate runtime state, alter evidence, or bypass the independent governor.
    """

    agent_id = "llm-diagnostic-deliberation-agent"
    SYNTHESIZE = "synthesize_hypotheses"
    _CAPABILITIES = (
        DiagnosticCapability(
            capability_id="inspect_evidence_status",
            description="Summarize executable evidence facts, roles, statuses, and missing items.",
            input_artifact_ids=["online.evidence.v1"],
            output_kind="EVIDENCE_STATUS_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_candidate_fusion",
            description="Inspect the governed data/rule/KG candidate fusion result.",
            input_artifact_ids=["online.fusion.v1"],
            output_kind="FUSION_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_kg_support",
            description="Inspect episode-bound KG evidence and directional associations.",
            input_artifact_ids=["online.kg_evidence.v1"],
            output_kind="KG_SUPPORT_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_retrieval_context",
            description="Inspect admitted pre-generation document and ontology context.",
            input_artifact_ids=["online.reasoning.v1.pre_generation"],
            output_kind="RETRIEVAL_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_experience_context",
            description="Inspect separately approved experience and memory reuse context.",
            input_artifact_ids=["online.reasoning.v1.pre_generation.experience"],
            output_kind="EXPERIENCE_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_deterministic_core",
            description="Inspect the independent deterministic diagnosis and its evidence references.",
            input_artifact_ids=["online.deterministic_core.v1"],
            output_kind="CORE_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_verification_plan",
            description="Inspect unresolved evidence and the deterministic verification plan.",
            input_artifact_ids=["online.verification.v1"],
            output_kind="VERIFICATION_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="inspect_probe_catalog",
            description="Inspect host-approved deterministic diagnostic probes available for this episode.",
            input_artifact_ids=["online.evidence.v1"],
            output_kind="DIAGNOSTIC_PROBE_CATALOG_SUMMARY",
        ),
        DiagnosticCapability(
            capability_id="execute_diagnostic_probe",
            description="Execute exactly one host-bound probe by probe_id; model-defined channels, windows and thresholds are forbidden.",
            input_artifact_ids=["diagnostic.probe.catalog.v1"],
            output_kind="SHADOW_DIAGNOSTIC_PROBE_RESULT",
            permission="DETERMINISTIC_SHADOW_EXECUTE",
        ),
        DiagnosticCapability(
            capability_id=SYNTHESIZE,
            description="Stop investigation and synthesize hypotheses from the recorded trajectory.",
            input_artifact_ids=["llm.diagnostic.trajectory"],
            output_kind="TERMINATION",
        ),
    )

    @staticmethod
    def _constrained_proposal_model(
        *, available_ids: list[str], observation_ids: list[str], citation_required: bool,
        probe_ids: list[str] | None = None,
    ) -> type[DiagnosticActionProposal]:
        """Build a decoder-level allowlist for backends that support JSON-schema guidance."""
        capability_literal = Literal.__getitem__(tuple(available_ids))
        field_definitions: dict[str, Any] = {
            "capability_id": (capability_literal, ...),
        }
        if observation_ids:
            observation_literal = Literal.__getitem__(tuple(observation_ids))
            field_definitions["cited_observation_ids"] = (
                list[observation_literal],
                Field(min_length=1) if citation_required else Field(default_factory=list),
            )
        if probe_ids:
            probe_literal = Literal.__getitem__(tuple(probe_ids))
            field_definitions["probe_id"] = (probe_literal | None, None)
        return create_model(
            "ConstrainedDiagnosticActionProposal",
            __base__=DiagnosticActionProposal,
            **field_definitions,
        )

    @classmethod
    def capabilities(cls) -> list[DiagnosticCapability]:
        return list(cls._CAPABILITIES)

    @staticmethod
    def _summarize(capability_id: str, artifacts: dict[str, Any]) -> dict[str, Any]:
        if capability_id == "inspect_evidence_status":
            bundle = artifacts.get("evidence") or {}
            facts = bundle.get("evidence_facts") or []
            return {
                "episode_id": (bundle.get("episode") or {}).get("episode_id"),
                "fact_count": len(facts),
                "facts": [
                    {
                        "evidence_fact_id": item.get("evidence_fact_id"),
                        "evidence_definition_id": item.get("evidence_definition_id"),
                        "status": item.get("status"),
                        "value": item.get("value"),
                    }
                    for item in facts
                ],
            }
        if capability_id == "inspect_candidate_fusion":
            fusion = artifacts.get("fusion")
            return {"configured": fusion is not None, "candidate_fusion": fusion}
        if capability_id == "inspect_kg_support":
            kg = artifacts.get("kg") or {}
            return {
                "configured": bool(kg.get("kg_evidence")),
                "kg_evidence": kg.get("kg_evidence"),
                "forward_associations": kg.get("forward_associations", []),
                "backward_associations": kg.get("backward_associations", []),
                "degraded_to_reference": kg.get("degraded_to_reference", False),
            }
        if capability_id == "inspect_retrieval_context":
            retrieval = artifacts.get("retrieval")
            if retrieval is None:
                return {"configured": False, "retrieval": None}
            return {
                "configured": True,
                "status": retrieval.get("status"),
                "local_hits": [
                    item for item in retrieval.get("local_hits", [])
                    if item.get("lane") != "EXPERIENCE"
                ],
                "ontology_paths": retrieval.get("ontology_paths", []),
                "matched_concept_ids": retrieval.get("matched_concept_ids", []),
            }
        if capability_id == "inspect_experience_context":
            retrieval = artifacts.get("retrieval")
            if retrieval is None:
                return {"configured": False, "experience_hits": []}
            return {
                "configured": bool(retrieval.get("experience_hit_count", 0)),
                "experience_hits": [
                    item for item in retrieval.get("local_hits", [])
                    if item.get("lane") == "EXPERIENCE"
                ],
                "experience_source_ids": retrieval.get("experience_source_ids", []),
                "experience_reuse_receipt": retrieval.get("experience_reuse_receipt"),
            }
        if capability_id == "inspect_deterministic_core":
            return {"deterministic_core": artifacts.get("core")}
        if capability_id == "inspect_verification_plan":
            return {"verification_plan": artifacts.get("verification")}
        if capability_id == "inspect_probe_catalog":
            catalog = artifacts.get("probe_catalog") or {}
            parsed_catalog = DiagnosticProbeCatalog.model_validate(catalog)
            admitted_ids = {
                item.probe_id for item in parsed_catalog.probes
                if is_admitted_telemetry_probe(item, parsed_catalog)
            }
            return {
                "configured": bool(catalog.get("probes")),
                "telemetry_admission_status": catalog.get("telemetry_admission_status", "NOT_ADMITTED"),
                "probes": [
                    {
                        "probe_id": item.get("probe_id"),
                        "kind": item.get("kind"),
                        "title": item.get("title"),
                        "objective": item.get("objective"),
                        "target_fault_ids": item.get("target_fault_ids", []),
                        "evidence_definition_id": item.get("evidence_definition_id"),
                        "feature_result_id": item.get("feature_result_id"),
                        "formal_evidence_eligible": False,
                        "creates_new_raw_observation": item.get("probe_id") in admitted_ids,
                    }
                    for item in catalog.get("probes", [])
                ],
                "claim_boundary": catalog.get("claim_boundary"),
            }
        raise ValueError(f"no executor for diagnostic capability: {capability_id}")

    @staticmethod
    def _audit(
        steps: list[DiagnosticDeliberationStep], *, final_synthesis_consumes_trajectory: bool
    ) -> LLMMethodSubstanceAudit:
        executed = [item for item in steps if item.observation is not None]
        distinct_executed = {item.observation.capability_id for item in executed if item.observation}
        replans = sum(
            1
            for item in steps[1:]
            if item.proposal.cited_observation_ids
        )
        reasons: list[str] = []
        if len(distinct_executed) < 2:
            reasons.append("FEWER_THAN_TWO_LLM_SELECTED_CAPABILITIES_EXECUTED")
        if replans < 1:
            reasons.append("NO_OBSERVATION_CONDITIONED_REPLAN")
        if not final_synthesis_consumes_trajectory:
            reasons.append("FINAL_SYNTHESIS_DID_NOT_CONSUME_TRAJECTORY")
        return LLMMethodSubstanceAudit(
            status=(
                LLMMethodSubstanceStatus.SUBSTANTIVE_METHOD_PATH
                if not reasons
                else LLMMethodSubstanceStatus.WRAPPER_RISK
            ),
            llm_selected_action_count=len(steps),
            executed_diagnostic_capability_count=len(distinct_executed),
            observation_conditioned_replan_count=replans,
            final_synthesis_consumes_trajectory=final_synthesis_consumes_trajectory,
            reason_codes=reasons or ["STRUCTURAL_LLM_DIAGNOSTIC_LOOP_PRESENT"],
        )

    @staticmethod
    def _contribution_audit(steps: list[DiagnosticDeliberationStep]) -> DiagnosticContributionAudit:
        probe_contents = [
            step.observation.content
            for step in steps
            if step.observation is not None and step.proposal.capability_id == "execute_diagnostic_probe"
        ]
        raw_count = sum(bool(item.get("creates_new_raw_observation")) for item in probe_contents)
        formal_count = sum(bool(item.get("formal_evidence_generated")) for item in probe_contents)
        probe_count = len(probe_contents)
        if probe_count == 0:
            status = DiagnosticContributionStatus.NO_NEW_INFORMATION
            reasons = ["NO_DIAGNOSTIC_PROBE_EXECUTED"]
        elif raw_count == 0 and formal_count == 0:
            status = DiagnosticContributionStatus.DERIVED_OBSERVATION_ONLY
            reasons = ["PROBE_RECOMPUTED_ALREADY_ADMITTED_INFORMATION_ONLY"]
        else:
            status = DiagnosticContributionStatus.NEW_INFORMATION_NO_DIAGNOSTIC_CHANGE
            reasons = ["NEW_OBSERVATION_REQUIRES_DOWNSTREAM_OUTCOME_COMPARISON"]
        return DiagnosticContributionAudit(
            status=status,
            probe_execution_count=probe_count,
            new_observation_count=probe_count,
            new_raw_observation_count=raw_count,
            formal_evidence_generated_count=formal_count,
            reason_codes=reasons,
        )

    def run(
        self,
        *,
        backend: StructuredReasoningBackend,
        episode_id: str,
        artifacts: dict[str, Any],
        max_steps: int = 4,
        allowed_capability_ids: list[str] | None = None,
        observation_feedback_enabled: bool = True,
        probe_catalog: DiagnosticProbeCatalog | None = None,
        probe_executor: DiagnosticProbeExecutor | None = None,
    ) -> LLMControlledDiagnosticTrajectory:
        catalog = {item.capability_id: item for item in self._CAPABILITIES}
        probe_catalog = probe_catalog or DiagnosticProbeCatalog(episode_id=episode_id, probes=())
        probe_executor = probe_executor or DiagnosticProbeExecutor()
        artifacts = dict(artifacts)
        artifacts["probe_catalog"] = probe_catalog.model_dump(mode="json")
        selected_ids = (
            [
                item.capability_id for item in self._CAPABILITIES
                if item.capability_id != "execute_diagnostic_probe" or probe_catalog.probes
            ]
            if allowed_capability_ids is None
            else list(allowed_capability_ids)
        )
        if "execute_diagnostic_probe" in selected_ids and not probe_catalog.probes:
            selected_ids.remove("execute_diagnostic_probe")
        if self.SYNTHESIZE not in selected_ids:
            selected_ids.append(self.SYNTHESIZE)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("allowed diagnostic capability IDs must be unique")
        unknown = sorted(set(selected_ids) - set(catalog))
        if unknown:
            raise ValueError(f"unknown allowed diagnostic capability IDs: {unknown}")
        if len(set(selected_ids) - {self.SYNTHESIZE}) < 2:
            raise ValueError("controlled diagnostic loop requires at least two inspection capabilities")
        capabilities = tuple(catalog[item] for item in selected_ids)
        if not getattr(backend, "supports_diagnostic_deliberation", False):
            return self._build(
                episode_id=episode_id,
                status=DiagnosticLoopStatus.BACKEND_UNSUPPORTED,
                steps=[],
                termination_reason="BACKEND_DID_NOT_DECLARE_CONTROLLED_DELIBERATION_SUPPORT",
                final_synthesis_consumes_trajectory=False,
                allowed_capability_ids=selected_ids,
            )
        allowed = set(selected_ids)
        steps: list[DiagnosticDeliberationStep] = []
        observations: list[DiagnosticObservation] = []
        status = DiagnosticLoopStatus.BUDGET_EXHAUSTED
        reason = "MAXIMUM_DELIBERATION_STEPS_REACHED"
        consumed = False
        for sequence in range(1, max_steps + 1):
            executed_capability_ids = {
                item.capability_id for item in observations
            }
            available_capabilities = [
                item
                for item in capabilities
                if (
                    (item.capability_id not in executed_capability_ids or item.capability_id == "execute_diagnostic_probe")
                    and (item.capability_id != self.SYNTHESIZE or len(observations) >= 2)
                    and (item.capability_id != "execute_diagnostic_probe" or len({p.probe_id for p in probe_catalog.probes} - {
                        step.proposal.probe_id for step in steps if step.observation is not None and step.proposal.probe_id
                    }) > 0)
                )
            ]
            if sequence == max_steps and len(observations) >= 2:
                available_capabilities = [
                    item for item in capabilities if item.capability_id == self.SYNTHESIZE
                ]
            available_ids = {item.capability_id for item in available_capabilities}
            observation_ids = [item.observation_id for item in observations]
            output_model: type[DiagnosticActionProposal] = DiagnosticActionProposal
            if getattr(backend, "supports_dynamic_diagnostic_action_schema", False):
                output_model = self._constrained_proposal_model(
                    available_ids=sorted(available_ids),
                    observation_ids=observation_ids,
                    citation_required=bool(observations) and observation_feedback_enabled,
                    probe_ids=[item.probe_id for item in probe_catalog.probes],
                )
            generated_proposal = backend.generate_structured(
                agent_id=self.agent_id,
                output_model=output_model,
                payload={
                    "episode_id": episode_id,
                    "allowed_capabilities": [
                        item.model_dump(mode="json") for item in available_capabilities
                    ],
                    "already_executed_capability_ids": sorted(executed_capability_ids),
                    "observations": (
                        [item.model_dump(mode="json") for item in observations]
                        if observation_feedback_enabled else []
                    ),
                    "observation_feedback": (
                        [item.model_dump(mode="json") for item in observations]
                        if observation_feedback_enabled else []
                    ),
                    "observation_index": [
                        {"observation_id": item.observation_id, "capability_id": item.capability_id}
                        for item in observations
                    ],
                    "probe_index": [
                        {
                            "probe_id": item.probe_id,
                            "kind": item.kind.value,
                            "title": item.title,
                            "diagnostic_question": item.objective,
                            "target_fault_ids": list(item.target_fault_ids),
                            "creates_new_raw_observation": is_admitted_telemetry_probe(item, probe_catalog),
                        }
                        for item in probe_catalog.probes
                    ],
                    "citation_requirement": (
                        "For this replan, cited_observation_ids must contain at least one of: "
                        + ", ".join(observation_ids)
                        if observation_ids and observation_feedback_enabled
                        else "No prior observation exists; cited_observation_ids may be empty."
                    ),
                    "remaining_step_budget": max_steps - sequence + 1,
                    "instruction": (
                        "Select exactly one allowed capability. When observation_feedback is non-empty, cite "
                        "at least one supplied observation ID when replanning. Use synthesize_hypotheses only after "
                        "at least two diagnostic capabilities have executed. Compare whether any admitted raw "
                        "telemetry probe could distinguish competing hypotheses; use a relevant probe when it has "
                        "information value, but do not run one just to satisfy a quota. KG, retrieval and existing "
                        "evidence cannot substitute for a new raw observation. For execute_diagnostic_probe, select "
                        "exactly one supplied probe_id; never invent probe IDs, channels, windows, thresholds, "
                        "or identifiers. A probe result remains shadow-only, never formal evidence."
                    ),
                },
            )
            proposal = DiagnosticActionProposal.model_validate(
                generated_proposal.model_dump(mode="json")
            )
            known_probe_ids = {item.probe_id for item in probe_catalog.probes}
            if proposal.capability_id == "execute_diagnostic_probe":
                if not proposal.probe_id or proposal.probe_id not in known_probe_ids:
                    steps.append(DiagnosticDeliberationStep(
                        sequence=sequence, proposal=proposal, policy_status="REJECTED_UNKNOWN_PROBE_ID"
                    ))
                    status = DiagnosticLoopStatus.POLICY_BLOCKED
                    reason = "MODEL_SELECTED_PROBE_OUTSIDE_HOST_CATALOG"
                    break
                if proposal.probe_id in {
                    step.proposal.probe_id for step in steps
                    if step.proposal.capability_id == "execute_diagnostic_probe" and step.observation is not None
                }:
                    steps.append(DiagnosticDeliberationStep(
                        sequence=sequence, proposal=proposal, policy_status="REJECTED_DUPLICATE_PROBE"
                    ))
                    status = DiagnosticLoopStatus.POLICY_BLOCKED
                    reason = "MODEL_REPEATED_DIAGNOSTIC_PROBE"
                    break
            elif proposal.probe_id is not None:
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence, proposal=proposal, policy_status="REJECTED_UNEXPECTED_PROBE_ID"
                ))
                status = DiagnosticLoopStatus.POLICY_BLOCKED
                reason = "PROBE_ID_SUPPLIED_FOR_NON_PROBE_CAPABILITY"
                break
            if proposal.capability_id not in allowed:
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence, proposal=proposal, policy_status="REJECTED_UNKNOWN_CAPABILITY"
                ))
                status = DiagnosticLoopStatus.POLICY_BLOCKED
                reason = "MODEL_SELECTED_CAPABILITY_OUTSIDE_ALLOWLIST"
                break
            if proposal.capability_id not in available_ids:
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence,
                    proposal=proposal,
                    policy_status="REJECTED_CAPABILITY_NOT_CURRENTLY_AVAILABLE",
                ))
                status = DiagnosticLoopStatus.POLICY_BLOCKED
                reason = "MODEL_SELECTED_CAPABILITY_OUTSIDE_DYNAMIC_BUDGET_POLICY"
                break
            if proposal.capability_id in executed_capability_ids and proposal.capability_id != "execute_diagnostic_probe":
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence,
                    proposal=proposal,
                    policy_status="REJECTED_DUPLICATE_CAPABILITY",
                ))
                status = DiagnosticLoopStatus.POLICY_BLOCKED
                reason = "MODEL_REPEATED_DIAGNOSTIC_CAPABILITY"
                break
            known_observations = {item.observation_id for item in observations}
            if not set(proposal.cited_observation_ids).issubset(known_observations):
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence, proposal=proposal, policy_status="REJECTED_UNKNOWN_OBSERVATION"
                ))
                status = DiagnosticLoopStatus.POLICY_BLOCKED
                reason = "MODEL_CITED_UNKNOWN_OBSERVATION"
                break
            if (
                observation_feedback_enabled
                and sequence > 1
                and observations
                and not proposal.cited_observation_ids
            ):
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence, proposal=proposal, policy_status="REJECTED_UNGROUNDED_REPLAN"
                ))
                status = DiagnosticLoopStatus.POLICY_BLOCKED
                reason = "REPLAN_DID_NOT_CITE_PRIOR_OBSERVATION"
                break
            if proposal.capability_id == self.SYNTHESIZE:
                policy_status = "ACCEPTED" if len(observations) >= 2 else "REJECTED_PREMATURE_SYNTHESIS"
                steps.append(DiagnosticDeliberationStep(
                    sequence=sequence, proposal=proposal, policy_status=policy_status
                ))
                if len(observations) < 2:
                    status = DiagnosticLoopStatus.POLICY_BLOCKED
                    reason = "SYNTHESIS_REQUESTED_BEFORE_MINIMUM_DIAGNOSTIC_ACTIONS"
                else:
                    status = DiagnosticLoopStatus.COMPLETED
                    reason = "MODEL_REQUESTED_GROUNDED_HYPOTHESIS_SYNTHESIS"
                    consumed = True
                break
            if proposal.capability_id == "execute_diagnostic_probe":
                spec = probe_catalog.get(str(proposal.probe_id))
                if is_active_telemetry_probe(spec) and not isinstance(probe_executor, ActiveTelemetryProbeExecutor):
                    raise ValueError("raw telemetry probe requires the admission-aware executor")
                probe_result = probe_executor.execute(catalog=probe_catalog, probe_id=str(proposal.probe_id))
                content = probe_result.model_dump(mode="json")
                policy_status = "EXECUTED_SHADOW_PROBE"
            else:
                content = self._summarize(proposal.capability_id, artifacts)
                policy_status = "EXECUTED_READ_ONLY"
            observation = DiagnosticObservation(
                observation_id=f"OBS-{sequence:02d}-{_digest(content)[:12]}",
                capability_id=proposal.capability_id,
                artifact_sha256=_digest(content),
                content=content,
            )
            observations.append(observation)
            steps.append(DiagnosticDeliberationStep(
                sequence=sequence,
                proposal=proposal,
                policy_status=policy_status,
                observation=observation,
            ))
        return self._build(
            episode_id=episode_id,
            status=status,
            steps=steps,
            termination_reason=reason,
            final_synthesis_consumes_trajectory=consumed,
            allowed_capability_ids=selected_ids,
            observation_feedback_enabled=observation_feedback_enabled,
        )

    def _build(
        self,
        *,
        episode_id: str,
        status: DiagnosticLoopStatus,
        steps: list[DiagnosticDeliberationStep],
        termination_reason: str,
        final_synthesis_consumes_trajectory: bool,
        allowed_capability_ids: list[str] | None = None,
        observation_feedback_enabled: bool = True,
        contribution_audit_override: DiagnosticContributionAudit | None = None,
    ) -> LLMControlledDiagnosticTrajectory:
        allowed_ids = allowed_capability_ids or [item.capability_id for item in self._CAPABILITIES]
        substance_audit = self._audit(
            steps, final_synthesis_consumes_trajectory=final_synthesis_consumes_trajectory
        ).model_copy(update=(
            {}
            if observation_feedback_enabled
            else {
                "status": LLMMethodSubstanceStatus.WRAPPER_RISK,
                "observation_conditioned_replan_count": 0,
                "reason_codes": ["OBSERVATION_FEEDBACK_ABLATED"],
            }
        ))
        contribution_audit = contribution_audit_override or self._contribution_audit(steps)
        content = {
            "episode_id": episode_id,
            "status": status.value,
            "steps": [item.model_dump(mode="json") for item in steps],
            "termination_reason": termination_reason,
            "allowed_capability_ids": allowed_ids,
            "final_synthesis_consumes_trajectory": final_synthesis_consumes_trajectory,
            "substance_audit": substance_audit.model_dump(mode="json"),
            "contribution_audit": contribution_audit.model_dump(mode="json"),
        }
        return LLMControlledDiagnosticTrajectory(
            episode_id=episode_id,
            status=status,
            steps=steps,
            termination_reason=termination_reason,
            allowed_capability_ids=allowed_ids,
            final_synthesis_consumes_trajectory=final_synthesis_consumes_trajectory,
            trajectory_sha256=_digest(content),
            substance_audit=substance_audit,
            contribution_audit=contribution_audit,
        )

    def record_shadow_revision(
        self, trajectory: LLMControlledDiagnosticTrajectory, revision: ShadowCandidateRevision,
    ) -> LLMControlledDiagnosticTrajectory:
        """Audit candidate refinement without changing formal evidence or disposition."""
        if not revision.active_telemetry_attributable:
            return trajectory
        contribution = trajectory.contribution_audit.model_copy(update={
            "status": DiagnosticContributionStatus.CANDIDATE_SET_REFINED,
            "candidate_set_changed": True,
            "decision_changed": False,
            "reason_codes": [
                *trajectory.contribution_audit.reason_codes,
                "VALIDATED_SHADOW_CANDIDATE_REVISION_CITES_ACTIVE_TELEMETRY",
            ],
        })
        return self._build(
            episode_id=trajectory.episode_id,
            status=trajectory.status,
            steps=trajectory.steps,
            termination_reason=trajectory.termination_reason,
            final_synthesis_consumes_trajectory=trajectory.final_synthesis_consumes_trajectory,
            allowed_capability_ids=trajectory.allowed_capability_ids,
            contribution_audit_override=contribution,
        )

    def record_synthesis_failure(
        self,
        trajectory: LLMControlledDiagnosticTrajectory,
        *,
        error_type: str,
    ) -> LLMControlledDiagnosticTrajectory:
        """Invalidate a requested synthesis when the downstream model call did not return."""
        return self._build(
            episode_id=trajectory.episode_id,
            status=DiagnosticLoopStatus.POLICY_BLOCKED,
            steps=trajectory.steps,
            termination_reason=f"FINAL_SYNTHESIS_FAILED:{error_type}",
            final_synthesis_consumes_trajectory=False,
            allowed_capability_ids=trajectory.allowed_capability_ids,
        )


__all__ = [
    "ControlledDiagnosticDeliberationService",
    "DiagnosticActionProposal",
    "DiagnosticContributionAudit",
    "DiagnosticContributionStatus",
    "DiagnosticCapability",
    "DiagnosticDeliberationStep",
    "DiagnosticLoopStatus",
    "DiagnosticObservation",
    "LLMControlledDiagnosticTrajectory",
    "LLMMethodSubstanceAudit",
    "LLMMethodSubstanceStatus",
]
