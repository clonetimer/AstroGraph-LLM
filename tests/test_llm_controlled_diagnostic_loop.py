from __future__ import annotations

from astrograph.orchestration.llm_diagnostic_loop import (
    ControlledDiagnosticDeliberationService,
    DiagnosticActionProposal,
    DiagnosticLoopStatus,
    LLMMethodSubstanceStatus,
)
from pydantic import ValidationError
import pytest


class GroundedBackend:
    model_id = "scripted-deliberation"
    model_version = "test-v1"
    supports_diagnostic_deliberation = True

    def generate_structured(self, *, agent_id, output_model, payload):
        assert output_model is DiagnosticActionProposal
        observations = payload["observations"]
        if not observations:
            content = {"capability_id": "inspect_evidence_status", "objective": "Inspect facts"}
        elif len(observations) == 1:
            content = {
                "capability_id": "inspect_kg_support",
                "objective": "Cross-check graph support",
                "cited_observation_ids": [observations[-1]["observation_id"]],
            }
        else:
            content = {
                "capability_id": "synthesize_hypotheses",
                "objective": "Synthesize from recorded observations",
                "cited_observation_ids": [item["observation_id"] for item in observations],
            }
        return output_model.model_validate(content)


class UnknownCapabilityBackend(GroundedBackend):
    def generate_structured(self, *, agent_id, output_model, payload):
        return output_model.model_validate(
            {"capability_id": "run_shell", "objective": "Bypass policy"}
        )


def _artifacts():
    return {
        "evidence": {
            "episode": {"episode_id": "EP-1"},
            "evidence_facts": [
                {
                    "evidence_fact_id": "EF-1",
                    "evidence_definition_id": "ED-1",
                    "status": "SATISFIED",
                    "value": 1.0,
                }
            ],
        },
        "fusion": {"birrf_result": {"top_fault_id": "FAULT-A"}},
        "kg": {"kg_evidence": {"fault_id": "FAULT-A"}},
        "retrieval": {"status": "SUFFICIENT"},
        "core": {"predicted_case_id": "fault_a"},
        "verification": {"tasks": []},
    }


def test_llm_selects_two_read_only_capabilities_then_grounded_synthesis():
    result = ControlledDiagnosticDeliberationService().run(
        backend=GroundedBackend(), episode_id="EP-1", artifacts=_artifacts()
    )

    assert result.status == DiagnosticLoopStatus.COMPLETED
    assert [step.proposal.capability_id for step in result.steps] == [
        "inspect_evidence_status",
        "inspect_kg_support",
        "synthesize_hypotheses",
    ]
    assert result.substance_audit.status == LLMMethodSubstanceStatus.SUBSTANTIVE_METHOD_PATH
    assert result.substance_audit.executed_diagnostic_capability_count == 2
    assert result.final_synthesis_consumes_trajectory is True


def test_unknown_capability_is_blocked_without_execution():
    result = ControlledDiagnosticDeliberationService().run(
        backend=UnknownCapabilityBackend(), episode_id="EP-1", artifacts=_artifacts()
    )
    assert result.status == DiagnosticLoopStatus.POLICY_BLOCKED
    assert result.steps[0].policy_status == "REJECTED_UNKNOWN_CAPABILITY"
    assert result.substance_audit.status == LLMMethodSubstanceStatus.WRAPPER_RISK


def test_backend_must_explicitly_support_controlled_deliberation():
    class LegacyBackend:
        model_id = "legacy"
        model_version = "v1"

    result = ControlledDiagnosticDeliberationService().run(
        backend=LegacyBackend(), episode_id="EP-1", artifacts=_artifacts()  # type: ignore[arg-type]
    )
    assert result.status == DiagnosticLoopStatus.BACKEND_UNSUPPORTED
    assert result.steps == []


def test_last_step_offers_synthesis_only_after_two_observations():
    class BudgetAwareBackend(GroundedBackend):
        def generate_structured(self, *, agent_id, output_model, payload):
            available = payload["allowed_capabilities"]
            selected = available[0]["capability_id"]
            return output_model.model_validate({
                "capability_id": selected,
                "objective": "Follow dynamic budget policy",
                "cited_observation_ids": [
                    item["observation_id"] for item in payload["observations"]
                ],
            })

    result = ControlledDiagnosticDeliberationService().run(
        backend=BudgetAwareBackend(), episode_id="EP-1", artifacts=_artifacts(), max_steps=3
    )
    assert [item.proposal.capability_id for item in result.steps] == [
        "inspect_evidence_status",
        "inspect_candidate_fusion",
        "synthesize_hypotheses",
    ]
    assert result.status == DiagnosticLoopStatus.COMPLETED


def test_dynamic_schema_hides_premature_synthesis_and_requires_known_citation():
    class SchemaAwareBackend(GroundedBackend):
        supports_dynamic_diagnostic_action_schema = True

        def generate_structured(self, *, agent_id, output_model, payload):
            schema = output_model.model_json_schema()
            capability_schema = schema["properties"]["capability_id"]
            available = set(capability_schema.get("enum", [capability_schema.get("const")]))
            observations = payload["observations"]
            if len(observations) < 2:
                assert "synthesize_hypotheses" not in available
            content = {
                "capability_id": sorted(available)[0],
                "objective": "Use decoder-constrained action",
            }
            if observations:
                citation_schema = schema["properties"]["cited_observation_ids"]
                assert citation_schema["minItems"] == 1
                content["cited_observation_ids"] = [observations[-1]["observation_id"]]
            return output_model.model_validate(content)

    result = ControlledDiagnosticDeliberationService().run(
        backend=SchemaAwareBackend(), episode_id="EP-1", artifacts=_artifacts(), max_steps=3
    )
    assert result.status == DiagnosticLoopStatus.COMPLETED
    assert result.substance_audit.status == LLMMethodSubstanceStatus.SUBSTANTIVE_METHOD_PATH


def test_trajectory_hash_rejects_tampering():
    result = ControlledDiagnosticDeliberationService().run(
        backend=GroundedBackend(), episode_id="EP-1", artifacts=_artifacts()
    )
    tampered = result.model_dump(mode="json")
    tampered["termination_reason"] = "TAMPERED"
    with pytest.raises(ValidationError, match="trajectory_sha256 mismatch"):
        type(result).model_validate(tampered)


def test_observation_feedback_ablation_is_explicit_wrapper_risk():
    class NoFeedbackBackend(GroundedBackend):
        def generate_structured(self, *, agent_id, output_model, payload):
            selected = payload["allowed_capabilities"][0]["capability_id"]
            assert payload["observations"] == []
            return output_model.model_validate({
                "capability_id": selected,
                "objective": "Select without observation content",
                "cited_observation_ids": [],
            })

    result = ControlledDiagnosticDeliberationService().run(
        backend=NoFeedbackBackend(),
        episode_id="EP-1",
        artifacts=_artifacts(),
        max_steps=3,
        observation_feedback_enabled=False,
    )
    assert result.status == DiagnosticLoopStatus.COMPLETED
    assert result.substance_audit.status == LLMMethodSubstanceStatus.WRAPPER_RISK
    assert result.substance_audit.reason_codes == ["OBSERVATION_FEEDBACK_ABLATED"]


def test_capability_mask_is_enforced_and_requires_two_inspection_capabilities():
    service = ControlledDiagnosticDeliberationService()
    with pytest.raises(ValueError, match="at least two inspection"):
        service.run(
            backend=GroundedBackend(), episode_id="EP-1", artifacts=_artifacts(),
            allowed_capability_ids=["inspect_kg_support"],
        )
