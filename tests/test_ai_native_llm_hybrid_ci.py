"""Phase 0 CI harness: run the whole-spacecraft online chain in rule / llm /
hybrid reasoner modes with the deterministic reference backend (no vLLM).
Verifies the llm/hybrid CI entry point required by the AI-Native evolution
roadmap Phase 0 and asserts rule-vs-llm decision consistency.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.delivery_evidence import require_delivery_evidence

from astrograph.orchestration.models import (
    ExecutionPolicy,
    ExecutionStatus,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import (
    WholeSpacecraftOnlineDiagnosisWorkflow,
)
from astrograph.services import ReferenceStructuredReasoningBackend
from astrograph.simulation import SimulationRunCompleted


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_base"
PROTOCOL = ROOT / "protocols" / "phm_project_a"


def _permissions():
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _invoke(tmp_path: Path, mode: str, backend=None):
    event = SimulationRunCompleted(
        event_id=f"SIM-AI-NATIVE-{mode}",
        run_id="heater_stuck_on",
        run_dir=str(FIXTURE / "heater_stuck_on"),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    permissions = _permissions()
    envelope = TaskEnvelope(
        task_id=f"AI-NATIVE-{mode}",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / mode),
            "shadow_mode": True,
            "reasoner_mode": mode,
        },
        execution_policy=ExecutionPolicy(deterministic_only=mode == "rule", allow_llm=mode != "rule"),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
        run_id="heater_stuck_on",
        permissions=permissions,
    )
    return WholeSpacecraftOnlineDiagnosisWorkflow(reasoning_backend=backend).invoke(envelope, context)


def _decision(result):
    payload = result.output_payload or {}
    record = payload.get("decision_record") or {}
    return {
        "disposition": record.get("disposition"),
        "accepted_fault_id": record.get("accepted_fault_id"),
        "requires_human_review": record.get("requires_human_review"),
    }


def test_rule_mode_succeeds_deterministically(tmp_path):
    result = _invoke(tmp_path, "rule")
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["reasoner_mode"] == "rule"
    assert result.output_payload["llm_runtime_snapshot"]["status"] == "DETERMINISTIC_REFERENCE"


def test_llm_mode_with_reference_backend_succeeds_without_vllm(tmp_path):
    backend = ReferenceStructuredReasoningBackend()
    result = _invoke(tmp_path, "llm", backend=backend)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["reasoner_mode"] == "llm"
    assert backend.last_call_metadata["status"] == "SUCCEEDED"
    assert result.output_payload["llm_runtime_snapshot"]["status"] == "LLM_BACKEND"


def test_hybrid_mode_with_reference_backend_succeeds(tmp_path):
    backend = ReferenceStructuredReasoningBackend()
    result = _invoke(tmp_path, "hybrid", backend=backend)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["reasoner_mode"] == "hybrid"


def test_rule_and_llm_decisions_are_consistent(tmp_path):
    rule = _invoke(tmp_path, "rule")
    llm = _invoke(tmp_path, "llm", backend=ReferenceStructuredReasoningBackend())
    assert rule.status == ExecutionStatus.SUCCEEDED
    assert llm.status == ExecutionStatus.SUCCEEDED
    assert _decision(rule) == _decision(llm)


def test_reference_backend_rejects_unexpected_output_model():
    backend = ReferenceStructuredReasoningBackend()
    with pytest.raises(TypeError):
        backend.generate_structured(agent_id="x", output_model=dict, payload={})


def test_reference_backend_requires_reference_hypothesis_set():
    from astrograph.investigation import HypothesisSet
    backend = ReferenceStructuredReasoningBackend()
    with pytest.raises(ValueError):
        backend.generate_structured(
            agent_id="x", output_model=HypothesisSet, payload={}
        )


def test_ai_native_matrix_and_baseline_are_valid():
    require_delivery_evidence(ROOT, "governance/evidence/ai_native/ai_native_maturity_matrix.json")
    require_delivery_evidence(ROOT, "governance/evidence/ai_native/feedback_loop_baseline.json")
    script = ROOT / "scripts" / "ai_native" / "validate_ai_native_matrix.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr

def _invoke_llm(tmp_path: Path, deviation: str | None) -> tuple:
    backend = ReferenceStructuredReasoningBackend(deviation=deviation)
    result = _invoke(tmp_path, "llm", backend=backend)
    return result, backend


def _record(result):
    payload = result.output_payload or {}
    record = payload.get("decision_record") or {}
    return {
        "disposition": record.get("disposition"),
        "accepted_fault_id": record.get("accepted_fault_id"),
        "requires_human_review": record.get("requires_human_review"),
    }


def test_llm_invented_fault_is_rejected_fail_closed(tmp_path):
    # A backend that fabricates an unregistered fault must not release a diagnosis.
    result, backend = _invoke_llm(tmp_path, "invent_fault")
    assert result.status == ExecutionStatus.FAILED
    assert "UNREGISTERED_FAULT_XYZ" in (result.error.message if result.error else "")
    assert backend.last_call_metadata["deviation"] == "invent_fault"


def test_llm_corrupted_evidence_degrades_to_human_review(tmp_path):
    # Fabricated evidence ids must not lead to an automatic acceptance.
    result, _ = _invoke_llm(tmp_path, "corrupt_evidence")
    assert result.status == ExecutionStatus.SUCCEEDED
    assert _record(result)["disposition"] == "HUMAN_REVIEW"
    assert _record(result)["accepted_fault_id"] is None


def test_llm_dropped_faults_degrades_to_human_review(tmp_path):
    # A model that silently drops every registered fault must not auto-accept.
    result, _ = _invoke_llm(tmp_path, "drop_all_faults")
    assert result.status == ExecutionStatus.SUCCEEDED
    assert _record(result)["disposition"] == "HUMAN_REVIEW"
    assert _record(result)["accepted_fault_id"] is None


def test_llm_baseline_decision_matches_rule(tmp_path):
    rule = _invoke(tmp_path, "rule")
    llm, _ = _invoke_llm(tmp_path, None)
    assert rule.status == ExecutionStatus.SUCCEEDED
    assert llm.status == ExecutionStatus.SUCCEEDED
    assert _record(rule) == _record(llm)
    assert _record(rule)["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"

def test_navigator_walks_episode_evidence_graph():
    from astrograph.evidence import EpisodeEvidenceBundle
    from astrograph.services import EvidenceNavigationPlanningService

    bundle = EpisodeEvidenceBundle.model_validate_json(
        (ROOT / "tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json")
        .read_text(encoding="utf-8")
    )
    result = EvidenceNavigationPlanningService().navigate_episode(bundle)
    assert result.episode_id == bundle.episode.episode_id
    assert result.paths
    summary = result.fault_summary.get("THERMAL_HEATER_STUCK_ON")
    assert summary is not None
    assert summary["has_support"] is True
    definition_ids = {item.evidence_definition_id for item in bundle.evidence_definitions}
    feature_ids = {item.feature_definition_id for item in bundle.feature_definitions}
    for path in result.paths:
        assert path.evidence_definition_id in definition_ids
        assert path.feature_definition_id in feature_ids


def test_navigator_filters_candidate_faults():
    from astrograph.evidence import EpisodeEvidenceBundle
    from astrograph.services import EvidenceNavigationPlanningService

    bundle = EpisodeEvidenceBundle.model_validate_json(
        (ROOT / "tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json")
        .read_text(encoding="utf-8")
    )
    result = EvidenceNavigationPlanningService().navigate_episode(
        bundle, candidate_fault_ids=["NONEXISTENT_FAULT"],
    )
    assert result.paths == []


def test_online_workflow_records_navigation_trace(tmp_path):
    result = _invoke(tmp_path, "rule")
    assert result.status == ExecutionStatus.SUCCEEDED
    payload = result.output_payload or {}
    trace = payload.get("agent_trace") or {}
    events = trace.get("events") or []
    actions = [event.get("action") for event in events]
    assert "navigate_episode_evidence_graph" in actions
