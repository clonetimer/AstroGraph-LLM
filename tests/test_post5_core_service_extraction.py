from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from astrograph.agents.agentic import DiagnosticReasonerAgent, SafetyCriticAgent
from astrograph.diagnosis import DiagnosisPolicyGate, ReleaseQualificationGate
from astrograph.fusion import BiRRFEngine, OpenSetFusionPolicy, RuntimeFusionService
from astrograph.investigation import DecisionGovernor, DiagnosticDecisionGovernor
from astrograph.knowledge import KGEvidenceRuntimeService
from astrograph.services import (
    DeterministicReferenceReasoner,
    EvidenceIntegrityAuditor,
    HypothesisSafetyCritic,
    SafetyValidationService,
    StructuredHypothesisProposalService,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fusion_is_componentized_behind_runtime_service() -> None:
    service = RuntimeFusionService()
    assert isinstance(service.engine, BiRRFEngine)
    assert isinstance(service.open_set_policy, OpenSetFusionPolicy)
    workflow = (ROOT / "src/astrograph/orchestration/whole_spacecraft_online_workflow.py").read_text(encoding="utf-8")
    assert "RuntimeFusionService" in workflow
    assert "candidate_fusion.model_copy" not in workflow
    assert "build_candidate_fusion_package_from_graph_associations" not in workflow


def test_kg_backend_lifecycle_is_not_owned_by_online_workflow() -> None:
    workflow = (ROOT / "src/astrograph/orchestration/whole_spacecraft_online_workflow.py").read_text(encoding="utf-8")
    assert "GraphDatabase.driver" not in workflow
    assert "fallback nodes CSV hash" not in workflow
    assert KGEvidenceRuntimeService.__name__ == "KGEvidenceRuntimeService"


def test_reasoning_agents_are_compatibility_facades_over_services() -> None:
    reasoner = DiagnosticReasonerAgent()
    critic = SafetyCriticAgent()
    assert isinstance(reasoner.reference_service, DeterministicReferenceReasoner)
    assert isinstance(reasoner.proposal_service, StructuredHypothesisProposalService)
    assert isinstance(critic.service, SafetyValidationService)
    assert isinstance(critic.service.evidence_auditor, EvidenceIntegrityAuditor)
    assert isinstance(critic.service.hypothesis_critic, HypothesisSafetyCritic)


def test_data_driven_runtime_import_does_not_eager_load_lab_modules() -> None:
    code = f'''\nimport sys\nsys.path.insert(0, {str(ROOT / "src")!r})\nfrom astrograph.data_driven import DataDrivenModelRegistry\nmods = set(sys.modules)\nfor forbidden in (\n    "astrograph.data_driven.campaign",\n    "astrograph.data_driven.benchmark",\n    "astrograph.data_driven.cross_condition",\n    "astrograph.data_driven.whole_spacecraft_hgb",\n    "astrograph.data_driven.temporal_autoencoder",\n):\n    assert forbidden not in mods, forbidden\n'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_stable_online_import_does_not_eager_load_legacy_evidence_builder() -> None:
    code = f'''\nimport sys\nsys.path.insert(0, {str(ROOT / "src")!r})\nimport astrograph.tools.online\nassert "astrograph.simulation.runtime_adapter" not in sys.modules\n'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_decision_and_release_qualification_have_distinct_canonical_names() -> None:
    assert DecisionGovernor is DiagnosticDecisionGovernor
    assert DiagnosisPolicyGate is ReleaseQualificationGate
    assert DiagnosticDecisionGovernor.role == "DIAGNOSTIC_DECISION"
    assert ReleaseQualificationGate.role == "RELEASE_QUALIFICATION"


def test_online_workflow_no_longer_fabricates_safety_replay_from_core_comparison() -> None:
    workflow = (ROOT / "src/astrograph/orchestration/whole_spacecraft_online_workflow.py").read_text(encoding="utf-8")
    assert "CoreComparisonReference(" in workflow
    assert "safety_replay=None" in workflow
    assert "SafetyReplayReference(" not in workflow
