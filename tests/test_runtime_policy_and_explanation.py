from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.agents.operator_assistant import (
    ExplanationSelection,
    LockedExplanationFact,
    LockedExplanationPayload,
    OperatorAssistantAgent,
    RetrievedExplanationSource,
)
from astrograph.knowledge import build_forward_evidence_query_plan
from astrograph.policy_profiles import (
    FusionPolicyProfile,
    GovernedGraphQueryCompiler,
    RuntimePolicyRegistry,
)
from astrograph.orchestration.models import (
    ExecutionPolicy,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import (
    WholeSpacecraftOnlineDiagnosisWorkflow,
    WholeSpacecraftOnlineInput,
)
from astrograph.simulation import SimulationRunCompleted


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "runtime_policies" / "registry.json"
SOURCE_HASH = hashlib.sha256(b"reviewed-source").hexdigest()


def test_runtime_policy_registry_resolves_hash_bound_query_and_fusion_profiles() -> None:
    registry = RuntimePolicyRegistry(REGISTRY)
    query = registry.resolve(profile_type="GRAPH_QUERY", profile_id="graph-query.read-only.v1")
    fusion = registry.resolve(profile_type="FUSION", profile_id="birrf.engineering-default.v1")

    compiler = GovernedGraphQueryCompiler(query)
    compiled = compiler.compile(build_forward_evidence_query_plan(["ED-1"]))
    assert compiled.read_only is True
    assert compiled.policy_id == "astrograph-evidence-graph-read-only-v1"
    assert fusion.profile.qualification_status == "ENGINEERING_DEFAULT"
    assert fusion.profile.config.data_weight == 1.0
    assert fusion.receipt.qualification_status == "ENGINEERING_DEFAULT"


def test_runtime_policy_registry_fails_closed_on_unknown_and_hash_drift(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown runtime policy"):
        RuntimePolicyRegistry(REGISTRY).resolve(profile_type="FUSION", profile_id="missing")

    copied = tmp_path / "runtime_policies"
    shutil.copytree(ROOT / "configs" / "runtime_policies", copied)
    index = json.loads((copied / "registry.json").read_text(encoding="utf-8"))
    index["asset_root"] = "."
    index["profiles"][0]["path"] = "graph_query_read_only.v1.json"
    (copied / "registry.json").write_text(json.dumps(index), encoding="utf-8")
    with (copied / "graph_query_read_only.v1.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        RuntimePolicyRegistry(copied / "registry.json").resolve(
            profile_type="GRAPH_QUERY", profile_id="graph-query.read-only.v1"
        )


def test_calibrated_fusion_claim_requires_receipt() -> None:
    raw = json.loads(
        (ROOT / "configs" / "runtime_policies" / "birrf_engineering_default.v1.json").read_text(
            encoding="utf-8"
        )
    )
    raw["profile_id"] = "invalid.calibrated"
    raw["qualification_status"] = "CALIBRATED"
    with pytest.raises(ValidationError, match="calibration receipt"):
        FusionPolicyProfile.model_validate(raw)


def test_online_runtime_policy_selection_fails_closed_without_registry(tmp_path: Path) -> None:
    event = SimulationRunCompleted(
        event_id="runtime-policy-missing",
        run_id="runtime-policy-missing",
        run_dir=str(tmp_path / "run"),
        nominal_run_dir=str(tmp_path / "nominal"),
    )
    permissions = list(PermissionScope)
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow()
    result = workflow.invoke(
        TaskEnvelope(
            task_id="runtime-policy-missing",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(ROOT / "protocols/phm_project_a"),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(tmp_path / "artifacts"),
                "graph_query_policy_profile_id": "graph-query.read-only.v1",
            },
            execution_policy=ExecutionPolicy(),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="runtime-policy-missing",
            permissions=permissions,
        ),
    )
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_RUNTIME_POLICY_REGISTRY_REQUIRED"


def test_inline_and_profile_fusion_configuration_cannot_be_mixed(tmp_path: Path) -> None:
    event = SimulationRunCompleted(
        event_id="runtime-policy-conflict",
        run_id="runtime-policy-conflict",
        run_dir=str(tmp_path / "run"),
        nominal_run_dir=str(tmp_path / "nominal"),
    )
    with pytest.raises(ValidationError, match="cannot be combined with inline birrf_config"):
        WholeSpacecraftOnlineInput.model_validate(
            {
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(ROOT / "protocols/phm_project_a"),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(tmp_path / "artifacts"),
                "fusion_policy_profile_id": "birrf.engineering-default.v1",
                "birrf_config": {"data_weight": 2.0},
            }
        )


def test_soft_calibration_asset_requires_hash_and_cannot_override_profile(tmp_path: Path) -> None:
    event = SimulationRunCompleted(
        event_id="soft-policy-input",
        run_id="soft-policy-input",
        run_dir=str(tmp_path / "run"),
        nominal_run_dir=str(tmp_path / "nominal"),
    )
    direct = {
        "simulation_event": event.model_dump(mode="json"),
        "protocol_root": str(ROOT / "protocols/phm_project_a"),
        "evidence_package_path": str(
            ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json"
        ),
        "runtime_evidence_builder_id": "configured.package.v1",
        "artifact_output_dir": str(tmp_path / "artifacts"),
    }
    with pytest.raises(ValidationError, match="path and SHA-256 must be provided together"):
        WholeSpacecraftOnlineInput.model_validate(
            {**direct, "soft_evidence_calibration_policy_path": "/tmp/policy.json"}
        )
    accepted = WholeSpacecraftOnlineInput.model_validate(
        {
            **direct,
            "soft_evidence_calibration_policy_path": "/tmp/policy.json",
            "soft_evidence_calibration_policy_sha256": "a" * 64,
        }
    )
    assert accepted.soft_evidence_calibration_policy_sha256 == "a" * 64
    with pytest.raises(ValidationError, match="direct protocol/evidence assets"):
        WholeSpacecraftOnlineInput.model_validate(
            {
                "simulation_event": event.model_dump(mode="json"),
                "dataset_profile_id": "phm.v6_0",
                "artifact_output_dir": str(tmp_path / "artifacts"),
                "soft_evidence_calibration_policy_path": "/tmp/policy.json",
                "soft_evidence_calibration_policy_sha256": "a" * 64,
            }
        )


def _payload() -> LockedExplanationPayload:
    return LockedExplanationPayload(
        diagnosis_id="DX-1",
        diagnosis_payload_sha256=hashlib.sha256(b"locked-diagnosis").hexdigest(),
        facts=[
            LockedExplanationFact(
                fact_id="FACT-1",
                text="The locked governor disposition is ABSTAIN.",
                source_ids=["SRC-1"],
            )
        ],
        retrieved_sources=[
            RetrievedExplanationSource(
                source_id="SRC-1",
                title="Reviewed knowledge release",
                uri="kg://release/v1/node/ED-1",
                sha256=SOURCE_HASH,
                snippet="The evidence definition is review-only engineering evidence.",
                knowledge_version="v1",
            )
        ],
        allowed_caveat_codes=["ENGINEERING_EVIDENCE_ONLY"],
    )


def _selection(payload: LockedExplanationPayload, **updates):
    raw = {
        "locked_envelope_hash": payload.envelope_hash,
        "ordered_fact_ids": ["FACT-1"],
        "context_source_ids": ["SRC-1"],
        "caveat_codes": ["ENGINEERING_EVIDENCE_ONLY"],
    }
    raw.update(updates)
    return raw


def test_operator_explanation_is_deterministically_rendered_from_locked_ids() -> None:
    payload = _payload()
    result = OperatorAssistantAgent().explain(payload, draft_selection=_selection(payload))
    assert result.diagnosis_payload_sha256 == payload.diagnosis_payload_sha256
    assert result.selected_fact_ids == ["FACT-1"]
    assert result.cited_source_ids == ["SRC-1"]
    assert result.cited_source_sha256 == {"SRC-1": SOURCE_HASH}
    assert "ABSTAIN" in result.rendered_markdown
    assert "kg://release/v1/node/ED-1" in result.rendered_markdown


def test_operator_explanation_rejects_payload_drift_and_invented_ids() -> None:
    payload = _payload()
    agent = OperatorAssistantAgent()
    with pytest.raises(ValueError, match="locked payload"):
        agent.explain(
            payload,
            draft_selection=_selection(payload, locked_envelope_hash="0" * 64),
        )
    with pytest.raises(ValueError, match="unknown locked facts"):
        agent.explain(payload, draft_selection=_selection(payload, ordered_fact_ids=["INVENTED"]))
    with pytest.raises(ValueError, match="unknown retrieval sources"):
        agent.explain(payload, draft_selection=_selection(payload, context_source_ids=["INVENTED"]))
    with pytest.raises(ValidationError):
        ExplanationSelection.model_validate(
            {**_selection(payload), "replacement_fault_id": "MODEL_INVENTED_FAULT"}
        )


class _SelectionBackend:
    model_id = "structured-selector"
    model_version = "1"

    def __init__(self, selection: ExplanationSelection) -> None:
        self.selection = selection

    def generate_structured(self, *, agent_id, output_model, payload):
        assert agent_id == "operator-assistant-agent"
        assert output_model is ExplanationSelection
        assert "cypher" not in payload
        return self.selection


def test_model_backend_can_select_but_cannot_rewrite_locked_content() -> None:
    payload = _payload()
    selection = ExplanationSelection.model_validate(_selection(payload))
    result = OperatorAssistantAgent(backend=_SelectionBackend(selection)).explain(payload)
    assert result.model_id == "structured-selector"
    assert result.diagnosis_payload_sha256 == payload.diagnosis_payload_sha256
    assert "The locked governor disposition is ABSTAIN." in result.rendered_markdown
