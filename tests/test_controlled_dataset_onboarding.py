from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from astrograph.agents.dataset_onboarding import (
    ControlledDatasetOnboardingWorkflow,
    DatasetProfilerAgent,
    SchemaMappingProposal,
    SchemaMappingProposalAgent,
)


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SCHEMA = ROOT / "configs" / "semantics" / "core_semantic_schema.json"


def _telemetry(path: Path) -> Path:
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "wheel_speed_rpm": [100.0, 101.0, 102.0],
            "label.fault_id": ["NORMAL", "FAULT", "FAULT"],
        }
    ).to_csv(path, index=False)
    return path


def _draft(*, schema_sha: str, **candidate_updates: Any) -> dict[str, Any]:
    candidate = {
        "source_field": "wheel_speed_rpm",
        "target_quantity_kind_id": "mechanical.angular_rate",
        "component_instance": "reaction_wheel_0",
        "source_unit": "rpm",
        "canonical_unit": "rad/s",
        "transform_operator_id": "unit_convert",
        "confidence": 0.97,
        "ambiguous": False,
        "rationale": "The source dictionary identifies wheel rotational speed.",
        "source_references": ["vendor.telemetry_dictionary:wheel_speed_rpm"],
    }
    candidate.update(candidate_updates)
    return {
        "dataset_id": "new.dataset",
        "source_schema_sha256": schema_sha,
        "target_semantic_schema_id": "astrograph.core.physical-quantities.v1",
        "candidates": [candidate],
        "unresolved_source_fields": ["time_s"],
        "proposer_id": "draft",
        "proposer_version": "1",
    }


APPROVED_COMPONENTS = ["reaction_wheel_0"]
APPROVED_REFERENCES = ["vendor.telemetry_dictionary:wheel_speed_rpm"]


def _run_kwargs() -> dict[str, list[str]]:
    return {
        "approved_component_instances": APPROVED_COMPONENTS,
        "approved_source_references": APPROVED_REFERENCES,
    }


def test_controlled_onboarding_is_read_only_and_ready_only_for_human_review(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path / "telemetry.csv")
    profile = DatasetProfilerAgent().profile(
        dataset_id="new.dataset",
        telemetry_path=telemetry,
        declared_units={"time_s": "s", "wheel_speed_rpm": "rpm"},
    )
    before = sorted(path.name for path in tmp_path.iterdir())
    result = ControlledDatasetOnboardingWorkflow().run(
        dataset_id="new.dataset",
        telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        declared_units={"time_s": "s", "wheel_speed_rpm": "rpm"},
        draft_proposal=_draft(schema_sha=profile.schema_sha256),
        **_run_kwargs(),
    )
    after = sorted(path.name for path in tmp_path.iterdir())

    assert result.validation.status == "READY_FOR_HUMAN_REVIEW"
    assert result.validation.publication_allowed is False
    assert result.validation.requires_human_approval is True
    assert result.files_written == []
    assert before == after == ["telemetry.csv"]
    assert [step.agent_id for step in result.trace] == [
        "dataset-profiler-agent",
        "schema-mapping-proposal-agent",
        "mapping-contract-critic-agent",
    ]
    prohibited = {item.field for item in result.profile.columns if item.prohibited_for_runtime_mapping}
    assert prohibited == {"label.fault_id"}


def test_contract_critic_blocks_each_safety_boundary(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path / "telemetry.csv")
    profile = DatasetProfilerAgent().profile(dataset_id="new.dataset", telemetry_path=telemetry)
    cases = {
        "PROHIBITED_SOURCE_FIELD": {"source_field": "label.fault_id"},
        "UNKNOWN_SEMANTIC_TARGET": {"target_quantity_kind_id": "invented.quantity"},
        "INCOMPATIBLE_UNIT_DIMENSION": {"source_unit": "K"},
        "CANONICAL_UNIT_MISMATCH": {"canonical_unit": "K"},
        "UNREGISTERED_OPERATOR": {"transform_operator_id": "execute_generated_python"},
        "LOW_CONFIDENCE": {"confidence": 0.2},
        "AMBIGUOUS_MAPPING": {"ambiguous": True},
        "UNAPPROVED_COMPONENT_INSTANCE": {"component_instance": "invented_component"},
        "UNAPPROVED_SOURCE_REFERENCE": {"source_references": ["invented_reference"]},
    }
    for expected_code, updates in cases.items():
        result = ControlledDatasetOnboardingWorkflow().run(
            dataset_id="new.dataset",
            telemetry_path=telemetry,
            semantic_schema_path=SEMANTIC_SCHEMA,
            draft_proposal=_draft(schema_sha=profile.schema_sha256, **updates),
            **_run_kwargs(),
        )
        assert result.validation.status == "BLOCKED"
        assert expected_code in {item.code for item in result.validation.findings}

    missing_unit_result = ControlledDatasetOnboardingWorkflow().run(
        dataset_id="new.dataset",
        telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        draft_proposal=_draft(schema_sha=profile.schema_sha256),
        **_run_kwargs(),
    )
    assert "MISSING_DECLARED_SOURCE_UNIT" in {
        item.code for item in missing_unit_result.validation.findings
    }

    declared_result = ControlledDatasetOnboardingWorkflow().run(
        dataset_id="new.dataset",
        telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        declared_units={"wheel_speed_rpm": "deg/s"},
        draft_proposal=_draft(schema_sha=profile.schema_sha256),
        **_run_kwargs(),
    )
    assert "SOURCE_DECLARED_UNIT_MISMATCH" in {
        item.code for item in declared_result.validation.findings
    }


class _RecordingBackend:
    model_id = "test-structured-model"
    model_version = "1"

    def __init__(self, proposal: SchemaMappingProposal) -> None:
        self.proposal = proposal
        self.payload: dict[str, Any] | None = None

    def generate_structured(self, *, agent_id: str, output_model: type, payload: dict[str, Any]):
        assert agent_id == "schema-mapping-proposal-agent"
        assert output_model is SchemaMappingProposal
        self.payload = payload
        return self.proposal


def test_llm_backend_receives_schema_only_and_cannot_self_publish(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path / "telemetry.csv")
    profile = DatasetProfilerAgent().profile(dataset_id="new.dataset", telemetry_path=telemetry)
    proposal = SchemaMappingProposal.model_validate(_draft(schema_sha=profile.schema_sha256))
    backend = _RecordingBackend(proposal)
    workflow = ControlledDatasetOnboardingWorkflow(
        proposer=SchemaMappingProposalAgent(backend=backend)
    )
    result = workflow.run(
        dataset_id="new.dataset",
        telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        declared_units={"time_s": "s", "wheel_speed_rpm": "rpm"},
        approved_component_instances=APPROVED_COMPONENTS,
        approved_source_references=APPROVED_REFERENCES,
    )

    assert backend.payload is not None
    disclosed_fields = {item["field"] for item in backend.payload["source_columns"]}
    assert disclosed_fields == {"time_s", "wheel_speed_rpm"}
    assert "label.fault_id" not in disclosed_fields
    assert "source_values" not in backend.payload
    assert result.proposal.proposer_mode == "LLM"
    assert result.proposal.proposer_id == backend.model_id
    assert result.validation.publication_allowed is False


def test_schema_hash_drift_is_blocked(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path / "telemetry.csv")
    profile = DatasetProfilerAgent().profile(dataset_id="new.dataset", telemetry_path=telemetry)
    draft = _draft(schema_sha="0" * 64)
    result = ControlledDatasetOnboardingWorkflow().run(
        dataset_id="new.dataset",
        telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        draft_proposal=draft,
        **_run_kwargs(),
    )
    assert result.validation.status == "BLOCKED"
    assert "SOURCE_SCHEMA_HASH_MISMATCH" in {item.code for item in result.validation.findings}
    assert profile.schema_sha256 != "0" * 64


def test_ambiguous_mode_is_raised_for_role_review_but_remains_unmappable(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    pd.DataFrame({
        "time_s": [0.0, 1.0],
        "wheel_speed_rpm": [100.0, 101.0],
        "mode": ["NORMAL", "FAULT"],
        "label.fault_id": ["NORMAL", "FAULT"],
    }).to_csv(telemetry, index=False)
    profile = DatasetProfilerAgent().profile(dataset_id="new.dataset", telemetry_path=telemetry)
    mode = next(item for item in profile.columns if item.field == "mode")
    assert mode.prohibited_for_runtime_mapping is True
    assert mode.requires_source_role_review is True
    proposal = SchemaMappingProposal.model_validate({
        **_draft(schema_sha=profile.schema_sha256),
        "role_review_questions": ["Does mode describe an independently logged operating condition?"],
    })
    backend = _RecordingBackend(proposal)
    result = ControlledDatasetOnboardingWorkflow(
        proposer=SchemaMappingProposalAgent(backend=backend)
    ).run(
        dataset_id="new.dataset", telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        declared_units={"time_s": "s", "wheel_speed_rpm": "rpm"},
        **_run_kwargs(),
    )
    assert backend.payload is not None
    assert backend.payload["role_review_candidates"][0]["field"] == "mode"
    assert "mode" not in {item["field"] for item in backend.payload["source_columns"]}
    assert "SOURCE_ROLE_REVIEW_REQUIRED" in {item.code for item in result.validation.findings}
    assert result.validation.publication_allowed is False
    attempted_mapping = ControlledDatasetOnboardingWorkflow().run(
        dataset_id="new.dataset", telemetry_path=telemetry,
        semantic_schema_path=SEMANTIC_SCHEMA,
        draft_proposal=_draft(schema_sha=profile.schema_sha256, source_field="mode"),
        **_run_kwargs(),
    )
    assert attempted_mapping.validation.status == "BLOCKED"
    assert "PROHIBITED_SOURCE_FIELD" in {item.code for item in attempted_mapping.validation.findings}
