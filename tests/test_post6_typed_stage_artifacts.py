from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.audit import NullAuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.orchestration.models import ExecutionPolicy, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.orchestration.whole_spacecraft_stage_executor import ONLINE_STAGE_DEPENDENCIES
from astrograph.runtime import (
    RuntimeRequest,
    RuntimeStageArtifact,
    RuntimeStatus,
    WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    build_application_runtime,
)
from astrograph.runtime.stages import WholeSpacecraftOnlineDagStage
from astrograph.simulation import SimulationRunCompleted

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_base"
PROTOCOL = ROOT / "protocols" / "phm_project_a"


def _permissions() -> list[PermissionScope]:
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _runtime():
    return build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(), audit_store=NullAuditStore(), project_root=ROOT,
    )


def _request(tmp_path: Path, *, request_id: str = "post6") -> RuntimeRequest:
    permissions = _permissions()
    event = SimulationRunCompleted(
        event_id=f"SIM-{request_id}", run_id="heater_stuck_on",
        run_dir=str(FIXTURE / "heater_stuck_on"), nominal_run_dir=str(FIXTURE / "nominal"),
    )
    envelope = TaskEnvelope(
        task_id=f"TASK-{request_id}", task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / request_id),
            "shadow_mode": True, "reasoner_mode": "rule",
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
        run_id="heater_stuck_on", permissions=permissions,
    )
    return RuntimeRequest(
        request_id=request_id, pipeline_id=WHOLE_SPACECRAFT_PIPELINE_ID,
        payload={"envelope": envelope.model_dump(mode="json"), "context": context.model_dump(mode="json")},
    )


def _load_artifacts(output_dir: Path) -> dict[str, RuntimeStageArtifact]:
    index = json.loads((output_dir / "runtime" / "stage_artifact_index.json").read_text(encoding="utf-8"))
    return {
        stage_id: RuntimeStageArtifact.model_validate(
            json.loads((output_dir / record["path"]).read_text(encoding="utf-8"))
        )
        for stage_id, record in index["artifacts"].items()
    }


def test_online_runtime_emits_content_addressed_typed_artifacts(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = _runtime().execute(request)
    assert result.status == RuntimeStatus.SUCCEEDED
    assert len(result.stages) == 19
    assert all(item.artifact_type and item.artifact_sha256 and item.artifact_semantic_sha256 for item in result.stages)
    index = json.loads((tmp_path / "post6" / "runtime" / "stage_artifact_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "astrograph-runtime-stage-artifact-index.v1"
    # Persist is created after the persistence transaction, so the on-disk replay index
    # intentionally contains the 18 upstream artifacts. The Runtime receipt has all 19.
    assert len(index["artifacts"]) == 18
    artifacts = _load_artifacts(tmp_path / "post6")
    evidence_content = artifacts["online.evidence.v1"].content
    assert evidence_content["soft_evidence"]["schema_version"] == "astrograph-soft-evidence-bundle.v1"
    assert evidence_content["soft_evidence"]["bundle_sha256"]
    assert evidence_content["soft_evidence"]["calibrated_fact_count"] == 0
    fusion_content = artifacts["online.fusion.v1"].content
    assert fusion_content["soft_graph_shadow"]["suggested_action"] == "NO_OP"
    assert fusion_content["soft_graph_shadow"]["runtime_effect_applied"] is False
    assert fusion_content["document_claim_shadow"]["status"] == "DISABLED"
    assert fusion_content["document_claim_shadow"]["candidate_fusion_modified"] is False


def test_online_runtime_rebuilds_real_document_graph_claims_as_zero_vote_shadow(
    tmp_path: Path,
) -> None:
    trace = ROOT / (
        "governance/evidence/research/document_corpus/"
        "document_graph_retrieval_bridge_r001/document_claim_admission_trace.json"
    )
    mappings = ROOT / (
        "governance/evidence/research/document_corpus/"
        "document_claim_fusion_adapter_r001/reviewed_mapping_bundle.json"
    )
    request = _request(tmp_path, request_id="document-shadow")
    payload = request.model_dump(mode="json")
    input_payload = payload["payload"]["envelope"]["input_payload"]
    input_payload["document_claim_shadow"] = {
        "enabled": True,
        "claim_trace_path": str(trace),
        "claim_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "mapping_bundle_path": str(mappings),
        "mapping_bundle_sha256": hashlib.sha256(mappings.read_bytes()).hexdigest(),
        "source_artifact_id": "document-claim-shadow:r015:r001",
    }
    result = _runtime().execute(RuntimeRequest.model_validate(payload))
    assert result.status == RuntimeStatus.SUCCEEDED
    artifacts = _load_artifacts(tmp_path / "document-shadow")
    shadow = artifacts["online.fusion.v1"].content["document_claim_shadow"]
    assert shadow["status"] == "SHADOW_ZERO_VOTE"
    assert shadow["ranking_build"]["consumed_claim_ids"] == []
    assert shadow["candidate_fusion_modified"] is False
    persisted_shadow = json.loads(
        (tmp_path / "document-shadow" / "diagnosis" / "document_claim_shadow.json")
        .read_text(encoding="utf-8")
    )
    assert persisted_shadow == shadow
    structured = json.loads(
        (tmp_path / "document-shadow" / "report" / "structured_diagnosis_output.json")
        .read_text(encoding="utf-8")
    )
    final_shadow = structured["knowledge_and_retrieval_summary"][
        "document_graph_diagnosis_shadow"
    ]
    assert final_shadow["status"] == "SHADOW_ZERO_VOTE"
    assert final_shadow["ignored_ineligible_claim_count"] == 24
    assert final_shadow["candidate_fusion_modified"] is False
    assert "document_claim_shadow" in structured["lineage_sha256"]


def test_stage_artifact_hash_tampering_is_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert _runtime().execute(request).status == RuntimeStatus.SUCCEEDED
    artifacts = _load_artifacts(tmp_path / "post6")
    raw = artifacts["online.evidence.v1"].model_dump(mode="json")
    raw["content"]["builder_version"] = "tampered"
    with pytest.raises(ValidationError, match="content hash mismatch"):
        RuntimeStageArtifact.model_validate(raw)


def test_single_stage_replay_matches_deterministic_core_semantics(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runtime = _runtime()
    result = runtime.execute(request)
    assert result.status == RuntimeStatus.SUCCEEDED
    artifacts = _load_artifacts(tmp_path / "post6")
    replay, receipt = runtime.replay_stage(
        request, stage_id="online.deterministic_core.v1",
        upstream_artifacts={"online.evidence.v1": artifacts["online.evidence.v1"]},
    )
    original = next(item for item in result.stages if item.stage_id == "online.deterministic_core.v1")
    assert replay.artifact is not None
    assert receipt.artifact_semantic_sha256 == original.artifact_semantic_sha256


def test_single_stage_replay_requires_all_declared_inputs(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runtime = _runtime()
    assert runtime.execute(request).status == RuntimeStatus.SUCCEEDED
    artifacts = _load_artifacts(tmp_path / "post6")
    with pytest.raises(ValueError, match="missing replay upstream artifacts"):
        runtime.replay_stage(
            request, stage_id="online.evidence.v1",
            upstream_artifacts={"online.prepare.v1": artifacts["online.prepare.v1"]},
        )


def test_artifact_provenance_exactly_matches_declared_dag_inputs(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert _runtime().execute(request).status == RuntimeStatus.SUCCEEDED
    artifacts = _load_artifacts(tmp_path / "post6")
    for stage_id, artifact in artifacts.items():
        assert set(artifact.provenance.upstream_artifact_sha256) == set(
            WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES[stage_id]
        )
    assert ONLINE_STAGE_DEPENDENCIES == WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES


def test_evidence_stage_replay_is_semantically_stable(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runtime = _runtime()
    result = runtime.execute(request)
    artifacts = _load_artifacts(tmp_path / "post6")
    deps = WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES["online.evidence.v1"]
    replay, _ = runtime.replay_stage(
        request, stage_id="online.evidence.v1",
        upstream_artifacts={dep: artifacts[dep] for dep in deps},
    )
    original = next(item for item in result.stages if item.stage_id == "online.evidence.v1")
    assert replay.artifact is not None
    assert replay.artifact.semantic_sha256 == original.artifact_semantic_sha256


def test_cross_request_artifact_replay_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime()
    first = _request(tmp_path, request_id="scope-a")
    assert runtime.execute(first).status == RuntimeStatus.SUCCEEDED
    artifacts = _load_artifacts(tmp_path / "scope-a")
    second = _request(tmp_path, request_id="scope-b")
    with pytest.raises(ValueError, match="execution_scope mismatch"):
        runtime.replay_stage(
            second, stage_id="online.deterministic_core.v1",
            upstream_artifacts={"online.evidence.v1": artifacts["online.evidence.v1"]},
        )


def test_canonical_runtime_stage_no_longer_advances_shared_generator() -> None:
    source = inspect.getsource(WholeSpacecraftOnlineDagStage)
    assert "next(iterator)" not in source
    assert "_session_key" not in source
    assert "_iter_runtime_dag" not in source
    assert "artifact_store=context.artifacts" in source
