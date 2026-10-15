from __future__ import annotations

import json
from pathlib import Path

from astrograph.architecture import load_stable_layer_policy, scan_stable_layers
from astrograph.audit import NullAuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.contracts import STABLE_CONTRACTS
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.runtime import (
    AstroGraphRuntime,
    DIAGNOSIS_PIPELINE_ID,
    UNIFIED_DIAGNOSIS_PIPELINE_ID,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    RuntimePipeline,
    RuntimeRequest,
    RuntimeStatus,
    StageResult,
    build_application_runtime,
)
from astrograph.service.core import AstroGraphService


ROOT = Path(__file__).resolve().parents[1]


class RecordingStage:
    stage_id = "test.recording.v1"
    stage_version = "1.0"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def execute(self, payload, context):
        self.seen.append(dict(payload))
        return StageResult.succeeded({
            "schema_version": "astrograph-diagnosis-api-response.v3",
            "request_id": payload["request_id"],
            "case_id": payload["case_id"],
            "diagnosis": {"status": "INSUFFICIENT_EVIDENCE"},
            "conflict_explanation": {},
            "report_markdown": "runtime-owned",
            "audit": {"audit_id": "runtime-test"},
            "runtime": {"pipeline_id": DIAGNOSIS_PIPELINE_ID, "stage_id": self.stage_id},
        })


def test_stable_contract_bundle_has_one_contract_per_control_plane() -> None:
    assert STABLE_CONTRACTS.model.schema_version == "astrograph-model-contract.v1"
    assert STABLE_CONTRACTS.evidence.schema_version == "astrograph-evidence-contract.v1"
    assert STABLE_CONTRACTS.knowledge.schema_version == "astrograph-knowledge-contract.v1"
    assert STABLE_CONTRACTS.policy.schema_version == "astrograph-policy-contract.v1"
    assert STABLE_CONTRACTS.model.raw_pickle_allowed is False
    assert STABLE_CONTRACTS.policy.dataset_profile_may_override is False


def test_application_runtime_is_the_single_pipeline_registry() -> None:
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data/knowledge_seed/causes.csv"),
        policy=DiagnosisPolicy(),
        audit_store=NullAuditStore(),
        project_root=ROOT,
    )
    assert runtime.pipeline_ids == tuple(sorted((
        DIAGNOSIS_PIPELINE_ID,
        UNIFIED_DIAGNOSIS_PIPELINE_ID,
        WHOLE_SPACECRAFT_PIPELINE_ID,
    )))
    description = runtime.describe()
    assert description["runtime_id"] == "astrograph.runtime"
    assert description["status"] == "PASS"


def test_unknown_pipeline_fails_closed() -> None:
    stage = RecordingStage()
    runtime = AstroGraphRuntime([RuntimePipeline("known", (stage,))])
    result = runtime.execute(RuntimeRequest(request_id="r-1", pipeline_id="missing", payload={}))
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_RUNTIME_PIPELINE_NOT_FOUND"
    assert stage.seen == []


def test_service_diagnosis_delegates_to_canonical_runtime(tmp_path: Path) -> None:
    stage = RecordingStage()
    runtime = AstroGraphRuntime([RuntimePipeline(DIAGNOSIS_PIPELINE_ID, (stage,))])
    service = AstroGraphService(
        repository=CsvKnowledgeRepository(tmp_path / "missing.csv"),
        policy=DiagnosisPolicy(),
        runtime=runtime,
    )
    from astrograph.schemas import WorkflowInput

    request = WorkflowInput(case_id="CASE-STABLE", request_id="req-stable", anomalies=[])
    response = service.diagnose(request)
    assert response["runtime"]["stage_id"] == stage.stage_id
    assert len(stage.seen) == 1
    assert stage.seen[0]["case_id"] == "CASE-STABLE"


def test_stable_layer_policy_has_no_dependency_inversions() -> None:
    policy = load_stable_layer_policy(ROOT / "configs/architecture/stable_layers.json")
    report = scan_stable_layers(ROOT, policy)
    assert report.status == "PASS", report.model_dump(mode="json")


def test_dataset_profiles_reference_policy_but_do_not_own_fusion_config() -> None:
    for filename in ("adcs_v5_2_2.json", "phm_v6_0.json"):
        payload = json.loads((ROOT / "configs/dataset_profiles" / filename).read_text(encoding="utf-8"))
        assert payload["fusion_policy_profile_id"] == "birrf.engineering-default.v1"
        assert "birrf_config" not in payload


def test_extension_perimeter_defaults_external_tools_offline() -> None:
    payload = json.loads((ROOT / "configs/extensions/stable_perimeter.json").read_text(encoding="utf-8"))
    assert {item["provider_name"] for item in payload["providers"]} == {"DOR", "FEC", "ISGN", "M2N"}
    assert all(item["online_default"] is False for item in payload["providers"])
