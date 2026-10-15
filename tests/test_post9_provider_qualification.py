from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from astrograph.data_driven.contracts import (
    DataDrivenModelManifest,
    DataDrivenTaskType,
    ModelCapability,
    ModelOutputSemantics,
    ModelRuntime,
)
from astrograph.data_driven.providers import (
    CallableRawModelProvider,
    ExternalServiceModelProvider,
    SklearnProbabilityProvider,
)
from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.knowledge.runtime import KGEvidenceRuntimeService
from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.provider_lifecycle import (
    BUILTIN_PROVIDER_QUALIFICATIONS,
    ProviderQualificationStatus,
)
from astrograph.runtime import (
    AstroGraphRuntime,
    CancellationMode,
    ExecutionBoundary,
    RuntimePipeline,
    RuntimeRequest,
    RuntimeStageContext,
    RuntimeStatus,
    SchedulerMode,
    StageExecutionProfile,
    StageResult,
    StageSideEffect,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    build_application_runtime,
)
from astrograph.audit import NullAuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository

ROOT = Path(__file__).resolve().parents[1]


class _PassStage:
    stage_version = "1"

    def __init__(self, stage_id: str) -> None:
        self.stage_id = stage_id

    def execute(self, payload: dict, context: RuntimeStageContext) -> StageResult:
        time.sleep(0.02)
        return StageResult.succeeded(dict(payload))


def _manifest(runtime: ModelRuntime) -> DataDrivenModelManifest:
    return DataDrivenModelManifest(
        model_id=f"m-{runtime.value.lower()}",
        model_version="1",
        display_name="test",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=runtime,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING],
        supported_fault_ids=["FAULT.A"],
        feature_schema_ids=["features.v1"],
    )


def test_builtin_provider_registry_is_conservative() -> None:
    local = BUILTIN_PROVIDER_QUALIFICATIONS.get("local-deterministic-isolated.v1")
    assert local.qualification_status == ProviderQualificationStatus.QUALIFIED
    assert local.stable_parallel_eligible is True

    for provider_id in (
        "external-cli-managed.v1",
        "neo4j-readonly.v1",
        "kg-csv-reference.v1",
        "http-json-provider.v1",
        "inprocess-model-runtime.v1",
    ):
        profile = BUILTIN_PROVIDER_QUALIFICATIONS.get(provider_id)
        assert profile.stable_parallel_eligible is False


def test_runtime_description_exposes_provider_qualification_and_stage_binding() -> None:
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(),
        audit_store=NullAuditStore(),
        project_root=ROOT,
    )
    description = runtime.describe()
    ids = {item["provider_id"] for item in description["provider_qualifications"]}
    assert "external-cli-managed.v1" in ids
    pipeline = next(item for item in description["pipelines"] if item["pipeline_id"] == WHOLE_SPACECRAFT_PIPELINE_ID)
    profiles = {item["stage_id"]: item["execution_profile"] for item in pipeline["stages"]}
    assert profiles["online.kg_evidence.v1"]["provider_qualification_ids"] == [
        "neo4j-readonly.v1", "kg-csv-reference.v1"
    ]
    assert profiles["online.reasoning.v1"]["provider_qualification_ids"] == ["http-json-provider.v1"]
    assert profiles["online.spatiotemporal.v1"]["provider_qualification_ids"] == ["local-deterministic-isolated.v1"]


def test_unqualified_provider_binding_blocks_parallel_admission_even_with_hard_cancel_claim() -> None:
    stages = (_PassStage("a"), _PassStage("b"))
    profiles = {
        stage.stage_id: StageExecutionProfile(
            stage_id=stage.stage_id,
            parallel_safe=True,
            replay_equivalence_certified=True,
            payload_passthrough=True,
            cancellation_mode=CancellationMode.COOPERATIVE,
            execution_boundary=ExecutionBoundary.IN_PROCESS,
            parallel_execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
            isolation_adapter_id="whole_spacecraft.online.v1",
            cancellation_qualification_id="test-hard-cancel",
            provider_qualification_ids=("http-json-provider.v1",),
            side_effect=StageSideEffect.READ_ONLY,
        )
        for stage in stages
    }
    runtime = AstroGraphRuntime([
        RuntimePipeline(
            pipeline_id="provider-gate",
            stages=stages,
            dependencies={"a": (), "b": ()},
            execution_profiles=profiles,
        )
    ])
    result = runtime.execute(RuntimeRequest(
        request_id="provider-gate",
        pipeline_id="provider-gate",
        payload={},
        metadata={"execution_control": {
            "scheduler_mode": SchedulerMode.SAFE_PARALLEL.value,
            "max_parallel_stages": 2,
            "max_resource_units": 2,
            "require_hard_cancel_for_parallel": True,
        }},
    ))
    assert result.status == RuntimeStatus.SUCCEEDED
    control = result.metadata["execution_control"]
    assert control["parallel_stage_count"] == 0
    assert control["parallel_activation_withheld"] is True


def test_data_driven_provider_lifecycle_is_explicit() -> None:
    callable_provider = CallableRawModelProvider(_manifest(ModelRuntime.CUSTOM_CALLABLE), lambda request: {})
    sklearn_provider = SklearnProbabilityProvider(_manifest(ModelRuntime.SCIKIT_LEARN), object(), ["x"])
    external_provider = ExternalServiceModelProvider(_manifest(ModelRuntime.EXTERNAL_SERVICE), "http://127.0.0.1:1")
    assert callable_provider.lifecycle_profile.provider_id == "inprocess-model-runtime.v1"
    assert sklearn_provider.lifecycle_profile.provider_id == "inprocess-model-runtime.v1"
    assert external_provider.lifecycle_profile.provider_id == "http-json-provider.v1"


def test_kg_and_llm_provider_lifecycle_is_explicit() -> None:
    assert KGEvidenceRuntimeService.lifecycle_profile().provider_id == "neo4j-readonly.v1"
    assert KGEvidenceRuntimeService.lifecycle_profile(selected_backend="csv_reference_fallback").provider_id == "kg-csv-reference.v1"
    backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        base_url="http://127.0.0.1:1234/v1",
        api_key="test",
        model="test",
    ))
    assert backend.lifecycle_profile.provider_id == "http-json-provider.v1"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group runtime qualification")
def test_managed_external_cli_timeout_terminates_descendant_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "tree.py"
    script.write_text(
        "import pathlib,subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_managed_subprocess(
            [sys.executable, str(script), str(child_pid_path)],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not Path(f"/proc/{child_pid}").exists():
            break
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()


def test_integrations_use_managed_subprocess_boundary() -> None:
    allowed = {ROOT / "src" / "astrograph" / "integrations" / "process_control.py"}
    offenders = []
    for path in (ROOT / "src" / "astrograph" / "integrations").glob("*.py"):
        if path in allowed:
            continue
        if "subprocess.run(" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []
