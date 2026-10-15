from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from astrograph.agents.agentic import DiagnosticReasonerAgent, SafetyCriticAgent
from astrograph.data.delivery_archive import materialize_d3_delivery
from astrograph.evidence import D3RWExecutableEvidenceBuilder, D3RWExecutableEvidenceCatalog
from astrograph.investigation import DecisionGovernor, DecisionDisposition
from astrograph.knowledge.evidence_graph import build_episode_evidence_graph
from astrograph.orchestration.agentic_investigation_workflow import AgenticInvestigationWorkflow
from astrograph.orchestration.models import (
    ExecutionPolicy,
    ExecutionStatus,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.semantics import TelemetrySemanticsRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = PROJECT_ROOT / "protocols" / "phm_project_a"
CATALOG_PATH = PROJECT_ROOT / "configs" / "reaction_wheel_evidence.json"


def _row(run_id: str, t: float, *, friction_wheel: int | None = None, torque_wheel: int | None = None) -> dict:
    row: dict[str, float | str] = {"record_id": f"{run_id}-{t}", "run_id": run_id, "time_s": t}
    for wheel in (0, 1, 2):
        active_friction = friction_wheel == wheel and 2 <= t <= 4
        active_torque = torque_wheel == wheel and 2 <= t <= 4
        row[f"sensor.adcs.rw.speed_rad_s_{wheel}"] = 0.5 if (active_friction or active_torque) else 0.0
        row[f"sensor.adcs.rw.motor_current_a_{wheel}"] = 0.2 if (active_friction or active_torque) else 0.0
        row[f"sensor.adcs.rw.winding_temp_c_{wheel}"] = 0.02 if active_friction else 0.0
        row[f"status.adcs.rw.available_{wheel}"] = 0.0 if active_torque else 1.0
        row[f"status.adcs.rw.driver_code_{wheel}"] = 2.0 if active_torque else 0.0
        row[f"adcs.rw.command_torque_nm_{wheel}"] = 0.1 if (active_friction or active_torque) else 0.0
        row[f"estimate.adcs.rw.actual_torque_nm_{wheel}"] = 0.1 if active_torque else 0.0
        row[f"estimate.adcs.rw.friction_torque_nm_{wheel}"] = 0.01 if active_friction else 0.0
    for axis in "xyz":
        row[f"sensor.adcs.gyro_rate_rad_s_{axis}"] = 0.001 if (friction_wheel is not None or torque_wheel is not None) and 2 <= t <= 4 else 0.0
    return row


def _write_fake_delivery(root: Path) -> Path:
    delivery = root / "PHM_D3_RW_FAKE"
    aggregate = delivery / "02_dataset" / "aggregate"
    audit = delivery / "03_run_audit"
    aggregate.mkdir(parents=True)
    runs = [
        {
            "run_id": "nominal-run",
            "case_type": "nominal",
            "case_id": "nominal",
            "fault_id": None,
            "repeat_index": 0,
            "run_seed": 1,
            "pair_seed": 1,
            "paired_nominal_run_id": None,
        },
        {
            "run_id": "friction-run",
            "case_type": "fault",
            "case_id": "rw_friction_increase",
            "fault_id": "ADCS_RW_FRICTION_INCREASE",
            "repeat_index": 0,
            "run_seed": 2,
            "pair_seed": 1,
            "paired_nominal_run_id": "nominal-run",
        },
        {
            "run_id": "torque-run",
            "case_type": "fault",
            "case_id": "rw_torque_authority_loss",
            "fault_id": "ADCS_RW_TORQUE_AUTHORITY_LOSS",
            "repeat_index": 0,
            "run_seed": 3,
            "pair_seed": 1,
            "paired_nominal_run_id": "nominal-run",
        },
    ]
    (delivery / "02_dataset" / "campaign_plan.json").write_text(
        json.dumps({"runs": runs}), encoding="utf-8"
    )
    (delivery / "02_dataset" / "dataset_manifest.json").write_text(
        json.dumps({"dataset_id": "fake-d3-rw"}), encoding="utf-8"
    )
    (delivery / "02_dataset" / "dataset_split_manifest.json").write_text(
        json.dumps(
            {
                "train_run_ids": ["nominal-run", "friction-run", "torque-run"],
                "validation_run_ids": [],
                "test_run_ids": [],
                "zero_shot_validation_run_ids": [],
            }
        ),
        encoding="utf-8",
    )
    rows: list[dict] = []
    for t in range(6):
        rows.append(_row("nominal-run", float(t)))
        rows.append(_row("friction-run", float(t), friction_wheel=1))
        rows.append(_row("torque-run", float(t), torque_wheel=2))
    pd.DataFrame(rows).to_csv(aggregate / "telemetry_adcs_raw.csv", index=False)
    for run in runs:
        directory = audit / run["run_id"]
        directory.mkdir(parents=True)
        manifest = {
            **run,
            "duration_s": 5.0,
            "recording_periods_s": {"adcs": 1.0, "eps": 5.0, "comm": 10.0, "thermal": 30.0},
        }
        (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        values = {
            "actual_fault_onset_s": 2.0,
            "actual_fault_duration_s": 2.0,
            "wheel_index": 1 if run["run_id"] == "friction-run" else 2,
        }
        (directory / "parameter_sample.json").write_text(
            json.dumps({"sampled_values": values}), encoding="utf-8"
        )
    return delivery


def _context() -> WorkflowContext:
    return WorkflowContext(
        workflow_id=AgenticInvestigationWorkflow.workflow_id,
        run_id="test-run",
        permissions=[
            PermissionScope.READ_DATA,
            PermissionScope.READ_KNOWLEDGE,
            PermissionScope.RUN_DIAGNOSIS,
            PermissionScope.WRITE_AUDIT,
        ],
    )


def test_production_semantics_registry_closes_current_field_protocol():
    registry = TelemetrySemanticsRegistry(PROTOCOL_ROOT)
    assert registry.validation.passed is True
    assert registry.validation.field_count == 67
    assert registry.event_window_semantics["interval"] == "closed"
    assert registry.rw_installation["0"]["body_axis"] == "+Bx"
    assert registry.get("estimate.adcs.rw.friction_torque_nm_1").component_instance == "RW1"
    assert "must not be interpreted" in registry.get(
        "estimate.adcs.rw.friction_torque_nm_1"
    ).limitations


def test_friction_evidence_bundle_is_label_free_and_closed_window(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    semantics = TelemetrySemanticsRegistry(PROTOCOL_ROOT)
    catalog = D3RWExecutableEvidenceCatalog(CATALOG_PATH)
    bundle = D3RWExecutableEvidenceBuilder(
        delivery, semantics_registry=semantics, catalog=catalog
    ).build_run("friction-run")
    assert bundle.selected_wheel_index == 1
    assert bundle.episode.interval == "closed"
    assert bundle.episode.start_s == 2.0
    assert bundle.episode.end_s == 4.0
    assert bundle.episode.source_is_label_free is True
    assert bundle.evaluation_metadata["injected_wheel_is_model_input"] is False
    facts = {item.evidence_definition_id: item.status.value for item in bundle.evidence_facts}
    assert facts["ED-RW-FRIC-DIRECT-V1"] == "SATISFIED"
    assert facts["ED-RW-FRIC-AVAILABLE-V1"] == "SATISFIED"
    assert facts["ED-RW-TORQUE-AVAILABLE-DROP-V1"] == "NOT_SATISFIED"


def test_reasoner_critic_governor_and_dynamic_graph_match_friction(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    semantics = TelemetrySemanticsRegistry(PROTOCOL_ROOT)
    catalog = D3RWExecutableEvidenceCatalog(CATALOG_PATH)
    builder = D3RWExecutableEvidenceBuilder(delivery, semantics_registry=semantics, catalog=catalog)
    bundle = builder.build_run("friction-run")
    hypotheses = DiagnosticReasonerAgent().reason(bundle)
    status = {item.target_fault_id: item.status.value for item in hypotheses.hypotheses}
    assert status["ADCS_RW_FRICTION_INCREASE"] == "SUPPORTED"
    assert status["ADCS_RW_TORQUE_AUTHORITY_LOSS"] == "CONTRADICTED"
    critique = SafetyCriticAgent().critique(hypotheses, bundle)
    assert critique.blocking is False
    decision = DecisionGovernor().decide(
        hypothesis_set=hypotheses,
        critique=critique,
        deterministic_core_result={
            "diagnosis_status": "DIAGNOSED",
            "predicted_case_id": "rw_friction_increase",
        },
        protocol_versions={"astrograph": "0.5.0"},
    )
    assert decision.disposition == DecisionDisposition.ACCEPT
    assert decision.accepted_fault_id == "ADCS_RW_FRICTION_INCREASE"
    assert decision.comparison_status == "MATCH"
    assert decision.formal_release_allowed is False
    graph = build_episode_evidence_graph(bundle, hypotheses, decision)
    assert graph.persistent_raw_telemetry is False
    assert any(edge.relation_type == "SUPPORTED_BY" for edge in graph.edges)
    assert any(node.node_type == "DecisionRecord" for node in graph.nodes)


def test_agentic_workflow_matches_torque_core_without_changing_official_result(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    envelope = TaskEnvelope(
        task_id="task-torque",
        task_type="agentic_investigation",
        input_payload={
            "d3_delivery_path": str(delivery),
            "run_id": "torque-run",
            "protocol_root": str(PROTOCOL_ROOT),
            "evidence_catalog_path": str(CATALOG_PATH),
            "shadow_mode": True,
        },
    )
    result = AgenticInvestigationWorkflow().invoke(envelope, _context())
    assert result.status == ExecutionStatus.SUCCEEDED
    output = result.output_payload
    assert output["mode"] == "SHADOW"
    assert output["formal_result_changed"] is False
    assert output["episode_evidence_bundle"]["selected_wheel_index"] == 2
    assert output["decision_record"]["accepted_fault_id"] == "ADCS_RW_TORQUE_AUTHORITY_LOSS"
    assert output["decision_record"]["comparison_status"] == "MATCH"
    assert output["decision_record"]["formal_release_allowed"] is False
    assert len(output["agent_trace"]["events"]) == 10
    assert [event["tool_id"] for event in output["agent_trace"]["events"] if event["tool_id"]] == [
        "semantics.validate",
        "evidence.catalog.read",
        "d3_rw.evidence.build",
        "diagnosis.invoke_core",
        "evidence_graph.build",
    ]


def test_agentic_workflow_permission_and_llm_policy_fail_closed(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    payload = {
        "d3_delivery_path": str(delivery),
        "run_id": "friction-run",
        "protocol_root": str(PROTOCOL_ROOT),
        "evidence_catalog_path": str(CATALOG_PATH),
        "shadow_mode": True,
    }
    denied = AgenticInvestigationWorkflow().invoke(
        TaskEnvelope(task_id="denied", task_type="agentic", input_payload=payload),
        WorkflowContext(workflow_id="wf", run_id="denied", permissions=[]),
    )
    assert denied.status == ExecutionStatus.FAILED
    assert denied.error is not None
    assert denied.error.code == "ASTROGRAPH_AGENTIC_PERMISSION_DENIED"

    class FakeBackend:
        model_id = "fake"
        model_version = "1"

        def generate_structured(self, **kwargs):  # pragma: no cover - blocked before invocation
            raise AssertionError("backend must not be called")

    blocked = AgenticInvestigationWorkflow(reasoning_backend=FakeBackend()).invoke(
        TaskEnvelope(task_id="llm", task_type="agentic", input_payload=payload),
        _context(),
    )
    assert blocked.status == ExecutionStatus.FAILED
    assert blocked.error is not None
    assert blocked.error.code == "ASTROGRAPH_LLM_BLOCKED_BY_DETERMINISTIC_POLICY"

    no_permission = AgenticInvestigationWorkflow(reasoning_backend=FakeBackend()).invoke(
        TaskEnvelope(
            task_id="llm2",
            task_type="agentic",
            input_payload=payload,
            execution_policy=ExecutionPolicy(deterministic_only=False, allow_llm=False),
        ),
        _context(),
    )
    assert no_permission.status == ExecutionStatus.FAILED
    assert no_permission.error is not None
    assert no_permission.error.code == "ASTROGRAPH_LLM_NOT_ALLOWED"


def test_windows_backslash_delivery_zip_is_materialized(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path / "source")
    archive_path = tmp_path / "delivery.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in delivery.rglob("*"):
            if path.is_file():
                relative = path.relative_to(delivery.parent).as_posix().replace("/", "\\")
                archive.write(path, relative)
    with materialize_d3_delivery(archive_path) as materialized:
        assert (materialized / "02_dataset" / "dataset_manifest.json").exists()
        assert (materialized / "03_run_audit" / "friction-run" / "run_manifest.json").exists()
        assert not (materialized / "unrelated-source-snapshot").exists()


def test_agentic_tool_registry_is_read_only_and_allow_listed():
    workflow = AgenticInvestigationWorkflow()
    specs = {item.tool_id: item for item in workflow.tool_registry.list_specs()}
    assert set(specs) == {
        "semantics.validate",
        "evidence.catalog.read",
        "d3_rw.evidence.build",
        "diagnosis.invoke_core",
        "evidence_graph.build",
    }
    assert all(item.deterministic for item in specs.values())
    assert all(item.side_effect_level.value == "READ_ONLY" for item in specs.values())
    try:
        workflow.tool_registry.get("python.execute_arbitrary")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("arbitrary Python must not be registered")


def test_structured_reasoning_backend_is_policy_gated_and_allowlist_checked(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    payload = {
        "d3_delivery_path": str(delivery),
        "run_id": "friction-run",
        "protocol_root": str(PROTOCOL_ROOT),
        "evidence_catalog_path": str(CATALOG_PATH),
        "shadow_mode": True,
    }

    class ReferenceBackend:
        model_id = "structured-reference-test"
        model_version = "1"

        def generate_structured(self, *, agent_id, output_model, payload):
            return output_model.model_validate(payload["reference_hypothesis_set"])

    allowed = AgenticInvestigationWorkflow(reasoning_backend=ReferenceBackend()).invoke(
        TaskEnvelope(
            task_id="llm-allowed",
            task_type="agentic",
            input_payload=payload,
            execution_policy=ExecutionPolicy(deterministic_only=False, allow_llm=True),
        ),
        _context(),
    )
    assert allowed.status == ExecutionStatus.SUCCEEDED
    assert allowed.output_payload["agent_trace"]["model_id"] == "structured-reference-test"

    class InventedFaultBackend(ReferenceBackend):
        model_id = "invented-fault-test"

        def generate_structured(self, *, agent_id, output_model, payload):
            document = dict(payload["reference_hypothesis_set"])
            document["hypotheses"] = [dict(item) for item in document["hypotheses"]]
            document["hypotheses"][0]["target_fault_id"] = "INVENTED_FAULT"
            return output_model.model_validate(document)

    rejected = AgenticInvestigationWorkflow(reasoning_backend=InventedFaultBackend()).invoke(
        TaskEnvelope(
            task_id="llm-rejected",
            task_type="agentic",
            input_payload=payload,
            execution_policy=ExecutionPolicy(deterministic_only=False, allow_llm=True),
        ),
        _context(),
    )
    assert rejected.status == ExecutionStatus.FAILED
    assert rejected.error is not None
    assert rejected.error.code == "ASTROGRAPH_AGENTIC_INVESTIGATION_FAILED"
    assert "invented unregistered fault" in rejected.error.message


def test_nominal_run_returns_no_registered_fault_without_claiming_absolute_health(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    result = AgenticInvestigationWorkflow().invoke(
        TaskEnvelope(
            task_id="task-nominal",
            task_type="agentic_investigation",
            input_payload={
                "d3_delivery_path": str(delivery),
                "run_id": "nominal-run",
                "protocol_root": str(PROTOCOL_ROOT),
                "evidence_catalog_path": str(CATALOG_PATH),
                "shadow_mode": True,
            },
        ),
        _context(),
    )
    assert result.status == ExecutionStatus.SUCCEEDED
    output = result.output_payload
    assert output["decision_record"]["disposition"] == "NO_REGISTERED_FAULT"
    assert output["decision_record"]["accepted_fault_id"] is None
    assert output["decision_record"]["comparison_status"] == "MATCH"
    no_fault = [
        item
        for item in output["hypothesis_set"]["hypotheses"]
        if item["hypothesis_type"] == "NO_REGISTERED_FAULT"
    ]
    assert len(no_fault) == 1
    assert no_fault[0]["status"] == "SUPPORTED"
    assert "不表示硬件绝对健康" in no_fault[0]["reasoning_summary"]
    assert output["episode_evidence_bundle"]["episode"]["source"] == "full_run_no_mechanism_episode"
    assert output["episode_evidence_bundle"]["analysis_wheel_index"] == 0
    assert output["episode_evidence_bundle"]["selected_wheel_index"] is None
    assert output["episode_evidence_bundle"]["wheel_selection_status"] == "NO_MECHANISM_DETECTED"


def test_v052_hypothesis_metrics_match_reference(tmp_path: Path):
    from astrograph.evaluation import evaluate_hypothesis_set

    delivery = _write_fake_delivery(tmp_path)
    semantics = TelemetrySemanticsRegistry(PROTOCOL_ROOT)
    catalog = D3RWExecutableEvidenceCatalog(CATALOG_PATH)
    bundle = D3RWExecutableEvidenceBuilder(
        delivery, semantics_registry=semantics, catalog=catalog
    ).build_run("friction-run")
    reference = DiagnosticReasonerAgent.reference_hypotheses(bundle)
    metrics = evaluate_hypothesis_set(reference, reference)
    assert metrics.status_match_rate == 1.0
    assert metrics.supporting_evidence.f1 == 1.0
    assert metrics.contradicting_evidence.f1 == 1.0
    assert metrics.missing_evidence.f1 == 1.0
    assert metrics.no_fault_semantic_match is True
    assert metrics.exact_structural_match is True


def test_v052_unit_mismatch_is_blocked_and_routed_to_human_review(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    envelope = TaskEnvelope(
        task_id="task-unit-perturbation",
        task_type="agentic_investigation",
        input_payload={
            "d3_delivery_path": str(delivery),
            "run_id": "friction-run",
            "protocol_root": str(PROTOCOL_ROOT),
            "evidence_catalog_path": str(CATALOG_PATH),
            "shadow_mode": True,
            "evidence_perturbations": [
                {
                    "perturbation_id": "P-UNIT",
                    "perturbation_type": "UNIT_MISMATCH",
                    "feature_definition_id": "FD-RW-FRICTION-PEAK-ABS-V1",
                    "replacement_unit": "rpm",
                }
            ],
        },
    )
    result = AgenticInvestigationWorkflow().invoke(envelope, _context())
    assert result.status == ExecutionStatus.SUCCEEDED
    output = result.output_payload
    codes = {item["code"] for item in output["critique_result"]["findings"]}
    assert "FEATURE_UNIT_MISMATCH" in codes
    assert "EVIDENCE_UNIT_MISMATCH" in codes
    assert output["critique_result"]["blocking"] is True
    assert output["decision_record"]["disposition"] == "HUMAN_REVIEW"
    assert output["decision_record"]["requires_human_review"] is True
    assert len(output["evaluation_perturbations"]) == 1


def test_v052_missing_critical_evidence_does_not_silently_accept_fault(tmp_path: Path):
    delivery = _write_fake_delivery(tmp_path)
    envelope = TaskEnvelope(
        task_id="task-drop-perturbation",
        task_type="agentic_investigation",
        input_payload={
            "d3_delivery_path": str(delivery),
            "run_id": "friction-run",
            "protocol_root": str(PROTOCOL_ROOT),
            "evidence_catalog_path": str(CATALOG_PATH),
            "shadow_mode": True,
            "evidence_perturbations": [
                {
                    "perturbation_id": "P-DROP",
                    "perturbation_type": "DROP_EVIDENCE",
                    "evidence_definition_id": "ED-RW-FRIC-DIRECT-V1",
                }
            ],
        },
    )
    result = AgenticInvestigationWorkflow().invoke(envelope, _context())
    assert result.status == ExecutionStatus.SUCCEEDED
    output = result.output_payload
    friction = next(
        item for item in output["hypothesis_set"]["hypotheses"]
        if item["target_fault_id"] == "ADCS_RW_FRICTION_INCREASE"
    )
    assert friction["status"] == "INSUFFICIENT_EVIDENCE"
    assert "ED-RW-FRIC-DIRECT-V1" in friction["missing_evidence_definition_ids"]
    assert output["decision_record"]["disposition"] == "HUMAN_REVIEW"
