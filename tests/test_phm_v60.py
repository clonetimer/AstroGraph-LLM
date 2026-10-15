from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import hashlib

import pytest
from jsonschema import Draft202012Validator

from astrograph.data_driven.phm_v60 import audit_split, is_allowed_channel, load_records
from astrograph.evidence import (
    ConfiguredRuntimeEvidenceBuilder,
    SoftCalibrationSelectionReceipt,
    SoftEvidenceCalibrationPolicy,
    SoftEvidencePolicyStatus,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get(
    "ASTROGRAPH_PHM_V60_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0"),
))
EXTERNAL_DATASET_AVAILABLE = (DATASET / "sample_index.csv").is_file()
requires_external_dataset = pytest.mark.skipif(
    not EXTERNAL_DATASET_AVAILABLE,
    reason="optional PHM v6.0 external dataset is not configured; set ASTROGRAPH_PHM_V60_DATA_ROOT",
)


@requires_external_dataset
def test_phm_v60_group_split_is_disjoint_and_frozen() -> None:
    report = audit_split(load_records(DATASET))
    assert report["status"] == "PASS"
    assert report["group_overlap_count"] == 0
    assert [report["splits"][name]["sample_count"] for name in ("train", "validation", "test")] == [336, 112, 112]


def test_phm_v60_feature_allowlist_rejects_answer_proxies() -> None:
    assert is_allowed_channel("thermal.electronics_heater_power_w")
    for name in (
        "label.fault_active", "fault.condition", "event.active_effect", "battery.remaining_ratio",
        "fuel.leak_rate", "truth.component",
    ):
        assert not is_allowed_channel(name)


@requires_external_dataset
def test_phm_v60_hierarchy_and_strict_mapping_coverages_are_explicit() -> None:
    hierarchy = json.loads((ROOT / "configs/datasets/phm_v6_0/diagnosis_hierarchy.json").read_text(encoding="utf-8"))
    class_map = json.loads((ROOT / "configs/datasets/phm_v6_0/fault_class_map.json").read_text(encoding="utf-8"))
    coverage = json.loads((ROOT / "configs/datasets/phm_v6_0/knowledge_coverage.json").read_text(encoding="utf-8"))
    faults = {row.condition_id for row in load_records(DATASET)} - {"nominal"}
    assert set(hierarchy["fault_to_component"]) == faults
    assert set(class_map["mappings"]) <= faults
    assert len(class_map["mappings"]) == coverage["strict_registered_mapping_count"] == 5
    assert coverage["generic_tbox_binding_count"] == len(faults) == 17
    assert set(coverage["formal_fault_registration_or_runtime_evidence_required"]) == (
        faults - set(class_map["mappings"])
    )
    assert coverage["runtime_evidence_candidate_count"] == 5


def test_phm_v60_configured_evidence_package_is_valid() -> None:
    builder = ConfiguredRuntimeEvidenceBuilder(
        ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json"
    )
    assert builder.package.domain_id == "PHM_V6_0"
    assert builder.package.version == "2.0.0"
    assert all(item.max_missing_fraction == pytest.approx(0.01) for item in builder.package.channels)
    assert {item.target_fault_id for item in builder.package.evidence_definitions} == {
        "COMM_DOWNLINK_SERVICE_LOSS",
        "EPS_BATTERY_CAPACITY_LOSS",
        "EPS_SOLAR_ARRAY_STRING_FAILURE",
        "THERMAL_HEATER_STUCK_ON",
        "THERMAL_RADIATOR_REJECTION_LOSS",
    }


def test_phm_v60_evidence_calibration_receipt_pins_source_assets() -> None:
    receipt = json.loads(
        (ROOT / "configs/datasets/phm_v6_0/evidence_calibration_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["decision"] == "ACCEPT_FOR_E0_DATASET_SCOPED_RUNTIME_EVIDENCE"
    assert receipt["selection_protocol"]["development_group_ids"] == [3, 8]
    if EXTERNAL_DATASET_AVAILABLE:
        for relative, expected in receipt["source_hashes"].items():
            actual = hashlib.sha256((DATASET / relative).read_bytes()).hexdigest()
            assert actual == expected
    for row in receipt["faults"].values():
        assert row["development"]["joint_non_target_pass_count"] == 0
        assert row["untouched_screen"]["joint_non_target_pass_count"] == 0
        assert row["development"]["joint_target_pass_count"] == row["development"]["target_count"]
        assert row["untouched_screen"]["joint_target_pass_count"] == row["untouched_screen"]["target_count"]


def test_phm_soft_calibration_candidate_is_auditable_but_not_profile_bound() -> None:
    artifact_root = ROOT / "research/evidence/phm_v6_0_soft_calibration_r001"
    policy = SoftEvidenceCalibrationPolicy.model_validate_json(
        (artifact_root / "development_selected_policy.json").read_text(encoding="utf-8")
    )
    receipt = SoftCalibrationSelectionReceipt.model_validate_json(
        (artifact_root / "selection_receipt.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        (ROOT / "configs/dataset_profiles/phm_v6_0.json").read_text(encoding="utf-8")
    )
    assert policy.status is SoftEvidencePolicyStatus.DEVELOPMENT_SELECTED
    assert policy.selected_on_groups == [3, 8]
    assert policy.selection_receipt_sha256 == receipt.receipt_sha256
    assert receipt.group_episode_counts == {"3": 51, "8": 51}
    assert sum(receipt.group_episode_counts.values()) == 102
    assert all(item.observation_count == 714 for item in receipt.candidate_metrics)
    assert receipt.freeze_eligible is False
    assert "soft_evidence_calibration_policy" not in profile["assets"]
    for name in (
        "soft_evidence_calibration_selection_protocol.schema.json",
        "soft_evidence_calibration_selection_receipt.schema.json",
    ):
        Draft202012Validator.check_schema(
            json.loads((ROOT / "configs/schemas" / name).read_text(encoding="utf-8"))
        )


@requires_external_dataset
@pytest.mark.parametrize(
    ("condition_id", "fault_id", "component"),
    [
        ("eps_battery_capacity_loss", "EPS_BATTERY_CAPACITY_LOSS", "BATTERY_PACK"),
        ("eps_solar_string_open", "EPS_SOLAR_ARRAY_STRING_FAILURE", "SOLAR_ARRAY"),
        ("comm_data_downlink_link_loss", "COMM_DOWNLINK_SERVICE_LOSS", "COMM_DOWNLINK"),
        ("thermal_heater_stuck_on", "THERMAL_HEATER_STUCK_ON", "THERMAL_HEATER"),
        (
            "thermal_radiator_rejection_loss",
            "THERMAL_RADIATOR_REJECTION_LOSS",
            "THERMAL_RADIATOR",
        ),
    ],
)
def test_phm_v60_all_strict_faults_build_executable_evidence(
    condition_id: str,
    fault_id: str,
    component: str,
) -> None:
    from astrograph.evidence import EvidenceFactStatus
    from astrograph.simulation import SimulationRunCompleted

    records = load_records(DATASET)
    sample = next(
        row for row in records if row.condition_id == condition_id and row.sample_index == 3
    )
    nominal = next(
        row
        for row in records
        if row.condition_id == "nominal"
        and row.subsystem == sample.subsystem
        and row.sample_index == sample.sample_index
    )
    builder = ConfiguredRuntimeEvidenceBuilder(
        ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json"
    )
    bundle, gate = builder.build(
        SimulationRunCompleted(
            event_id=f"evidence-{condition_id}",
            run_id=f"evidence-{condition_id}",
            run_dir=str(sample.telemetry_path.parent),
            nominal_run_dir=str(nominal.telemetry_path.parent),
        )
    )

    target_facts = [item for item in bundle.evidence_facts if item.target_fault_id == fault_id]
    assert gate.passed
    assert target_facts
    assert all(item.status is EvidenceFactStatus.SATISFIED for item in target_facts)
    assert bundle.selected_component_instance == component
    assert bundle.analysis_metadata["satisfied_critical_components"] == [component]


@requires_external_dataset
def test_phm_profile_drives_online_workflow_and_persists_receipt(tmp_path: Path) -> None:
    from astrograph.dataset_profiles import DatasetProfileRegistry
    from astrograph.policy_profiles import RuntimePolicyRegistry
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
    from astrograph.simulation import SimulationRunCompleted

    records = load_records(DATASET)
    sample = next(row for row in records if row.condition_id == "thermal_heater_stuck_on")
    nominal = next(
        row for row in records
        if row.condition_id == "nominal"
        and row.subsystem == sample.subsystem
        and row.sample_index == sample.sample_index
    )
    run_dir = tmp_path / "run"
    nominal_dir = tmp_path / "nominal"
    run_dir.mkdir()
    nominal_dir.mkdir()
    shutil.copy2(sample.telemetry_path, run_dir / "telemetry.csv")
    shutil.copy2(nominal.telemetry_path, nominal_dir / "telemetry.csv")
    event = SimulationRunCompleted(
        event_id="profile-phm-test",
        run_id="profile-phm-test",
        run_dir=str(run_dir),
        nominal_run_dir=str(nominal_dir),
    )
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow(
        dataset_profile_registry=DatasetProfileRegistry(
            index_path=ROOT / "configs/dataset_profiles/registry.json"
        ),
        runtime_policy_registry=RuntimePolicyRegistry(
            ROOT / "configs/runtime_policies/registry.json"
        ),
    )
    result = workflow.invoke(
        TaskEnvelope(
            task_id="PROFILE-PHM-TEST",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "dataset_profile_id": "phm.v6_0",
                "artifact_output_dir": str(tmp_path / "artifacts"),
                "reasoner_mode": "rule",
            },
            execution_policy=ExecutionPolicy(),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="profile-phm-test",
            permissions=permissions,
        ),
    )
    assert result.status == ExecutionStatus.SUCCEEDED
    receipt = result.output_payload["dataset_profile_receipt"]
    assert receipt["profile_id"] == "phm.v6_0"
    assert receipt["reference_scope"]["asset_id"] == "simulated-spacecraft.phm-v6_0"
    retrieval = result.output_payload["pre_reasoning_retrieval"]
    assert retrieval["query_construction_mode"] == "EVIDENCE_AWARE"
    assert retrieval["grounding_fault_ids"] == ["THERMAL_HEATER_STUCK_ON"]
    assert retrieval["active_graph_modified"] is False
    assert (tmp_path / "artifacts/input/dataset_profile_receipt.json").is_file()
