from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from astrograph.evaluation.data_exposure import dataset_use


SCRIPT = Path(__file__).parents[1] / "scripts" / "integration" / "run_adcs_v522_regression.py"
SPEC = importlib.util.spec_from_file_location("adcs_v522_regression", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

EXTERNAL_DATASET_AVAILABLE = (MODULE.DEFAULT_DATASET / "sample_index.csv").is_file()
requires_external_dataset = pytest.mark.skipif(
    not EXTERNAL_DATASET_AVAILABLE,
    reason="optional ADCS v5.2.2 external dataset is not configured; set ASTROGRAPH_ADCS_V522_DATA_ROOT",
)


@requires_external_dataset
def test_frozen_group_split_counts_and_no_overlap():
    rows = MODULE.load_trajectories(MODULE.DEFAULT_DATASET)
    report = MODULE.audit_split(rows)
    assert report["status"] == "PASS"
    assert report["group_overlap_count"] == 0
    assert report["splits"]["train"]["trajectory_count"] == 546
    assert report["splits"]["validation"]["trajectory_count"] == 182
    assert report["splits"]["test"]["trajectory_count"] == 182


def test_train_only_scaler_uses_only_supplied_training_windows():
    training = np.asarray([[[0.0], [2.0]], [[2.0], [4.0]]], dtype=np.float32)
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=["a", "b"]):
        impute, mean, scale = MODULE.fit_train_scaler(training)
    np.testing.assert_allclose(impute, [2.0])
    np.testing.assert_allclose(mean, [2.0])
    np.testing.assert_allclose(scale, [np.sqrt(2.0)])


def test_train_only_imputer_removes_non_finite_values():
    training = np.asarray([[[1.0], [np.nan]], [[3.0], [5.0]]], dtype=np.float32)
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=["a", "b"]):
        impute, mean, scale = MODULE.fit_train_scaler(training)
    transformed = MODULE.scale_windows(
        MODULE.WindowSet(training, np.zeros(2, dtype=int), np.full(2, -1, dtype=int),
                         ["a", "b"], np.zeros(2), np.zeros(2)),
        impute, mean, scale,
    )
    assert np.isfinite(transformed.x).all()


def test_split_assignment_is_total_and_disjoint():
    assert [MODULE.split_for_index(index) for index in range(10)] == [
        "train", "train", "train", "validation", "test",
        "train", "train", "train", "validation", "test",
    ]


def test_only_exact_ontology_correspondences_are_bound():
    assert MODULE.CLASS_TO_REGISTERED["adcs_rw_jamming"] == "ADCS_RW_JAM"
    assert MODULE.CLASS_TO_REGISTERED["adcs_control_loop_failure"] == (
        "ADCS_CONTROL_LOOP_AUTHORITY_LOSS"
    )
    assert set(MODULE.CLASS_TO_REGISTERED) == set(MODULE.FAULT_TO_COMPONENT)


@requires_external_dataset
def test_diagnosis_hierarchy_covers_every_dataset_fault_once():
    rows = MODULE.load_trajectories(MODULE.DEFAULT_DATASET)
    dataset_faults = {
        row.diagnostic_class for row in rows if row.diagnostic_class != "nominal"
    }
    assert set(MODULE.FAULT_TO_COMPONENT) == dataset_faults
    assert set(MODULE.FAULT_TO_COMPONENT.values()) == set(MODULE.COMPONENT_NAMES)
    assert "nominal" not in MODULE.FAULT_TO_COMPONENT


def test_predicted_component_strictly_routes_fault_classification():
    class_names = sorted(MODULE.FAULT_TO_COMPONENT)
    class_by_component = MODULE.component_class_indices(
        class_names, MODULE.COMPONENT_NAMES
    )
    all_class_prob = np.full((2, len(class_names)), 1.0 / len(class_names))
    component_prob = np.zeros((2, len(MODULE.COMPONENT_NAMES)))
    component_prob[0, MODULE.COMPONENT_NAMES.index("reaction_wheel")] = 1.0
    component_prob[1, MODULE.COMPONENT_NAMES.index("sun_sensor")] = 1.0
    conditioned, predicted_component = MODULE._conditioned_class_probabilities(
        all_class_prob, component_prob, class_by_component
    )
    for row_index, component_index in enumerate(predicted_component):
        allowed = set(class_by_component[int(component_index)])
        assert np.isclose(conditioned[row_index].sum(), 1.0)
        assert {
            index for index, value in enumerate(conditioned[row_index]) if value > 0
        } == allowed


def test_runtime_evidence_package_is_loadable_and_versioned():
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder

    builder = ConfiguredRuntimeEvidenceBuilder(MODULE.EVIDENCE_PACKAGE)
    assert builder.package.package_id == "whole-spacecraft-adcs-v5.2.2"
    assert builder.package.version == "1.0.0"
    assert len(MODULE.CLASS_TO_REGISTERED) == 14
    assert len({item.target_fault_id for item in builder.package.evidence_definitions}) == 4
    assert len(builder.package_sha256) == 64


def test_runtime_evidence_package_rejects_label_channels():
    from pydantic import ValidationError
    from astrograph.evidence import ConfiguredRuntimeEvidencePackage

    document = json.loads(MODULE.EVIDENCE_PACKAGE.read_text(encoding="utf-8"))
    document["channels"][0]["source_field"] = "label.fault_active"
    try:
        ConfiguredRuntimeEvidencePackage.model_validate(document)
    except ValidationError as exc:
        assert "cannot read label" in str(exc)
    else:
        raise AssertionError("label channel was accepted into runtime evidence mapping")


@requires_external_dataset
def test_configured_builder_uses_raw_adcs_schema_without_synthetic_fill(tmp_path):
    from astrograph.agents.agentic import DiagnosticReasonerAgent
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    from astrograph.investigation import HypothesisStatus
    from astrograph.simulation import SimulationRunCompleted

    rows = MODULE.load_trajectories(MODULE.DEFAULT_DATASET)
    sample = next(
        row for row in rows
        if row.sample_id == "attitude_acquisition_transient:adcs_rw_jamming:mc_04"
    )
    nominal = next(
        row for row in rows
        if row.diagnostic_class == "nominal"
        and row.trajectory_pair_id == sample.trajectory_pair_id
    )
    run_dir = tmp_path / "run"
    nominal_dir = tmp_path / "nominal"
    run_dir.mkdir()
    nominal_dir.mkdir()
    shutil.copy2(sample.telemetry_path, run_dir / "telemetry.csv")
    shutil.copy2(nominal.telemetry_path, nominal_dir / "telemetry.csv")
    event = SimulationRunCompleted(
        event_id="configured-adcs-test",
        run_id="configured-adcs-test",
        run_dir=str(run_dir),
        nominal_run_dir=str(nominal_dir),
    )
    bundle, gate = ConfiguredRuntimeEvidenceBuilder(MODULE.EVIDENCE_PACKAGE).build(event)
    assert gate.passed
    assert gate.field_count == 147
    assert bundle.source_protocol_hashes["configured_runtime_evidence_package"]
    hypotheses = DiagnosticReasonerAgent.reference_hypotheses(bundle)
    supported = {
        item.target_fault_id for item in hypotheses.hypotheses
        if item.status == HypothesisStatus.SUPPORTED
    }
    assert supported == {"ADCS_RW_JAM"}


@requires_external_dataset
def test_adcs_profile_drives_online_workflow_and_persists_receipt(tmp_path):
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

    rows = MODULE.load_trajectories(MODULE.DEFAULT_DATASET)
    sample = next(
        row for row in rows
        if row.sample_id == "attitude_acquisition_transient:adcs_rw_jamming:mc_04"
    )
    nominal = next(
        row for row in rows
        if row.diagnostic_class == "nominal"
        and row.trajectory_pair_id == sample.trajectory_pair_id
    )
    run_dir = tmp_path / "run"
    nominal_dir = tmp_path / "nominal"
    run_dir.mkdir()
    nominal_dir.mkdir()
    shutil.copy2(sample.telemetry_path, run_dir / "telemetry.csv")
    shutil.copy2(nominal.telemetry_path, nominal_dir / "telemetry.csv")
    event = SimulationRunCompleted(
        event_id="profile-adcs-test",
        run_id="profile-adcs-test",
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
            index_path=MODULE.DATASET_PROFILE_REGISTRY
        ),
        runtime_policy_registry=RuntimePolicyRegistry(
            MODULE.ROOT / "configs/runtime_policies/registry.json"
        ),
    )
    result = workflow.invoke(
        TaskEnvelope(
            task_id="PROFILE-ADCS-TEST",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "dataset_profile_id": "adcs.v5_2_2",
                "artifact_output_dir": str(tmp_path / "artifacts"),
                "reasoner_mode": "rule",
            },
            execution_policy=ExecutionPolicy(),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="profile-adcs-test",
            permissions=permissions,
        ),
    )
    assert result.status == ExecutionStatus.SUCCEEDED
    receipt = result.output_payload["dataset_profile_receipt"]
    assert receipt["profile_id"] == "adcs.v5_2_2"
    assert receipt["reference_scope"]["asset_id"] == "simulated-spacecraft.adcs-v5_2_2"
    assert len(receipt["verified_asset_sha256"]) == 10
    assert "fault_catalog" in receipt["verified_asset_sha256"]
    persisted = tmp_path / "artifacts/input/dataset_profile_receipt.json"
    persisted_receipt = json.loads(persisted.read_text(encoding="utf-8"))
    assert persisted_receipt["dataset_id"] == "whole_spacecraft_adcs_paired_dataset_v5_2_2"
    assert persisted_receipt["reference_scope"]["kind"] == "ASSET_LOCAL"
