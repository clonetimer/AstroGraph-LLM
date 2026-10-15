from __future__ import annotations

import os
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from astrograph.integrations.isgn_observation import (
    ISGNObservationAuditRequest,
    ISGNObservationAuditService,
    ISGNObservationInputContract,
    ISGNObservationStatus,
    isgn_observation_release_blockers,
)


ISGN = Path(os.environ.get("ASTROGRAPH_ISGN_EXECUTABLE", "__optional_isgn_not_found__"))
ROOT = Path(__file__).resolve().parents[1]


def _simulate_motor(tmp_path: Path) -> Path:
    dataset = tmp_path / "motor.npz"
    completed = subprocess.run(
        [
            str(ISGN),
            "simulate",
            "--system",
            "motor",
            "--n",
            "32",
            "--steps",
            "160",
            "--dt",
            "0.05",
            "--seed",
            "101",
            "--out",
            str(dataset),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return dataset


def _write_contract(
    tmp_path: Path,
    dataset: Path,
    *,
    state_names: list[str] | None = None,
    require_measurement_chain: bool = False,
    require_observed_timebase: bool = False,
    require_temporal_sufficiency: bool = True,
) -> tuple[Path, str]:
    contract = tmp_path / "observation_contract.json"
    source_refs = ["isgn simulate motor fixture"]
    sample_time_source_ref = None
    if require_observed_timebase:
        sample_time_source_ref = "test fixture physical sample clock"
        source_refs.append(sample_time_source_ref)
    contract.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-isgn-observation-input.v2",
                "dataset_id": "isgn-motor-fixture",
                "system_id": "motor-fixture",
                "dataset_path": dataset.name,
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "source_role": "NORMAL_MECHANISM_DISCOVERY",
                "production_mechanism_eligible": True,
                "state_names": state_names or ["i", "w"],
                "input_names": ["V", "TL"],
                "require_measurement_chain": require_measurement_chain,
                "require_observed_timebase": require_observed_timebase,
                "require_temporal_sufficiency": require_temporal_sufficiency,
                "sample_time_source_ref": sample_time_source_ref,
                "source_refs": source_refs,
                "claim_boundary": "Test-only normal mechanism-discovery dataset.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return contract, hashlib.sha256(contract.read_bytes()).hexdigest()


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_real_isgn_observation_qualifies_bound_normal_dataset(tmp_path: Path) -> None:
    dataset = _simulate_motor(tmp_path)
    contract, digest = _write_contract(tmp_path, dataset)
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(ISGN),
        )
    )

    assert audit.status == ISGNObservationStatus.QUALIFIED
    assert audit.eligible_for_mechanism_certification
    assert audit.report["summary"]["n_trajectories"] == 32
    assert audit.runtime is not None and audit.runtime.distribution == "isgn-toolkit"
    assert audit.runtime.version
    assert audit.runtime.module_version == audit.runtime.version
    assert len(audit.runtime.package_source_sha256) == 64
    assert audit.timebase_status == "N/A"
    assert audit.temporal_sufficiency_status == "PASS"
    assert not audit.sample_time_present
    assert isgn_observation_release_blockers(audit, "FAULT_A") == []


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_required_measurement_contract_fails_closed_when_isgn_reports_na(tmp_path: Path) -> None:
    dataset = _simulate_motor(tmp_path)
    contract, digest = _write_contract(tmp_path, dataset, require_measurement_chain=True)
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(ISGN),
        )
    )

    assert audit.report["status"] == "QUALIFIED"
    assert audit.status == ISGNObservationStatus.CONTRACT_INSUFFICIENT
    assert not audit.eligible_for_mechanism_certification
    assert isgn_observation_release_blockers(audit, "FAULT_A") == [
        "ISGN_OBSERVATION_NOT_QUALIFIED"
    ]


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_label_named_state_is_rejected_before_observation_execution(tmp_path: Path) -> None:
    dataset = _simulate_motor(tmp_path)
    with np.load(dataset, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    schema = json.loads(str(values["schema_json"]))
    schema["states"][0]["name"] = "fault_label"
    values["schema_json"] = np.array(json.dumps(schema))
    np.savez_compressed(dataset, **values)
    contract, digest = _write_contract(tmp_path, dataset, state_names=["fault_label", "w"])
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(tmp_path / "missing-isgn"),
        )
    )

    assert audit.status == ISGNObservationStatus.ERROR
    assert "prohibited label/truth fields" in audit.errors[0]


def _rewrite_dataset_with_timebase(dataset: Path, *, mode: str = "regular") -> None:
    with np.load(dataset, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    dt = float(values["dt"])
    sample_time = np.broadcast_to(
        np.arange(values["X"].shape[1], dtype=np.float64) * dt,
        values["X"].shape[:2],
    ).copy()
    if mode == "duplicate":
        sample_time[:, 20] = sample_time[:, 19]
    elif mode == "wrong_shape":
        sample_time = sample_time[:, :-1]
    values["sample_time_s"] = sample_time
    np.savez_compressed(dataset, **values)


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_observed_timebase_is_preserved_and_qualified(tmp_path: Path) -> None:
    dataset = _simulate_motor(tmp_path)
    _rewrite_dataset_with_timebase(dataset)
    contract, digest = _write_contract(
        tmp_path,
        dataset,
        require_observed_timebase=True,
    )
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(ISGN),
        )
    )

    assert audit.status == ISGNObservationStatus.QUALIFIED
    assert audit.sample_time_present
    assert audit.sample_time_source_ref == "test fixture physical sample clock"
    assert audit.timebase_status == "PASS"
    assert audit.temporal_sufficiency_status == "PASS"


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_required_observed_timebase_fails_closed_when_missing(tmp_path: Path) -> None:
    dataset = _simulate_motor(tmp_path)
    contract, digest = _write_contract(
        tmp_path,
        dataset,
        require_observed_timebase=True,
    )
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(ISGN),
        )
    )

    assert audit.status == ISGNObservationStatus.CONTRACT_INSUFFICIENT
    assert not audit.eligible_for_mechanism_certification
    assert audit.timebase_status == "N/A"


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
@pytest.mark.parametrize("mode", ["duplicate", "wrong_shape"])
def test_malformed_observed_timebase_is_rejected_before_isgn(
    tmp_path: Path,
    mode: str,
) -> None:
    dataset = _simulate_motor(tmp_path)
    _rewrite_dataset_with_timebase(dataset, mode=mode)
    contract, digest = _write_contract(
        tmp_path,
        dataset,
        require_observed_timebase=True,
    )
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(tmp_path / "must-not-run-isgn"),
        )
    )

    assert audit.status == ISGNObservationStatus.ERROR
    assert "sample_time_s" in audit.errors[0]


def test_required_observed_timebase_requires_explicit_provenance() -> None:
    with pytest.raises(ValueError, match="sample_time_source_ref provenance"):
        ISGNObservationInputContract(
            dataset_id="test",
            system_id="test",
            dataset_path="test.npz",
            dataset_sha256="0" * 64,
            source_role="NORMAL_MECHANISM_DISCOVERY",
            production_mechanism_eligible=True,
            state_names=["x"],
            input_names=[],
            require_observed_timebase=True,
            source_refs=["dataset manifest"],
            claim_boundary="test-only invalid contract",
        )


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_hidden_state_history_gain_blocks_mechanism_certification(tmp_path: Path) -> None:
    dataset = _simulate_motor(tmp_path)
    with np.load(dataset, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    schema = json.loads(str(values["schema_json"]))
    schema["states"] = [schema["states"][0]]
    values["X"] = np.asarray(values["X"])[:, :, :1]
    values["schema_json"] = np.array(json.dumps(schema))
    np.savez_compressed(dataset, **values)
    contract, digest = _write_contract(tmp_path, dataset, state_names=["i"])
    audit = ISGNObservationAuditService().audit(
        ISGNObservationAuditRequest(
            contract_path=str(contract),
            expected_contract_sha256=digest,
            executable_path=str(ISGN),
        )
    )

    assert audit.status == ISGNObservationStatus.OBSERVATION_INSUFFICIENT
    assert audit.temporal_sufficiency_status == "FAIL"
    assert not audit.eligible_for_mechanism_certification


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_online_workflow_persists_observation_qualification(tmp_path: Path) -> None:
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

    dataset = _simulate_motor(tmp_path)
    contract, digest = _write_contract(tmp_path, dataset)
    scene_root = ROOT / "tests/fixtures/spacecraft_scenarios"
    event = SimulationRunCompleted(
        event_id="ISGN-OBS-ONLINE",
        run_id="heater_stuck_on",
        run_dir=str(scene_root / "heater_stuck_on"),
        nominal_run_dir=str(scene_root / "nominal"),
    )
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    artifact_dir = tmp_path / "artifacts"
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow()
    result = workflow.invoke(
        TaskEnvelope(
            task_id="ISGN-OBS-ONLINE",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(ROOT / "protocols/phm_project_a"),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(artifact_dir),
                "reasoner_mode": "rule",
                "isgn_observation_audit": {
                    "contract_path": str(contract),
                    "expected_contract_sha256": digest,
                    "executable_path": str(ISGN),
                },
            },
            execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="ISGN-OBS-ONLINE",
            permissions=permissions,
        ),
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["isgn_observation_audit"]["status"] == "QUALIFIED"
    audit_path = artifact_dir / "audit/isgn_observation_qualification.json"
    assert audit_path.is_file()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert manifest["files"]["isgn_observation_audit"]["sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
