from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.data_driven import ClassProbability, DataDrivenPrediction
from astrograph.dataset_profiles import DatasetReferenceScope
from astrograph.diagnosis import DiagnosisPolicyGate
from astrograph.integrations.fec_evidence import (
    FECEvidenceAuditRequest,
    FECEvidenceAuditService,
    FECEvidenceMode,
    FECEvidenceStatus,
    fec_evidence_release_blockers,
)


FEC = Path(os.environ.get("ASTROGRAPH_FEC_EXECUTABLE", "__optional_fec_not_found__"))
ROOT = Path(__file__).resolve().parents[1]


def _prediction(*, task_type: str = "MULTICLASS_FAULT_CLASSIFICATION") -> DataDrivenPrediction:
    return DataDrivenPrediction(
        prediction_id="PRED-FEC-1",
        episode_id="EP-FEC-1",
        model_id="fixture-model",
        model_version="1.0",
        task_type=task_type,
        output_semantics="PROBABILITIES",
        classes=[
            ClassProbability(class_id="FAULT_A", probability=0.9, rank=1),
            ClassProbability(class_id="NORMAL", probability=0.1, rank=2),
        ],
        normal_class_id="NORMAL",
        feature_schema_id="FEATURES-1",
        input_data_sha256="a" * 64,
    )


def _write_reference(tmp_path: Path, payload: dict) -> tuple[Path, str]:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _local_scope(asset_id: str = "simulated-spacecraft.test") -> dict:
    return {
        "kind": "ASSET_LOCAL",
        "identity_basis": "SYNTHETIC_DATASET_DOMAIN",
        "asset_id": asset_id,
        "fleet_group_id": None,
        "member_asset_ids": [],
        "commissioning_receipt_asset_role": None,
        "claim_boundary": "Test reference is valid only for its declared synthetic asset domain.",
    }


def _fleet_scope() -> dict:
    return {
        "kind": "FLEET_GROUP",
        "identity_basis": "PHYSICAL_ASSET_REGISTRY",
        "asset_id": None,
        "fleet_group_id": "fleet-test-a",
        "member_asset_ids": ["asset-a", "asset-b"],
        "commissioning_receipt_asset_role": "fleet_commissioning_receipt",
        "claim_boundary": "Test fleet reference requires independent per-asset commissioning.",
    }


def _class_reference(*, context: bool = False, scope: dict | None = None) -> dict:
    reference_scope = _local_scope() if scope is None else scope
    if context:
        return {
            "contract": "FEC-CONTEXT-CLASS-CONFORMAL-REFERENCE/1.0",
            "class_frame": ["FAULT_A", "NORMAL"],
            "context_columns": ["scenario"],
            "mask_columns": [],
            "alpha": 0.1,
            "min_group_size": 2,
            "groups": [
                {
                    "context": ["steady"],
                    "mask": [1],
                    "n_reference": 20,
                    "supported": True,
                    "threshold": 0.2,
                    "alpha": 0.1,
                }
            ],
            "calibration_score_sha256": "b" * 64,
            "source_score_contract_sha256": None,
            "provenance": "independent-calibration-fixture",
            "astrograph_reference_scope": reference_scope,
        }
    return {
        "contract": "FEC-CLASS-CONFORMAL-REFERENCE/1.0",
        "class_frame": ["FAULT_A", "NORMAL"],
        "mask_columns": [],
        "alpha": 0.1,
        "lambda_vacuity": 0.0,
        "min_group_size": 2,
        "groups": [{"mask": [1], "n_reference": 20, "threshold": 0.2, "alpha": 0.1}],
        "calibration_score_sha256": "b" * 64,
        "source_score_contract_sha256": None,
        "provenance": "independent-calibration-fixture",
        "astrograph_reference_scope": reference_scope,
    }


@pytest.mark.skipif(not FEC.is_file(), reason="optional external FEC runtime is not installed")
def test_class_reference_qualifies_prediction_set_and_records_runtime(tmp_path: Path) -> None:
    reference, digest = _write_reference(tmp_path, _class_reference())
    expected_scope = DatasetReferenceScope.model_validate(_local_scope())
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(FEC),
            expected_reference_scope=expected_scope,
        ),
    )

    assert audit.status == FECEvidenceStatus.QUALIFIED
    assert audit.calibration_supported
    assert audit.eligible_for_model_evidence
    assert audit.prediction_set == ["FAULT_A"]
    assert audit.runtime is not None
    assert audit.runtime.fec_version
    assert audit.runtime.package_source_sha256
    assert audit.declared_reference_scope == expected_scope
    assert fec_evidence_release_blockers(audit, "FAULT_A") == []
    assert fec_evidence_release_blockers(audit, "FAULT_B")[0].startswith(
        "FEC_PREDICTION_SET_FAULT_MISMATCH"
    )


@pytest.mark.skipif(not FEC.is_file(), reason="optional external FEC runtime is not installed")
def test_unseen_exact_context_abstains_and_blocks_release(tmp_path: Path) -> None:
    reference, digest = _write_reference(tmp_path, _class_reference(context=True))
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CONTEXT_CLASS,
            executable_path=str(FEC),
            context={"scenario": "transient"},
        ),
    )

    assert audit.status == FECEvidenceStatus.ABSTAIN
    assert not audit.calibration_supported
    assert fec_evidence_release_blockers(audit, "FAULT_A") == ["FEC_MODEL_EVIDENCE_UNSUPPORTED"]


@pytest.mark.skipif(not FEC.is_file(), reason="optional external FEC runtime is not installed")
def test_anomaly_reference_keeps_scalar_decision_separate_from_fault_type(tmp_path: Path) -> None:
    reference, digest = _write_reference(
        tmp_path,
        {
            "contract": "FEC-ANOMALY-CONTEXT-REFERENCE/1.0",
            "context_columns": ["scenario"],
            "alpha": 0.1,
            "min_group_size": 2,
            "groups": [
                {
                    "key": ["steady"],
                    "n_reference": 20,
                    "supported": True,
                    "scores": [index / 100.0 for index in range(20)],
                }
            ],
            "calibration_score_sha256": "d" * 64,
            "source_score_contract_sha256": None,
            "provenance": "trusted-normal-fixture",
        },
    )
    audit = FECEvidenceAuditService().audit(
        _prediction(task_type="ANOMALY_ATTRIBUTION"),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.ANOMALY,
            executable_path=str(FEC),
            context={"scenario": "steady"},
        ),
    )

    assert audit.status == FECEvidenceStatus.QUALIFIED
    assert audit.anomaly_decision == "ABNORMAL"
    assert audit.p_value is not None and audit.p_value <= 0.1
    assert audit.prediction_set == []
    assert audit.reference_metadata["provenance"] == "trusted-normal-fixture"
    assert fec_evidence_release_blockers(audit, "FAULT_A") == []


def test_reference_hash_mismatch_fails_closed_before_fec_execution(tmp_path: Path) -> None:
    reference, _ = _write_reference(tmp_path, _class_reference())
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256="0" * 64,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(tmp_path / "missing-fec"),
        ),
    )

    assert audit.status == FECEvidenceStatus.ERROR
    assert "SHA-256 mismatch" in audit.errors[0]
    assert fec_evidence_release_blockers(audit, "FAULT_A") == ["FEC_MODEL_EVIDENCE_ERROR"]


def test_profile_scope_mismatch_fails_closed_before_fec_execution(tmp_path: Path) -> None:
    reference, digest = _write_reference(tmp_path, _class_reference())
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(tmp_path / "missing-fec"),
            expected_reference_scope=DatasetReferenceScope.model_validate(
                _local_scope("simulated-spacecraft.other")
            ),
        ),
    )

    assert audit.status == FECEvidenceStatus.ERROR
    assert "scope does not match" in audit.errors[0]


def test_profile_bound_reference_requires_embedded_scope(tmp_path: Path) -> None:
    payload = _class_reference()
    payload.pop("astrograph_reference_scope")
    reference, digest = _write_reference(tmp_path, payload)
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(tmp_path / "missing-fec"),
            expected_reference_scope=DatasetReferenceScope.model_validate(_local_scope()),
        ),
    )

    assert audit.status == FECEvidenceStatus.ERROR
    assert "lacks the Dataset Profile reference scope" in audit.errors[0]


def test_fleet_reference_requires_commissioning_receipt(tmp_path: Path) -> None:
    fleet_scope = DatasetReferenceScope.model_validate(_fleet_scope())
    reference, digest = _write_reference(tmp_path, _class_reference(scope=_fleet_scope()))
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(tmp_path / "missing-fec"),
            expected_reference_scope=fleet_scope,
        ),
    )

    assert audit.status == FECEvidenceStatus.ERROR
    assert "requires a hash-bound fleet commissioning receipt" in audit.errors[0]


def test_asset_local_reference_rejects_fleet_receipt_fields(tmp_path: Path) -> None:
    reference, digest = _write_reference(tmp_path, _class_reference())
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(tmp_path / "missing-fec"),
            expected_reference_scope=DatasetReferenceScope.model_validate(_local_scope()),
            expected_fleet_commissioning_receipt_sha256="f" * 64,
        ),
    )

    assert audit.status == FECEvidenceStatus.ERROR
    assert "ASSET_LOCAL reference cannot carry" in audit.errors[0]


@pytest.mark.skipif(not FEC.is_file(), reason="optional external FEC runtime is not installed")
def test_admitted_fleet_reference_records_commissioning_receipt(tmp_path: Path) -> None:
    fleet_payload = _fleet_scope()
    fleet_scope = DatasetReferenceScope.model_validate(fleet_payload)
    reference, digest = _write_reference(tmp_path, _class_reference(scope=fleet_payload))
    receipt_path = tmp_path / "fleet-commissioning-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-fec-fleet-commissioning-receipt.v1",
                "reference_sha256": digest,
                "fec_decision": {
                    "fleet_group_id": "fleet-test-a",
                    "bundle_id": "bundle-test-a",
                    "accepted": True,
                    "state": "validated",
                    "reason": "Every declared asset passed its independent sentinel audit.",
                    "required_asset_ids": ["asset-a", "asset-b"],
                    "asset_validations": [
                        {
                            "asset_id": "asset-a",
                            "accepted": True,
                            "state": "healthy",
                            "reason": "Independent sentinel passed.",
                            "supported_fraction": 1.0,
                            "health": {"status": "healthy"},
                            "sentinel_data_ids": ["sentinel-a-1"],
                        },
                        {
                            "asset_id": "asset-b",
                            "accepted": True,
                            "state": "healthy",
                            "reason": "Independent sentinel passed.",
                            "supported_fraction": 1.0,
                            "health": {"status": "healthy"},
                            "sentinel_data_ids": ["sentinel-b-1"],
                        },
                    ],
                    "power_limited": False,
                    "promotion_guard_id": None,
                    "planned_variant_count": None,
                    "promotion_pending": False,
                },
                "claim_boundary": "Fixture receipt only; no physical fleet claim.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    audit = FECEvidenceAuditService().audit(
        _prediction(),
        FECEvidenceAuditRequest(
            reference_path=str(reference),
            expected_reference_sha256=digest,
            mode=FECEvidenceMode.CLASS,
            executable_path=str(FEC),
            expected_reference_scope=fleet_scope,
            fleet_commissioning_receipt_path=str(receipt_path),
            expected_fleet_commissioning_receipt_sha256=receipt_digest,
        ),
    )

    assert audit.status == FECEvidenceStatus.QUALIFIED
    assert audit.declared_reference_scope == fleet_scope
    assert audit.fleet_commissioning_receipt_sha256 == receipt_digest


def test_policy_gate_has_distinct_calibration_blocker_channel() -> None:
    import inspect

    assert "calibration_gate_blockers" in inspect.signature(DiagnosisPolicyGate.decide).parameters


@pytest.mark.skipif(not FEC.is_file(), reason="optional external FEC runtime is not installed")
def test_online_workflow_persists_fec_audit_without_changing_birrf(tmp_path: Path) -> None:
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
    from astrograph.simulation import SimulationRunCompleted, WholeSpacecraftRuntimeEvidenceBuilder

    scene_root = ROOT / "tests/fixtures/spacecraft_scenarios"
    protocol_root = ROOT / "protocols/phm_project_a"
    event = SimulationRunCompleted(
        event_id="FEC-ONLINE-TEST",
        run_id="heater_stuck_on",
        run_dir=str(scene_root / "heater_stuck_on"),
        nominal_run_dir=str(scene_root / "nominal"),
    )
    preview, _ = WholeSpacecraftRuntimeEvidenceBuilder(protocol_root).build(event)
    registered = sorted({item.target_fault_id for item in preview.evidence_definitions})
    remaining = [item for item in registered if item != "THERMAL_HEATER_STUCK_ON"]
    ordered = [("THERMAL_HEATER_STUCK_ON", 0.8), ("NORMAL", 0.1)]
    ordered.extend((fault_id, 0.1 / len(remaining)) for fault_id in remaining)
    prediction = DataDrivenPrediction(
        prediction_id="PRED-FEC-ONLINE",
        episode_id=preview.episode.episode_id,
        model_id="fixture-model",
        model_version="1.0",
        output_semantics="PROBABILITIES",
        classes=[
            ClassProbability(class_id=class_id, probability=probability, rank=index)
            for index, (class_id, probability) in enumerate(ordered, start=1)
        ],
        normal_class_id="NORMAL",
        feature_schema_id="FEATURES-ONLINE",
        input_data_sha256="c" * 64,
    )
    reference_payload = _class_reference()
    reference_payload["class_frame"] = [class_id for class_id, _ in ordered]
    reference_payload["groups"][0]["threshold"] = 0.25
    reference, digest = _write_reference(tmp_path, reference_payload)
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
            task_id="FEC-ONLINE-TEST",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(protocol_root),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(artifact_dir),
                "reasoner_mode": "rule",
                "data_driven_prediction": prediction.model_dump(mode="json"),
                "fec_evidence_audit": {
                    "reference_path": str(reference),
                    "expected_reference_sha256": digest,
                    "mode": "CLASS",
                    "executable_path": str(FEC),
                },
            },
            execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="FEC-ONLINE-TEST",
            permissions=permissions,
        ),
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["fec_evidence_audit"]["status"] == "QUALIFIED"
    assert result.output_payload["candidate_fusion"]["birrf_result"]["top_fault_id"] == (
        "THERMAL_HEATER_STUCK_ON"
    )
    assert result.output_payload["diagnosis_evidence_bundle"]["model_evidence"]["calibrated"] is True
    audit_path = artifact_dir / "audit/fec_model_evidence_audit.json"
    assert audit_path.is_file()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert manifest["files"]["fec_evidence_audit"]["sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
