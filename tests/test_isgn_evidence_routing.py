from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.integrations.isgn_routing import (
    ISGNRoutingAuditRequest,
    ISGNRoutingAuditService,
    ISGNRoutingPackage,
    ISGNRoutingStatus,
    isgn_routing_release_blockers,
)


ROOT = Path(__file__).resolve().parents[1]
ISGN = Path(os.environ.get("ASTROGRAPH_ISGN_EXECUTABLE", "__optional_isgn_not_found__"))


def _bundle() -> EpisodeEvidenceBundle:
    return EpisodeEvidenceBundle.model_validate_json(
        (
            ROOT
            / "tests/fixtures/evidence/reaction_wheel_friction_increase/diagnosis/episode_evidence_bundle.json"
        ).read_text(encoding="utf-8")
    )


def _write_package(tmp_path: Path, *, completion_status: str) -> tuple[Path, str]:
    atlas = tmp_path / "atlas.json"
    atlas.write_text(
        json.dumps(
            {
                "format": "isgn-mechanism-atlas-v1",
                "system_id": "fixture-spacecraft",
                "scopes": {
                    "rw_scope": {
                        "scope_id": "rw_scope",
                        "target_states": ["rw_drag", "rw_current", "rw_torque_available"],
                        "scope_kind": "dynamic",
                        "observed_drivers": [],
                        "latent_boundaries": [],
                        "graph": {},
                        "observation_status": "PASS",
                        "completion_status": completion_status,
                        "parameter_policy": {},
                        "phm_coverage": {"evidence_kinds": ["THRESHOLD_EVENT"]},
                        "metadata": {},
                    }
                },
                "interfaces": [],
                "uncovered_regions": [],
                "metadata": {"provenance": "test-fixture"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bindings = [
        ("ED-ONLINE-RW-FRICTION-DIRECT-V1", "rw_drag"),
        ("ED-ONLINE-RW-FRICTION-CURRENT-V1", "rw_current"),
        ("ED-ONLINE-RW-TORQUE-AVAILABLE-V1", "rw_torque_available"),
    ]
    package = tmp_path / "routing.json"
    package.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-isgn-routing-package.v1",
                "package_id": "fixture-routing",
                "version": "1.0.0",
                "system_id": "fixture-spacecraft",
                "atlas_path": "atlas.json",
                "atlas_sha256": hashlib.sha256(atlas.read_bytes()).hexdigest(),
                "bindings": [
                    {
                        "evidence_definition_id": evidence_id,
                        "evidence_kind": "THRESHOLD_EVENT",
                        "variables": [variable],
                    }
                    for evidence_id, variable in bindings
                ],
                "require_all_active_evidence_mapped": True,
                "claim_boundary": "Test-only routing package.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return package, hashlib.sha256(package.read_bytes()).hexdigest()


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_real_isgn_routes_active_evidence_only_to_complete_scope(tmp_path: Path) -> None:
    package, digest = _write_package(tmp_path, completion_status="COMPLETE")
    audit = ISGNRoutingAuditService().audit(
        _bundle(),
        ISGNRoutingAuditRequest(
            package_path=str(package),
            expected_package_sha256=digest,
            executable_path=str(ISGN),
        ),
    )

    assert audit.status == ISGNRoutingStatus.QUALIFIED
    assert audit.active_evidence_count == 3
    assert audit.mapped_evidence_count == 3
    assert {item["decision"] for item in audit.route_decisions} == {"ROUTED_CERTIFIED_SCOPE"}
    assert audit.runtime is not None and audit.runtime.distribution == "isgn-toolkit"
    assert audit.runtime.version
    assert audit.runtime.module_version == audit.runtime.version
    assert len(audit.runtime.package_source_sha256) == 64
    assert isgn_routing_release_blockers(audit, "ADCS_RW_FRICTION_INCREASE") == []


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_uncertified_scope_is_preserved_as_blocked_not_promoted(tmp_path: Path) -> None:
    package, digest = _write_package(tmp_path, completion_status="UNCERTIFIED")
    audit = ISGNRoutingAuditService().audit(
        _bundle(),
        ISGNRoutingAuditRequest(
            package_path=str(package),
            expected_package_sha256=digest,
            executable_path=str(ISGN),
        ),
    )

    assert audit.status == ISGNRoutingStatus.BLOCKED
    assert {item["decision"] for item in audit.route_decisions} == {"BLOCKED_UNCERTIFIED_SCOPE"}
    assert isgn_routing_release_blockers(audit, "ADCS_RW_FRICTION_INCREASE") == [
        "ISGN_MECHANISM_SCOPE_UNSUPPORTED"
    ]


def test_routing_package_hash_mismatch_fails_before_isgn_execution(tmp_path: Path) -> None:
    package, _ = _write_package(tmp_path, completion_status="COMPLETE")
    audit = ISGNRoutingAuditService().audit(
        _bundle(),
        ISGNRoutingAuditRequest(
            package_path=str(package),
            expected_package_sha256="0" * 64,
            executable_path=str(tmp_path / "missing-isgn"),
        ),
    )

    assert audit.status == ISGNRoutingStatus.ERROR
    assert "SHA-256 mismatch" in audit.errors[0]


def test_adcs_routing_configuration_does_not_claim_certified_mechanisms() -> None:
    package_path = ROOT / "configs/datasets/adcs_v5_2_2/isgn_evidence_routing.json"
    package = ISGNRoutingPackage.model_validate_json(package_path.read_text(encoding="utf-8"))
    atlas_path = package_path.parent / package.atlas_path
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(atlas_path.read_bytes()).hexdigest() == package.atlas_sha256
    assert package.require_all_active_evidence_mapped
    assert len(package.bindings) == 8
    assert {scope["completion_status"] for scope in atlas["scopes"].values()} == {"UNCERTIFIED"}
    assert atlas["metadata"]["status"] == "DRAFT_UNCERTIFIED"


@pytest.mark.skipif(not ISGN.is_file(), reason="optional external ISGN runtime is not installed")
def test_online_workflow_persists_isgn_routing_audit(tmp_path: Path) -> None:
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

    package, digest = _write_package(tmp_path, completion_status="COMPLETE")
    scene_root = ROOT / "tests/fixtures/spacecraft_scenarios"
    event = SimulationRunCompleted(
        event_id="ISGN-ONLINE-TEST",
        run_id="reaction_wheel_friction_increase",
        run_dir=str(scene_root / "reaction_wheel_friction_increase"),
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
            task_id="ISGN-ONLINE-TEST",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(ROOT / "protocols/phm_project_a"),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(artifact_dir),
                "reasoner_mode": "rule",
                "isgn_routing_audit": {
                    "package_path": str(package),
                    "expected_package_sha256": digest,
                    "executable_path": str(ISGN),
                },
            },
            execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="ISGN-ONLINE-TEST",
            permissions=permissions,
        ),
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["isgn_routing_audit"]["status"] == "BLOCKED"
    assert result.output_payload["isgn_routing_audit"]["unmapped_active_evidence_ids"]
    audit_path = artifact_dir / "audit/isgn_mechanism_routing_audit.json"
    assert audit_path.is_file()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert manifest["files"]["isgn_routing_audit"]["sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
