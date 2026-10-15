from __future__ import annotations

import json
from pathlib import Path

from astrograph.evaluation.external_dataset_contract import file_identity, verify_seal

ROOT = Path(__file__).resolve().parents[1]


def test_r28_protocol_is_sealed_and_records_inherited_r27_zip_identity_mismatch():
    plan = json.loads((ROOT / "configs/evaluation/r28_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    verify_seal(plan)
    assert plan["patch_id"] == "PATCH-R28"
    assert plan["base_zip_byte_identity_matches_artifact_record"] is False
    assert plan["base_identity_finding"] == "BASE-R28-01_INHERITED_PACKAGING_IDENTITY_MISMATCH"
    assert plan["base_source_internal_integrity"] == {"verified_files": 2760, "errors": 0, "status": "PASS"}
    assert plan["dxc09_sample"]["agent_required"] is False
    assert plan["dxc09_sample"]["runtime_default_changed"] is False
    assert plan["dxc09_sample"]["competition_or_rematch_regenerated"] is False
    assert plan["dxc09_sample"]["new_holdout_supplied"] is False
    assert plan["agent"]["diagnostic_authority"] is False
    assert plan["agent"]["formal_evidence_eligible"] is False
    assert plan["agent"]["may_rerun_or_reconfigure_forward_model"] is False
    for relative, expected in plan["source_bindings"].items():
        # The shared Diagnostic Probe host is an additive extension point. Later
        # patches may add new read-only probe kinds without rewriting the sealed
        # R28 receipt. Preserve the historical R28 capability rather than
        # requiring the current shared file to equal its R28 byte hash.
        if relative == "src/astrograph/investigation/diagnostic_probes.py":
            current = (ROOT / relative).read_text(encoding="utf-8")
            assert "DXC_FORWARD_SIMULATION_REVIEW" in current
            continue
        assert file_identity(ROOT / relative) == expected


def test_r28_development_receipt_is_post_contact_and_resolution_is_not_hidden():
    plan = json.loads((ROOT / "configs/evaluation/r28_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "governance/evidence/research/r28_forward_simulation/dxc_forward_simulation_development.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["receipt_sha256"] == plan["dxc09_sample"]["development_receipt_sha256"]
    assert receipt["status"] == "POST_CONTACT_DEVELOPMENT_FORWARD_MODEL_AUDIT_ONLY"
    assert receipt["competition_or_rematch_regenerated"] is False
    assert receipt["runtime_default_changed"] is False
    assert receipt["agent_required"] is False
    metrics = receipt["combined_post_contact_development"]["r28_metrics"]
    assert metrics["diagnostic_resolution_rate"] == 0.96
    assert metrics["resolved_exact_fault_set_rate"] == 1.0
    assert receipt["unresolved_case_count"] == 1
    assert receipt["stop_rule"] == "STOP_AFTER_FIXED_R28_MODEL_AUDIT_NO_FURTHER_CASE_SPECIFIC_RULE_OPTIMIZATION"


def test_r28_architecture_receipt_keeps_forward_model_core_non_agent_and_non_physics_claim():
    arch = json.loads((ROOT / "governance/evidence/research/r28_forward_simulation/r28_algorithm_architecture_receipt.json").read_text(encoding="utf-8"))
    verify_seal(arch)
    assert arch["r28_agent_required"] is False
    assert arch["agent_review_read_only"] is True
    assert arch["continuous_physics_simulator"] is False
    assert arch["runtime_default_changed"] is False
    assert arch["qualification"] == "NOT_EXTERNAL_VALIDATION_NOT_PRODUCTION_DEFAULT_NOT_FLIGHT_QUALIFIED"
