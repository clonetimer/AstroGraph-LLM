from __future__ import annotations

import json
from pathlib import Path

from astrograph.evaluation.external_dataset_contract import file_identity, verify_seal

ROOT = Path(__file__).resolve().parents[1]


def test_r29_protocol_is_sealed_and_binds_reduced_order_core_without_agent_authority():
    plan = json.loads((ROOT / "configs/evaluation/r29_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    verify_seal(plan)
    assert plan["patch_id"] == "PATCH-R29"
    assert plan["base_artifact_sha256"] == "46750823384f5d6aa43d117a5ce1677bdb3674798f323495d4b4b97d34638e0a"
    assert plan["dxc09_sample"]["agent_required"] is False
    assert plan["dxc09_sample"]["runtime_default_changed"] is False
    assert plan["dxc09_sample"]["competition_or_rematch_regenerated"] is False
    assert plan["dxc09_sample"]["new_holdout_supplied"] is False
    assert plan["agent"]["diagnostic_authority"] is False
    assert plan["agent"]["formal_evidence_eligible"] is False
    assert plan["simulation"]["numerical_time_series_simulation"] is True
    assert plan["simulation"]["continuous_network_solver"] is False
    assert plan["simulation"]["virtual_adapt_or_simulink_equivalent"] is False
    assert plan["simulation"]["candidate_semantics"] == "HOST_BOUND_SIMULATION_HYPOTHESIS_NOT_DIAGNOSIS_PROOF"
    for relative, expected in plan["source_bindings"].items():
        # Diagnostic Probe is a shared additive extension host. R31 adds a new
        # read-only WaterPump review without rewriting the sealed R29 receipt.
        # Preserve the historical R29 capability; R31 independently binds the
        # current shared file bytes.
        if relative == "src/astrograph/investigation/diagnostic_probes.py":
            current = (ROOT / relative).read_text(encoding="utf-8")
            assert "DXC_REDUCED_ORDER_SIMULATION_REVIEW" in current
            continue
        assert file_identity(ROOT / relative) == expected


def test_r29_development_receipt_is_post_contact_and_does_not_claim_external_validation():
    plan = json.loads((ROOT / "configs/evaluation/r29_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "governance/evidence/research/r29_reduced_order_simulation/dxc_reduced_order_development.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["receipt_sha256"] == plan["dxc09_sample"]["development_receipt_sha256"]
    assert receipt["status"] == "POST_CONTACT_DEVELOPMENT_REDUCED_ORDER_SIMULATION_AUDIT_ONLY"
    assert receipt["competition_or_rematch_regenerated"] is False
    assert receipt["new_holdout_supplied"] is False
    assert receipt["runtime_default_changed"] is False
    assert receipt["agent_required"] is False
    assert receipt["changed_case_count"] == 1
    assert receipt["physics_application_case_count"] == 1
    assert receipt["combined_post_contact_development"]["r29_metrics"]["fault_element_f1_micro"] == 1.0
    assert receipt["qualification"] == "NOT_EXTERNAL_VALIDATION_NOT_PRODUCTION_DEFAULT_NOT_FLIGHT_QUALIFIED"


def test_r29_architecture_receipt_records_numeric_reduced_order_scope_and_neutral_candidates():
    receipt = json.loads((ROOT / "governance/evidence/research/r29_reduced_order_simulation/r29_algorithm_architecture_receipt.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["numerical_time_series_simulation"] is True
    assert receipt["continuous_network_solver"] is False
    assert receipt["virtual_adapt_or_simulink_equivalent"] is False
    assert receipt["simulation_candidate_semantics"] == "HOST_BOUND_SIMULATION_HYPOTHESIS_NOT_DIAGNOSIS_PROOF"
    assert receipt["r29_agent_required"] is False
    assert receipt["agent_review_read_only"] is True
    assert receipt["core_truth_or_case_id_dependency"] is False
    assert receipt["runtime_default_changed"] is False
