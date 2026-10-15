from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.semantics import ArchetypeFeatureCatalog, FeaturePrimitive


ROOT = Path(__file__).parents[1]
CATALOG_PATH = (
    ROOT / "configs" / "semantics" / "sat_agent_archetype_feature_contracts.s0.v1.json"
)


def test_sat_agent_feature_contracts_are_truth_blind_and_differ_by_archetype() -> None:
    catalog = ArchetypeFeatureCatalog.model_validate_json(
        CATALOG_PATH.read_text(encoding="utf-8")
    )
    assert catalog.source_id == "SAT_AGENT_PLATFORM"
    assert catalog.source_count == 1
    # S1 fit authorized by project owner 2026-08-04; one data-driven baseline fit
    # (loop-15/16) under grouped-holdout discipline.  The catalog records the
    # materialized fit state rather than the pre-S1 S0 state.
    assert catalog.fitted_model_count == 1
    assert catalog.s1_fit_authorized is True
    assert catalog.feature_selection_evidence_used is False
    assert {item.component_archetype_id for item in catalog.archetype_contracts} == {
        "storage.energy_buffer",
        "actuator.controlled_effort",
        "actuator.spinning_rotor",
        "thermal.lumped_capacity",
        "sensor.measurement",
        "converter.power",
        "link.transport",
        "queue.bounded_inventory",
        "actuator.thrust_impulse",
    }
    recipes = [
        tuple(item.primitive_id for item in contract.primitives)
        for contract in catalog.archetype_contracts
    ]
    assert len(recipes) == len(set(recipes)) == 9
    assert {len(item.primitives) for item in catalog.archetype_contracts} == {3, 4, 5}
    assert all(
        item.output_shape_policy
        == "VARIABLE_BY_ARCHETYPE_FIXED_ONLY_AFTER_S1_FREEZE"
        for item in catalog.archetype_contracts
    )
    assert catalog.evidence_token_contract.labels_used is False
    assert catalog.evidence_token_contract.truth_used is False
    assert catalog.evidence_token_contract.identity_fields_allowed is False
    assert "provenance_hash" not in catalog.evidence_token_contract.model_input_fields
    assert catalog.evidence_token_contract.audit_sidecar_fields == ["provenance_hash"]


@pytest.mark.parametrize(
    "role",
    ["dataset.id", "case_role", "fault_label", "seed", "split", "hidden_truth"],
)
def test_feature_primitive_rejects_identity_or_truth_inputs(role: str) -> None:
    with pytest.raises(ValueError, match="prohibited"):
        FeaturePrimitive(
            primitive_id="test.bad",
            operator_family="ROBUST_LEVEL",
            input_port_role_ids=[role],
            output_token_type="STATE_LEVEL",
            availability_policy="REQUIRED",
        )


@pytest.mark.parametrize(
    "role",
    ["target.mode", "health_status", "anomaly_score", "fault_onset", "provenance_hash"],
)
def test_feature_primitive_rejects_additional_answer_or_identity_inputs(role: str) -> None:
    with pytest.raises(ValueError, match="prohibited"):
        FeaturePrimitive(
            primitive_id="test.bad",
            operator_family="ROBUST_LEVEL",
            input_port_role_ids=[role],
            output_token_type="STATE_LEVEL",
            availability_policy="REQUIRED",
        )


def test_archetype_recipe_cannot_differ_only_by_primitive_names() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    first = payload["archetype_contracts"][0]
    second = payload["archetype_contracts"][1]
    second["required_port_role_ids"] = list(first["required_port_role_ids"])
    second["optional_port_role_ids"] = list(first["optional_port_role_ids"])
    second["primitives"] = [
        {**item, "primitive_id": f"renamed.{index}"}
        for index, item in enumerate(first["primitives"])
    ]
    with pytest.raises(ValueError, match="identical feature recipe"):
        ArchetypeFeatureCatalog.model_validate(payload)
