from __future__ import annotations

import json
from pathlib import Path

from astrograph.evaluation.external_dataset_contract import file_identity, verify_seal
from astrograph.governance.agent_tool_specification import build_agent_tool_governance_catalog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'governance/evidence/research/r31_waterpump_simulation/R31_PROTOCOL.json'


def test_r31_protocol_is_sealed_and_binds_second_numerical_mechanism_family():
    doc = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    verify_seal(doc)
    assert doc['patch_id'] == 'PATCH-R31'
    assert doc['base_r30_zip_sha256'] == 'c42fcda321d225e2d8be557674a2bec683ce68e3614e66829797b5fdfddbe7a4'
    assert doc['waterpump_provider_id'] == 'adapt-waterpump-reduced-order-v1'
    assert doc['auto_override_enabled'] is False
    assert doc['agent_required'] is False
    assert doc['formal_evidence_eligible'] is False
    assert doc['data_exposure']['dxc_competition_rematch_regenerated'] is False
    assert doc['data_exposure']['esa_seen_external_regenerated'] is False
    assert doc['stop_rule'] == 'NO_PARAMETER_OR_RULE_OPTIMIZATION_ON_R31_PUMP_BLIND_OR_DXC09_SAMPLE_OUTCOMES'
    for relative, expected in doc['source_bindings'].items():
        assert file_identity(ROOT / relative)["sha256"] == expected


def test_r31_mechanism_registry_numerical_coverage_increases_without_claiming_full_simulation():
    doc = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    coverage = doc['mechanism_registry']['coverage']
    assert coverage['catalog_fault_mode_instance_count'] == 222
    assert coverage['modeled_fault_mode_instance_count'] == 222
    assert coverage['numerical_reduced_order_count'] == 14
    assert coverage['qualitative_state_response_count'] == 203
    assert coverage['declared_only_count'] == 5
    assert abs(coverage['numerical_reduced_order_coverage_of_fault_universe'] - 14/222) < 1e-15


def test_r31_live_tool_catalog_matches_frozen_r31_catalog_and_keeps_agent_non_authoritative():
    frozen = json.loads((ROOT/'governance/evidence/research/r31_waterpump_simulation/agent_tool_governance_catalog.json').read_text(encoding='utf-8'))
    live = build_agent_tool_governance_catalog().model_dump(mode='json')
    assert frozen == live
    assert len(live['tools']) == 46
    row = next(item for item in live['tools'] if item['tool_key']=='probe:DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW')
    assert row['evidence_authority'] == 'SHADOW_DERIVED_OBSERVATION_ONLY'
    assert row['diagnosis_authority'] == 'SHADOW_REVIEW_ONLY'
    assert row['core_prediction_depends_on_agent_call'] is False
    assert row['formal_release_authority'] is False


def test_r31_blind_prediction_artifact_contains_no_hidden_truth_and_report_is_protocol_only():
    pred = json.loads((ROOT/'governance/evidence/research/r31_waterpump_simulation/predictions_before_truth_join.json').read_text(encoding='utf-8'))
    raw = json.dumps(pred, sort_keys=True)
    assert 'hidden_fault_id' not in raw
    report = json.loads((ROOT/'governance/evidence/research/r31_waterpump_simulation/report.json').read_text(encoding='utf-8'))
    assert report['case_count'] == 80
    assert report['accuracy'] == 1.0
    assert report['macro_f1'] == 1.0
    assert report['status'] == 'SYNTHETIC_BLIND_EVENT_CONDITIONAL_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION'


def test_r31_public_sample_mechanism_audit_is_explicitly_truth_aligned_not_online_performance():
    audit = json.loads((ROOT/'governance/evidence/research/r31_waterpump_simulation/public_sample_mechanism_audit.json').read_text(encoding='utf-8'))
    verify_seal(audit)
    assert audit['case_count'] == 2
    assert audit['mode_agreement_rate'] == 1.0
    assert audit['status'] == 'TRUTH_ALIGNED_PUBLIC_SAMPLE_MECHANISM_FIT_AUDIT_NOT_ONLINE_DIAGNOSIS'
    assert audit['window_alignment'] == 'FAULT_INJECTION_TRUTH_TIMESTAMP_USED_FOR_MECHANISM_FIT_AUDIT_ONLY'
