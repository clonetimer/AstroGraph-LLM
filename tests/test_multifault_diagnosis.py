from __future__ import annotations

import json
from pathlib import Path

import pytest

import pandas as pd

from astrograph.diagnosis.sat_sim_multifault import MultiFaultStatus, diagnose_paired_multifault
from astrograph.knowledge.sat_sim_fault_catalog import build_fault_catalog_graph, load_sat_sim_fault_catalog

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "tests/fixtures/multifault"


def _case(campaign: str, scenario_id: str) -> tuple[Path, Path, set[str]]:
    root=DATA_ROOT/campaign
    truth=json.loads((root/'hidden_truth.json').read_text(encoding="utf-8"))
    cid=next(k for k,v in truth.items() if v['scenario_id']==scenario_id)
    d=next((root/'public').glob(f'*/{cid}'))
    return d/'nominal.csv', d/'effect.csv', set(truth[cid]['effect_ids'])


def test_multifault_exact_on_development_and_holdout() -> None:
    for campaign in ('development','holdout'):
        root=DATA_ROOT/campaign
        truth=json.loads((root/'hidden_truth.json').read_text(encoding="utf-8"))
        for cid,item in truth.items():
            d=next((root/'public').glob(f'*/{cid}'))
            result=diagnose_paired_multifault(case_id='opaque',nominal_csv=d/'nominal.csv',effect_csv=d/'effect.csv')
            assert result.status == MultiFaultStatus.KNOWN_FAULT
            assert set(result.detected_effect_ids)==set(item['effect_ids'])
            assert result.truth_used is False
            assert result.labels_used is False
            assert result.prohibited_columns_used == []


def test_multifault_rejects_label_column(tmp_path: Path) -> None:
    nominal,effect,_=_case('development','eps_capacity_solar')
    n=pd.read_csv(nominal);e=pd.read_csv(effect);n['label.answer']='x';e['label.answer']='x'
    np=tmp_path/'n.csv';ep=tmp_path/'e.csv';n.to_csv(np,index=False);e.to_csv(ep,index=False)
    result=diagnose_paired_multifault(case_id='opaque',nominal_csv=np,effect_csv=ep)
    assert result.status == MultiFaultStatus.DATA_ERROR


def test_multifault_same_trace_is_not_confirmed(tmp_path: Path) -> None:
    nominal,_,_=_case('development','eps_capacity_solar')
    result=diagnose_paired_multifault(case_id='opaque',nominal_csv=nominal,effect_csv=nominal)
    assert result.status in {MultiFaultStatus.AMBIGUOUS, MultiFaultStatus.INSUFFICIENT_EVIDENCE}
    assert result.detected_effect_ids == []


def test_catalog_covers_all_26_effects() -> None:
    catalog=load_sat_sim_fault_catalog(ROOT / 'reference' / 'knowledge' / 'sat_sim_fault_catalog.json')
    graph=build_fault_catalog_graph(catalog)
    assert len(catalog.entries)==26
    assert len(graph.effect_coverage)==26
    assert sum(len(entry.validated_propagation_paths) for entry in catalog.entries)==12
    assert sum(edge.assertion_status.value=='DERIVED' and edge.relation_type.value=='PROPAGATES_TO' for edge in graph.edges)>=12
    assert all(edge.evidence_source_id for edge in graph.edges)
    assert all('causal effect' not in str(edge.properties).lower() for edge in graph.edges)


def test_rw_jam_relative_torque_threshold_accepts_holdout_boundary() -> None:
    import pandas as pd
    from astrograph.diagnosis.sat_sim_multifault import _rw_relative_torque_evidence
    nominal = pd.DataFrame({
        "rw_motor_torque_nm_0": [1.0, 1.0],
        "rw_motor_torque_nm_1": [1.0, 1.0],
        "rw_motor_torque_nm_2": [1.0, 1.0],
    })
    effect = pd.DataFrame({
        "rw_motor_torque_nm_0": [0.94, 0.94],
        "rw_motor_torque_nm_1": [1.0, 1.0],
        "rw_motor_torque_nm_2": [1.0, 1.0],
    })
    evidence = _rw_relative_torque_evidence(nominal, effect)
    assert evidence.passed is True
    assert evidence.observed_delta == pytest.approx(-0.06)


def test_cross_subsystem_cases_report_root_fault_and_observed_impacts() -> None:
    root=DATA_ROOT/'development'
    truth=json.loads((root/'hidden_truth.json').read_text(encoding="utf-8"))
    for case_id,item in truth.items():
        if item['category'] != 'cross_subsystem_propagation':
            continue
        case_dir=next((root/'public').glob(f'*/{case_id}'))
        result=diagnose_paired_multifault(case_id='opaque',nominal_csv=case_dir/'nominal.csv',effect_csv=case_dir/'effect.csv')
        assert result.status is MultiFaultStatus.KNOWN_FAULT
        assert result.detected_effect_ids == item['effect_ids']
        assert len(item['effect_ids']) == 1
        assert result.propagation_paths
        assert all(impact.claim_semantics == 'OBSERVED_ASSOCIATIVE_COUPLING_NOT_CAUSAL_EFFECT_ESTIMATE'
                   for impact in result.observed_secondary_impacts)
