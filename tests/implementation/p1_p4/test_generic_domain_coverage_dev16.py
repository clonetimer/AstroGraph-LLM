from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
from astrograph.data.adapter import SIGNAL_TO_ANOMALY, SIGNAL_TO_UNIT, engineering_unit_matches
from astrograph.knowledge.repository import CsvKnowledgeRepository
ROOT=Path(__file__).resolve().parents[3]


def test_every_enabled_anomaly_rule_has_generic_semantics_and_unit_contract():
    rules=json.loads((ROOT/'configs/anomaly_rules.json').read_text(encoding='utf-8'))['rules']
    enabled=[x for x in rules if x.get('enabled',True)]
    assert len(enabled)==19
    for row in enabled:
        channel=row['channel_id']
        assert channel in SIGNAL_TO_ANOMALY
        assert SIGNAL_TO_ANOMALY[channel][0]==row['anomaly_id']
        assert channel in SIGNAL_TO_UNIT
        assert engineering_unit_matches(SIGNAL_TO_UNIT[channel],SIGNAL_TO_UNIT[channel])


def test_generated_generic_edges_are_reproducible_and_source_bound(tmp_path: Path):
    out=tmp_path/'edges.csv'
    subprocess.run([sys.executable,str(ROOT/'scripts/knowledge/build_generic_series_edges.py'),'--project-root',str(ROOT),'--output',str(out)],check=True)
    committed=ROOT/'configs/knowledge/generic_series_fault_catalog_edges.csv'
    assert out.read_bytes()==committed.read_bytes()
    rows=list(csv.DictReader(committed.open(encoding='utf-8')))
    assert len(rows)==24
    derived=[x for x in rows if x['evidence_source']=='astrograph_lab_fault_catalog_exact_channel_match']
    assert len(derived)==23
    assert all(x['review_status']=='provisional' and x['provisional']=='true' and x['evidence_level']=='E0' for x in derived)
    assert all('lab_fault_catalog.json#' in x['source_ref'] for x in derived)


def test_expanded_generic_repository_has_expected_cross_domain_candidates():
    repo=CsvKnowledgeRepository(ROOT/'configs/knowledge/generic_series_fault_catalog_edges.csv',knowledge_version='astrograph-generic-series-e0.2')
    gyro=repo.retrieve(['ANOM_GYRO_X_DEVIATION'])
    assert {x.fault_id for x in gyro}=={'ADCS_GYRO_BIAS_DRIFT'}
    solar=repo.retrieve(['ANOM_SOLAR_CURRENT_DEVIATION'])
    assert {x.fault_id for x in solar}=={'EPS_SOLAR_ARRAY_STRING_FAILURE','EPS_SOLAR_ARRAY_EFFICIENCY_LOSS'}
    thermal=repo.retrieve(['ANOM_PAYLOAD_TEMP_DEVIATION'])
    assert {'THERMAL_RADIATOR_REJECTION_LOSS','THERMAL_TEMPERATURE_SENSOR_BIAS','THERMAL_TEMPERATURE_SENSOR_DRIFT'} <= {x.fault_id for x in thermal}


def test_context_only_exact_channel_does_not_become_fault_edge():
    repo=CsvKnowledgeRepository(ROOT/'configs/knowledge/generic_series_fault_catalog_edges.csv',knowledge_version='astrograph-generic-series-e0.2')
    # battery current is CONTEXT-only in the existing fault catalog and therefore cannot become a cause edge.
    assert repo.retrieve(['ANOM_BATTERY_CURRENT_DEVIATION'])==[]
    # command torque is CONTEXT-only for torque-authority loss.
    assert repo.retrieve(['ANOM_RW0_TORQUE_DEVIATION'])==[]
