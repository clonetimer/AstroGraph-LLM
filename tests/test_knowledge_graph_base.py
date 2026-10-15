from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/knowledge_graph/online_base'

def read_csv(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))

def test_v057_kg_gold_structure():
    nodes=read_csv('nodes.csv');edges=read_csv('edges.csv');neg=read_csv('negative_constraints.csv')
    node_ids={x['node_id'] for x in nodes}
    assert len(node_ids)==len(nodes)
    assert len({x['edge_id'] for x in edges})==len(edges)
    assert all(x['source_id'] in node_ids and x['target_id'] in node_ids for x in edges)
    assert len(neg)>=6

def test_v057_each_fault_has_required_evidence_and_queries():
    faults={
      'ADCS_RW_JAM','EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE','EPS_SOLAR_ARRAY_STRING_FAILURE',
      'ADCS_GYRO_BIAS_DRIFT','COMM_TRANSMITTER_OUTPUT_POWER_LOSS','DATA_STORAGE_WRITE_FAILURE'}
    edges=read_csv('edges.csv')
    for f in faults:
        assert any(x['source_id']==f and x['relation']=='REQUIRES_EVIDENCE' for x in edges)
    queries=json.loads((ROOT/'query_gold.json').read_text(encoding='utf-8'))['queries']
    assert len(queries)==12
