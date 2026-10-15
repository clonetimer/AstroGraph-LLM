"""Safety and format compatibility checks for unreviewed NASA/ECSS reference graph."""
from __future__ import annotations
from pathlib import Path
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'reference' / 'public_spacecraft_knowledge_seed'


def test_public_reference_graph_typed_and_clearly_unreviewed():
    graph = json.loads((ROOT / 'graph.json').read_text(encoding='utf8'))
    assert graph['graph_kind'] == 'UNREVIEWED_PUBLIC_REFERENCE_SEED'
    assert not graph['is_flight_qualified']
    assert len(graph['entities']) == 206 and len(graph['relations']) == 225
    assert len(graph['clusters']) == 30
    allowed = {tuple(x) for x in graph['ontology']['allowed_triples']}
    types = {n['id']: n['entity_type'] for n in graph['entities']}
    for edge in graph['relations']:
        assert edge['source'] in types and edge['target'] in types
        assert (types[edge['source']], edge['relation_type'], types[edge['target']]) in allowed
        assert edge['review_status'] == 'PUBLIC_REFERENCE_SEED_UNREVIEWED'
        assert edge['evidence_form'] == 'CHINESE_PARAPHRASE_NOT_VERBATIM'
        assert edge['is_confirmed_root_cause'] is False
        assert edge['automatic_control_authorized'] is False
        assert edge['source_ids'] and edge['source_locator']


def test_public_reference_graph_csv_graphml_counts():
    for filename, rows_expected in [('nodes.csv', 206), ('edges.csv', 225), ('expert_review_template.csv', 225)]:
        with (ROOT / filename).open(encoding='utf-8-sig', newline='') as f:
            assert len(list(csv.DictReader(f))) == rows_expected
    root = ET.parse(ROOT / 'graph.graphml').getroot()
    ns = {'g':'http://graphml.graphdrawing.org/xmlns'}
    assert len(root.findall('.//g:node', ns)) == 206
    assert len(root.findall('.//g:edge', ns)) == 225


def test_public_reference_graph_cli_and_importer_are_read_only_by_default():
    command = [sys.executable, str(ROOT / 'query_graph.py'), '--fault', 'ADCS-WHEEL-SAT', '--json']
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    entries = json.loads(proc.stdout)
    assert len(entries) == 1 and entries[0]['component'] == '反作用飞轮'
    proc = subprocess.run([sys.executable, str(ROOT / 'neo4j_import_seed.py')], capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    assert info['status'] == 'DRY_RUN_NO_DATABASE_WRITE'
    assert info['flight_authorized'] is False
    assert info['edge_count'] == 225
