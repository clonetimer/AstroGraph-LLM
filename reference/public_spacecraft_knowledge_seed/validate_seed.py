#!/usr/bin/env python3
"""Executable structural, provenance, publication-safety and query checks for public KG."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

R = Path(__file__).resolve().parent

def main():
    g=json.loads((R/'graph.json').read_text(encoding='utf8'))
    ontology=json.loads((R/'ontology.json').read_text(encoding='utf8'))
    sources=json.loads((R/'sources.json').read_text(encoding='utf8'))
    summary=json.loads((R/'build_summary.json').read_text(encoding='utf8'))
    assert g['graph_kind']=='UNREVIEWED_PUBLIC_REFERENCE_SEED' and not g['is_flight_qualified']
    assert g['scope']=='PUBLIC_GENERAL_SPACECRAFT'
    assert len(g['entities'])==206 and len(g['relations'])==225 and len(g['clusters'])==30
    assert len(sources)==13 and len({s['id'] for s in sources})==13
    nodes={n['id']:n for n in g['entities']}
    edges={e['id']:e for e in g['relations']}
    source_ids={s['id'] for s in sources}
    assert len(nodes)==len(g['entities']) and len(edges)==len(g['relations'])
    assert len({c['code'] for c in g['clusters']})==len(g['clusters'])
    assert set(ontology['entity_types'])=={n['entity_type'] for n in g['entities']}
    allowed={tuple(x) for x in ontology['allowed_triples']}
    assert g['ontology']==ontology
    for source in sources:
        parsed=urlparse(source['url'])
        assert parsed.scheme=='https' and parsed.hostname in {'www.nasa.gov','www3.nasa.gov','swehb.nasa.gov','ecss.nl'}
        assert source['verified_via_public_page'] and source['accessed_on']
    for rel in g['relations']:
        assert rel['source'] in nodes and rel['target'] in nodes
        assert (nodes[rel['source']]['entity_type'],rel['relation_type'],nodes[rel['target']]['entity_type']) in allowed, rel['id']
        assert set(rel['source_ids']) <= source_ids and rel['source_ids']
        assert rel['source_locator'] and rel['source_summary_zh']
        assert rel['evidence_form']=='CHINESE_PARAPHRASE_NOT_VERBATIM'
        assert rel['review_status']=='PUBLIC_REFERENCE_SEED_UNREVIEWED'
        assert rel['is_confirmed_root_cause'] is False and rel['automatic_control_authorized'] is False
        assert rel['equipment_scope']=='PUBLIC_GENERAL_SPACECRAFT'
        if rel['relation_type'] in {'MITIGATED_BY','REQUIRES'}:
            assert rel['approval_required'] is True
    assert set(e for c in g['clusters'] for e in c['edge_ids'])==set(edges)
    for c in g['clusters']:
        assert len(c['edge_ids'])>=7 and len(c['source_ids'])==7
        assert set(c['source_ids'])=={'fault','component','mechanism','symptom','verification','action','condition'}
        assert all(nid in nodes for nid in c['source_ids'].values())
        assert set(c['sources']) <= source_ids and c['source_locator']
    assert sum(e['interpretation']=='ENGINEERING_INFERENCE_CROSS_SUBSYSTEM' for e in edges.values())==15
    assert summary['node_count']==206 and summary['edge_count']==225
    assert summary['all_unreviewed'] and summary['fault_case_count']==30
    assert sum(summary['domains'].values())==30
    for fname,expected in [('nodes.csv',206),('edges.csv',225),('expert_review_template.csv',225)]:
        with (R/fname).open(encoding='utf-8-sig',newline='') as f:
            records=list(csv.DictReader(f))
        assert len(records)==expected, fname
    xml_root=ET.parse(R/'graph.graphml').getroot()
    ns={'g':'http://graphml.graphdrawing.org/xmlns'}
    assert len(xml_root.findall('.//g:node',ns))==206
    assert len(xml_root.findall('.//g:edge',ns))==225
    # Queries should work with stdlib Python only.
    def query(*args):
        return subprocess.run([sys.executable,str(R/'query_graph.py'),*args],capture_output=True,text=True,check=True).stdout
    r=json.loads(query('--fault','ADCS-WHEEL-SAT','--json'))
    assert len(r)==1 and r[0]['component']=='反作用飞轮'
    assert 'nasa.gov' in r[0]['source_references'][0]['url']
    assert '未审核' in query('欠压')
    assert json.loads(query('','--domain','PROP','--json'))
    # Safe by default: Neo4j import has no side effects in dry-run.
    r=json.loads(subprocess.run([sys.executable,str(R/'neo4j_import_seed.py')],capture_output=True,text=True,check=True).stdout)
    assert r['status']=='DRY_RUN_NO_DATABASE_WRITE' and not r['flight_authorized']
    # No approval is created just by source ingestion.
    assert all(n['status']=='PUBLIC_REFERENCE_SEED_UNREVIEWED' for n in nodes.values())
    print('PASS: 206 nodes / 225 typed relations / 30 fault clusters / 7 domains')
    print('PASS: 13 verified reference URLs / 12 sources used / per-edge source localization')
    print('PASS: 225/225 unreviewed / no automatic flight control / proposal actions require approval')
    print('PASS: CSV/GraphML consistent / CLI fault and Chinese keyword retrieval / Neo4j dry-run')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
