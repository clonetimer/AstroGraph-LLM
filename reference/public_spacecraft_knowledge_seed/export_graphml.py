#!/usr/bin/env python3
"""Export the public, unreviewed graph to standard GraphML (Gephi/yEd/NetworkX)."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
G = json.loads((ROOT / 'graph.json').read_text(encoding='utf-8'))
NS='http://graphml.graphdrawing.org/xmlns'
ET.register_namespace('', NS)
def e(tag, attrs=None, parent=None, text=None):
    result=ET.Element('{%s}%s' % (NS,tag), attrs or {}) if parent is None else ET.SubElement(parent,'{%s}%s' % (NS,tag),attrs or {})
    if text is not None:result.text=str(text)
    return result

root=e('graphml')
node_keys={'label':'节点名称','type':'实体类型','domains':'所属领域','review':'审核状态'}
edge_keys={'relation':'关系类型','interpretation':'解释类型','sources':'资料来源ID','locator':'文献定位','approval':'要求审核','review_status':'审核状态','hint':'证据摘要/适用边界'}
for key,name in node_keys.items():e('key',{'id':'n_'+key,'for':'node','attr.name':key,'attr.type':'string'},root)
for key,name in edge_keys.items():e('key',{'id':'e_'+key,'for':'edge','attr.name':key,'attr.type':'string'},root)
graph=e('graph',{'id':'public-spacecraft-reference-seed','edgedefault':'directed'},root)
for node in G['entities']:
    elem=e('node',{'id':node['id']},graph)
    vals={'label':node['name'],'type':node['entity_type'],'domains':';'.join(node['domains']),'review':node['status']}
    for key,value in vals.items():e('data',{'key':'n_'+key},elem,value)
for rel in G['relations']:
    elem=e('edge',{'id':rel['id'],'source':rel['source'],'target':rel['target']},graph)
    vals={'relation':rel['relation_type'],'interpretation':rel['interpretation'],'sources':';'.join(rel['source_ids']),
          'locator':rel['source_locator'],'approval':str(rel['approval_required']).lower(),
          'review_status':rel['review_status'],'hint':rel['source_summary_zh']}
    for key,value in vals.items():e('data',{'key':'e_'+key},elem,value)
output=ROOT/'graph.graphml'
ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
print(f'GraphML: {len(G["entities"])} nodes, {len(G["relations"])} edges, {output.stat().st_size} bytes')
