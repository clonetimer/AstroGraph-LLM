#!/usr/bin/env python3
"""No-dependency offline lookup of the unreviewed public spacecraft KG."""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load():
 return json.loads((ROOT/'graph.json').read_text(encoding='utf8'))

def search(graph, query='', domain=None, limit=12):
 nodes={n['id']:n for n in graph['entities']}
 results=[]
 q=query.casefold().strip()
 for c in graph['clusters']:
  if domain and c['domain'] != domain.upper():continue
  vals=[c['title'],c['code'],c['summary']]+[nodes[nid]['name'] for nid in c['source_ids'].values()]
  vals += sum((nodes[nid].get('aliases',[]) for nid in c['source_ids'].values()),[])
  score=0 if not q else (20 if c['code'].casefold()==q else 12 if c['title'].casefold()==q else 0)
  score+=4*sum(q in str(v).casefold() for v in vals) if q else 1
  if score:results.append((score,c))
 return [v for _,v in sorted(results,key=lambda t:(-t[0],t[1]['code']))[:limit]]

def lookup(graph, c):
 nodes={n['id']:n for n in graph['entities']}
 byid={r['id']:r for r in graph['relations']}
 relations=[byid[e] for e in c['edge_ids']]
 keys=['component','mechanism','symptom','verification','action','condition']
 output={'code':c['code'],'domain':c['domain'],'fault':c['title'],
         'component':nodes[c['source_ids']['component']]['name'],
         'mechanism':nodes[c['source_ids']['mechanism']]['name'],
         'symptom':nodes[c['source_ids']['symptom']]['name'],
         'verification':nodes[c['source_ids']['verification']]['name'],
         'action':nodes[c['source_ids']['action']]['name'],
         'condition':nodes[c['source_ids']['condition']]['name'],
         'alternative_mechanisms':[dict(mechanism=nodes[r['target']]['name'],explanation=r['source_summary_zh']) for r in relations if r['interpretation']=='ENGINEERING_INFERENCE_CROSS_SUBSYSTEM'],
         'source_references':[{'title':next(s['title'] for s in graph['sources'] if s['id']==id),'url':next(s['url'] for s in graph['sources'] if s['id']==id)} for id in c['sources']],
         'source_locator':c['source_locator'],'source_summary_zh':c['summary'],
         'review_status':'UNREVIEWED_PUBLIC_REFERENCE_SEED',
         'note':'症状/机理是鉴别线索，不是确诊；检查与措施均应经型号专家和飞行程序审查。'}
 return output

def related_nodes(graph, node_id, max_hops=2):
 index=defaultdict(list);nodes={n['id']:n for n in graph['entities']}
 for r in graph['relations']:
  index[r['source']].append((r['target'],r));index[r['target']].append((r['source'],r))
 visited={node_id:0};q=deque([node_id]); out=[]
 while q:
  curr=q.popleft()
  if visited[curr]>=max_hops:continue
  for nxt,r in index[curr]:
   if nxt not in visited:
    visited[nxt]=visited[curr]+1;q.append(nxt)
   out.append({'from':nodes[curr]['name'],'edge':r['relation_type'],'to':nodes[nxt]['name'],'hop':visited[curr]+1})
 return out

def main(argv=None):
 p=argparse.ArgumentParser(description='Offline query of unreviewed public spacecraft failure reference graph')
 p.add_argument('query',nargs='?',default='',help='fault code, equipment keyword, symptoms, mechanism')
 p.add_argument('--fault',help='exact fault code, e.g. ADCS-WHEEL-SAT')
 p.add_argument('--domain',choices=['ADCS','EPS','THERMAL','PROP','COMM','CDH','PAYLOAD'])
 p.add_argument('--hops',type=int,default=0,choices=[0,1,2,3])
 p.add_argument('--limit',type=int,default=8)
 p.add_argument('--json',action='store_true',help='machine-readable output')
 p.add_argument('--stats',action='store_true')
 a=p.parse_args(argv)
 g=load()
 if a.stats:
  d=json.loads((ROOT/'build_summary.json').read_text(encoding='utf8'))
  print(json.dumps(d,ensure_ascii=False,indent=2));return 0
 groups=search(g,a.fault or a.query,domain=a.domain,limit=max(1,a.limit))
 if a.fault:groups=[c for c in groups if c['code']==a.fault]
 if not groups:
  print('未找到匹配故障条目（该图谱覆盖范围有限）。')
  return 1
 entries=[lookup(g,c) for c in groups]
 if a.hops:
  for c,e in zip(groups,entries):e['local_links']=related_nodes(g,c['source_ids']['fault'],a.hops)
 if a.json:print(json.dumps(entries,ensure_ascii=False,indent=2));return 0
 for e in entries:
  print('\n'+e['code']+'  '+e['fault']+' ['+e['domain']+'] [未审核公共知识]')
  for key,label in [('component','相关部件'),('mechanism','候选机理'),('symptom','异常线索'),('verification','建议核查'),('action','可能措施'),('condition','执行前提')]:print('  '+label+'：'+e[key])
  for v in e['alternative_mechanisms']:print('  +跨系统假设：'+v['mechanism'])
  print('  证据定位：'+e['source_locator'])
  for v in e['source_references']:print('  文献：'+v['url'])
 print('\n注意：所有结论均待型号级审核，不作为实际确诊或飞行操作授权。')
 return 0
if __name__=='__main__':raise SystemExit(main())
