#!/usr/bin/env python3
"""Write seed to *separate* Neo4j labels, never to AstroGraph reviewed KG.

DRY RUN by default. For write: --confirm-write and host opt-in env, a dedicated
non-default DB with a restricted principal. This is NOT a release/publisher.
"""
import argparse
from hashlib import sha256
import json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NODE_CYPHER='''
UNWIND $rows AS row
MERGE (n:AstroPublicSeedEntity {seed_id: row.id, seed_scope: $scope})
SET n.name = row.name, n.entity_type = row.entity_type,
    n.domains = row.domains, n.aliases = row.aliases,
    n.status = 'PUBLIC_REFERENCE_SEED_UNREVIEWED',n.seed_digest = $digest
'''
EDGE_CYPHER='''
UNWIND $rows AS row
MATCH (s:AstroPublicSeedEntity {seed_id: row.source,seed_scope:$scope})
MATCH (t:AstroPublicSeedEntity {seed_id: row.target,seed_scope:$scope})
MERGE (r:AstroPublicSeedClaim {seed_id: row.id,seed_scope:$scope})
SET r.relation_type=row.relation_type,r.source_ids=row.source_ids,
    r.source_locator=row.source_locator,r.source_summary_zh=row.source_summary_zh,
    r.interpretation=row.interpretation,r.status='PUBLIC_REFERENCE_SEED_UNREVIEWED',
    r.is_confirmed_root_cause=false,r.automatic_control_authorized=false,
    r.approval_required=row.approval_required,r.seed_digest=$digest
MERGE (s)-[:ASTRO_PUBLIC_SEED_SUBJECT]->(r)
MERGE (r)-[:ASTRO_PUBLIC_SEED_OBJECT]->(t)
'''

def prepare(root=ROOT):
 raw=(root/'graph.json').read_bytes();g=json.loads(raw)
 if g.get('graph_kind')!='UNREVIEWED_PUBLIC_REFERENCE_SEED' or not g.get('relations'):
  raise ValueError('expected unreviewed public seed graph')
 assert all(not e['automatic_control_authorized'] and not e['is_confirmed_root_cause'] for e in g['relations'])
 return g,sha256(raw).hexdigest()

def main(argv=None,driver=None):
 ap=argparse.ArgumentParser();ap.add_argument('--confirm-write',action='store_true');ap.add_argument('--batch-size',type=int,default=100)
 a=ap.parse_args(argv)
 g,digest=prepare()
 result=dict(scope=g['scope'],node_count=len(g['entities']),edge_count=len(g['relations']),sha256=digest,
             labels=['AstroPublicSeedEntity','AstroPublicSeedClaim'],flight_authorized=False)
 if not a.confirm_write:
  print(json.dumps(dict(result,status='DRY_RUN_NO_DATABASE_WRITE'),ensure_ascii=False,indent=2));return 0
 if os.environ.get('ASTROGRAPH_PUBLIC_SEED_NEO4J_WRITE_ENABLED')!='1':raise PermissionError('host opt-in required')
 database=os.environ.get('ASTROGRAPH_PUBLIC_SEED_NEO4J_DATABASE','').strip()
 if not database or database.casefold() in {'neo4j','system'}:raise PermissionError('dedicated Neo4j database required')
 uri=os.environ.get('ASTROGRAPH_PUBLIC_SEED_NEO4J_URI','').strip()
 user=os.environ.get('ASTROGRAPH_PUBLIC_SEED_NEO4J_USER','').strip()
 password=os.environ.get('ASTROGRAPH_PUBLIC_SEED_NEO4J_PASSWORD','')
 if driver is None:
  if not uri.startswith(('neo4j+s://','bolt+s://','bolt://','neo4j://')) or not user or not password:
   raise PermissionError('trusted host URI/user/password required')
  from neo4j import GraphDatabase
  driver=GraphDatabase.driver(uri,auth=(user,password),connection_timeout=8.0)
  own=True
 else:own=False
 try:
  with driver.session(database=database) as session:
   bs=min(300,max(1,a.batch_size))
   for k,query in [('entities',NODE_CYPHER),('relations',EDGE_CYPHER)]:
    for start in range(0,len(g[k]),bs):
     session.execute_write(lambda tx,rows:qwrite(tx,query,rows,g['scope'],digest),g[k][start:start+bs])
 finally:
  if own:driver.close()
 print(json.dumps(dict(result,status='IMPORTED_AS_UNREVIEWED_PUBLIC_SEED',database=database),ensure_ascii=False,indent=2))
 return 0

def qwrite(tx,query,rows,scope,digest):
 tx.run(query,rows=rows,scope=scope,digest=digest).consume()

if __name__=='__main__':raise SystemExit(main())
