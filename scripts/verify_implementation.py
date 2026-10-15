"""Execute independent regression groups concurrently, then join and write one receipt.

Only fixed project-owned commands are executed. A failed group does not cancel
unrelated groups, and every node start/finish is recorded in a hash-chained ledger.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
from astrograph.orchestration.dag import ChangeLedger,DAGExecutor,DAGPlan,TaskNode
ROOT=Path(__file__).resolve().parents[1]
GROUPS={
 'ContractsDAG':['tests/implementation/test_online_contracts.py','tests/implementation/test_dag_execution.py','tests/implementation/test_dual_source_dag.py','tests/implementation/test_dag_cli.py'],
 'Plugins':['tests/implementation/test_plugin_execution.py','tests/test_controlled_tools.py','tests/test_online_hybrid_tool_registry.py'],
 'Memory':['tests/implementation/test_memory_binding.py','tests/test_case_registry_lifecycle.py'],
 'Physical':['tests/implementation/test_physical_entry_readiness.py','tests/test_physical_diagnosis.py','tests/test_repair_oag_runtime.py'],
 'Fusion':['tests/test_data_graph_birrf.py','tests/test_fusion_diagnostic_channels.py','tests/test_document_claim_fusion_adapter.py'],
 'DOR':['tests/test_dor_control_plane_bridge.py','tests/test_dor_semantic_bridge.py','tests/test_dor_multitask_bridge.py','tests/test_dor_r5_active_discovery_contracts.py'],
 'Compatibility':['tests/test_clean_distribution.py','tests/test_verify_source_inventory.py','tests/test_platform_knowledge_enrichment.py','tests/test_knowledge_graph_base.py','tests/test_knowledge_graph_extended.py','tests/test_object_plugin_dag_and_second_object.py','tests/test_object_plugin_factory.py'],
}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--group-timeout',type=int,default=180)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'events.jsonl').exists():parser.error('use a fresh run directory to preserve prior receipts')
    ledger=ChangeLedger(args.output/'events.jsonl')
    run_id='validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    def test_handler(group):
        def execute(_):
            xml=args.output/(group+'.xml');log=args.output/(group+'.log')
            command=[sys.executable,'-m','pytest','-q',*GROUPS[group],'--disable-warnings','--junitxml='+str(xml.resolve())]
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            start=time.monotonic();status='FINISHED';code=None
            with log.open('w',encoding='utf-8') as stream:
                try:code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=args.group_timeout).returncode
                except subprocess.TimeoutExpired:status='TIMED_OUT'
            counts={}
            if xml.exists():
                tree=ET.parse(xml)
                suites=tree.getroot().findall('testsuite') if tree.getroot().tag=='testsuites' else [tree.getroot()]
                counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
            receipt={'group':group,'command':command,'status':status,'exit_code':code,'duration_s':time.monotonic()-start,'counts':counts}
            (args.output/(group+'.receipt.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
            if code!=0:raise RuntimeError('REGRESSION_GROUP_FAILED:'+group)
            return receipt
        return execute
    def join(deps):
        rows={name:json.loads((args.output/(name+'.receipt.json')).read_text()) for name in GROUPS}
        total={key:sum(r['counts'].get(key,0) for r in rows.values()) for key in ('tests','failures','errors','skipped')}
        total['passed']=total['tests']-total['failures']-total['errors']-total['skipped']
        receipt={'run_id':run_id,'status':'PASS' if all(v.status=='SUCCEEDED' for v in deps.values()) else 'FAIL',
                 'groups':rows,'counts':total,'scope':'Targeted regression, not full-suite or accuracy benchmark.'}
        (args.output/'summary.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');return receipt
    plan=DAGPlan('parallel-implementation-regression',tuple(TaskNode(name) for name in GROUPS)+(TaskNode('Receipt',tuple(GROUPS),allow_failed_dependencies=True),))
    (args.output/'plan.json').write_text(json.dumps(plan.to_dict(),indent=2),encoding='utf-8')
    (args.output/'plan.mmd').write_text(plan.mermaid(),encoding='utf-8')
    result=DAGExecutor(plan,{**{g:test_handler(g) for g in GROUPS},'Receipt':join},max_workers=args.workers,ledger=ledger).run(run_id=run_id)
    (args.output/'outcomes.json').write_text(json.dumps({k:asdict(v) for k,v in result.items()},indent=2),encoding='utf-8')
    print(json.dumps(result['Receipt'].output,ensure_ascii=False,indent=2))
    return 0 if all(v.status=='SUCCEEDED' for v in result.values()) else 1
if __name__=='__main__':raise SystemExit(main())
