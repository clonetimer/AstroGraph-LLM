"""Run the opt-in READ-ONLY candidate DAG using an operator-approved plugin snapshot.

This is not the production Decision Governor. Approval/configuration files are
operator-controlled, and must not be writable by an Agent or retrieved document.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from astrograph.orchestration.dual_source_dag import DualSourceDAGWorkflow,build_dual_source_candidate_runtime,CANDIDATE_PIPELINE_ID
from astrograph.runtime import RuntimeRequest,RuntimeStatus
from astrograph.orchestration.dag import ChangeLedger
from astrograph.orchestration.models import WorkflowContext,PermissionScope
from astrograph.semantics.online_contracts import DiagnosticRequest
from astrograph.tools.plugins import discover_plugin_specs,load_approved_snapshot

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True,type=Path);p.add_argument('--plugins',required=True,type=Path)
    p.add_argument('--approved-config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--backend',choices=('native','langgraph'),default='native');p.add_argument('--ledger',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists; use a new artifact path')
    cfg=json.loads(a.approved_config.read_text(encoding='utf-8'))
    if cfg.get('schema_version')!='astrograph-readonly-dag-config.v1':p.error('unsupported operator config')
    req=DiagnosticRequest.model_validate_json(a.request.read_text(encoding='utf-8'))
    snapshot=load_approved_snapshot(a.plugins,discover_plugin_specs(a.plugins),
        approved_hashes=frozenset(cfg['approved_artifact_sha256']),approved_manifest_hashes=frozenset(cfg['approved_descriptor_sha256']))
    context=WorkflowContext(workflow_id='dual-source-research-dag',run_id=req.request_id,actor='operator-cli',
        permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE,
                     PermissionScope.READ_MEMORY,PermissionScope.READ_STATE,PermissionScope.READ_ARTIFACTS])
    workflow=DualSourceDAGWorkflow(snapshot,data_tool=cfg['data_tool'],knowledge_tool=cfg['knowledge_tool'],
        fault_universe=tuple(cfg['fault_universe']),investigation_tools=tuple(cfg.get('investigation_tools',[])),
        investigation_contracts=cfg.get('investigation_contracts',{}),
        backend=a.backend,max_rounds=cfg.get('max_rounds',2),max_tool_calls=cfg.get('max_tool_calls',6),
        ledger=ChangeLedger(a.ledger) if a.ledger else None)
    runtime=build_dual_source_candidate_runtime(workflow,context)
    result=runtime.execute(RuntimeRequest(request_id=req.request_id,pipeline_id=CANDIDATE_PIPELINE_ID,payload=req.model_dump(mode='json')))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create prevents another concurrent invocation overwriting the same artifact.
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result.model_dump(mode='json'),stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result.status.value,'formal_decision':None}))
    return 0 if result.status==RuntimeStatus.SUCCEEDED else 2
if __name__=='__main__':raise SystemExit(main())
