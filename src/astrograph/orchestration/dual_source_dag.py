"""Opt-in parallel candidate-fusion DAG with a bounded evidence acquisition loop.

Data inference and knowledge reasoning receive the same admitted observations, not
each other's Top-1. Existing BiRRF math is reused unchanged. This research entrypoint
returns candidates and trace, NOT a replacement authorization for Decision Governor.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from astrograph.data_driven.multitask_contract import MultiTaskPluginResult
from astrograph.fusion.birrf import BiRRFEngine
from astrograph.fusion.models import BiRRFConfig, FaultRanking, RankedFault, RankingSource, ScoreSemantics
from astrograph.orchestration.models import WorkflowContext, SideEffectLevel, PermissionScope
from astrograph.semantics.online_contracts import DiagnosticRequest
from astrograph.tools.plugins import PluginSnapshot
from .dag import ChangeLedger, DAGPlan, TaskNode, TaskOutcome, DAGExecutor


class KnowledgeToolResult(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    input_sha256: str=Field(pattern=r'^[0-9a-f]{64}$')
    knowledge_sha256: str=Field(pattern=r'^[0-9a-f]{64}$')
    forward: FaultRanking
    backward: FaultRanking

    @model_validator(mode='after')
    def branches(self) -> Self:
        if self.forward.source!=RankingSource.KG_FORWARD or self.backward.source!=RankingSource.KG_BACKWARD:
            raise ValueError('knowledge ranking source mismatch')
        return self


class DerivedObservation(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)
    evidence_id: str=Field(min_length=1)
    source_input_sha256: str=Field(pattern=r'^[0-9a-f]{64}$')
    metric_id: str=Field(min_length=1)
    unit: str=Field(min_length=1)
    value: float
    available_at: datetime
    source_evidence_ids: tuple[str,...]=Field(min_length=1)

    def information_key(self) -> str:
        # Renaming a fact or re-stamping availability is not new information.
        body=self.model_dump(mode='json',exclude={'evidence_id','available_at'})
        body['source_evidence_ids']=sorted(set(body['source_evidence_ids']))
        return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _empty(request_id: str, source: RankingSource) -> FaultRanking:
    return FaultRanking(ranking_id=f'{request_id}:{source.value}:unavailable',episode_id=request_id,source=source,
                        producer_id='unavailable-branch',producer_version='1',candidates=[])


def _model_ranking(value: MultiTaskPluginResult, request: DiagnosticRequest) -> FaultRanking:
    if value.input_sha256!=request.observation.payload_sha256 or value.model_release!=request.model_release:
        raise ValueError('model result is not bound to this input/model release')
    rows=sorted(zip(value.fault.class_ids,value.fault.values),key=lambda pair:(-pair[1],pair[0]))
    return FaultRanking(ranking_id=request.request_id+':model',episode_id=request.request_id,
        source=RankingSource.DATA_DRIVEN,producer_id=value.model_release.release_id,producer_version='plugin-contract.v1',
        score_semantics=ScoreSemantics.RANKING_SCORE,
        candidates=[RankedFault(fault_id=fault,rank=index+1,score=score,source=RankingSource.DATA_DRIVEN,
                     source_evidence_ids=[request.observation.payload_sha256],
                     correlation_group_ids=['observation:'+request.observation.payload_sha256],lineage_complete=True,
                     contributes_to_fusion=value.fault.status=='AVAILABLE') for index,(fault,score) in enumerate(rows)],
        metadata={'source_score_kind':value.fault.score_kind,'task_status':value.fault.status})


class DualSourceDAGWorkflow:
    """Trusted deterministic planner; all actual tools still go through one registry.

    Each investigation round is an acyclic fan-out/fan-in graph. The outer loop is
    explicitly bounded. LangGraph is selected explicitly and never silently mocked.
    """
    def __init__(self, snapshot: PluginSnapshot, *, data_tool: str, knowledge_tool: str,
                 fault_universe: tuple[str,...], investigation_tools: tuple[str,...]=(),
                 investigation_contracts: dict[str,dict[str,str]] | None=None,
                 config: BiRRFConfig | None=None, backend: Literal['native','langgraph']='native',
                 max_rounds: int=2, max_tool_calls: int=6, max_workers: int=2,
                 ledger: ChangeLedger | None=None):
        if backend not in ('native','langgraph') or max_rounds<1 or max_rounds>4 or max_tool_calls<2 or max_workers<1:
            raise ValueError('invalid bounded workflow configuration')
        if not fault_universe or len(set(fault_universe))!=len(fault_universe):
            raise ValueError('a unique registered fault universe is required')
        for tool_id in (data_tool,knowledge_tool,*investigation_tools):
            spec=snapshot.registry.get(tool_id).spec
            if spec.side_effect_level not in (SideEffectLevel.NONE,SideEffectLevel.READ_ONLY):
                raise PermissionError('online investigation only accepts approved read-only tools')
        contracts=investigation_contracts or {}
        if set(contracts)!=set(investigation_tools) or any(not metrics or any(not k or not v for k,v in metrics.items()) for metrics in contracts.values()):
            raise ValueError('every investigation tool needs an operator-approved metric/unit contract')
        self.investigation_contracts={tool:dict(metrics) for tool,metrics in contracts.items()}
        self.snapshot=snapshot;self.data_tool=data_tool;self.knowledge_tool=knowledge_tool
        self.universe=fault_universe;self.investigation_tools=tuple(dict.fromkeys(investigation_tools))
        self.config=config or BiRRFConfig(expected_fault_universe=fault_universe)
        if self.config.expected_fault_universe and set(self.config.expected_fault_universe)!=set(fault_universe):
            raise ValueError('fusion and workflow fault universes differ')
        self.backend=backend;self.max_rounds=max_rounds;self.max_tool_calls=max_tool_calls
        self.max_workers=max_workers;self.ledger=ledger

    def invoke(self, request: DiagnosticRequest, context: WorkflowContext) -> dict:
        # Revalidate even caller-built model_copy/model_construct instances at boundary.
        request=DiagnosticRequest.model_validate_json(request.model_dump_json())
        if PermissionScope.RUN_DIAGNOSIS not in context.permissions:
            raise PermissionError('RUN_DIAGNOSIS permission required')
        if request.plugin_release.sha256!=self.snapshot.snapshot_sha256:
            raise ValueError('plugin snapshot differs from request pin')
        context=context.model_copy(deep=True)
        context.metadata={**context.metadata,'tool_versions':dict(self.snapshot.tool_versions)}
        evidence: list[DerivedObservation]=[]
        seen_ids: dict[str,str]={};seen_information: set[str]=set()
        used_tools: set[str]=set();trace=[];calls=0;rounds=[];final=None
        stop='ROUND_BUDGET_EXHAUSTED'
        for round_index in range(self.max_rounds):
            if calls+2>self.max_tool_calls:
                stop='TOOL_BUDGET_EXHAUSTED';break
            payload={'request':request.model_dump(mode='json'),
                     'registered_fault_ids':list(self.universe),
                     'derived_observations':[item.model_dump(mode='json') for item in evidence]}
            def data(_):
                raw=self.snapshot.registry.invoke(self.data_tool,payload,context)
                result=MultiTaskPluginResult.model_validate(raw)
                ranking=_model_ranking(result,request)
                self._validate_ranking(ranking,request)
                return {'ranking':ranking.model_dump(mode='json'),'three_task':raw}
            def knowledge(_):
                raw=self.snapshot.registry.invoke(self.knowledge_tool,payload,context)
                result=KnowledgeToolResult.model_validate(raw)
                if result.input_sha256!=request.observation.payload_sha256 or result.knowledge_sha256!=request.knowledge_release.sha256:
                    raise ValueError('knowledge result input/release mismatch')
                for rank in (result.forward,result.backward):self._validate_ranking(rank,request)
                return result.model_dump(mode='json')
            def fuse(parents):
                dr,kr=parents['Data'],parents['Knowledge']
                data_rank=FaultRanking.model_validate(dr.output['ranking']) if dr.status=='SUCCEEDED' else _empty(request.request_id,RankingSource.DATA_DRIVEN)
                if kr.status=='SUCCEEDED':
                    kg=KnowledgeToolResult.model_validate(kr.output);forward,backward=kg.forward,kg.backward
                else:forward,backward=_empty(request.request_id,RankingSource.KG_FORWARD),_empty(request.request_id,RankingSource.KG_BACKWARD)
                result=BiRRFEngine().fuse(data_rank,forward,backward,config=self.config)
                # Preserve engine conflicts: candidate rank is not a safety authorization.
                return result.model_dump(mode='json')
            plan=DAGPlan('diagnostic-candidate-round',(
                TaskNode('Data'),TaskNode('Knowledge'),TaskNode('Fuse',('Data','Knowledge'),allow_failed_dependencies=True)))
            handlers={'Data':data,'Knowledge':knowledge,'Fuse':fuse}
            run_id=f'{request.request_id}:round:{round_index}'
            if self.backend=='langgraph':
                from .langgraph_dag import run_langgraph_dag
                outcomes=run_langgraph_dag(plan,handlers,max_workers=self.max_workers,run_id=run_id,ledger=self.ledger)
            else:
                outcomes=DAGExecutor(plan,handlers,max_workers=self.max_workers,ledger=self.ledger).run(run_id=run_id)
            calls+=2
            rounds.append({key:asdict(value) for key,value in outcomes.items()})
            if outcomes['Fuse'].status!='SUCCEEDED':
                stop='FUSION_FAILED';break
            final=outcomes['Fuse'].output
            data_top=self._top(outcomes['Data'],'ranking')
            kg_top=self._top(outcomes['Knowledge'],'forward')
            conflict=(data_top is None or kg_top is None or data_top!=kg_top or bool(final['blocking_conflicts']))
            trace.append({'round':round_index,'data_top':data_top,'knowledge_top':kg_top,
                          'candidate_top':final['top_fault_id'],'conflict':conflict,'evidence_count':len(evidence)})
            if not final['candidates']:
                stop='NO_AVAILABLE_CANDIDATES';break
            if not conflict:
                stop='BRANCHES_AGREE';break
            if round_index+1>=self.max_rounds:
                stop='ROUND_BUDGET_EXHAUSTED';break
            if calls+3>self.max_tool_calls:
                stop='TOOL_BUDGET_EXHAUSTED';break
            tool=next((item for item in self.investigation_tools if item not in used_tools),None)
            if tool is None:
                stop='NO_UNUSED_QUALIFIED_TOOL';break
            used_tools.add(tool);calls+=1
            try:
                raw=self.snapshot.registry.invoke(tool,payload,context)
                if set(raw)!={'observations'} or not isinstance(raw['observations'],list):
                    raise ValueError('investigation tool must return structured observations')
                new=[DerivedObservation.model_validate(item) for item in raw['observations']]
                admitted=[]
                # Validate the entire batch before admitting any part of it.
                batch_ids=dict(seen_ids);batch_information=set(seen_information)
                for item in new:
                    if self.investigation_contracts[tool].get(item.metric_id)!=item.unit:
                        raise ValueError('unregistered metric or unit mismatch')
                    allowed_sources={request.observation.artifact_id,request.observation.payload_sha256,*seen_ids}
                    if not set(item.source_evidence_ids)<=allowed_sources:
                        raise ValueError('unknown source evidence reference')
                    if item.source_input_sha256!=request.observation.payload_sha256:
                        raise ValueError('evidence belongs to another input')
                    if item.available_at.tzinfo is None or item.available_at.utcoffset() is None or item.available_at>request.cutoff_at:
                        raise ValueError('evidence is not yet available')
                    key=item.information_key()
                    if item.evidence_id in batch_ids and batch_ids[item.evidence_id]!=key:
                        raise ValueError('evidence ID reused with different content')
                    batch_ids[item.evidence_id]=key
                    if key not in batch_information:
                        batch_information.add(key);admitted.append(item)
                if not admitted:
                    stop='NO_NEW_EVIDENCE';break
                evidence.extend(admitted);seen_ids=batch_ids;seen_information=batch_information
            except Exception as exc:
                trace.append({'tool_id':tool,'error_type':type(exc).__name__})
                stop='INVESTIGATION_FAILED';break
        return {'schema_version':'astrograph-dual-source-dag-result.v1',
                'status':'CANDIDATES_READY' if final and final['candidates'] else 'UNAVAILABLE',
                'backend':self.backend,'request_id':request.request_id,'plugin_snapshot_sha256':self.snapshot.snapshot_sha256,
                'tool_calls':calls,'rounds':rounds,'planning_trace':trace,'stopping_reason':stop,
                'derived_observations':[item.model_dump(mode='json') for item in evidence],
                'final_fusion':final,'formal_decision':None,
                'claim_boundary':'Research candidate DAG. Existing Decision Governor is not bypassed or replaced; rank scores are not diagnosis probabilities.'}

    def _validate_ranking(self, ranking, request):
        if ranking.episode_id!=request.request_id:
            raise ValueError('ranking episode differs from request')
        if not {item.fault_id for item in ranking.candidates}<=set(self.universe):
            raise ValueError('unregistered candidate')

    @staticmethod
    def _top(outcome: TaskOutcome, field: str) -> str | None:
        if outcome.status!='SUCCEEDED':return None
        rows=outcome.output[field]['candidates']
        active=[row for row in rows if row['contributes_to_fusion']]
        if not active:return None
        rank=min(row['rank'] for row in active)
        top=[row['fault_id'] for row in active if row['rank']==rank]
        return top[0] if len(top)==1 else None


CANDIDATE_PIPELINE_ID = 'diagnosis.dual_source.research.v1'


def build_dual_source_candidate_runtime(workflow: DualSourceDAGWorkflow, operator_context: WorkflowContext):
    """Expose this optional subgraph behind the existing canonical Runtime envelope.

    Permissions come from trusted factory input, never from observation payloads.
    The default application pipeline and its Governor are not reconfigured here.
    """
    from astrograph.runtime import AstroGraphRuntime, RuntimePipeline, StageResult

    class CandidateStage:
        stage_id = 'dual_source.candidate_subgraph.v1'
        stage_version = '1.0-dev1'

        def execute(self, payload, context):
            request = DiagnosticRequest.model_validate(payload)
            trusted = operator_context.model_copy(deep=True)
            trusted.run_id = request.request_id
            result = workflow.invoke(request, trusted)
            if result['status'] == 'UNAVAILABLE':
                # Retain the full trace as diagnostic details, never fabricate NORMAL.
                return StageResult.failed(code='CANDIDATES_UNAVAILABLE',message='No admitted diagnostic candidates',
                    stage_id=self.stage_id,details={'candidate_result':result})
            return StageResult.succeeded(result, metadata={'formal_decision_published':False,'opt_in_research':True})

    return AstroGraphRuntime([RuntimePipeline(pipeline_id=CANDIDATE_PIPELINE_ID,stages=(CandidateStage(),),
        description='Opt-in read-only dual-source candidate subgraph; canonical Runtime boundary retained.')])
