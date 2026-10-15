from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from astrograph.agents.mission.context import ContextBuilder, ContextBudgetExceeded
from astrograph.agents.mission.context_projection import project_diagnostic_context, restore_diagnostic_context
from astrograph.agents.mission.models import InvestigationPlan, CheckChoice, AgentRequest
from astrograph.agents.mission.planner import Planner
from astrograph.agents.mission.storage import canonical, sha
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.llm.token_budget import TokenizerLoadError


def large_ranking(count=14, lineage=1090):
    group = [f"CORR-OBS-{i:040x}" for i in range(lineage)]
    return {"data_ranking": {"source": "DATA_DRIVEN", "candidates": [
        {"fault_id": f"FAULT-{i}", "rank": i+1, "score": 1/(i+2),
         "correlation_group_ids": list(group), "source_evidence_ids": [f"EF-{i}"],
         "value": -0.0005, "unit": "rad/s", "status": "NOT_SATISFIED",
         "contradicting_evidence_ids": [f"COUNTER-{i}"]}
        for i in range(count)]}}


def writer(digest, value):
    return {"name": f"context-{digest}.json", "sha256": digest}


def test_1090_lineages_shared_by_14_candidates_are_lossless_and_not_new_votes():
    payload = {"diagnostic_context": large_ranking()}
    before = copy.deepcopy(payload)
    view, stats = project_diagnostic_context(payload, full_payload=payload, full_artifact=writer(sha(payload),payload))
    assert payload == before
    assert stats["lineage_occurrences"] == 14
    assert stats["distinct_lineage_lists"] == 1
    assert stats["lineage_catalogue"]["L1"]["count"] == 1090
    assert view["_context_encoding"]["diagnostic_votes_added"] is False
    assert canonical(restore_diagnostic_context(view, full_payload=payload)) == canonical(before)
    assert len(canonical(view)) < len(canonical(payload))/10


def test_partial_overlap_and_different_order_are_not_declared_independent():
    payload = {"diagnostic_context":large_ranking(count=3,lineage=80)}
    candidates=payload["diagnostic_context"]["data_ranking"]["candidates"]
    candidates[1]["correlation_group_ids"] = list(reversed(candidates[0]["correlation_group_ids"]))
    candidates[2]["correlation_group_ids"] = candidates[0]["correlation_group_ids"][40:]+[f"other-{i}" for i in range(40)]
    view,stats=project_diagnostic_context(payload,full_payload=payload,full_artifact=writer(sha(payload),payload))
    overlap=view["_context_encoding"]["lineage_pairwise_overlap"]
    assert [x["intersection_count"] for x in overlap] == [80,40,40]
    assert [x["union_count"] for x in overlap] == [80,120,120]
    assert restore_diagnostic_context(view,full_payload=payload)==payload


def test_absent_artifact_does_not_silently_drop_lineage():
    payload={"diagnostic_context":large_ranking()}
    view,stats=project_diagnostic_context(payload,full_payload=payload)
    assert stats["lineage_occurrences"]==0
    assert restore_diagnostic_context(view,full_payload=payload)==payload
    with pytest.raises(ContextBudgetExceeded):
        ContextBuilder().build(InvestigationPlan,payload)


@pytest.mark.parametrize("fault",["artifact_digest","unsafe_name","lineage_hash","lineage_count","full_artifact","table_width","missing_shared"])
def test_projection_tampering_is_rejected(fault):
    payload={"diagnostic_context":large_ranking(count=3,lineage=80)}
    binding=writer(sha(payload),payload)
    if fault in {"artifact_digest","unsafe_name"}:
        binding["sha256" if fault=="artifact_digest" else "name"]="wrong" if fault=="artifact_digest" else "../escape.json"
        with pytest.raises(ValueError):project_diagnostic_context(payload,full_payload=payload,full_artifact=binding)
        return
    view,_=project_diagnostic_context(payload,full_payload=payload,full_artifact=binding)
    if fault=="lineage_hash":view["_correlation_lineage_sets"]["L1"]["sha256"]="0"*64
    if fault=="lineage_count":view["_correlation_lineage_sets"]["L1"]["count"]+=1
    if fault=="full_artifact":payload["modified"]=True
    if fault=="table_width":view["diagnostic_context"]["data_ranking"]["candidates"]["$table"]["rows"][0].append("extra")
    if fault=="missing_shared":view["diagnostic_context"]={"$shared":"missing"}
    with pytest.raises((ValueError,KeyError)):restore_diagnostic_context(view,full_payload=payload)


def test_missing_null_false_zero_and_negation_are_not_merged():
    candidates=[{"fault_id":str(i),"flag":v,"counter":"NOT measured, not ruled out","unit":"rad/s"} for i,v in enumerate([False,0,None,True,1])]
    payload={"diagnosis":{"candidate_explanations":candidates}}
    view,_=project_diagnostic_context(payload,full_payload=payload)
    assert canonical(restore_diagnostic_context(view,full_payload=payload))==canonical(payload)
    candidates.append({"fault_id":"missing","counter":"not measured","unit":"rad/s"})
    view,_=project_diagnostic_context(payload,full_payload=payload)
    assert canonical(restore_diagnostic_context(view,full_payload=payload))==canonical(payload)


@pytest.mark.parametrize("key",["_context_encoding","$table","$lineage","$shared"])
def test_input_cannot_forge_codec_markers(key):
    with pytest.raises(ValueError,match="RESERVED"):
        project_diagnostic_context({"diagnosis":{key:"forged"}},full_payload={})


class CountingTokenizer:
    """Protocol fixture, not the deployed Qwen tokenizer."""
    def __init__(self):self.calls=[]
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt):
        self.calls.append(copy.deepcopy(messages))
        assert tokenize and add_generation_prompt
        return list(range(800))


def test_mission_uses_actual_backend_preflight_not_an_earlier_byte_rejection():
    tokenizer=CountingTokenizer()
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model="fixture",context_length=4096,max_tokens=1000,tokenizer_mode="required",repair_attempts=0),tokenizer=tokenizer)
    seen=[]
    backend._post=lambda body:(seen.append(body) or {"choices":[{"message":{"content":json.dumps({"task":"research","actions":[],"rationale":"Fixture"})},"finish_reason":"stop"}]})
    planner=Planner(backend,mode="llm_assisted",context_builder=ContextBuilder(max_tokens=4096,output_reserve=1000))
    # Deliberately too large for the old byte threshold, but tokenizer fixture reports fit.
    plan=planner.prepare(AgentRequest(task_id="budget",task="research"),[],diagnostic_context={"counterevidence":"not confirmed "*1200})
    assert plan.task=="research" and planner.calls==1 and len(seen)==1
    record=planner.context_builder.records[-1]
    assert record["original_bytes"]>4096 and record["budget_measurement"]["fits"]
    assert tokenizer.calls[-1]==seen[0]["messages"]
    assert "not confirmed "*1200 in seen[0]["messages"][1]["content"]


def test_required_tokenizer_failure_never_uses_looser_estimator():
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model="fixture",tokenizer_mode="required",tokenizer_path="/nonexistent/local/model"))
    with pytest.raises(TokenizerLoadError):backend.measure_structured_context(output_model=CheckChoice,payload={"work_mode":"thinking","eligible_checks":[{"tool_id":"fixture.only"}]})


def test_true_counterevidence_over_budget_is_still_rejected_before_http():
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model="fixture",context_length=4096,max_tokens=1000,tokenizer_mode="disabled",repair_attempts=0))
    backend._post=lambda body:pytest.fail("must not send oversized mandatory context")
    planner=Planner(backend,mode="llm_assisted",context_builder=ContextBuilder(max_tokens=4096,output_reserve=1000))
    planner.prepare(AgentRequest(task_id="overflow"),[],diagnostic_context={"counterevidence":"indispensable negative evidence "*1000})
    assert planner.calls==0
    assert "MANDATORY_CONTEXT_TOO_LARGE" in planner.records[-1]["reason"]


def test_transport_escaping_is_not_counted_as_model_input_tokens():
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model="fixture",tokenizer_mode="disabled"))
    messages=[{"role":"system","content":"规则"},{"role":"user","content":'"quote"\\\\\n'*100}]
    result=backend._measure_prompt_tokens(payload={"work_mode":"thinking","eligible_checks":[{"tool_id":"fixture.only"}]},messages=messages)
    expected=sum(len(m['role'].encode())+len(m['content'].encode()) for m in messages)+512
    assert result["selected_prompt_tokens"]==expected
    assert result["tokenizer_prompt_tokens"] is None
    assert result["guarded_prompt_tokens"]==expected+backend.config.prompt_token_reserve


def test_non_http_compat_import_does_not_require_fastapi():
    code='''
import builtins
real=builtins.__import__
def guarded(name,*args,**kw):
    if name=="fastapi" or name.startswith("fastapi."): raise ModuleNotFoundError("isolated optional api")
    return real(name,*args,**kw)
builtins.__import__=guarded
import astrograph.compat
from astrograph.evaluation import three_task_replay
assert "fastapi" not in __import__("sys").modules
'''
    process=subprocess.run([sys.executable,"-c",code],env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[3]/'src')},capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stderr


def test_saved_context_replay_preserves_original_workspace_and_sends_no_generation(tmp_path):
    from scripts.acceptance.context_budget import replay
    task=tmp_path/'old';task.mkdir()
    payload={'goal':'Check competing hypotheses','fixed_task':'diagnose','max_actions':0,'diagnostic_context':large_ranking()}
    digest=sha(payload);artifact=writer(digest,payload)
    (task/artifact['name']).write_bytes(canonical(payload))
    (task/'context-receipts.json').write_text(json.dumps([{'full_payload_artifact':artifact,'full_payload_sha256':digest,'status':'MANDATORY_CONTEXT_TOO_LARGE'}]),encoding='utf-8')
    before={p.name:p.read_bytes() for p in task.iterdir()}
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='offline',context_length=16384,max_tokens=4096,tokenizer_mode='disabled'))
    backend._post=lambda body:pytest.fail('offline replay must not generate')
    result=replay(task,tmp_path/'new',mode='thinking',backend=backend,context_length=16384,output_reserve=4096)
    assert result['status']=='READY_FOR_LIVE_RETEST' and result['generation_requests_sent']==0
    assert before=={p.name:p.read_bytes() for p in task.iterdir()}


def test_saved_context_replay_rejects_tampered_input(tmp_path):
    from scripts.acceptance.context_budget import replay
    task=tmp_path/'old';task.mkdir()
    (task/'context-bad.json').write_text('{"fixed_task":"research"}',encoding='utf-8')
    (task/'context-receipts.json').write_text(json.dumps([{'full_payload_artifact':{'name':'context-bad.json','sha256':'0'*64},'full_payload_sha256':'0'*64}]),encoding='utf-8')
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='offline',tokenizer_mode='disabled'))
    result=replay(task,tmp_path/'new',mode='thinking',backend=backend,context_length=16384,output_reserve=4096)
    assert result['status']=='BLOCKED' and 'hash mismatch' in result['rows'][0]['reason']


def test_saved_context_replay_will_not_overwrite_results(tmp_path):
    from scripts.acceptance.context_budget import replay
    with pytest.raises(FileExistsError):
        replay(tmp_path,tmp_path,mode='thinking',backend=None,context_length=16384,output_reserve=4096)


def test_configure_local_tokenizer_is_explicit_and_does_not_load_weights(tmp_path):
    import argparse
    from scripts.acceptance.llm_service import configure
    tokenpath=tmp_path/'tokenizer';tokenpath.mkdir()
    args=argparse.Namespace(base_url='http://127.0.0.1:8000/v1',model='served-model',api_key_env=None,
         output_dir=str(tmp_path/'config'),max_tokens=4096,timeout=120,context_length=16384,
         strategy='json_schema',tokenizer_path=str(tokenpath))
    assert configure(args)==0
    cfg=json.loads((tmp_path/'config'/'llm.local.json').read_text(encoding='utf-8'))
    assert cfg['tokenizer_mode']=='required' and cfg['tokenizer_local_files_only'] is True
    assert cfg['tokenizer_path']==str(tokenpath) and cfg['tokenizer_trust_remote_code'] is False
