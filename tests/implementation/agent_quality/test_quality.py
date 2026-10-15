from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from astrograph.evaluation.agent_quality.metrics import retrieval_novelty, aggregate, inspect_task, review_template
from astrograph.evaluation.agent_quality.io import read_json, write_new, file_sha, safe_child
from astrograph.evaluation.agent_quality.corpus import evaluate_corpus
from astrograph.evaluation.agent_quality.fixtures import build_corpus_fixture, build_numeric_cases
from astrograph.evaluation.agent_quality.experiment import create_experiment, run_experiment, score_experiment, verify_experiment
from astrograph.agents.mission.models import AgentRequest,AgentSettings,InvestigationPlan,CheckChoice,AgentNarrative,ExplorationProposal
from astrograph.agents.mission.service import MissionAgent

ROOT=Path(__file__).resolve().parents[3]


def hit(cid="D1",text="Exact evidence, not an answer"):
    return {"citation_id":cid,"text":text,"content_sha256":hashlib.sha256(text.encode()).hexdigest()}


def test_duplicate_same_text_new_id_not_new_information():
    r=retrieval_novelty([{"hits":[hit()]},{"hits":[hit("D2")]},{"status":"FAILED","hits":[]}])
    assert [x["new_texts"] for x in r["calls"]]==[1,0,0]
    assert [x["new_citation_content_pairs"] for x in r["calls"]]==[1,1,0]
    assert r["semantic_information_gain"] is None and r["executed_calls"]==3


def test_changed_text_under_same_id_is_visible():
    r=retrieval_novelty([{"hits":[hit()]},{"hits":[hit(text="different")] }])
    assert r["calls"][1]["citation_content_changes"]==["D1"]
    assert r["unique_texts_total"]==2


@pytest.mark.parametrize("change",[{"text":{}},{"citation_id":3},{"content_sha256":"0"*64}])
def test_bad_hit_fails_closed(change):
    with pytest.raises(ValueError):retrieval_novelty([{"hits":[{**hit(),**change}]}])


@pytest.mark.parametrize("name",["../escape","/etc/passwd","x/../../escape"])
def test_evaluator_paths_confined(tmp_path,name):
    with pytest.raises(ValueError):safe_child(tmp_path,name)


def test_immutable_reports(tmp_path):
    p=tmp_path/"report.json";write_new(p,{"failed":True})
    with pytest.raises(FileExistsError):write_new(p,{"failed":False})
    assert read_json(p)["failed"] is True


def minimal_quality(status="SUCCEEDED"):
    return {"mission_status":status,"llm_status":"SUCCEEDED","http_attempts":3,"fallbacks":0,"abstentions":0,"full_wire_pairs":True,
            "diagnosis":{"status":"MODEL_ONLY","selected":"A","candidate_fault_ids":["A"]},
            "retrieval":{"executed_calls":2,"zero_new_text_calls":1},"checks":{"executed":1}}


def test_all_planned_denominators_and_incomplete_pairs():
    cells=[{"case_id":"c","mode":"thinking","planner":"llm_assisted","repeat":i,"status":s,"quality":q,"elapsed_s":t}
           for i,s,q,t in [(1,"SUCCEEDED",minimal_quality(),1),(2,"RUNNER_ERROR",None,2),(3,"NOT_RUN",None,None)]]
    r=aggregate(cells);a=r["arms"][0]
    assert a["scheduled"]==3 and a["started"]==2 and a["llm_contract_succeeded"]==1
    assert a["llm_success_rate_scheduled"]==pytest.approx(1/3)
    assert all(x["status"]=="PAIR_INCOMPLETE" for x in r["pairs"])


@pytest.mark.parametrize("update",[{"fallbacks":1},{"abstentions":1},{"http_attempts":0},{"full_wire_pairs":False},{"llm_status":"PARTIAL"}])
def test_contract_rate_does_not_count_fallback_empty_or_script(update):
    q={**minimal_quality(),**update}
    r=aggregate([{"case_id":"c","mode":"thinking","planner":"llm_assisted","repeat":1,"status":"PARTIAL","quality":q,"elapsed_s":1}])
    assert r["arms"][0]["llm_contract_succeeded"]==0


def test_pair_difference_is_not_accuracy():
    cells=[{"case_id":"c","mode":"thinking","planner":p,"repeat":1,"status":"SUCCEEDED","quality":minimal_quality(),"elapsed_s":1}
           for p in ("deterministic","llm_assisted")]
    pair=aggregate(cells)["pairs"][0]
    assert pair["diagnosis_equal"] and pair["corrected_errors"] is None


@pytest.fixture
def numeric(tmp_path):
    return build_numeric_cases(ROOT,tmp_path/"numeric")


class Scripted:
    def __init__(self, common_first=False):self.common_first=common_first
    def generate_structured(self,*,agent_id,output_model,payload):
        if output_model is InvestigationPlan:return output_model(task=payload["fixed_task"],actions=[],rationale="Fixture core investigation.")
        if output_model is CheckChoice:
            options=payload["eligible_checks"]
            choice=next((x for x in options if self.common_first and x["tool_id"].endswith("F_common")),options[0])
            return output_model(tool_id=choice["tool_id"],rationale="Explicit fixture, not live model intelligence.")
        if output_model is AgentNarrative:return output_model(candidate_fault_id=payload["required_candidate_fault_id"],excerpt_ids=[payload["sources_untrusted"][0]["excerpt_id"]])
        if output_model is ExplorationProposal:return output_model(hypotheses=[])
        raise ValueError("not scripted")


def test_multicheck_numeric_effectiveness_not_single_tool_obedience(tmp_path,numeric):
    req=AgentRequest.model_validate(read_json(Path(numeric[0]["request_path"])))
    results=[]
    for i,common in enumerate((False,True)):
        settings=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/f"runs{i}"),
                               planner="llm_assisted",max_checks=2,max_optional_actions=0)
        result=MissionAgent(settings,backend=Scripted(common)).run(req)
        report=inspect_task(Path(settings.output_root)/req.task_id)
        assert report["checks"]["multiple_option_opportunities"]>=1
        assert report["http_attempts"]==0 and not report["full_wire_pairs"]
        assert report["diagnosis"]["correct"] is None
        results.append(result.output["diagnosis_result"])
    assert results[0]["diagnosis"]["selected_fault_id"]=="B"
    assert results[1]["diagnosis"]["selected_fault_id"]!= "B"
    assert all(r["checks_executed"]==2 for r in results)


def test_trace_no_mutation_and_tamper_rejected(tmp_path,numeric):
    req=AgentRequest.model_validate(read_json(Path(numeric[2]["request_path"])))
    s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/"run"),max_optional_actions=0)
    result=MissionAgent(s).run(req);root=Path(s.output_root)/req.task_id
    before={p.name:file_sha(p) for p in root.glob('*') if p.is_file()}
    report=inspect_task(root);review=review_template(report)
    assert review["status"]=="UNREVIEWED" and review["reviewer"] is None
    assert {p.name:file_sha(p) for p in root.glob('*') if p.is_file()}==before
    (root/"step-001.json").write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match="ARTIFACT_HASH"):inspect_task(root)


def test_corpus_negative_and_counterevidence_are_reported(tmp_path):
    m=build_corpus_fixture(tmp_path/"corpus");r=evaluate_corpus(m,tmp_path/"report")
    assert r["queries"]==4 and r["mean_passage_recall_at_k"]==1
    assert r["no_answer_rejection_rate"]==0.5  # Do not tune away the lexical false positive.
    assert r["rows"][0]["counterevidence_recall"]==1
    assert r["generation_requests_sent"]==0 and not r["qrels_sent_to_retriever"]
    assert r["rows"][-1]["answer_correctness"] is None


@pytest.mark.parametrize("mutation",["document","quote","empty","duplicate","k","role"])
def test_corpus_judgment_integrity(tmp_path,mutation):
    p=build_corpus_fixture(tmp_path/"c");m=read_json(p)
    if mutation=="document":(p.parent/"imu.md").write_text("changed",encoding="utf-8")
    if mutation=="quote":m["queries"][0]["relevant"][0]["quote"]="invented text"
    if mutation=="empty":m["queries"]=[]
    if mutation=="duplicate":m["queries"].append(m["queries"][0])
    if mutation=="k":m["top_k"]=0
    if mutation=="role":m["queries"][0]["relevant"][0]["role"]="ground_truth"
    p.write_text(json.dumps(m),encoding="utf-8")
    with pytest.raises(ValueError):evaluate_corpus(p,tmp_path/"out")


@pytest.fixture
def experiment(tmp_path,numeric):
    config=tmp_path/"llm.json";write_new(config,{"model":"TEST_NOT_REAL","base_url":"http://127.0.0.1:1/v1","api_key":"do-not-export-secret"})
    s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/"unused"),
                    planner="llm_assisted",llm_config=str(config),max_checks=2,max_optional_actions=0)
    settings=tmp_path/"host.json";write_new(settings,s.model_dump(mode="json"))
    out=tmp_path/"experiment"
    manifest=create_experiment(out,project=ROOT,settings_path=settings,cases=numeric[:1],repeats=2,modes=("thinking",),check_budget=2)
    return out,manifest


def test_freeze_no_secret_no_extra_grant_no_live_by_default(experiment):
    root,m=experiment
    assert "do-not-export-secret" not in (root/"experiment.json").read_text(encoding="utf-8")
    assert len(m["cells"])==4 and [c["planner"] for c in m["cells"]]==["deterministic","llm_assisted","llm_assisted","deterministic"]
    with pytest.raises(PermissionError,match="ALLOW_LIVE"):run_experiment(root)
    assert not (root/"trials").exists()
    assert score_experiment(root)["status_counts"]=={"NOT_RUN":4}


def test_repeat_runner_resumes_only_committed_keeps_failures(experiment):
    root,m=experiment
    r=run_experiment(root,backend_factory=lambda c:Scripted(),max_runs=2)
    assert r["status_counts"]=={"NOT_RUN":2,"SUCCEEDED":2}
    first=root/"trials"/m["cells"][0]["cell_id"]/"trial.json";digest=file_sha(first)
    r2=run_experiment(root,backend_factory=lambda c:Scripted(),max_runs=2)
    assert r2["status_counts"]=={"SUCCEEDED":4} and file_sha(first)==digest
    assert score_experiment(root)==r2
    assert all(a["llm_contract_succeeded"] in (0,None) for a in r2["arms"])  # injected code is not HTTP proof


@pytest.mark.parametrize("what",["input","manifest","config","runtime"])
def test_change_invalidates_frozen_experiment(experiment,monkeypatch,what):
    root,m=experiment
    if what=="input":(root/next(iter(m["input_files"]))).write_text("changed",encoding="utf-8")
    if what=="manifest":(root/"experiment.json").write_text("{}",encoding="utf-8")
    if what=="config":Path(m["llm_config_path"]).write_text("{}",encoding="utf-8")
    if what=="runtime":monkeypatch.setattr("astrograph.evaluation.agent_quality.experiment.runtime_fingerprint",lambda _: {})
    with pytest.raises(ValueError):verify_experiment(root)


def test_interrupted_slot_not_reexecuted(experiment):
    root,m=experiment;d=root/"trials"/m["cells"][0]["cell_id"];d.mkdir(parents=True)
    with pytest.raises(RuntimeError,match="REQUIRES_REVIEW"):run_experiment(root,backend_factory=lambda c:Scripted())
    assert score_experiment(root)["status_counts"]=={"INTERRUPTED":1,"NOT_RUN":3}


def test_exception_slot_is_preserved_in_denominator(experiment):
    root,m=experiment
    def broken(_):raise RuntimeError("private endpoint details must not be copied")
    r=run_experiment(root,backend_factory=broken,max_runs=2)
    assert r["status_counts"]=={"NOT_RUN":2,"RUNNER_ERROR":1,"SUCCEEDED":1}
    raw=(root/"trials"/m["cells"][1]["cell_id"]/"trial.json").read_text(encoding="utf-8")
    assert "private endpoint" not in raw


def tiny_review_quality():
    return {"task_id":"t","result_sha256":"a"*64,"retrieval":{"calls":[{"step":1,"query":"q","returned_content_sha256":["b"*64]}]},
            "exploration":{"hypotheses":[]},"checks":{"choices":[]}}


def test_unreviewed_is_not_negative_or_success():
    from astrograph.evaluation.agent_quality.reviews import score_review
    q=tiny_review_quality();r=score_review(q,review_template(q))
    assert all(x["positive_rate_reviewed"] is None and x["review_coverage"]==0 for x in r["counts"].values())
    assert r["diagnostic_accuracy"] is None


@pytest.mark.parametrize("mutation",["hash","id","items","subject","reviewer","time","note","bool","missing"])
def test_review_requires_binding_attribution_and_complete_fields(mutation):
    from astrograph.evaluation.agent_quality.reviews import score_review
    q=tiny_review_quality();r=review_template(q)
    r.update(status="REVIEWED",reviewer="independent-test-reviewer",reviewed_at="2026-09-13T00:00:00+00:00")
    for k in ("useful","answers_question","adds_distinguishing_information","tool_corpus_fit"):r["items"][0][k]=True
    r["items"][0]["evidence_note"]="Test attribution to source; not a factual review."
    if mutation=="hash":r["result_sha256"]="x"
    if mutation=="id":r["task_id"]="other"
    if mutation=="items":r["items"]*=2
    if mutation=="subject":r["items"][0]["query"]="changed"
    if mutation=="reviewer":r["reviewer"]=None
    if mutation=="time":r["reviewed_at"]="2026-09-13T00:00:00"
    if mutation=="note":r["items"][0]["evidence_note"]=""
    if mutation=="bool":r["items"][0]["useful"]="yes"
    if mutation=="missing":r["items"][0]["useful"]=None
    with pytest.raises((ValueError,TypeError)):score_review(q,r)


def test_review_explicit_low_usefulness_not_hidden_by_contract_success():
    from astrograph.evaluation.agent_quality.reviews import score_review
    q=tiny_review_quality();r=review_template(q)
    r.update(status="REVIEWED",reviewer="test-only-reviewer",reviewed_at="2026-09-13T00:00:00+00:00")
    for k in ("useful","answers_question","adds_distinguishing_information","tool_corpus_fit"):r["items"][0][k]=False
    r["items"][0]["evidence_note"]="Negative fixture."
    out=score_review(q,r)
    assert all(v["positive_rate_reviewed"]==0 for v in out["counts"].values())
    assert out["confirmed_memory_write"] is False
