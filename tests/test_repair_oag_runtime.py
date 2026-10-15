"""Real native Runtime+KG+BiRRF tests; HTTP fixtures are protocol doubles, not live AI."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import threading
import pytest
from pydantic import ValidationError
from astrograph.runtime import AstroGraphRuntime, RuntimeRequest, RuntimeStatus
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.oag import OAGInput, build_csv_oag_runtime
from astrograph.oag.cli import demo_input, main
from astrograph.oag.contracts import LiteratureDocument, OAGExplanation, OntologyProposal
from astrograph.oag.runtime import OAG_PIPELINE_ID
from astrograph.oag.paperseek_http import PaperSeekHTTPClient, PaperSeekConfig

ROOT=Path(__file__).resolve().parents[1]
GRAPH=ROOT/"benchmarks/knowledge_graph/online_extended"


def run(data=None, **providers):
    runtime=build_csv_oag_runtime(GRAPH, **providers)
    assert isinstance(runtime, AstroGraphRuntime)
    return runtime.execute(RuntimeRequest(request_id="repair-test",pipeline_id=OAG_PIPELINE_ID,
                           payload=(data or demo_input(ROOT)).model_dump(mode="json")))


@contextmanager
def http_fixture(callback):
    """A local protocol-only test server; never counted as real model execution."""
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path,payload))
            status,value=callback(self.path,payload)
            raw=value if isinstance(value,bytes) else json.dumps(value).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}",calls
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


def paper_reply():
    return {"ranked":[{"id":"test-doc","title":"Protocol test material",
             "abstract":"Bearing friction affects wheel speed.","url":"https://example.org/test-material"}]}


def llm_config(url):
    return OpenAICompatibleConfig(base_url=url+"/v1",model="test-protocol-double-not-qwen",
               api_key="local-test-key",timeout_s=3,tokenizer_mode="disabled",
               response_format_strategy="json_object",repair_attempts=0)


def test_native_runtime_all_seven_stages_execute():
    result=run()
    assert result.status == RuntimeStatus.SUCCEEDED
    assert len(result.stages)==7
    out=result.output
    assert out["summary"]["generation"]=="NATIVE_GRAPH_QUERY_PLUS_BIRRF_PROJECTION"
    assert out["summary"]["suspected_fault_id"]=="ADCS_RW_FRICTION_INCREASE"
    assert out["summary"]["calibrated_probability"] is None
    assert out["formal_diagnosis_publishable"] is False
    assert out["literature"]["status"]=="NOT_CONFIGURED"
    assert out["ontology"]["active_kg_modified"] is False
    assert out["generation"]["explanation"] is None
    assert out["generation"]["semantic_hallucination_assessment"]=="NOT_EVALUATED"


def test_summary_hash_is_reproducible():
    assert run().output["summary"]==run().output["summary"]


@pytest.mark.parametrize("field", ["fault","label","ground_truth"])
def test_inference_request_forbids_extra_answer_fields(field):
    data=demo_input(ROOT).model_dump();data[field]="LEAKED-ANSWER"
    with pytest.raises(ValidationError): OAGInput.model_validate(data)


def test_evaluation_metadata_is_rejected_not_used_for_diagnosis():
    data=demo_input(ROOT).model_dump()
    data["bundle"]["evaluation_metadata"]={"fault":"LEAKED-ANSWER"}
    with pytest.raises(ValidationError,match="evaluation_metadata"): OAGInput.model_validate(data)


def test_duplicate_fact_id_is_rejected():
    data=demo_input(ROOT).model_dump()
    data["bundle"]["evidence_facts"].append(data["bundle"]["evidence_facts"][0])
    with pytest.raises(ValidationError,match="duplicate evidence"): OAGInput.model_validate(data)


def test_mismatched_episode_is_rejected():
    data=demo_input(ROOT).model_dump();data["diagnostic_ranking"]["episode_id"]="OTHER"
    with pytest.raises(ValidationError,match="same episode"): OAGInput.model_validate(data)


def test_no_positive_facts_means_no_fabricated_bearing_fault():
    data=demo_input(ROOT).model_dump()
    for fact in data["bundle"]["evidence_facts"]:
        fact["status"]="MISSING_DATA"
        fact["feature_result_id"]="MISSING"
    result=run(OAGInput.model_validate(data))
    assert result.status == RuntimeStatus.SUCCEEDED
    assert result.output["summary"]["suspected_fault_id"] is None
    assert not any(c["eligible_for_explanation"] for c in result.output["summary"]["candidates"])


@pytest.mark.parametrize("flag",["require_llm","require_paperseek","require_ontology_extraction"])
def test_required_but_unconfigured_backend_fails(flag):
    data=demo_input(ROOT).model_dump();data[flag]=True
    assert run(OAGInput.model_validate(data)).status==RuntimeStatus.FAILED


def test_configured_backend_error_does_not_fallback():
    class BrokenClient:
        def search(self,query): raise ConnectionError("deliberate protocol failure")
    result=run(paperseek_client=BrokenClient())
    assert result.status==RuntimeStatus.FAILED
    assert "deliberate protocol failure" in str(result.error)


def test_paperseek_http_matches_uploaded_api_contract():
    with http_fixture(lambda path,data:(200,paper_reply())) as (url,calls):
        result=PaperSeekHTTPClient(PaperSeekConfig(base_url=url)).search("wheel evidence")
    assert calls[0][0]=="/api/search"
    assert calls[0][1]["question"]=="wheel evidence"
    assert calls[0][1]["fetch_abstracts"] is True
    assert len(result["documents"])==1
    doc=LiteratureDocument.model_validate(result["documents"][0])
    assert doc.role=="EXTERNAL_KNOWLEDGE_NOT_OBSERVATION"
    assert hashlib.sha256(doc.text.encode()).hexdigest()==doc.source_content_sha256


@pytest.mark.parametrize("reply", [{"ranked":[]}, {"ranked":[{"title":"no abstract","url":"https://example.org/a"}]}])
def test_no_papers_does_not_manufacture_a_reference(reply):
    with http_fixture(lambda path,data:(200,reply)) as (url,_):
        result=PaperSeekHTTPClient(PaperSeekConfig(base_url=url)).search("no evidence")
    assert result["documents"]==[]


def test_duplicate_papers_are_not_independent_context():
    reply=paper_reply();reply["ranked"]*=3
    with http_fixture(lambda path,data:(200,reply)) as (url,_):
        result=PaperSeekHTTPClient(PaperSeekConfig(base_url=url)).search("same paper")
    assert len(result["documents"])==1


@pytest.mark.parametrize("reply", [[],{"papers":[]},b"not-json"])
def test_malformed_paperseek_responses_raise(reply):
    with http_fixture(lambda path,data:(200,reply)) as (url,_):
        with pytest.raises((ValueError,json.JSONDecodeError)):
            PaperSeekHTTPClient(PaperSeekConfig(base_url=url)).search("query")


def test_paperseek_size_limit_and_http_error():
    with http_fixture(lambda path,data:(200,b" "*2000)) as (url,_):
        with pytest.raises(ValueError,match="size limit"):
            PaperSeekHTTPClient(PaperSeekConfig(base_url=url,max_response_bytes=1024)).search("query")
    with http_fixture(lambda path,data:(503,{})) as (url,_):
        with pytest.raises(Exception):
            PaperSeekHTTPClient(PaperSeekConfig(base_url=url)).search("query")


@pytest.mark.parametrize("url",["file:///secret","http://user:password@host/","http://host/?token=x"])
def test_invalid_service_config_rejected(url):
    with pytest.raises(ValidationError): PaperSeekConfig(base_url=url)


def test_full_oag_over_actual_http_clients_with_explicit_test_double():
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in GRAPH.glob("*.csv")}
    def respond(path,body):
        if path=="/api/search":return 200,paper_reply()
        assert path=="/v1/chat/completions"
        user=json.loads(body["messages"][-1]["content"])
        payload=user["payload"]
        assert "evaluation_metadata" not in json.dumps(payload)
        assert "scenario_for_evaluation_only" not in json.dumps(payload)
        if "summary" in payload:
            s=payload["summary"]
            answer={"candidate_fault_id":s["suspected_fault_id"],"abstained":False,
                    "claims":[{"kind":"OBSERVATION","text":"Observed evidence is available.",
                               "citation_ids":[s["evidence"][0]["evidence_id"]]}],
                    "limitations":["Protocol test only, semantic assertions are not evaluated."]}
        else:
            doc=payload["documents"][0]
            answer={"triples":[{"subject":"Bearing friction","predicate":"affects","object":"wheel speed",
                     "source_document_id":doc["document_id"],"source_quote":doc["text"]}]}
        return 200,{"choices":[{"message":{"content":json.dumps(answer)},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":1,"completion_tokens":1}}
    with http_fixture(respond) as (url,calls):
        backend=OpenAICompatibleStructuredBackend(llm_config(url))
        result=run(paperseek_client=PaperSeekHTTPClient(PaperSeekConfig(base_url=url)),
                   llm_backend=backend,ontology_backend=backend)
    assert result.status==RuntimeStatus.SUCCEEDED,str(result.error)
    assert len(calls)==3
    assert result.output["ontology"]["status"]=="REVIEW_REQUIRED"
    assert result.output["ontology"]["active_kg_modified"] is False
    assert result.output["generation"]["status"]=="SUCCEEDED"
    assert result.output["generation"]["semantic_hallucination_assessment"]=="NOT_EVALUATED"
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in GRAPH.glob("*.csv")}
    assert "local-test-key" not in json.dumps(result.model_dump(mode="json"))


@pytest.mark.parametrize("mode",["override_fault","unknown_citation","literature_as_observation"])
def test_invalid_model_output_does_not_override_evidence(mode):
    class BadBackend:
        def generate_structured(self,*,agent_id,output_model,payload):
            s=payload["summary"]
            if mode=="override_fault":
                return output_model(candidate_fault_id="UNKNOWN-FAULT",abstained=False)
            return output_model(candidate_fault_id=s["suspected_fault_id"],abstained=False,
                   claims=[{"kind":"OBSERVATION","text":"unsupported", "citation_ids":["DOC-FAKE"]}])
    assert run(llm_backend=BadBackend()).status==RuntimeStatus.FAILED


def test_cli_runs_and_writes_real_output(tmp_path):
    out=tmp_path/"demo.json"
    assert main(["demo","--project-root",str(ROOT),"--output",str(out)])==0
    assert json.loads(out.read_text(encoding="utf-8"))["output"]["summary"]["suspected_fault_id"]=="ADCS_RW_FRICTION_INCREASE"
    assert main(["demo","--project-root",str(ROOT),"--output",str(out),"--require-llm"])==2

@pytest.mark.parametrize("defect",["unknown_source","invented_quote","missing_entity","publish_active","duplicate_triple"])
def test_ontology_proposals_require_real_source_spans_and_cannot_publish(defect):
    class DocumentClient:
        def search(self,query):
            text="Bearing friction affects wheel speed."
            doc=LiteratureDocument(document_id="DOC-TEST",title="test",text=text,
                source_url="https://example.org/test",retrieved_at="2026-09-07T00:00:00Z",
                source_content_sha256=hashlib.sha256(text.encode()).hexdigest())
            return {"status":"SUCCEEDED","documents":[doc.model_dump(mode="json")]}
    class Extractor:
        def generate_structured(self,*,agent_id,output_model,payload):
            triple={"subject":"Bearing friction","predicate":"affects","object":"wheel speed",
                    "source_document_id":"DOC-TEST","source_quote":"Bearing friction affects wheel speed."}
            if defect=="unknown_source":triple["source_document_id"]="DOC-NOT-RETRIEVED"
            if defect=="invented_quote":triple["source_quote"]="Invented wheel speed proof."
            if defect=="missing_entity":triple["object"]="invented entity"
            obj={"triples":[triple]}
            if defect=="publish_active":obj["active_kg_modified"]=True
            if defect=="duplicate_triple":obj["triples"].append(dict(triple))
            return output_model(**obj)
    result=run(paperseek_client=DocumentClient(),ontology_backend=Extractor())
    assert result.status==RuntimeStatus.FAILED


def test_normal_fixture_does_not_create_a_fault():
    from astrograph.evidence import EpisodeEvidenceBundle
    from astrograph.fusion import FaultRanking, RankingSource
    data=json.loads((ROOT/"tests/fixtures/evidence/nominal/diagnosis/episode_evidence_bundle.json").read_text(encoding="utf-8"))
    data["evaluation_metadata"]={}
    bundle=EpisodeEvidenceBundle.model_validate(data)
    ranking=FaultRanking(ranking_id="EMPTY-UPSTREAM",episode_id=bundle.episode.episode_id,
        source=RankingSource.DATA_DRIVEN,producer_id="not-configured",producer_version="1",candidates=[])
    result=run(OAGInput(bundle=bundle,diagnostic_ranking=ranking))
    assert result.status==RuntimeStatus.SUCCEEDED
    assert result.output["summary"]["suspected_fault_id"] is None
