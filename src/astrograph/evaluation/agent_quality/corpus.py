"""Independent qrels evaluation of the deployed DocumentTools retrieval path."""
from __future__ import annotations
import math
from pathlib import Path
from astrograph.agents.mission.models import AgentSettings
from astrograph.agents.mission.storage import Workspace
from astrograph.agents.mission.documents import DocumentTools
from .io import read_json, write_new, file_sha, sha, safe_child
from .metrics import retrieval_novelty


def evaluate_corpus(manifest_path: Path, output: Path, *, settings_path: Path | None=None) -> dict:
    m=read_json(manifest_path);base=manifest_path.resolve().parent
    if m.get("schema_version")!="astrograph-retrieval-quality.v1":raise ValueError("BAD_RETRIEVAL_MANIFEST")
    if not 1<=m.get("top_k",0)<=20 or not m.get("corpus") or not m.get("queries"):
        raise ValueError("NONEMPTY_CORPUS_QUERIES_AND_BOUNDED_TOP_K_REQUIRED")
    if len({q["query_id"] for q in m["queries"]})!=len(m["queries"]):raise ValueError("DUPLICATE_QUERY_ID")
    docs=[];doc_by_sha={}
    for e in m["corpus"]:
        p=safe_child(base,e["path"])
        if file_sha(p)!=e["sha256"]:raise ValueError("CORPUS_HASH_MISMATCH")
        if p.suffix.lower() not in {".txt",".md",".markdown"}:raise ValueError("CONVERT_DOCUMENT_BEFORE_CORPUS_EVALUATION")
        docs.append(str(p));doc_by_sha[e["sha256"]]=p.read_text(encoding="utf-8-sig")
    for q in m["queries"]:
        if not isinstance(q.get("query"),str) or not q["query"].strip():raise ValueError("EMPTY_QUERY")
        seen=set()
        for judgment in q["relevant"]:
            key=(judgment["source_sha256"],judgment["quote"])
            if key in seen or not judgment["quote"].strip() or judgment.get("role") not in {"support","counter","context"}:
                raise ValueError("BAD_OR_DUPLICATE_RELEVANCE_JUDGMENT")
            seen.add(key)
            if judgment["quote"] not in doc_by_sha.get(judgment["source_sha256"],""):
                raise ValueError("JUDGMENT_QUOTE_NOT_IN_PINNED_CORPUS")
    if output.exists():raise FileExistsError("NEW_CORPUS_REPORT_DIRECTORY_REQUIRED")
    output.mkdir(parents=True)
    settings=AgentSettings.model_validate(read_json(settings_path)) if settings_path else AgentSettings()
    settings=settings.model_copy(update={"read_roots":list(dict.fromkeys([*settings.read_roots,str(base)])),
        "output_root":str(output),"memory_db":None,"memory_policy":None,"allow_mineru":False,"planner":"deterministic"})
    ws=Workspace(output,settings.read_roots,"corpus-work");ws.root.mkdir()
    tool=DocumentTools(ws,settings);tool.convert(docs);tool.clean(tool.documents)
    rows=[];calls=[]
    # Each query gets a separate relevance judgement; qrels never enter tool inputs.
    for q in m["queries"]:
        response=tool.retrieve(q["query"],top_k=m["top_k"])
        calls.append({**response,"step":q["query_id"]})
        hits=response.get("hits",[]);gold=q["relevant"];matched=set();ranks=[];grades=[]
        for rank,hit in enumerate(hits,1):
            keys=[i for i,g in enumerate(gold) if g["source_sha256"]==hit["source_sha256"] and g["quote"] in hit["text"]]
            if keys:ranks.append(rank)
            grades.append(1 if keys else 0);matched.update(keys)
        counters={i for i,g in enumerate(gold) if g["role"]=="counter"}
        relevant_chunks=sum(grades)
        # Recall counts gold passages; precision counts retrieved chunks. Do not
        # conflate a chunk containing several golden quotes with several hits.
        row={"query_id":q["query_id"],"query":q["query"],"status":response["status"],"hits":hits,
             "retrieval_mode":response["retrieval_mode"],"warnings":response.get("warnings",[]),
             "answerable":bool(gold),"gold_passages":len(gold),"retrieved_relevant_passages":len(matched),
             "passage_recall_at_k":len(matched)/len(gold) if gold else None,
             "chunk_precision_at_k":relevant_chunks/len(hits) if hits else (0.0 if gold else None),
             "reciprocal_rank":1/min(ranks) if ranks else (0.0 if gold else None),
             "counterevidence_recall":len(matched&counters)/len(counters) if counters else None,
             "no_answer_rejected":not hits if not gold else None,
             "answer_correctness":None,"diagnostic_improvement":None}
        rows.append(row)
    positive=[r for r in rows if r["answerable"]];negative=[r for r in rows if not r["answerable"]]
    def avg(values):return sum(values)/len(values) if values else None
    report={"schema_version":"astrograph-retrieval-quality-report.v1","scope":m["scope"],
            "manifest_sha256":file_sha(manifest_path),"rows":rows,"queries":len(rows),
            "mean_passage_recall_at_k":avg([r["passage_recall_at_k"] for r in positive]),
            "mean_reciprocal_rank":avg([r["reciprocal_rank"] for r in positive]),
            "no_answer_rejection_rate":avg([int(r["no_answer_rejected"]) for r in negative]),
            "novelty":retrieval_novelty(calls),"cache":tool.cache_stats,
            "execution_status":"COMPLETED","quality_verdict":"MEASURED_NOT_THRESHOLD_TUNED",
            "generation_requests_sent":0,"qrels_sent_to_retriever":False,"ragas_used":False,
            "judgment_limit":"Gold quotes are evaluator annotations, not independent mission ground truth. Nonmatching hits are relevance errors; no assertion about fact truth or diagnosis."}
    write_new(output/"report.json",report)
    return report
