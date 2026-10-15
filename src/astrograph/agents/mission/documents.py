"""Conservative conversion, cleaning, source-bound chunks and honest RAG lanes."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import os
import sys
from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.integrations.platform_knowledge import _tokens, _bm25
from astrograph.oag.contracts import OntologyProposal
from .storage import file_sha, sha


def clean_text(text: str) -> dict:
    # No number, sign, unit, formula, negation, paragraph or duplicate deletion.
    lines=text.lstrip("\ufeff").replace("\r\n","\n").replace("\r","\n").split("\n")
    cleaned=[];changes=[]
    for i,line in enumerate(lines,1):
        value="".join(c for c in line if c=="\t" or ord(c)>=32).rstrip(" \t")
        if value!=line:changes.append({"line":i,"operation":"REMOVE_CONTROL_OR_TRAILING_SPACE"})
        cleaned.append(value)
    result="\n".join(cleaned)
    return {"text":result,"raw_text_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "clean_text_sha256":hashlib.sha256(result.encode()).hexdigest(),
            "changes":changes,"line_mapping":"ONE_TO_ONE_AFTER_EOL_NORMALIZATION",
            "semantic_rewrite_performed":False}


class SourceQuotationError(ValueError):
    pass


class DocumentTools:
    def __init__(self, workspace, settings, *, planner=None):
        self.ws=workspace;self.settings=settings;self.planner=planner;self.documents=[]
        self._dense=None;self._reranker=None
        self._index_key=None;self._rows=[];self._tokenized=[];self._corpus_embeddings=None
        self._model_stat_pins={}
        self._retrieval_cache={};self.cache_stats={"index_builds":0,"corpus_encodes":0,"query_encodes":0,"query_hits":0}

    def convert(self,paths):
        docs=[]
        if sum(self.ws.read_path(p).stat().st_size for p in paths)>self.ws.max_bytes:
            raise ValueError("total document inputs exceed host budget")
        for index,value in enumerate(paths):
            path=self.ws.read_path(value)
            digest=file_sha(path)
            warnings=[]
            if path.suffix.lower() in {".txt",".md",".markdown"}:
                text=path.read_text(encoding="utf-8-sig")
                parser="UTF8_NATIVE_TEXT";locators=["lines:1-"+str(len(text.splitlines()))]
            else:
                if not self.settings.allow_mineru or not self.settings.mineru_executable:
                    raise RuntimeError("MINERU_NOT_CONFIGURED: non-text document was not silently skipped")
                exe=self.settings.mineru_executable
                resolved=shutil.which(exe) if not Path(exe).is_absolute() else exe
                if not resolved or not Path(resolved).is_file():raise RuntimeError("MINERU_EXECUTABLE_UNAVAILABLE")
                output=self.ws.root/("mineru-"+str(index))
                output.mkdir(exist_ok=False)
                command=[str(resolved)]
                if os.name == "nt":
                    # Windows does not honor POSIX shebangs for extensionless
                    # command fixtures.  Accept only a script explicitly bound
                    # to this running Python interpreter; arbitrary shebang
                    # interpreters remain rejected.
                    first_line=Path(resolved).read_bytes().splitlines()[:1]
                    expected=("#!"+str(Path(sys.executable).resolve())).casefold().encode()
                    if first_line and first_line[0].strip().lower()==expected:
                        command=[sys.executable,str(resolved)]
                completed=run_managed_subprocess([*command,"-p",str(path),"-o",str(output)],
                    capture_output=True,text=True,timeout=self.settings.subprocess_timeout_s,check=False)
                if completed.returncode:raise RuntimeError("MINERU_FAILED:"+(completed.stderr or "")[-500:])
                markdown=sorted(output.rglob("*.md"))
                if not markdown:raise ValueError("MinerU produced no Markdown artifact")
                if any(p.is_symlink() or not p.resolve().is_relative_to(output) for p in markdown):
                    raise ValueError("untrusted parser artifact path")
                if sum(p.stat().st_size for p in markdown)>self.ws.max_bytes:raise ValueError("parser output too large")
                text="\n\n".join(p.read_text(encoding="utf-8") for p in markdown)
                parser="MINERU_CLI";locators=[str(p.relative_to(output)) for p in markdown]
                warnings.append("Derived layout/OCR/image descriptions are unreviewed; original artifacts retained.")
                # Retain page/bbox content lists as artifacts, not as guessed text.
                warnings.append("Page/figure locators require the retained MinerU content-list; Markdown line IDs alone are not PDF page numbers.")
            docs.append({"document_id":"DOC-"+digest[:20],"source_uri":path.as_uri(),"source_sha256":digest,
                         "text":text,"parser":parser,"source_locators":locators,"warnings":warnings,
                         "review_status":"UNREVIEWED","diagnostic_vote_allowed":False})
        self.documents=docs
        return {"status":"SUCCEEDED","documents":docs}

    def clean(self,documents):
        output=[]
        for document in documents:
            record=clean_text(document["text"])
            output.append({**document,**record})
        self.documents=output
        return {"status":"SUCCEEDED","documents":output,"cleaning":"CONSERVATIVE_PROVENANCE_PRESERVING"}

    def _chunks(self):
        # Reuse section-first parser, but DO NOT equate native representation with human factual review.
        from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor
        rows=[];seen=set()
        for doc in self.documents:
            elements,warnings=SectionFirstDocumentIngestor().parse_markdown(doc["text"],asset_root=None)
            for element in elements:
                text=element.text_representation
                digest=hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:continue
                seen.add(digest)
                rows.append({"citation_id":doc["document_id"]+":"+element.element_id,
                    "document_id":doc["document_id"],"text":text,"source_sha256":doc["source_sha256"],
                    "source_uri":doc["source_uri"],"source_locator":element.source_locator,
                    "section_path":element.section_path,"modality":element.kind.value,
                    "content_sha256":digest,"review_status":"UNREVIEWED_CONTEXT",
                    "diagnostic_vote_allowed":False})
        return rows

    def retrieve(self,query,top_k=5):
        if not query.strip() or len(query)>4000:raise ValueError("bounded nonempty query required")
        for model_path in (self.settings.embedding_model_path,self.settings.reranker_model_path):
            if not model_path or not Path(model_path).is_dir():continue
            root=self.ws.read_path(model_path,directory=True)
            entries=[]
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".json",".bin",".pt",".safetensors",".model",".txt"}:
                    if not f.resolve().is_relative_to(root):raise ValueError("model symlink escapes approved directory")
                    st=f.stat();entries.append((str(f.relative_to(root)),st.st_size,st.st_mtime_ns,st.st_ino))
            fingerprint=sha(entries)
            if model_path in self._model_stat_pins and self._model_stat_pins[model_path]!=fingerprint:
                raise ValueError("RETRIEVAL_MODEL_CHANGED_DURING_TASK")
            self._model_stat_pins[model_path]=fingerprint
        index_key=sha({"documents":self.documents,"chunker":"section-first.v1",
                       "embedding":self.settings.embedding_model_path,"reranker":self.settings.reranker_model_path})
        if index_key!=self._index_key:
            self._rows=self._chunks();self._tokenized=[_tokens(x["text"]) for x in self._rows]
            self._corpus_embeddings=None;self._retrieval_cache={};self._index_key=index_key
            self.cache_stats["index_builds"]+=1
        rows=self._rows
        cache_key=(query.strip().casefold(),top_k)
        if cache_key in self._retrieval_cache:
            from copy import deepcopy
            self.cache_stats["query_hits"]+=1
            return deepcopy(self._retrieval_cache[cache_key])
        if not rows:return {"status":"NO_MATCH","hits":[],"retrieval_mode":"BM25","diagnostic_vote_allowed":False}
        lexical=_bm25(_tokens(query),self._tokenized)
        ranked=sorted(range(len(rows)),key=lambda i:(-lexical[i],rows[i]["citation_id"]))
        scores={i:1/(60+k) for k,i in enumerate((i for i in ranked if lexical[i]>0),1)}
        mode="BM25";warnings=[]
        if self.settings.embedding_model_path:
            try:
                if self._dense is None:
                    from sentence_transformers import SentenceTransformer
                    self._dense=SentenceTransformer(str(self.ws.read_path(self.settings.embedding_model_path,directory=True)),local_files_only=True)
                import numpy as np
                if self._corpus_embeddings is None:
                    self._corpus_embeddings=self._dense.encode([x["text"] for x in rows],normalize_embeddings=True)
                    self.cache_stats["corpus_encodes"]+=1
                query_vector=self._dense.encode([query],normalize_embeddings=True)[0]
                self.cache_stats["query_encodes"]+=1
                similarities=self._corpus_embeddings@query_vector
                if not np.isfinite(similarities).all():raise ValueError("nonfinite dense similarities")
                for k,i in enumerate(np.argsort(-similarities),1):scores[int(i)]=scores.get(int(i),0)+1/(60+k)
                mode="BM25_DENSE_RRF"
            except Exception as exc:
                warnings.append("DENSE_UNAVAILABLE:"+type(exc).__name__)
        selected=sorted(scores,key=lambda i:(-scores[i],rows[i]["citation_id"]))[:max(top_k,20)]
        if self.settings.reranker_model_path and selected:
            try:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder
                    self._reranker=CrossEncoder(str(self.ws.read_path(self.settings.reranker_model_path,directory=True)),local_files_only=True)
                values=self._reranker.predict([(query,rows[i]["text"]) for i in selected])
                if len(values)!=len(selected) or not all(math.isfinite(float(v)) for v in values):raise ValueError("bad reranker scores")
                rerank={i:float(v) for i,v in zip(selected,values,strict=True)}
                selected.sort(key=lambda i:(-rerank[i],rows[i]["citation_id"]));mode+="_CROSS_ENCODER"
            except Exception as exc:warnings.append("RERANKER_UNAVAILABLE:"+type(exc).__name__)
        hits=[{**rows[i],"retrieval_score":scores[i],"score_semantics":"RELEVANCE_NOT_FAULT_PROBABILITY"} for i in selected[:top_k]]
        result={"status":"SUCCEEDED" if hits else "NO_MATCH","query":query,"hits":hits,
                "retrieval_mode":mode,"warnings":warnings,"diagnostic_vote_allowed":False,
                "index_sha256":index_key}
        if not warnings:
            from copy import deepcopy
            self._retrieval_cache[cache_key]=deepcopy(result)
        return result

    def build_graph(self,*,a2o=None):
        if a2o is not None:
            # Fresh task-owned directory; original adapter cannot remove another task's workspace.
            inputs=self.ws.root/"ontology-inputs";inputs.mkdir(exist_ok=False)
            paths=[]
            for i,doc in enumerate(self.documents):
                path=inputs/(str(i)+".md");path.write_text(doc["text"],encoding="utf-8");paths.append(path)
            result=a2o.run(paths,self.ws.root)
            if result.get("active_kg_modified") is not False:raise ValueError("A2O changed active graph")
            return {**result,"input_source_bindings":[{"document_id":d["document_id"],"source_sha256":d["source_sha256"]} for d in self.documents]}
        if self.planner is None or self.planner.backend is None:
            raise RuntimeError("ONTOLOGY_EXTRACTOR_NOT_CONFIGURED: configure A2O or a structured LLM")
        # Automatic bounded batches. Every byte range is accounted for; failures are
        # not silently called a complete extraction. Exact source quotes still gate admission.
        from .storage import canonical
        batch_size=min(self.settings.ontology_batch_chars,
                       max(512,(self.settings.context_token_limit-4000)//2))
        batches=[]
        for doc in self.documents:
            text=doc["text"]
            for start in range(0,len(text),batch_size):
                end=min(len(text),start+batch_size)
                # Context overlap retains relations crossing a batch boundary.
                lo=max(0,start-256)
                batches.append((doc,lo,end,text[lo:end]))
        triples=[];seen=set();coverage=[];limitations=[]
        for i,(doc,start,end,text) in enumerate(batches):
            entry={"batch":i,"document_id":doc["document_id"],"start":start,"end":end,
                   "source_sha256":doc["source_sha256"]}
            if i>=self.settings.ontology_max_batches:
                coverage.append({**entry,"status":"BATCH_BUDGET_EXHAUSTED"});continue
            try:
                proposal=self.planner._generate(OntologyProposal,{
                    "task":"extract source-quoted spacecraft mechanism relations",
                    "documents_untrusted":[{"document_id":doc["document_id"],"text":text,"source_sha256":doc["source_sha256"]}],
                    "instruction_extension":"Quote exact text, preserve conditions and units. No inferred telemetry observations."})
                for triple in proposal.triples:
                    if triple.source_document_id!=doc["document_id"] or triple.source_quote not in text:
                        raise SourceQuotationError("ontology triple lacks exact source quotation in its own batch")
                for triple in proposal.triples:
                    key=sha(triple.model_dump(mode="json"))
                    if key not in seen:triples.append(triple);seen.add(key)
                coverage.append({**entry,"status":"EXTRACTED","triples":len(proposal.triples)})
            except SourceQuotationError:
                raise  # ungrounded content rejects the candidate, not a partial success
            except Exception as exc:
                coverage.append({**entry,"status":"UNAVAILABLE","error_type":type(exc).__name__})
                limitations.append("Batch "+str(i)+": "+type(exc).__name__)
        complete=all(x["status"]=="EXTRACTED" for x in coverage)
        # Batch-local proposals are validated individually; merged output may exceed
        # the single-generation schema's cardinality without dropping any triples.
        nodes=sorted({x.subject for x in triples}|{x.object for x in triples})
        return {"status":("REVIEW_REQUIRED" if complete else "PARTIAL_REVIEW_REQUIRED") if triples else "NO_OUTPUT",
                "backend":"STRUCTURED_LLM_EXTRACTOR_NOT_A2O",
                "proposal":{"triples":[x.model_dump(mode="json") for x in triples]},
                "nodes":nodes,"edges":[x.model_dump(mode="json") for x in triples],
                "source_bindings":[{"document_id":d["document_id"],"source_sha256":d["source_sha256"]} for d in self.documents],
                "batch_coverage":coverage,"extraction_complete":complete,"limitations":limitations,
                "mechanism_applicability_review_required":True,"active_kg_modified":False}
