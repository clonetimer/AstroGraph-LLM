"""Deterministic, source-preserving context projection for Mission tasks.

No free-text recursive summarization of telemetry. Mandatory diagnosis, units,
negative evidence, IDs and check history remain whole. Only untrusted sources
are packed by complete passages; full inputs are content-addressed artifacts.
"""
from __future__ import annotations
import copy
import json
import re
from .storage import canonical, sha
from .context_projection import project_diagnostic_context
from .sources import normalize_sources, visible_ids


class ContextBudgetExceeded(ValueError):
    pass


class ContextBuilder:
    def __init__(self, *, max_tokens=16384, output_reserve=2400, artifact_writer=None):
        if max_tokens <= output_reserve + 1024:
            raise ValueError("context limit must leave room for schema and inputs")
        self.max_tokens=max_tokens; self.output_reserve=output_reserve
        self.artifact_writer=artifact_writer
        self.records=[]

    def build(self, model, payload, *, budget_measure=None):
        original=copy.deepcopy(payload)
        digest=sha(original)
        artifact=self.artifact_writer(digest,original) if self.artifact_writer else None
        projected=json.loads(canonical(payload))  # deterministic key order; arrays retain order
        source_keys=("sources_untrusted","retrieved_context_untrusted")
        pools={k:normalize_sources(projected.pop(k)) for k in source_keys if k in projected}
        # Explicitly non-semantic root diagnostic transport metadata may be externalized.
        externalized=[]
        for k in ("debug","transport_metadata","full_artifacts"):
            if k in projected:
                value=projected.pop(k)
                externalized.append({"field":k,"sha256":sha(value)})
        # RankedFault has first-class numeric scores and lineage IDs. Its optional
        # untyped metadata may contain huge debug strings; externalize only those
        # leaf strings, retaining every scalar, identifier and formal evidence field.
        diagnostic_context=projected.get("diagnostic_context") or {}
        ranking=diagnostic_context.get("data_ranking",{}) if isinstance(diagnostic_context,dict) else {}
        def metadata_projection(value,pointer):
            if isinstance(value,dict):return {k:metadata_projection(v,pointer+"/"+k) for k,v in value.items()}
            if isinstance(value,list):return [metadata_projection(v,pointer+"/"+str(i)) for i,v in enumerate(value)]
            if isinstance(value,str) and len(value.encode())>2048:
                record={"json_pointer":pointer,"sha256":sha(value),"bytes":len(value.encode())}
                externalized.append(record)
                return {"external_metadata":record,"full_payload_sha256":digest,"requires_lookup":True}
            return value
        if isinstance(ranking,dict):
            if "metadata" in ranking:
                ranking["metadata"]=metadata_projection(ranking["metadata"],"/diagnostic_context/data_ranking/metadata")
            for i,candidate in enumerate(ranking.get("candidates",[])):
                if isinstance(candidate,dict) and "metadata" in candidate:
                    candidate["metadata"]=metadata_projection(candidate["metadata"],f"/diagnostic_context/data_ranking/candidates/{i}/metadata")
        # Compact repeated keys and reference exact, persisted lineage lists.
        # This is a reversible LLM view; no data ranking or evidence is edited.
        for key in pools:
            projected[key] = []
        if pools:
            projected["allowed_citation_ids"] = []
        projected, projection = project_diagnostic_context(
            projected, full_payload=original, full_artifact=artifact)
        # Byte bound is deliberately conservative without an actual tokenizer.
        # Backend subsequently checks full chat template/schema with its tokenizer.
        byte_budget=self.max_tokens-self.output_reserve-1024
        byte_budget-=len(canonical(model.model_json_schema()))
        receipt={"schema_version":"astrograph-context-receipt.v1","full_payload_sha256":digest,
                 "full_payload_artifact":artifact,"original_bytes":len(canonical(original)),
                 "limit_tokens":self.max_tokens,"output_reserve":self.output_reserve,
                 "estimator":"CONSERVATIVE_UTF8_BYTE_BOUND_THEN_BACKEND_TOKEN_PREFLIGHT",
                 "protected_facts_rewritten":False,"externalized_nonsemantic_fields":externalized,
                 "included_passages":[],"omitted_sources":[]}
        measurements = {}
        def measure(value):
            digest_key = sha(value)
            if digest_key not in measurements:
                if budget_measure is not None:
                    result = budget_measure(value)
                else:
                    result = {"fits": len(canonical(value)) <= byte_budget,
                              "method": "CONSERVATIVE_UTF8_BYTES_PAYLOAD_PLUS_SCHEMA_RESERVE",
                              "mandatory_byte_budget": byte_budget}
                if not isinstance(result, dict) or not isinstance(result.get("fits"), bool):
                    raise ValueError("INVALID_CONTEXT_BUDGET_MEASUREMENT")
                measurements[digest_key] = result
            return measurements[digest_key]
        # Required host anchor is budgeted BEFORE optional literature. Opaque IDs
        # elsewhere in diagnosis are not citable source IDs.
        def refresh_ids():
            if pools:
                projected["allowed_citation_ids"] = visible_ids(projected)
        excerpt_number = 0
        required_seen = set()
        for key, sources in pools.items():
            for source in sources:
                if not source.get("required_for_grounding"):
                    continue
                text = source["text"]
                if not text:
                    raise ValueError("REQUIRED_SOURCE_EMPTY")
                excerpt_number += 1
                projected[key].append({"citation_id":source["citation_id"], "text":text,
                    "excerpt_id":"Q"+str(excerpt_number),
                    "source_role":source.get("source_role","EXTERNAL_CONTEXT")})
                required_seen.add((key, source["citation_id"]))
                receipt["included_passages"].append({"citation_id":source["citation_id"],
                    "excerpt_id":"Q"+str(excerpt_number), "start":0, "end":len(text),
                    "sha256":sha(text), "source_record_sha256":sha(source), "required_for_grounding":True})
        refresh_ids()
        receipt.update(projection=projection, schema_name=model.__name__,
                       budget_measurement=measure(projected),
                       budget_scope="COMPLETE_CHAT_MESSAGES_SCHEMA_AND_OUTPUT" if budget_measure else "PAYLOAD_SCHEMA_PLUS_RESERVE")
        receipt["estimator"] = receipt["budget_measurement"].get("prompt_token_measurement", {}).get("method", receipt["estimator"])
        if not receipt["budget_measurement"]["fits"]:
            receipt.update(status="MANDATORY_CONTEXT_TOO_LARGE",projected_bytes=len(canonical(projected)))
            self.records.append(receipt)
            raise ContextBudgetExceeded("MANDATORY_CONTEXT_TOO_LARGE: split task; no evidence was truncated")
        query=" ".join(re.findall(r"[\w.-]+",str(projected.get("goal", ""))+" "+str(projected.get("diagnosis",{}))))
        query_tokens=set(query.casefold().split())
        for key,sources in pools.items():
            ranked=[];seen={sha(x["text"]):True for x in projected[key]}
            for order,source in enumerate(sources):
                text=source["text"]
                if (key, source["citation_id"]) in required_seen:continue
                if not text:continue
                # Exact duplicate content has aliases, not independent additional support.
                content=sha(text)
                if content in seen:
                    receipt["omitted_sources"].append({"citation_id":source.get("citation_id"),"reason":"DUPLICATE_CONTENT","sha256":content})
                    continue
                seen[content]=True
                passages=list(re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*",text))
                if not passages:continue
                for match in passages:
                    passage=match.group(0)
                    terms=set(re.findall(r"[\w.-]+",passage.casefold()))
                    counter=bool(terms & {"not","without","counterevidence","contradicted","unknown","missing"}) or any(w in passage for w in ("反证","未决","不适用","缺失","不能"))
                    score=len(terms&query_tokens)+int(counter)
                    ranked.append((order, -score, match.start(), source, passage, content))
            # First passage from different documents before second passages: source diversity.
            ranked.sort(key=lambda x:(x[1],x[0],x[2]))
            bydoc={}
            for row in ranked:bydoc.setdefault(row[3].get("document_id",row[3].get("citation_id","")),[]).append(row)
            fair=[]
            while any(bydoc.values()):
                for rows in bydoc.values():
                    if rows:fair.append(rows.pop(0))
            for order,score,offset,source,passage,content in fair:
                # Provenance/offset hashes live in the full artifact and receipt;
                # the LLM needs the stable ID, literal passage and source role.
                item={"citation_id":source["citation_id"],"text":passage,
                      "excerpt_id":"Q"+str(excerpt_number+1),
                      "source_role":source.get("source_role","EXTERNAL_CONTEXT")}
                projected[key].append(item)
                refresh_ids()
                if not measure(projected)["fits"]:
                    projected[key].pop()
                    refresh_ids()
                    receipt["omitted_sources"].append({"citation_id":source.get("citation_id"),"start":offset,"end":offset+len(passage),"reason":"PASSAGE_BUDGET","sha256":content})
                else:
                    excerpt_number += 1
                    receipt["included_passages"].append({"excerpt_id":item["excerpt_id"],"citation_id":source.get("citation_id"),"start":offset,"end":offset+len(passage),"sha256":content,"source_record_sha256":sha(source)})
        receipt["budget_measurement"] = measure(projected)
        receipt["visible_citation_ids"] = visible_ids(projected)
        receipt["visible_source_sha256"] = sha({key:projected[key] for key in pools})
        receipt["projected_artifact"] = self.artifact_writer(sha(projected), projected) if self.artifact_writer else None
        receipt.update(status="PROJECTED" if projected!=original else "UNCHANGED",projected_bytes=len(canonical(projected)),projected_sha256=sha(projected))
        self.records.append(receipt)
        return projected


def extractive_narrative_valid(output, sources):
    """Safe default: an attachment may quote sources, but cannot invent assertions.

    Only COMPLETE visible passages qualify, preventing negation stripping or citation laundering.
    This is provenance validation, not proof that the quoted document is true.
    """
    lookup={}
    for source in sources:
        lookup.setdefault(source["citation_id"],[]).append(source.get("text",""))
    for paragraph in output.paragraphs:
        if not paragraph.citation_ids or any(c not in lookup for c in paragraph.citation_ids):
            return False
        if any(not any(paragraph.text == text for text in lookup[c]) for c in paragraph.citation_ids):
            return False
    return True
