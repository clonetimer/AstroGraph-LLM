"""Evaluate human judgements bound to immutable task results, never train on them here."""
from __future__ import annotations
from datetime import datetime
from .metrics import review_template

FIELDS={"RETRIEVAL":("useful","answers_question","adds_distinguishing_information","tool_corpus_fit"),
        "PROPOSAL":("testable","has_discriminating_test","requires_offline_or_human_approval","independently_supported"),
        "CHECK_CHOICE":("preferable_under_equal_budget",)}


def score_review(quality: dict, review: dict) -> dict:
    expected=review_template(quality)
    if review.get("schema_version")!=expected["schema_version"] or any(review.get(k)!=expected[k] for k in ("task_id","result_sha256")):
        raise ValueError("REVIEW_NOT_BOUND_TO_THIS_RESULT")
    if review.get("status") not in {"UNREVIEWED","PARTIAL","REVIEWED"}:raise ValueError("UNKNOWN_REVIEW_STATUS")
    if review["status"]!="UNREVIEWED":
        if not isinstance(review.get("reviewer"),str) or not review["reviewer"].strip():raise ValueError("REVIEWER_REQUIRED")
        at=datetime.fromisoformat(review.get("reviewed_at") or "")
        if at.utcoffset() is None:raise ValueError("TIMEZONE_AWARE_REVIEW_TIME_REQUIRED")
    actual={(x["kind"],x["item"]):x for x in review.get("items",[])}
    gold={(x["kind"],x["item"]):x for x in expected["items"]}
    if len(actual)!=len(review.get("items",[])) or set(actual)!=set(gold):raise ValueError("REVIEW_ITEMS_CHANGED")
    totals={}
    for key, template in gold.items():
        item=actual[key];fields=FIELDS[key[0]]
        mutable=set(fields)|{"evidence_note"}|({"source_role"} if key[0]=="PROPOSAL" else set())
        if set(item)!=set(template) or any(item[k]!=template[k] for k in template if k not in mutable):
            raise ValueError("REVIEW_SUBJECT_OR_EVIDENCE_CHANGED")
        vals=[item[f] for f in fields]
        if any(v is not None and type(v) is not bool for v in vals):raise ValueError("JUDGEMENT_MUST_BE_BOOLEAN_OR_NULL")
        if any(v is not None for v in vals):
            if review["status"]=="UNREVIEWED" or not str(item["evidence_note"]).strip():raise ValueError("JUDGEMENT_REQUIRES_ATTRIBUTION_AND_EVIDENCE_NOTE")
        if review["status"]=="REVIEWED" and any(v is None for v in vals):raise ValueError("COMPLETE_REVIEW_HAS_MISSING_JUDGEMENTS")
        if key[0]=="PROPOSAL" and item["source_role"] not in {None,"OBSERVATION_CONTEXT","MECHANISM_SUPPORT","COUNTEREVIDENCE","UNSUPPORTED","UNDETERMINED"}:
            raise ValueError("BAD_SOURCE_ROLE")
        for f,v in zip(fields,vals):
            name=key[0]+"."+f
            count=totals.setdefault(name,{"eligible":0,"reviewed":0,"positive":0})
            count["eligible"]+=1;count["reviewed"]+=int(v is not None);count["positive"]+=int(v is True)
    for count in totals.values():
        count["review_coverage"]=count["reviewed"]/count["eligible"] if count["eligible"] else None
        count["positive_rate_reviewed"]=count["positive"]/count["reviewed"] if count["reviewed"] else None
    return {"schema_version":"astrograph-quality-review-score.v1","task_id":quality["task_id"],
            "result_sha256":quality["result_sha256"],"review_status":review["status"],"counts":totals,
            "reviewer":review.get("reviewer"),"reviewer_identity_independently_verified":False,
            "active_knowledge_write":False,"confirmed_memory_write":False,"diagnostic_accuracy":None,
            "interpretation":"Reviewer judgements with coverage, not automatically verified truth; downstream approval remains separate."}
