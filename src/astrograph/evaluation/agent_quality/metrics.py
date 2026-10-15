"""Observe completed mission traces without turning contract success into truth."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import math
from pathlib import Path
from typing import Any
from astrograph.llm.trace_verification import verify_task
from .io import read_json, verify_mission, file_sha, sha

ACCEPTED = {"ACCEPTED", "STRUCTURE_AND_CITATIONS_VALIDATED", "QUARANTINED_PROPOSAL"}


def retrieval_novelty(calls: list[dict]) -> dict:
    """Two metrics: new citation+content pairs AND truly new text content.

    Empty/failed calls stay in the denominator. Changed content under the same
    citation is reported, never erased. Text novelty is NOT information gain.
    """
    content_seen: set[str] = set()
    pair_seen: set[tuple] = set()
    citation_versions: dict[str, set[str]] = defaultdict(set)
    rows = []
    for call in calls:
        hits = call.get("hits", [])
        contents, pairs, conflicts = set(), set(), set()
        for hit in hits:
            if not isinstance(hit.get("text"), str) or not isinstance(hit.get("citation_id"), str):
                raise ValueError("INVALID_RETRIEVAL_HIT_CONTRACT")
            digest = hashlib.sha256(hit["text"].encode("utf-8")).hexdigest()
            if hit.get("content_sha256") and hit["content_sha256"] != digest:
                raise ValueError("RETRIEVAL_CONTENT_HASH_MISMATCH")
            cid = hit["citation_id"]
            if citation_versions[cid] and digest not in citation_versions[cid]:
                conflicts.add(cid)
            citation_versions[cid].add(digest)
            contents.add(digest)
            pairs.add((cid, digest))
        new_content, new_pairs = contents - content_seen, pairs - pair_seen
        rows.append({"step": call.get("step"), "tool_id": call.get("tool_id", "rag.retrieve"),
                     "query": call.get("query"), "status": call.get("status", "UNKNOWN"),
                     "retrieval_mode": call.get("retrieval_mode"), "returned_hits": len(hits),
                     "unique_texts": len(contents), "new_texts": len(new_content),
                     "new_citation_content_pairs": len(new_pairs),
                     "returned_citation_ids": [h["citation_id"] for h in hits],
                     "returned_content_sha256": [hashlib.sha256(h["text"].encode()).hexdigest() for h in hits],
                     "citation_content_changes": sorted(conflicts),
                     "quality_status": "NOT_JUDGED", "answers_investigation_question": None})
        content_seen.update(contents)
        pair_seen.update(pairs)
    return {"calls": rows, "executed_calls": len(rows),
            "zero_new_text_calls": sum(r["new_texts"] == 0 for r in rows),
            "nonempty_but_no_new_text_calls": sum(r["returned_hits"] > 0 and r["new_texts"] == 0 for r in rows),
            "unique_texts_total": len(content_seen), "citation_content_pairs_total": len(pair_seen),
            "semantic_information_gain": None,
            "interpretation": "CONTENT_NOVELTY_NOT_ANSWER_RELEVANCE_OR_NEW_DIAGNOSTIC_EVIDENCE"}


def inspect_task(task_dir: Path) -> dict:
    root = Path(task_dir).resolve()
    checked = verify_mission(root)
    result, events = checked["result"], checked["events"]
    records = result.get("plan", {}).get("planner_records", [])
    core = result.get("output", {}).get("diagnosis_result", {})
    initial, final = core.get("initial_diagnosis", {}), core.get("diagnosis", {})
    calls = []
    for event in events:
        if event["event"] != "TOOL_STARTED" or event.get("tool_id") not in {"rag.retrieve", "literature.search"}:
            continue
        path = root / f"step-{event['step']:03d}.json"
        if path.exists():
            step = read_json(path)
            payload = step["result"]
            hits = payload.get("hits", [])
            if event["tool_id"] == "literature.search":
                hits = [{"citation_id": d["document_id"], "text": d["text"]}
                        for d in payload.get("documents", [])]
            calls.append({**payload, "hits": hits, "step": event["step"], "tool_id": event["tool_id"]})
        else:
            # A failed invocation has no step file; it must not disappear.
            calls.append({"step": event["step"], "tool_id": event["tool_id"], "status": "FAILED_OR_INTERRUPTED", "hits": []})
    trace = core.get("planning_trace", [])
    choices = []
    for r in records:
        if r.get("phase") != "CHECK_SELECTION":
            continue
        contract = r.get("tool_contract") or {}
        options = contract.get("selectable_tool_ids", [])
        choices.append({"outcome": r["outcome"], "selected_tool_id": r.get("tool_id"),
                        "selectable_tool_ids": options, "n_choices": len(options),
                        "choice_opportunity": "MULTIPLE" if len(options) > 1 else ("SINGLE" if options else "UNKNOWN_OR_NONE"),
                        "tool_contract_sha256": r.get("tool_contract_sha256"),
                        "best_choice_independently_judged": None})
    wire = verify_task(root)
    full = bool(wire["rows"]) and all(
        all(row.get(k, {}).get("status") == "VERIFIED" for k in ("request", "response"))
        for row in wire["rows"])
    all_attempts = result.get("execution_budget", {}).get("http_attempts", 0)
    if wire["rows"] and len(wire["rows"]) != all_attempts:
        raise ValueError("OBSERVED_AND_REPORTED_HTTP_ATTEMPTS_DISAGREE")
    elapsed = (datetime.fromisoformat(events[-1]["at"]) - datetime.fromisoformat(events[0]["at"])).total_seconds()
    llm = result.get("capability_status", {}).get("llm", {})
    proposal = result.get("output", {}).get("exploration") or {}
    hypotheses = proposal.get("hypotheses", [])
    protection_keys = ("diagnostic_vote_allowed", "automatic_execution")
    isolated = (all(proposal.get(k) is False for k in protection_keys)
                and proposal.get("memory_write") == "NONE") if proposal else None
    return {
        "schema_version": "astrograph-agent-quality-trace.v1",
        "scope": "OFFLINE_OBSERVATION_NOT_MODEL_QUALITY_CERTIFICATION", "task_id": result["task_id"],
        "work_mode": result.get("work_mode"), "mission_status": result["status"],
        "result_sha256": checked["result_sha256"], "artifact_integrity": "VERIFIED",
        "request_sha256": result["request_sha256"], "settings_sha256": result["settings_sha256"],
        "core_status": result.get("capability_status", {}).get("core_diagnosis"),
        "llm_status": llm.get("status"), "logical_llm_calls": result.get("plan", {}).get("llm_calls", 0),
        "http_attempts": all_attempts, "accepted": sum(r["outcome"] in ACCEPTED for r in records),
        "fallbacks": sum(r["outcome"] == "DETERMINISTIC_FALLBACK" for r in records),
        "abstentions": sum(r["outcome"] == "ABSTAINED" for r in records),
        "wire_status": wire["status"], "full_wire_pairs": full,
        "backend_authenticity": "NOT_INDEPENDENTLY_VERIFIED_BY_HASHES", "elapsed_s": elapsed,
        "retrieval": retrieval_novelty(calls),
        "checks": {"executed": core.get("checks_executed", len(trace)), "choices": choices,
                   "multiple_option_opportunities": sum(c["n_choices"] > 1 for c in choices),
                   "outcomes": dict(Counter(x.get("outcome", "UNKNOWN") for x in trace)),
                   "admitted_feature_result_count": len({f for x in trace for f in x.get("admitted_evidence_ids", [])}),
                   "state_changed_checks": sum(x.get("after_decision", x.get("before_decision")) != x.get("before_decision") for x in trace),
                   "selected_changed_checks": sum(x.get("after_selected", x.get("before_selected")) != x.get("before_selected") for x in trace),
                   "planning_trace": trace, "best_check_accuracy": None},
        "diagnosis": {"initial_status": initial.get("status"), "status": final.get("status"),
                      "initial_selected": initial.get("selected_fault_id"), "selected": final.get("selected_fault_id"),
                      "selected_changed": initial.get("selected_fault_id") != final.get("selected_fault_id"),
                      "candidate_fault_ids": final.get("candidate_fault_ids", []),
                      "root_cause_confirmed": final.get("root_cause_confirmed"),
                      "correct": None, "truth_status": "NOT_PROVIDED_TO_THIS_TRACE_AUDIT",
                      "stopping_reason": core.get("stopping_reason"),
                      "knowledge_version": core.get("knowledge_version")},
        "exploration": {"proposals": len(hypotheses), "isolation_maintained": isolated,
                        "hypotheses": hypotheses, "mechanism_truth": None, "review_status": "UNREVIEWED"},
        "contract_success_is_usefulness": False, "generation_requests_sent": 0,
    }


def review_template(report: dict) -> dict:
    rows = []
    for r in report["retrieval"]["calls"]:
        rows.append({"kind": "RETRIEVAL", "item": str(r["step"]), "query": r["query"],
                     "source_content_sha256": r["returned_content_sha256"], "useful": None,
                     "answers_question": None, "adds_distinguishing_information": None,
                     "tool_corpus_fit": None, "evidence_note": ""})
    for i, h in enumerate(report["exploration"]["hypotheses"]):
        rows.append({"kind": "PROPOSAL", "item": str(i), "hypothesis": h["hypothesis"],
                     "source_role": None, "testable": None, "has_discriminating_test": None,
                     "requires_offline_or_human_approval": None, "independently_supported": None,
                     "evidence_note": ""})
    for i, c in enumerate(report["checks"]["choices"]):
        rows.append({"kind": "CHECK_CHOICE", "item": str(i), "selectable_tool_ids": c["selectable_tool_ids"],
                     "selected_tool_id": c["selected_tool_id"], "preferable_under_equal_budget": None,
                     "evidence_note": ""})
    return {"schema_version": "astrograph-quality-human-review.v1", "result_sha256": report["result_sha256"],
            "task_id": report["task_id"], "reviewer": None, "reviewed_at": None,
            "status": "UNREVIEWED", "items": rows,
            "rules": "Leave unknowns null; cite observations or adjudication evidence. This file is evaluation-only, never a planner input or confirmed-memory write."}


def aggregate(cells: list[dict]) -> dict:
    """All scheduled trials remain visible; no best-seed or success-only selection."""
    by_arm = defaultdict(list)
    for cell in cells:
        by_arm[(cell["case_id"], cell["mode"], cell["planner"])].append(cell)
    summaries = []
    for (case, mode, planner), rows in sorted(by_arm.items()):
        started = [r for r in rows if r["status"] != "NOT_RUN"]
        completed = [r for r in rows if r.get("quality") is not None]
        successful = [r for r in completed if r["quality"]["mission_status"] == "SUCCEEDED"]
        llm_ok = [r for r in completed if r["quality"]["llm_status"] == "SUCCEEDED" and r["quality"]["http_attempts"] > 0
                  and r["quality"]["fallbacks"] == 0 and r["quality"]["abstentions"] == 0 and r["quality"]["full_wire_pairs"]]
        outcomes = Counter((r["quality"]["diagnosis"]["status"], r["quality"]["diagnosis"]["selected"]) for r in completed)
        latencies = sorted(r["elapsed_s"] for r in started if r.get("elapsed_s") is not None)
        def quantile(p):
            if not latencies: return None
            pos = (len(latencies)-1)*p; lo = math.floor(pos); hi = math.ceil(pos)
            return latencies[lo] + (latencies[hi]-latencies[lo])*(pos-lo)
        summaries.append({"case_id": case, "mode": mode, "planner": planner,
                          "scheduled": len(rows), "started": len(started), "not_run": len(rows)-len(started),
                          "auditable_completed": len(completed), "incomplete_or_runner_errors": len(started)-len(completed),
                          "mission_succeeded": len(successful), "llm_contract_succeeded": len(llm_ok) if planner == "llm_assisted" else None,
                          "mission_success_rate_scheduled": len(successful)/len(rows),
                          "llm_success_rate_scheduled": len(llm_ok)/len(rows) if planner == "llm_assisted" else None,
                          "mission_success_rate_started": len(successful)/len(started) if started else None,
                          "elapsed_s": {"p50": quantile(.5), "p95": quantile(.95), "max": max(latencies) if latencies else None},
                          "diagnosis_distribution": [{"status": k[0], "selected": k[1], "n": v} for k,v in sorted(outcomes.items(),key=str)],
                          "retrieval_calls": sum(r["quality"]["retrieval"]["executed_calls"] for r in completed),
                          "retrieval_zero_new_text": sum(r["quality"]["retrieval"]["zero_new_text_calls"] for r in completed),
                          "repeated_case_frequency_is_generalization": False})
    pairs = defaultdict(dict)
    for r in cells:
        pairs[(r["case_id"], r["mode"], r["repeat"])][r["planner"]] = r
    comparisons = []
    for key, arms in sorted(pairs.items()):
        a, b = arms.get("deterministic", {}), arms.get("llm_assisted", {})
        if not a.get("quality") or not b.get("quality"):
            comparisons.append({"case_id":key[0], "mode":key[1], "repeat":key[2], "status":"PAIR_INCOMPLETE"});continue
        da, db = a["quality"], b["quality"]
        comparisons.append({"case_id":key[0], "mode":key[1], "repeat":key[2], "status":"PAIRED_DESCRIPTIVE_COMPARISON",
                            "diagnosis_equal": all(da["diagnosis"].get(k)==db["diagnosis"].get(k) for k in ("status","selected","candidate_fault_ids")),
                            "llm_minus_deterministic_checks": db["checks"]["executed"]-da["checks"]["executed"],
                            "llm_minus_deterministic_retrieval_calls": db["retrieval"]["executed_calls"]-da["retrieval"]["executed_calls"],
                            "corrected_errors": None, "induced_errors": None})
    return {"schema_version":"astrograph-quality-experiment-report.v1", "arms":summaries, "pairs":comparisons,
            "quality_verdict":"NOT_AUTOMATICALLY_JUDGED", "evaluation_labels_sent_to_agent":False,
            "statistical_note":"Descriptive repeated-case frequencies only; no independence assumption, significance claim or cross-case accuracy. Unrun and failed cells stay in denominators."}
