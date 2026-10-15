"""Non-voting diagnosis/literature joint-context attachment."""
from __future__ import annotations

from typing import Any

from .storage import sha


def build_literature_context_assessment(core_result: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosis=core_result.get("diagnosis",{}) if isinstance(core_result,dict) else {}
    selected=diagnosis.get("selected_fault_id") or diagnosis.get("conclusion_fault_id")
    candidates=list(diagnosis.get("candidate_fault_ids") or [])
    if selected and selected not in candidates:
        candidates=[selected,*candidates]
    rows=[]
    for source in sources:
        citation=str(source.get("citation_id") or source.get("document_id") or "")
        if not citation:
            continue
        # Retrieval relevance is intentionally not promoted to factual support or contradiction.
        reviewed_polarity=source.get("reviewed_claim_polarity")
        reviewed_fault=source.get("reviewed_target_fault_id")
        reviewed=reviewed_polarity in {"SUPPORTS","CONTRADICTS"} and reviewed_fault in candidates
        rows.append({
            "citation_id":citation,
            "target_fault_id":reviewed_fault if reviewed else None,
            "stance":reviewed_polarity if reviewed else "UNASSESSED_CONTEXT",
            "reviewed_mapping":bool(reviewed),
            "diagnostic_vote_allowed":False,
            "source_sha256":source.get("source_sha256"),
            "source_locator":source.get("source_locator"),
        })
    return {
        "schema_version":"astrograph-data-literature-context.v1",
        "status":"REVIEWED_STANCE_CONTEXT_AVAILABLE" if any(x["reviewed_mapping"] for x in rows) else ("RETRIEVED_CONTEXT_ONLY" if rows else "NO_LITERATURE_CONTEXT"),
        "core_diagnosis_sha256":sha(core_result),
        "selected_fault_id":selected,
        "candidate_fault_ids":candidates,
        "assessments":rows,
        "diagnostic_vote_allowed":False,
        "core_diagnosis_modified":False,
        "claim_boundary":(
            "Retrieved literature is explanatory/contextual only. Relevance does not mean support. Support/contradiction "
            "labels require an explicit reviewed claim mapping and still do not modify this committed core diagnosis."
        ),
    }


__all__=["build_literature_context_assessment"]
