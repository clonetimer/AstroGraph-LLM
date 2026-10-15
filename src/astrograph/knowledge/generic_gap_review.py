"""Build expert-review packets for generic-series knowledge coverage gaps.

The helper never creates active knowledge edges. Exact CONTEXT-only catalog
occurrences are surfaced as role-review candidates; non-exact channel names are
shown only as review hints and can never be auto-admitted.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal
import json

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)


class CatalogHint(_Strict):
    fault_id: str
    channel: str
    role: str
    expected_effect: str | None=None
    similarity: float=Field(ge=0.0, le=1.0)
    automatic_relation_allowed: Literal[False]=False


class GapReviewItem(_Strict):
    channel: str
    anomaly_id: str
    unit: str | None
    gap_kind: Literal["EXACT_CONTEXT_ROLE_REVIEW","NO_SAFE_RELATION_CANDIDATE"]
    exact_context_candidates: tuple[dict[str,Any],...]=()
    related_catalog_hints: tuple[CatalogHint,...]=()
    automatic_aliasing_allowed: Literal[False]=False
    active_kg_write_allowed: Literal[False]=False
    expert_question: str


class GapReviewPacket(_Strict):
    schema_version: Literal["astrograph-generic-knowledge-gap-review.v1"]="astrograph-generic-knowledge-gap-review.v1"
    knowledge_version: str
    source_fault_catalog: str
    source_anomaly_rules: str
    items: tuple[GapReviewItem,...]
    active_kg_modified: Literal[False]=False
    automatic_promotion_allowed: Literal[False]=False
    claim_boundary: str="Review packet only. No relation is admitted until expert review and offline graph regression complete."


def _similarity(a: str,b: str) -> float:
    # Similarity is a navigation hint only; it never admits a relation.
    return float(SequenceMatcher(None,a.lower(),b.lower()).ratio())


def build_gap_review_packet(project_root: str|Path, gaps: list[dict[str,Any]], *, knowledge_version: str="astrograph-generic-series-e0.2") -> GapReviewPacket:
    root=Path(project_root).resolve()
    rules=json.loads((root/"configs/anomaly_rules.json").read_text(encoding="utf-8"))["rules"]
    faults=json.loads((root/"configs/fault_modes/lab_fault_catalog.json").read_text(encoding="utf-8"))["fault_modes"]
    rule_by_channel={str(x["channel_id"]):x for x in rules if x.get("enabled",True)}
    all_catalog=[]
    for fault in faults:
        for entry in fault.get("telemetry_channels",[]):
            all_catalog.append((str(fault["fault_id"]),entry))
    items=[]
    for gap in gaps:
        channel=str(gap["channel"]);anomaly_id=str(gap["anomaly_id"]);rule=rule_by_channel[channel]
        exact=[]
        for fault_id,entry in all_catalog:
            if str(entry.get("channel"))==channel and str(entry.get("role"))=="CONTEXT":
                exact.append({"fault_id":fault_id,"channel":channel,"role":"CONTEXT","expected_effect":entry.get("expected_effect"),
                              "source_ref":f"configs/fault_modes/lab_fault_catalog.json#{fault_id}:{channel}:CONTEXT",
                              "proposed_relation":"UNSPECIFIED_PENDING_EXPERT_ROLE_REVIEW"})
        hints=[]
        if not exact:
            for fault_id,entry in all_catalog:
                other=str(entry.get("channel",""));score=_similarity(channel,other)
                if score>=0.62:
                    hints.append(CatalogHint(fault_id=fault_id,channel=other,role=str(entry.get("role","")),
                                             expected_effect=entry.get("expected_effect"),similarity=round(score,6)))
            hints=sorted(hints,key=lambda x:(-x.similarity,x.fault_id,x.channel))[:8]
        kind="EXACT_CONTEXT_ROLE_REVIEW" if exact else "NO_SAFE_RELATION_CANDIDATE"
        question=(
            "Does this exact CONTEXT telemetry have an independently justified SUPPORTING/PRIMARY relationship to any listed fault under explicit operating conditions?"
            if exact else
            "No exact causal/supporting catalog relation exists. Should an expert define a semantic channel mapping or new fault relation from independent documentation/analysis?"
        )
        items.append(GapReviewItem(channel=channel,anomaly_id=anomaly_id,unit=rule.get("unit"),gap_kind=kind,
                                   exact_context_candidates=tuple(exact),related_catalog_hints=tuple(hints),expert_question=question))
    return GapReviewPacket(knowledge_version=knowledge_version,
        source_fault_catalog="configs/fault_modes/lab_fault_catalog.json",
        source_anomaly_rules="configs/anomaly_rules.json",items=tuple(items))
