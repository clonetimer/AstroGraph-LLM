"""Shadow BiRRF adapter for generic-series learned predictions.

Provisional E0 generic knowledge is retained as context only and is not promoted
into an executable KG vote by this adapter.  This prevents a source-derived
catalog from being silently double-counted as formal graph evidence.
"""
from __future__ import annotations

from typing import Any

from astrograph.data_driven import DataDrivenPrediction
from astrograph.evidence.executable import stable_id
from astrograph.fusion import BiRRFConfig, FaultRanking, RankedFault, RankingSource, ScoreSemantics, fuse_rankings


def _provisional_context_ranking(episode_id: str, diagnosis: dict[str, Any], source: RankingSource) -> FaultRanking:
    rows=[]
    for index, item in enumerate(diagnosis.get("fault_candidates", []) or [], start=1):
        fault_id=str(item.get("fault_id") or "").strip()
        if not fault_id: continue
        score=float(item.get("score") or 0.0)
        rows.append(RankedFault(
            fault_id=fault_id, rank=index, score=score, source=source,
            contributes_to_fusion=False, lineage_complete=False,
            metadata={"provisional_e0":True,"context_only":True},
        ))
    return FaultRanking(
        ranking_id=stable_id("RANK",{"episode_id":episode_id,"source":source.value,"faults":[x.fault_id for x in rows]}),
        episode_id=episode_id, source=source, score_semantics=ScoreSemantics.EVIDENCE_STRENGTH,
        candidates=rows, producer_id="astrograph-generic-e0-context", producer_version="1",
        claim_boundary="Provisional source-derived generic knowledge; context only and non-voting in shadow BiRRF.",
        metadata={"diagnostic_vote_allowed":False,"provisional_e0":True},
    )


def build_generic_shadow_fusion(execution: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    if execution.get("status") != "SUCCEEDED":
        return {"status":"NOT_AVAILABLE","diagnostic_vote_allowed":False,"reason":"NO_EXECUTED_DATA_DRIVEN_MODEL"}
    prediction=DataDrivenPrediction.model_validate(execution["prediction"])
    registered=[x.class_id for x in prediction.classes if x.class_id != prediction.normal_class_id]
    data=prediction.to_fault_ranking(registered)
    forward=_provisional_context_ranking(prediction.episode_id,diagnosis,RankingSource.KG_FORWARD)
    backward=FaultRanking(
        ranking_id=stable_id("RANK",{"episode_id":prediction.episode_id,"source":"KG_BACKWARD","empty":True}),
        episode_id=prediction.episode_id, source=RankingSource.KG_BACKWARD,
        score_semantics=ScoreSemantics.EVIDENCE_COVERAGE, candidates=[],
        producer_id="astrograph-generic-backward-unavailable", producer_version="1",
        claim_boundary="No reviewed executable generic backward-evidence package is bound; branch withheld.",
        metadata={"diagnostic_vote_allowed":False,"reason":"NO_REVIEWED_EXECUTABLE_BACKWARD_EVIDENCE"},
    )
    config=BiRRFConfig(data_weight=1.0,forward_weight=0.0,backward_weight=0.0,
                       expected_fault_universe=tuple(sorted(registered)))
    result=fuse_rankings(data,forward,backward,config=config,
                         normal_probability=prediction.normal_probability,normal_rank=prediction.normal_rank)
    return {
        "status":"SUCCEEDED_SHADOW",
        "diagnostic_vote_allowed":False,
        "data_ranking":data.model_dump(mode="json"),
        "forward_context_ranking":forward.model_dump(mode="json"),
        "backward_ranking":backward.model_dump(mode="json"),
        "birrf_result":result.model_dump(mode="json"),
        "claim_boundary":"BiRRF main-chain shadow execution only. Generic E0 and unavailable backward evidence carry zero formal vote.",
    }


__all__=["build_generic_shadow_fusion"]
