"""Offline group-aware audit of dependence and negative transfer between branches."""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BranchPrediction:
    sample_id: str
    group_id: str
    predicted_id: str


@dataclass(frozen=True)
class TruthRecord:
    sample_id: str
    group_id: str
    true_id: str


def _phi(xs: list[int], ys: list[int]) -> float | None:
    n11=sum(a and b for a,b in zip(xs,ys)); n10=sum(a and not b for a,b in zip(xs,ys))
    n01=sum((not a) and b for a,b in zip(xs,ys)); n00=sum((not a) and (not b) for a,b in zip(xs,ys))
    den=math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return None if den == 0 else (n11*n00-n10*n01)/den


def audit_branch_dependence(data: Sequence[BranchPrediction], kg: Sequence[BranchPrediction], truths: Sequence[TruthRecord]) -> dict:
    d={x.sample_id:x for x in data}; k={x.sample_id:x for x in kg}; t={x.sample_id:x for x in truths}
    if not (set(d)==set(k)==set(t)):
        raise ValueError("data, KG and truth sample sets must match")
    rows=[]; groups={}
    for sid in sorted(d):
        if d[sid].group_id != k[sid].group_id or d[sid].group_id != t[sid].group_id:
            raise ValueError("group mismatch")
        row=(d[sid].predicted_id==t[sid].true_id, k[sid].predicted_id==t[sid].true_id, d[sid].predicted_id!=k[sid].predicted_id)
        rows.append(row); groups.setdefault(d[sid].group_id,[]).append(row)
    per_group={}
    for g,rs in groups.items():
        dx=[int(r[0]) for r in rs]; kx=[int(r[1]) for r in rs]
        per_group[g]={
            "data_accuracy":sum(dx)/len(dx), "kg_accuracy":sum(kx)/len(kx),
            "disagreement":sum(r[2] for r in rs)/len(rs), "double_wrong":sum((not r[0]) and (not r[1]) for r in rs)/len(rs),
            "correctness_phi":_phi(dx,kx),
        }
    return {
        "schema_version":"astrograph-fusion-dependence-audit.v1", "sample_count":len(rows), "group_count":len(groups),
        "group_macro_data_accuracy":sum(v["data_accuracy"] for v in per_group.values())/len(per_group),
        "group_macro_kg_accuracy":sum(v["kg_accuracy"] for v in per_group.values())/len(per_group),
        "group_macro_disagreement":sum(v["disagreement"] for v in per_group.values())/len(per_group),
        "group_macro_double_wrong":sum(v["double_wrong"] for v in per_group.values())/len(per_group),
        "per_group":dict(sorted(per_group.items())),
    }


__all__=["BranchPrediction","TruthRecord","audit_branch_dependence"]


# R18: group-aware leave-branch-out audit for actual BiRRF behavior.
from astrograph.fusion import BiRRFConfig, FaultRanking, RankingSource, fuse_rankings
from .calibration_campaign import group_bootstrap_mean_ci


@dataclass(frozen=True)
class FusionEvaluationCase:
    case_id: str
    group_id: str
    true_fault_id: str
    data_ranking: FaultRanking
    forward_ranking: FaultRanking
    backward_ranking: FaultRanking

    def __post_init__(self) -> None:
        if not self.case_id or not self.group_id or not self.true_fault_id:
            raise ValueError("case_id, group_id and true_fault_id are required")
        if self.data_ranking.source != RankingSource.DATA_DRIVEN:
            raise ValueError("data_ranking must be DATA_DRIVEN")
        if self.forward_ranking.source != RankingSource.KG_FORWARD:
            raise ValueError("forward_ranking must be KG_FORWARD")
        if self.backward_ranking.source != RankingSource.KG_BACKWARD:
            raise ValueError("backward_ranking must be KG_BACKWARD")
        episode_ids = {
            self.data_ranking.episode_id,
            self.forward_ranking.episode_id,
            self.backward_ranking.episode_id,
        }
        if len(episode_ids) != 1:
            raise ValueError("all branch rankings in a fusion case must share episode_id")


def _exact_sign_two_sided(positive: int, negative: int) -> float:
    if positive < 0 or negative < 0:
        raise ValueError("sign-test counts must be nonnegative")
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _policy_configs(config: BiRRFConfig) -> dict[str, BiRRFConfig]:
    return {
        "FULL": config,
        "DATA_ONLY": config.model_copy(update={"data_weight": 1.0, "forward_weight": 0.0, "backward_weight": 0.0}),
        "FORWARD_ONLY": config.model_copy(update={"data_weight": 0.0, "forward_weight": 1.0, "backward_weight": 0.0}),
        "BACKWARD_ONLY": config.model_copy(update={"data_weight": 0.0, "forward_weight": 0.0, "backward_weight": 1.0}),
        "NO_DATA": config.model_copy(update={"data_weight": 0.0}),
        "NO_FORWARD": config.model_copy(update={"forward_weight": 0.0}),
        "NO_BACKWARD": config.model_copy(update={"backward_weight": 0.0}),
    }


def audit_birrf_leave_branch_out(
    cases: Sequence[FusionEvaluationCase],
    config: BiRRFConfig,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260918,
) -> dict:
    if not cases:
        raise ValueError("at least one fusion evaluation case is required")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("fusion case IDs must be unique")
    policy_configs = _policy_configs(config)
    policy_hits: dict[str, dict[str, bool]] = {policy: {} for policy in policy_configs}
    policy_top: dict[str, dict[str, str | None]] = {policy: {} for policy in policy_configs}
    group_by_case = {case.case_id: case.group_id for case in cases}

    for case in cases:
        for policy, policy_config in policy_configs.items():
            result = fuse_rankings(
                case.data_ranking,
                case.forward_ranking,
                case.backward_ranking,
                config=policy_config,
            )
            policy_top[policy][case.case_id] = result.top_fault_id
            policy_hits[policy][case.case_id] = result.top_fault_id == case.true_fault_id

    groups = sorted({case.group_id for case in cases})
    group_accuracy: dict[str, dict[str, float]] = {}
    for policy in policy_configs:
        group_accuracy[policy] = {}
        for group_id in groups:
            ids = [case.case_id for case in cases if case.group_id == group_id]
            group_accuracy[policy][group_id] = sum(policy_hits[policy][case_id] for case_id in ids) / len(ids)

    policy_summary = {}
    for policy in policy_configs:
        macro = group_bootstrap_mean_ci(
            group_accuracy[policy],
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + sorted(policy_configs).index(policy),
        )
        policy_summary[policy] = {
            "case_accuracy": sum(policy_hits[policy].values()) / len(cases),
            "equal_group_macro_accuracy": macro,
        }

    comparisons = {}
    for idx, variant in enumerate(("DATA_ONLY", "FORWARD_ONLY", "BACKWARD_ONLY", "NO_DATA", "NO_FORWARD", "NO_BACKWARD")):
        deltas = {group_id: group_accuracy["FULL"][group_id] - group_accuracy[variant][group_id] for group_id in groups}
        positive = sum(delta > 0 for delta in deltas.values())
        negative = sum(delta < 0 for delta in deltas.values())
        ties = sum(delta == 0 for delta in deltas.values())
        corrected = sum(policy_hits["FULL"][cid] and not policy_hits[variant][cid] for cid in case_ids)
        harmed = sum((not policy_hits["FULL"][cid]) and policy_hits[variant][cid] for cid in case_ids)
        comparisons[variant] = {
            "full_minus_variant_equal_group_accuracy_delta": group_bootstrap_mean_ci(
                deltas,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + 100 + idx,
            ),
            "positive_group_count": positive,
            "negative_group_count": negative,
            "tie_group_count": ties,
            "exact_group_direction_sign_p": _exact_sign_two_sided(positive, negative),
            "case_level_full_unique_corrections_descriptive": corrected,
            "case_level_full_unique_harms_descriptive": harmed,
            "top1_disagreement_case_count": sum(policy_top["FULL"][cid] != policy_top[variant][cid] for cid in case_ids),
        }

    pair_dependence = {}
    for pair_idx, (left, right) in enumerate((("DATA_ONLY", "FORWARD_ONLY"), ("DATA_ONLY", "BACKWARD_ONLY"), ("FORWARD_ONLY", "BACKWARD_ONLY"))):
        disagreement_by_group = {}
        double_wrong_by_group = {}
        for group_id in groups:
            ids = [case.case_id for case in cases if case.group_id == group_id]
            disagreement_by_group[group_id] = sum(policy_top[left][cid] != policy_top[right][cid] for cid in ids) / len(ids)
            double_wrong_by_group[group_id] = sum((not policy_hits[left][cid]) and (not policy_hits[right][cid]) for cid in ids) / len(ids)
        pair_dependence[f"{left}__{right}"] = {
            "equal_group_macro_top1_disagreement": group_bootstrap_mean_ci(
                disagreement_by_group, resamples=bootstrap_resamples, seed=bootstrap_seed + 200 + pair_idx * 2
            ),
            "equal_group_macro_double_wrong": group_bootstrap_mean_ci(
                double_wrong_by_group, resamples=bootstrap_resamples, seed=bootstrap_seed + 201 + pair_idx * 2
            ),
            "interpretation": "Descriptive branch dependence only; no independence or causal-value claim."
        }

    case_identity = hashlib.sha256(json.dumps({"cases": [
        {
            "case_id": case.case_id, "group_id": case.group_id, "true_fault_id": case.true_fault_id,
            "data_ranking": case.data_ranking.model_dump(mode="json"),
            "forward_ranking": case.forward_ranking.model_dump(mode="json"),
            "backward_ranking": case.backward_ranking.model_dump(mode="json"),
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()

    payload = {
        "schema_version": "astrograph-birrf-leave-branch-out-audit.v1",
        "case_identity_sha256": case_identity,
        "case_count": len(cases),
        "independent_group_count": len(groups),
        "baseline_config": config.model_dump(mode="json"),
        "policies": policy_summary,
        "full_vs_variants": comparisons,
        "branch_pair_dependence_descriptive": pair_dependence,
        "source_dependence_status": "STRUCTURAL_SENSITIVITY_NOT_CAUSAL_ATTRIBUTION",
        "calibration_status": "NOT_PROBABILITY_CALIBRATION",
        "claim_boundary": (
            "Offline leave-branch-out sensitivity on frozen independent groups. Removing a branch changes the "
            "fusion system and may change redundancy/correlation handling; deltas are not causal source values, "
            "posterior probability contributions, or flight qualification evidence."
        ),
    }
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


__all__ += ["FusionEvaluationCase", "audit_birrf_leave_branch_out"]
