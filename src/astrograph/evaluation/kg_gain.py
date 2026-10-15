from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrograph.agents.agentic import DiagnosticReasonerAgent, SafetyCriticAgent
from astrograph.diagnosis import (
    CapabilityAxesEvidence,
    DiagnosisEvidenceBundle,
    DiagnosisPolicyGate,
    KGEvidenceReference,
    ModelEvidenceReference,
    SafetyReplayReference,
)
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus, EvidenceRole
from astrograph.evidence.executable import stable_id
from astrograph.evaluation.perturbations import (
    EvidencePerturbationSpec,
    apply_evidence_perturbations,
)
from astrograph.fusion import (
    BiRRFConfig,
    CandidateFusionPackage,
    FaultRanking,
    RankedFault,
    RankingSource,
    build_backward_evidence_ranking,
    build_forward_graph_ranking,
    fuse_rankings,
)
from astrograph.investigation import DecisionDisposition, DecisionGovernor
from astrograph.knowledge import (
    CsvEvidenceGraphReferenceExecutor,
    ReadOnlyCypherCompiler,
    ReadOnlyGraphQueryExecutor,
    associations_from_query_result,
    build_backward_fault_query_plan,
    build_forward_evidence_query_plan,
)
from astrograph.tools.online import _invoke_online_core


ABLATION_ARMS = [
    "data_only",
    "kg_forward_only",
    "kg_backward_only",
    "data_plus_kg_forward",
    "data_plus_kg_backward",
    "full_birrf",
    "full_safety_chain",
]

DEFAULT_FUSION_CONFIG = BiRRFConfig()
DATA_PLUS_FORWARD_CONFIG = BiRRFConfig(backward_weight=0.0)
DATA_PLUS_BACKWARD_CONFIG = BiRRFConfig(forward_weight=0.0)


@dataclass(frozen=True)
class FixedCaseSpec:
    case_id: str
    source_scenario: str
    truth_fault_id: str | None
    condition_tags: tuple[str, ...]
    data_fault_order: tuple[str, ...]
    data_scores: tuple[float, ...]
    perturbation_mode: str = "none"
    is_unknown: bool = False


def v0530_fixed_case_specs() -> list[FixedCaseSpec]:
    return [
        FixedCaseSpec(
            case_id="clear_single_fault_heater",
            source_scenario="heater_stuck_on",
            truth_fault_id="THERMAL_HEATER_STUCK_ON",
            condition_tags=("clear_single_fault",),
            data_fault_order=("THERMAL_HEATER_STUCK_ON", "THERMAL_RADIATOR_REJECTION_LOSS"),
            data_scores=(0.92, 0.05),
        ),
        FixedCaseSpec(
            case_id="data_wrong_rank_heater",
            source_scenario="heater_stuck_on",
            truth_fault_id="THERMAL_HEATER_STUCK_ON",
            condition_tags=("data_model_low_confidence_or_wrong_rank",),
            data_fault_order=("ADCS_STAR_TRACKER_DROPOUT", "THERMAL_HEATER_STUCK_ON"),
            data_scores=(0.54, 0.42),
        ),
        FixedCaseSpec(
            case_id="evidence_conflict_heater",
            source_scenario="heater_stuck_on",
            truth_fault_id="THERMAL_HEATER_STUCK_ON",
            condition_tags=("evidence_conflict",),
            data_fault_order=("THERMAL_HEATER_STUCK_ON", "THERMAL_RADIATOR_REJECTION_LOSS"),
            data_scores=(0.88, 0.07),
            perturbation_mode="force_contradicted_required",
        ),
        FixedCaseSpec(
            case_id="critical_evidence_missing_star_tracker",
            source_scenario="star_tracker_dropout",
            truth_fault_id="ADCS_STAR_TRACKER_DROPOUT",
            condition_tags=("critical_evidence_missing", "dropout"),
            data_fault_order=("ADCS_STAR_TRACKER_DROPOUT", "ADCS_GYRO_BIAS_DRIFT"),
            data_scores=(0.74, 0.18),
            perturbation_mode="drop_critical_required",
        ),
        FixedCaseSpec(
            case_id="noise_near_threshold_rw_friction",
            source_scenario="reaction_wheel_friction_increase",
            truth_fault_id="ADCS_RW_FRICTION_INCREASE",
            condition_tags=("noise",),
            data_fault_order=("ADCS_RW_TORQUE_AUTHORITY_LOSS", "ADCS_RW_FRICTION_INCREASE"),
            data_scores=(0.47, 0.45),
            perturbation_mode="near_threshold",
        ),
        FixedCaseSpec(
            case_id="severity_timing_rw_friction",
            source_scenario="reaction_wheel_friction_increase",
            truth_fault_id="ADCS_RW_FRICTION_INCREASE",
            condition_tags=("severity_change", "fault_timing_change"),
            data_fault_order=("ADCS_RW_TORQUE_AUTHORITY_LOSS", "ADCS_RW_FRICTION_INCREASE"),
            data_scores=(0.51, 0.43),
            perturbation_mode="shift_feature_window",
        ),
        FixedCaseSpec(
            case_id="unknown_fault_false_accept_probe",
            source_scenario="heater_stuck_on",
            truth_fault_id=None,
            condition_tags=("unknown_fault_or_unknown_label",),
            data_fault_order=("THERMAL_HEATER_STUCK_ON", "THERMAL_RADIATOR_REJECTION_LOSS"),
            data_scores=(0.91, 0.05),
            is_unknown=True,
        ),
    ]


def load_bundle(artifact_root: Path, scenario: str) -> EpisodeEvidenceBundle:
    path = artifact_root / scenario / "diagnosis" / "episode_evidence_bundle.json"
    return EpisodeEvidenceBundle.model_validate_json(path.read_text(encoding="utf-8"))


def registered_fault_ids(bundle: EpisodeEvidenceBundle) -> list[str]:
    return sorted({item.target_fault_id for item in bundle.evidence_definitions})


def build_fixed_data_ranking(
    bundle: EpisodeEvidenceBundle,
    *,
    fault_order: tuple[str, ...],
    scores: tuple[float, ...],
) -> FaultRanking:
    registered = registered_fault_ids(bundle)
    score_by_fault = {fault: max(0.001, 0.01 / (index + 1)) for index, fault in enumerate(registered)}
    for fault_id, score in zip(fault_order, scores, strict=True):
        if fault_id in score_by_fault:
            score_by_fault[fault_id] = float(score)
    ordered = sorted(score_by_fault.items(), key=lambda item: (-item[1], item[0]))
    candidates = [
        RankedFault(
            fault_id=fault_id,
            rank=index,
            score=score,
            source=RankingSource.DATA_DRIVEN,
            metadata={"fixed_prediction_protocol": "astrograph-v0530-kg-gain"},
        )
        for index, (fault_id, score) in enumerate(ordered, start=1)
    ]
    return FaultRanking(
        ranking_id=stable_id(
            "RANK",
            {
                "episode_id": bundle.episode.episode_id,
                "source": RankingSource.DATA_DRIVEN.value,
                "scores": ordered,
                "protocol": "v0530-fixed-prediction",
            },
        ),
        episode_id=bundle.episode.episode_id,
        source=RankingSource.DATA_DRIVEN,
        candidates=candidates,
        producer_id="v0530-fixed-prediction-table",
        producer_version="0.5.30",
        metadata={
            "training_performed": False,
            "test_tuned": False,
            "claim_boundary": "Fixed deterministic ablation input, not a learned data-driven model.",
        },
    )


def _first_positive_definition(bundle: EpisodeEvidenceBundle, fault_id: str, *, critical_only: bool) -> str:
    candidates = [
        item
        for item in bundle.evidence_definitions
        if item.target_fault_id == fault_id
        and item.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING}
        and (item.critical or not critical_only)
    ]
    if not candidates:
        candidates = [
            item
            for item in bundle.evidence_definitions
            if item.target_fault_id == fault_id and item.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING}
        ]
    if not candidates:
        raise ValueError(f"no positive evidence definition found for {fault_id}")
    candidates.sort(key=lambda item: (not item.critical, item.evidence_definition_id))
    return candidates[0].evidence_definition_id


def apply_case_perturbation(
    bundle: EpisodeEvidenceBundle,
    spec: FixedCaseSpec,
) -> tuple[EpisodeEvidenceBundle, list[dict[str, Any]]]:
    if spec.perturbation_mode == "none" or spec.truth_fault_id is None:
        return bundle, []
    if spec.perturbation_mode == "force_contradicted_required":
        target = _first_positive_definition(bundle, spec.truth_fault_id, critical_only=True)
        specs = [
            EvidencePerturbationSpec(
                perturbation_id=f"{spec.case_id}-contradicted-required",
                perturbation_type="FORCE_STATUS",
                evidence_definition_id=target,
                forced_status=EvidenceFactStatus.CONTRADICTED,
                rationale="Fixed V0.5.30 evidence-conflict boundary case.",
            )
        ]
    elif spec.perturbation_mode == "drop_critical_required":
        target = _first_positive_definition(bundle, spec.truth_fault_id, critical_only=True)
        specs = [
            EvidencePerturbationSpec(
                perturbation_id=f"{spec.case_id}-drop-critical",
                perturbation_type="DROP_EVIDENCE",
                evidence_definition_id=target,
                rationale="Fixed V0.5.30 critical-evidence-missing boundary case.",
            )
        ]
    elif spec.perturbation_mode == "near_threshold":
        target = _first_positive_definition(bundle, spec.truth_fault_id, critical_only=False)
        specs = [
            EvidencePerturbationSpec(
                perturbation_id=f"{spec.case_id}-near-threshold",
                perturbation_type="NEAR_THRESHOLD",
                evidence_definition_id=target,
                threshold_offset_ratio=0.01,
                rationale="Fixed V0.5.30 noise/near-threshold boundary case.",
            )
        ]
    elif spec.perturbation_mode == "shift_feature_window":
        target = _first_positive_definition(bundle, spec.truth_fault_id, critical_only=False)
        definition = next(item for item in bundle.evidence_definitions if item.evidence_definition_id == target)
        specs = [
            EvidencePerturbationSpec(
                perturbation_id=f"{spec.case_id}-shift-window",
                perturbation_type="SHIFT_FEATURE_WINDOW",
                feature_definition_id=definition.feature_definition_id,
                window_shift_s=7.0,
                rationale="Fixed V0.5.30 timing-boundary case.",
            )
        ]
    else:
        raise ValueError(f"unsupported perturbation mode: {spec.perturbation_mode}")
    updated, audits = apply_evidence_perturbations(bundle, specs)
    return updated, [item.model_dump(mode="json") for item in audits]


def _ranking_top(ranking: FaultRanking) -> str | None:
    return ranking.candidates[0].fault_id if ranking.candidates else None


def _fused_top(data: FaultRanking, forward: FaultRanking, backward: FaultRanking, config: BiRRFConfig) -> tuple[str | None, list[str]]:
    result = fuse_rankings(data, forward, backward, config=config)
    return result.top_fault_id, list(result.conflict_flags)


def _backward_metadata(backward: FaultRanking, fault_id: str | None) -> dict[str, Any]:
    if fault_id is None:
        return {}
    for candidate in backward.candidates:
        if candidate.fault_id == fault_id:
            return candidate.metadata
    return {}


def _has_boundary_evidence_issue(metadata: dict[str, Any]) -> bool:
    if float(metadata.get("completeness", 0.0)) < 1.0:
        return True
    if float(metadata.get("consistency", 0.0)) < 1.0:
        return True
    if int(metadata.get("critical_missing_count", 0)) > 0:
        return True
    return False


def _ranking_signature(ranking: FaultRanking) -> dict[str, Any]:
    return {
        "top_fault_id": _ranking_top(ranking),
        "fault_order": [item.fault_id for item in ranking.candidates],
        "evidence_fact_ids": sorted({fact for item in ranking.candidates for fact in item.evidence_fact_ids}),
    }


def _spearman_rank_correlation(left: FaultRanking, right: FaultRanking) -> float | None:
    left_rank = {item.fault_id: item.rank for item in left.candidates}
    right_rank = {item.fault_id: item.rank for item in right.candidates}
    shared = sorted(set(left_rank) & set(right_rank))
    n = len(shared)
    if n < 2:
        return None
    squared = sum((left_rank[fault_id] - right_rank[fault_id]) ** 2 for fault_id in shared)
    return 1.0 - (6.0 * squared) / (n * (n * n - 1))


def _top_k_overlap(left: FaultRanking, right: FaultRanking, k: int) -> float:
    left_top = {item.fault_id for item in sorted(left.candidates, key=lambda item: item.rank)[:k]}
    right_top = {item.fault_id for item in sorted(right.candidates, key=lambda item: item.rank)[:k]}
    if not left_top and not right_top:
        return 1.0
    return len(left_top & right_top) / max(1, len(left_top | right_top))


def _evidence_overlap(left: FaultRanking, right: FaultRanking) -> float:
    left_ids = set(_ranking_signature(left)["evidence_fact_ids"])
    right_ids = set(_ranking_signature(right)["evidence_fact_ids"])
    if not left_ids and not right_ids:
        return 1.0
    return len(left_ids & right_ids) / max(1, len(left_ids | right_ids))


def _candidate_fusion_package(
    *,
    bundle: EpisodeEvidenceBundle,
    data: FaultRanking,
    forward: FaultRanking,
    backward: FaultRanking,
    config: BiRRFConfig,
) -> CandidateFusionPackage:
    fused = fuse_rankings(data, forward, backward, config=config)
    return CandidateFusionPackage(
        episode_id=bundle.episode.episode_id,
        data_ranking=data,
        forward_ranking=forward,
        backward_ranking=backward,
        birrf_result=fused,
        registered_fault_ids=registered_fault_ids(bundle),
        metadata={
            "v0530_r1_methodology": "fixed ablation package; safety chain receives no truth labels",
        },
    )


def _synthetic_kg_evidence_reference(bundle: EpisodeEvidenceBundle) -> KGEvidenceReference:
    query_hash = "0" * 64
    return KGEvidenceReference(
        release_id="KG-GOLD-ONLINE-V058",
        knowledge_version="astrograph-kg-gold-online-v058",
        nodes_sha256="0" * 64,
        edges_sha256="0" * 64,
        query_ids=[
            stable_id("GQF", {"episode_id": bundle.episode.episode_id}),
            stable_id("GQB", {"episode_id": bundle.episode.episode_id}),
        ],
        query_sha256_values=[query_hash],
        read_user="v0530-r1-offline-evaluation",
        row_count=max(1, len(bundle.evidence_definitions)),
        graph_backend="offline_csv_reference",
    )


def _policy_bundle(
    *,
    bundle: EpisodeEvidenceBundle,
    candidate_fusion: CandidateFusionPackage,
    decision: Any,
    policy_replay_status: str,
) -> DiagnosisEvidenceBundle:
    ranking = candidate_fusion.data_ranking
    model_evidence = ModelEvidenceReference(
        model_id=ranking.producer_id,
        model_version=ranking.producer_version,
        prediction_id=str(ranking.metadata.get("prediction_id") or ranking.ranking_id),
        probability_ranking_id=ranking.ranking_id,
        top_fault_id=ranking.candidates[0].fault_id if ranking.candidates else None,
        calibrated=False,
    )
    return DiagnosisEvidenceBundle(
        bundle_id=stable_id(
            "DEB",
            {
                "episode_id": bundle.episode.episode_id,
                "decision_id": decision.decision_id,
                "fusion_id": candidate_fusion.birrf_result.fusion_id,
                "protocol": "v0530-r1",
            },
        ),
        sample_id=bundle.episode.run_id,
        episode_id=bundle.episode.episode_id,
        split="ONLINE",
        model_evidence=model_evidence,
        kg_evidence=_synthetic_kg_evidence_reference(bundle),
        safety_replay=SafetyReplayReference(
            replay_id=decision.decision_id,
            status=policy_replay_status,
        ),
        capability_axes=CapabilityAxesEvidence(
            supported_axes=["seed"],
            unsupported_axes=["operating_condition", "initial_state", "satellite_instance", "flight_data"],
            requested_axes=[],
            capability_manifest_id="v0530-r1-fixed-ablation-capability-boundary",
        ),
        upstream_formal_release_allowed=(
            decision.disposition == DecisionDisposition.ACCEPT and not decision.requires_human_review
        ),
        metadata={
            "decision_disposition": decision.disposition.value,
            "accepted_fault_id": decision.accepted_fault_id,
            "truth_available_to_policy": False,
        },
    )


def _run_safety_chain(
    *,
    bundle: EpisodeEvidenceBundle,
    candidate_fusion: CandidateFusionPackage,
) -> dict[str, Any]:
    reasoner = DiagnosticReasonerAgent()
    critic = SafetyCriticAgent()
    governor = DecisionGovernor()
    policy_gate = DiagnosisPolicyGate()
    hypotheses = reasoner.reason(
        bundle,
        candidate_fusion=candidate_fusion.model_dump(mode="json"),
    )
    critique = critic.critique(hypotheses, bundle)
    core = _invoke_online_core({"bundle": bundle.model_dump(mode="json")})
    decision = governor.decide(
        hypothesis_set=hypotheses,
        critique=critique,
        deterministic_core_result=core,
        protocol_versions={
            "astrograph": "0.5.30-R1",
            "runtime_core": str(core["core_version"]),
            "safety_critic": critic.agent_version,
            "decision_governor": governor.governor_version,
        },
    )
    policy_replay_status = "PASS" if decision.comparison_status in {"MATCH", "CONSISTENT_NON_DIAGNOSIS"} else "FAIL"
    policy_bundle = _policy_bundle(
        bundle=bundle,
        candidate_fusion=candidate_fusion,
        decision=decision,
        policy_replay_status=policy_replay_status,
    )
    fused = candidate_fusion.birrf_result
    policy_decision = policy_gate.decide(
        policy_bundle,
        blocking_conflicts=fused.blocking_conflicts,
        quality_warnings=fused.quality_warnings,
        open_set_actions=fused.open_set_actions,
    )
    auto_accepted = (
        decision.disposition == DecisionDisposition.ACCEPT
        and not decision.requires_human_review
        and not policy_decision.blocking_reasons
    )
    return {
        "decision": "AUTO_ACCEPT" if auto_accepted else "HUMAN_REVIEW",
        "auto_accepted": auto_accepted,
        "accepted_fault_id": decision.accepted_fault_id if auto_accepted else None,
        "decision_record": decision.model_dump(mode="json"),
        "diagnosis_policy_decision": policy_decision.model_dump(mode="json"),
        "deterministic_core_result": {
            key: value for key, value in core.items() if key != "reference_hypothesis_set"
        },
        "safety_critique": critique.model_dump(mode="json"),
    }


def evaluate_kg_gain_ablation(
    *,
    artifact_root: Path,
    case_specs: list[FixedCaseSpec] | None = None,
) -> dict[str, Any]:
    specs = case_specs or v0530_fixed_case_specs()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        bundle = load_bundle(artifact_root, spec.source_scenario)
        bundle, perturbation_audit = apply_case_perturbation(bundle, spec)
        data = build_fixed_data_ranking(bundle, fault_order=spec.data_fault_order, scores=spec.data_scores)
        forward = build_forward_graph_ranking(bundle)
        backward = build_backward_evidence_ranking(bundle, candidate_fault_ids=registered_fault_ids(bundle))
        data_forward_top, data_forward_flags = _fused_top(data, forward, backward, DATA_PLUS_FORWARD_CONFIG)
        data_backward_top, data_backward_flags = _fused_top(data, forward, backward, DATA_PLUS_BACKWARD_CONFIG)
        full_top, full_flags = _fused_top(data, forward, backward, DEFAULT_FUSION_CONFIG)
        safety_package = _candidate_fusion_package(
            bundle=bundle,
            data=data,
            forward=forward,
            backward=backward,
            config=DEFAULT_FUSION_CONFIG,
        )
        safety_chain = _run_safety_chain(bundle=bundle, candidate_fusion=safety_package)
        arm_tops = {
            "data_only": _ranking_top(data),
            "kg_forward_only": _ranking_top(forward),
            "kg_backward_only": _ranking_top(backward),
            "data_plus_kg_forward": data_forward_top,
            "data_plus_kg_backward": data_backward_top,
            "full_birrf": full_top,
            "full_safety_chain": safety_chain["accepted_fault_id"],
        }
        arm_flags = {
            "data_only": [],
            "kg_forward_only": [],
            "kg_backward_only": [],
            "data_plus_kg_forward": data_forward_flags,
            "data_plus_kg_backward": data_backward_flags,
            "full_birrf": full_flags,
            "full_safety_chain": full_flags,
        }
        for arm in ABLATION_ARMS:
            top = arm_tops[arm]
            metadata = {} if arm == "data_only" else _backward_metadata(backward, top)
            if arm == "full_safety_chain":
                decision, auto_accepted = safety_chain["decision"], bool(safety_chain["auto_accepted"])
            else:
                decision, auto_accepted = "AUTO_ACCEPT", True
            rows.append(
                {
                    "case_id": spec.case_id,
                    "source_scenario": spec.source_scenario,
                    "condition_tags": list(spec.condition_tags),
                    "arm": arm,
                    "truth_fault_id": spec.truth_fault_id,
                    "predicted_fault_id": top,
                    "data_top_fault_id": _ranking_top(data),
                    "forward_top_fault_id": _ranking_top(forward),
                    "backward_top_fault_id": _ranking_top(backward),
                    "correct": bool(top == spec.truth_fault_id) if not spec.is_unknown else False,
                    "known_case": not spec.is_unknown,
                    "unknown_case": spec.is_unknown,
                    "decision": decision,
                    "auto_accepted": auto_accepted,
                    "wrong_auto_accept": bool(auto_accepted and not spec.is_unknown and top != spec.truth_fault_id),
                    "unknown_false_accept": bool(auto_accepted and spec.is_unknown),
                    "human_review": decision == "HUMAN_REVIEW",
                    "kg_recovered_wrong_rank": bool(_ranking_top(data) != spec.truth_fault_id and top == spec.truth_fault_id)
                    if not spec.is_unknown
                    else False,
                    "kg_wrong_override_correct_model": bool(_ranking_top(data) == spec.truth_fault_id and top != spec.truth_fault_id)
                    if not spec.is_unknown
                    else False,
                    "evidence_path_status": "NOT_APPLICABLE" if arm == "data_only" else "APPLICABLE",
                    "evidence_path_completeness": None
                    if arm == "data_only"
                    else float(metadata.get("completeness", 0.0)),
                    "evidence_path_coverage": None
                    if arm == "data_only"
                    else float(metadata.get("coverage", 0.0)),
                    "evidence_path_consistency": None
                    if arm == "data_only"
                    else float(metadata.get("consistency", 0.0)),
                    "critical_missing_count": int(metadata.get("critical_missing_count", 0)),
                    "conflict_flags": arm_flags[arm],
                    "perturbation_audit": perturbation_audit,
                    "safety_chain_trace": safety_chain if arm == "full_safety_chain" else None,
                    "kg_branch_diagnostics": {
                        "forward_backward_spearman": _spearman_rank_correlation(forward, backward),
                        "forward_backward_top1_overlap": _top_k_overlap(forward, backward, 1),
                        "forward_backward_top3_overlap": _top_k_overlap(forward, backward, 3),
                        "forward_backward_evidence_overlap": _evidence_overlap(forward, backward),
                    },
                }
            )
    return summarize_ablation(rows)


def _macro_f1(known_rows: list[dict[str, Any]]) -> float:
    labels = sorted({row["truth_fault_id"] for row in known_rows if row["truth_fault_id"]})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(row["truth_fault_id"] == label and row["predicted_fault_id"] == label for row in known_rows)
        fp = sum(row["truth_fault_id"] != label and row["predicted_fault_id"] == label for row in known_rows)
        fn = sum(row["truth_fault_id"] == label and row["predicted_fault_id"] != label for row in known_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def summarize_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in ABLATION_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        known_rows = [row for row in arm_rows if row["known_case"]]
        unknown_rows = [row for row in arm_rows if row["unknown_case"]]
        arm_metrics[arm] = {
            "known_case_count": len(known_rows),
            "unknown_case_count": len(unknown_rows),
            "accuracy": sum(row["correct"] for row in known_rows) / len(known_rows) if known_rows else 0.0,
            "macro_f1": _macro_f1(known_rows),
            "wrong_auto_accept_rate": sum(row["wrong_auto_accept"] for row in known_rows) / len(known_rows)
            if known_rows
            else 0.0,
            "unknown_fault_false_accept_rate": sum(row["unknown_false_accept"] for row in unknown_rows) / len(unknown_rows)
            if unknown_rows
            else 0.0,
            "human_review_trigger_rate": sum(row["human_review"] for row in arm_rows) / len(arm_rows)
            if arm_rows
            else 0.0,
            "kg_recovered_wrong_rank_count": sum(row["kg_recovered_wrong_rank"] for row in known_rows),
            "kg_wrong_override_correct_model_count": sum(row["kg_wrong_override_correct_model"] for row in known_rows),
            "evidence_path_complete_rate": _path_complete_rate(arm_rows),
            "mean_evidence_path_completeness": _mean_path_completeness(arm_rows),
            "evidence_path_applicable_count": sum(row["evidence_path_status"] == "APPLICABLE" for row in arm_rows),
        }
    kg_diagnostics = _kg_branch_diagnostics(rows)
    covered_conditions = sorted({tag for row in rows for tag in row["condition_tags"]})
    return {
        "schema_version": "astrograph-v0530-knowledge-gain-ablation.v1",
        "version": "0.5.30",
        "protocol": {
            "methodology_revision": "V0.5.30-R1",
            "fixed_data": True,
            "fixed_data_model_predictions": True,
            "fixed_safety_chain": True,
            "safety_chain_reads_truth": False,
            "test_tuning_performed": False,
            "test_labels_used_to_construct_graph": False,
            "new_data_driven_model_added": False,
            "arms": ABLATION_ARMS,
            "fusion_configs": {
                "data_plus_kg_forward": DATA_PLUS_FORWARD_CONFIG.model_dump(mode="json"),
                "data_plus_kg_backward": DATA_PLUS_BACKWARD_CONFIG.model_dump(mode="json"),
                "full_birrf": DEFAULT_FUSION_CONFIG.model_dump(mode="json"),
            },
        },
        "covered_conditions": covered_conditions,
        "case_count": len({row["case_id"] for row in rows}),
        "row_count": len(rows),
        "arm_metrics": arm_metrics,
        "kg_branch_diagnostics": kg_diagnostics,
        "rows": rows,
        "claim_boundary": (
            "This is a fixed E0 ablation over existing whole-spacecraft runtime artifacts and deterministic "
            "prediction tables. It measures relative KG contribution and failure boundaries only; it is not "
            "cross-satellite, flight, or deployment generalization evidence."
        ),
    }


def _path_values(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(row["evidence_path_completeness"])
        for row in rows
        if row.get("evidence_path_status") == "APPLICABLE" and row.get("evidence_path_completeness") is not None
    ]


def _path_complete_rate(rows: list[dict[str, Any]]) -> float | None:
    values = _path_values(rows)
    if not values:
        return None
    return sum(value >= 1.0 for value in values) / len(values)


def _mean_path_completeness(rows: list[dict[str, Any]]) -> float | None:
    values = _path_values(rows)
    return statistics.fmean(values) if values else None


def _kg_branch_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_case = {}
    for row in rows:
        if row["arm"] != "full_birrf":
            continue
        per_case[row["case_id"]] = row["kg_branch_diagnostics"]
    values = list(per_case.values())
    return {
        "case_count": len(values),
        "mean_forward_backward_spearman": statistics.fmean(
            item["forward_backward_spearman"] for item in values
        )
        if values
        else None,
        "mean_forward_backward_top1_overlap": statistics.fmean(
            item["forward_backward_top1_overlap"] for item in values
        )
        if values
        else None,
        "mean_forward_backward_top3_overlap": statistics.fmean(
            item["forward_backward_top3_overlap"] for item in values
        )
        if values
        else None,
        "mean_forward_backward_evidence_overlap": statistics.fmean(
            item["forward_backward_evidence_overlap"] for item in values
        )
        if values
        else None,
        "per_case": per_case,
    }


def build_failure_boundary_report(ablation: dict[str, Any]) -> dict[str, Any]:
    rows = ablation["rows"]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for tag in row["condition_tags"]:
            by_condition.setdefault(tag, []).append(row)
    condition_rows = []
    for tag, tagged in sorted(by_condition.items()):
        safety = [row for row in tagged if row["arm"] == "full_safety_chain"]
        birrf = [row for row in tagged if row["arm"] == "full_birrf"]
        condition_rows.append(
            {
                "condition": tag,
                "case_count": len({row["case_id"] for row in tagged}),
                "full_birrf_correct_count": sum(row["correct"] for row in birrf),
                "safety_human_review_count": sum(row["human_review"] for row in safety),
                "safety_wrong_auto_accept_count": sum(row["wrong_auto_accept"] for row in safety),
                "safety_unknown_false_accept_count": sum(row["unknown_false_accept"] for row in safety),
                "minimum_completeness": min(_path_values(safety)) if _path_values(safety) else None,
                "boundary_interpretation": _boundary_interpretation(tag, safety),
            }
        )
    return {
        "schema_version": "astrograph-v0530-kg-failure-boundary-report.v1",
        "version": "0.5.30",
        "conditions": condition_rows,
        "safety_chain_summary": ablation["arm_metrics"]["full_safety_chain"],
        "claim_boundary": ablation["claim_boundary"],
    }


def _boundary_interpretation(tag: str, safety_rows: list[dict[str, Any]]) -> str:
    if tag == "unknown_fault_or_unknown_label":
        return "Unknown labels must be rejected instead of converted into a registered fault."
    if any(row["human_review"] for row in safety_rows):
        return "The fixed safety chain blocks automatic acceptance under this boundary condition."
    return "The condition remains automatically acceptable only when KG evidence stays complete and non-conflicting."


def _latency_ms(callable_obj: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_obj()
    return result, (time.perf_counter() - started) * 1000.0


def evaluate_neo4j_fallback(
    *,
    artifact_root: Path,
    nodes_csv: Path,
    edges_csv: Path,
    scenario: str = "heater_stuck_on",
    uri: str = "bolt://127.0.0.1:7687",
    user: str = "neo4j",
    password: str | None = None,
    knowledge_version: str = "astrograph-kg-gold-online-v058",
) -> dict[str, Any]:
    bundle = load_bundle(artifact_root, scenario)
    evidence_ids = [item.evidence_definition_id for item in bundle.evidence_definitions]
    fault_ids = registered_fault_ids(bundle)
    compiler = ReadOnlyCypherCompiler()
    forward_plan = build_forward_evidence_query_plan(evidence_ids)
    backward_plan = build_backward_fault_query_plan(fault_ids)
    offline = CsvEvidenceGraphReferenceExecutor(nodes_csv, edges_csv)
    offline_forward, offline_forward_ms = _latency_ms(
        lambda: offline.execute(forward_plan, knowledge_version=knowledge_version)
    )
    offline_backward, offline_backward_ms = _latency_ms(
        lambda: offline.execute(backward_plan, knowledge_version=knowledge_version)
    )
    live_report: dict[str, Any]
    live_forward = None
    live_backward = None
    if password:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(user, password))
            try:
                driver.verify_connectivity()
                executor = ReadOnlyGraphQueryExecutor(driver)
                compiled_forward = compiler.compile(forward_plan, knowledge_version=knowledge_version)
                compiled_backward = compiler.compile(backward_plan, knowledge_version=knowledge_version)
                live_forward, live_forward_ms = _latency_ms(
                    lambda: executor.execute(compiled_forward, knowledge_version=knowledge_version)
                )
                live_backward, live_backward_ms = _latency_ms(
                    lambda: executor.execute(compiled_backward, knowledge_version=knowledge_version)
                )
            finally:
                driver.close()
            live_report = {
                "available": True,
                "uri": uri,
                "user": user,
                "forward_query_latency_ms": live_forward_ms,
                "backward_query_latency_ms": live_backward_ms,
                "forward_rows": live_forward.row_count,
                "backward_rows": live_backward.row_count,
            }
        except Exception as exc:  # pragma: no cover - depends on local service
            live_report = {
                "available": False,
                "uri": uri,
                "user": user,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    else:
        live_report = {
            "available": False,
            "uri": uri,
            "user": user,
            "error_type": "MissingPassword",
            "error": "No password was provided to the fallback evaluation.",
        }
    offline_associations = {
        "forward": [item.model_dump(mode="json") for item in associations_from_query_result(offline_forward)],
        "backward": [item.model_dump(mode="json") for item in associations_from_query_result(offline_backward)],
    }
    fallback_equivalent = None
    if live_forward is not None and live_backward is not None:
        live_associations = {
            "forward": [item.model_dump(mode="json") for item in associations_from_query_result(live_forward)],
            "backward": [item.model_dump(mode="json") for item in associations_from_query_result(live_backward)],
        }
        fallback_equivalent = live_associations == offline_associations
    return {
        "schema_version": "astrograph-v0530-neo4j-fallback-report.v1",
        "version": "0.5.30",
        "scenario": scenario,
        "knowledge_version": knowledge_version,
        "neo4j": live_report,
        "offline_fallback": {
            "available": True,
            "executor": "CsvEvidenceGraphReferenceExecutor",
            "forward_query_latency_ms": offline_forward_ms,
            "backward_query_latency_ms": offline_backward_ms,
            "forward_rows": offline_forward.row_count,
            "backward_rows": offline_backward.row_count,
        },
        "neo4j_unavailable_fallback": {
            "simulated_condition": "Neo4j driver is unavailable, not configured, or connectivity fails.",
            "selected_executor": "CsvEvidenceGraphReferenceExecutor",
            "result": "PASS",
            "forward_rows": offline_forward.row_count,
            "backward_rows": offline_backward.row_count,
            "forward_query_latency_ms": offline_forward_ms,
            "backward_query_latency_ms": offline_backward_ms,
        },
        "fallback_equivalent_to_live": fallback_equivalent,
        "fallback_result": "USED_OFFLINE_REFERENCE" if not live_report["available"] else "LIVE_AVAILABLE",
        "claim_boundary": (
            "Offline fallback validates deterministic association retrieval from the versioned CSV graph release. "
            "Neo4j latency is reported only when the configured local service is reachable."
        ),
    }


def write_v0530_reports(
    *,
    output_dir: Path,
    ablation: dict[str, Any],
    failure_boundary: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "knowledge_gain_ablation_json": output_dir / "knowledge_gain_ablation.json",
        "knowledge_gain_ablation_md": output_dir / "knowledge_gain_ablation.md",
        "kg_failure_boundary_report_json": output_dir / "kg_failure_boundary_report.json",
        "neo4j_fallback_report_json": output_dir / "neo4j_fallback_report.json",
        "implementation_report_md": output_dir / "V0.5.30实现报告.md",
    }
    paths["knowledge_gain_ablation_json"].write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["kg_failure_boundary_report_json"].write_text(
        json.dumps(failure_boundary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["neo4j_fallback_report_json"].write_text(
        json.dumps(fallback, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["knowledge_gain_ablation_md"].write_text(_ablation_markdown(ablation), encoding="utf-8")
    paths["implementation_report_md"].write_text(
        _implementation_markdown(ablation, failure_boundary, fallback),
        encoding="utf-8",
    )
    return paths


def _ablation_markdown(ablation: dict[str, Any]) -> str:
    lines = [
        "# AstroGraph V0.5.30 Knowledge Gain Ablation",
        "",
        "## Protocol",
        "",
        "- Fixed existing E0 online evidence artifacts.",
        "- Fixed deterministic prediction table; no training or test-time tuning.",
        "- Seven arms: " + ", ".join(ABLATION_ARMS) + ".",
        "",
        "## Metrics",
        "",
        "| Arm | Accuracy | Macro F1 | Wrong Auto Accept | Unknown False Accept | HUMAN_REVIEW | KG Recoveries | KG Wrong Overrides | Mean Path Completeness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ABLATION_ARMS:
        metrics = ablation["arm_metrics"][arm]
        lines.append(
            f"| {arm} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['wrong_auto_accept_rate']:.4f} | {metrics['unknown_fault_false_accept_rate']:.4f} | "
            f"{metrics['human_review_trigger_rate']:.4f} | {metrics['kg_recovered_wrong_rank_count']} | "
            f"{metrics['kg_wrong_override_correct_model_count']} | "
            f"{_metric_value(metrics['mean_evidence_path_completeness'])} |"
        )
    lines.extend(
        [
            "",
            "## KG Branch Diagnostics",
            "",
            f"- Forward/backward Spearman: {_metric_value(ablation['kg_branch_diagnostics']['mean_forward_backward_spearman'])}",
            f"- Forward/backward Top-1 overlap: {_metric_value(ablation['kg_branch_diagnostics']['mean_forward_backward_top1_overlap'])}",
            f"- Forward/backward Top-3 overlap: {_metric_value(ablation['kg_branch_diagnostics']['mean_forward_backward_top3_overlap'])}",
            f"- Forward/backward evidence overlap: {_metric_value(ablation['kg_branch_diagnostics']['mean_forward_backward_evidence_overlap'])}",
            "",
            "## Boundary",
            "",
            ablation["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _metric_value(value: Any) -> str:
    if value is None:
        return "NOT_APPLICABLE"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _implementation_markdown(
    ablation: dict[str, Any],
    failure_boundary: dict[str, Any],
    fallback: dict[str, Any],
) -> str:
    safety = ablation["arm_metrics"]["full_safety_chain"]
    birrf = ablation["arm_metrics"]["full_birrf"]
    lines = [
        "# AstroGraph V0.5.30 Implementation Report",
        "",
        "## Scope",
        "",
        "V0.5.30-R1 implements knowledge-graph gain and failure-boundary validation only.",
        "",
        "## Acceptance Summary",
        "",
        f"- Cases: {ablation['case_count']}",
        f"- Ablation rows: {ablation['row_count']}",
        f"- Conditions covered: {', '.join(ablation['covered_conditions'])}",
        f"- Full BiRRF accuracy: {birrf['accuracy']:.4f}",
        f"- Full safety-chain wrong auto-accept rate: {safety['wrong_auto_accept_rate']:.4f}",
        f"- Full safety-chain unknown false-accept rate: {safety['unknown_fault_false_accept_rate']:.4f}",
        f"- Full safety-chain HUMAN_REVIEW trigger rate: {safety['human_review_trigger_rate']:.4f}",
        f"- Forward/backward Top-1 overlap: {_metric_value(ablation['kg_branch_diagnostics']['mean_forward_backward_top1_overlap'])}",
        f"- Forward/backward Spearman: {_metric_value(ablation['kg_branch_diagnostics']['mean_forward_backward_spearman'])}",
        f"- Neo4j live available: {fallback['neo4j']['available']}",
        f"- Live/fallback selected path: {fallback['fallback_result']}",
        f"- Neo4j-unavailable fallback result: {fallback['neo4j_unavailable_fallback']['result']}",
        "",
        "## Failure Boundaries",
        "",
        "| Condition | Cases | Safety HUMAN_REVIEW | Wrong Auto Accept | Unknown False Accept |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in failure_boundary["conditions"]:
        lines.append(
            f"| {row['condition']} | {row['case_count']} | {row['safety_human_review_count']} | "
            f"{row['safety_wrong_auto_accept_count']} | {row['safety_unknown_false_accept_count']} |"
        )
    lines.extend(
        [
        "",
        "## Methodology Corrections",
        "",
        "- R1 safety-chain execution calls DiagnosticReasonerAgent, SafetyCriticAgent, Independent Core, DecisionGovernor, and DiagnosisPolicyGate.",
        "- R1 safety-chain execution does not read truth_fault_id or is_unknown; truth is used only after decisions for metric scoring.",
        "- The pseudo-unknown heater case is retained as a negative control and is no longer claimed as a valid unknown-fault rejection test.",
        "- Data-only evidence-path completeness is NOT_APPLICABLE because that arm does not use KG evidence paths.",
        "",
        "## Claim Boundary",
            "",
            ablation["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)
