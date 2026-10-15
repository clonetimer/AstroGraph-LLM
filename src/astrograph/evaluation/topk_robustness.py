from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from astrograph.llm.candidate_selection import (
    CandidateSelectionInput,
    CandidateSignal,
    resolve_candidate_top_k,
    select_candidates,
)

ACTIVE_STATUSES = {"SUPPORTED", "INSUFFICIENT_EVIDENCE", "UNDER_VERIFICATION"}
PERTURBATIONS = (
    "clean",
    "near_threshold",
    "support_contradiction_conflict",
    "critical_evidence_missing",
    "numerical_noise",
    "time_window_shift",
    "unit_state_mismatch",
)


def load_rule_hypothesis_sets(artifact_root: Path) -> list[tuple[str, dict[str, Any]]]:
    paths = sorted(artifact_root.glob("**/rule/*/diagnosis/hypothesis_set.json"))
    if not paths and artifact_root.name == "rule":
        paths = sorted(artifact_root.glob("*/diagnosis/hypothesis_set.json"))
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        scenario = path.parent.parent.name
        import json

        rows.append((scenario, json.loads(path.read_text(encoding="utf-8-sig"))))
    return rows


def signals_from_hypothesis_set(document: dict[str, Any]) -> list[CandidateSignal]:
    signals = []
    for item in document.get("hypotheses", []):
        if item.get("hypothesis_type") != "REGISTERED_FAULT" or not item.get("target_fault_id"):
            continue
        signals.append(CandidateSignal(
            fault_id=str(item["target_fault_id"]),
            status=str(item.get("status", "OUT_OF_SCOPE")),
            supporting_evidence_fact_ids=[str(v) for v in item.get("supporting_evidence_fact_ids", [])],
            contradicting_evidence_fact_ids=[str(v) for v in item.get("contradicting_evidence_fact_ids", [])],
            missing_evidence_definition_ids=[str(v) for v in item.get("missing_evidence_definition_ids", [])],
            source="DETERMINISTIC_PERTURBATION",
        ))
    return signals


def expected_targets(signals: Iterable[CandidateSignal]) -> list[str]:
    return sorted(item.fault_id for item in signals if item.status in ACTIVE_STATUSES)


def perturb_signals(
    signals: list[CandidateSignal],
    *,
    expected_fault_ids: list[str],
    perturbation: str,
) -> list[CandidateSignal]:
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation: {perturbation}")
    items = [CandidateSignal.model_validate(copy.deepcopy(item.model_dump())) for item in signals]
    targets = set(expected_fault_ids)
    distractors = [item for item in sorted(items, key=lambda x: x.fault_id) if item.fault_id not in targets]

    if perturbation == "clean" or not targets:
        return items

    for item in items:
        if item.fault_id not in targets:
            continue
        if perturbation == "near_threshold":
            item.status = "INSUFFICIENT_EVIDENCE"
            item.supporting_evidence_fact_ids = []
            item.missing_evidence_definition_ids = [f"SYNTH-NEAR-{item.fault_id}"]
        elif perturbation == "support_contradiction_conflict":
            item.contradicting_evidence_fact_ids = [f"SYNTH-CONFLICT-{item.fault_id}"]
        elif perturbation == "critical_evidence_missing":
            item.status = "INSUFFICIENT_EVIDENCE"
            item.supporting_evidence_fact_ids = []
            item.missing_evidence_definition_ids = [
                f"SYNTH-MISSING-A-{item.fault_id}",
                f"SYNTH-MISSING-B-{item.fault_id}",
            ]
        elif perturbation == "numerical_noise":
            item.status = "UNDER_VERIFICATION"
            item.supporting_evidence_fact_ids = [f"SYNTH-NOISY-{item.fault_id}"]
            item.contradicting_evidence_fact_ids = [f"SYNTH-NOISY-CON-{item.fault_id}"]
        elif perturbation == "time_window_shift":
            item.status = "UNDER_VERIFICATION"
            item.supporting_evidence_fact_ids = []
            item.missing_evidence_definition_ids = [f"SYNTH-WINDOW-{item.fault_id}"]
        elif perturbation == "unit_state_mismatch":
            item.status = "INSUFFICIENT_EVIDENCE"
            item.supporting_evidence_fact_ids = []
            item.contradicting_evidence_fact_ids = [
                f"SYNTH-UNIT-{item.fault_id}",
                f"SYNTH-STATE-{item.fault_id}",
            ]
            item.missing_evidence_definition_ids = [f"SYNTH-SEMANTICS-{item.fault_id}"]

    # Deterministic distractors emulate conflicting retrieval/evidence without inventing
    # a new formal fault. They remain inside the registered whitelist.
    if perturbation == "support_contradiction_conflict":
        for index, item in enumerate(distractors[:3]):
            item.status = "SUPPORTED"
            item.supporting_evidence_fact_ids = [
                f"SYNTH-DISTRACTOR-{index}-A",
                f"SYNTH-DISTRACTOR-{index}-B",
            ]
            item.contradicting_evidence_fact_ids = [f"SYNTH-DISTRACTOR-CON-{index}"]
    elif perturbation in {"numerical_noise", "time_window_shift"}:
        for index, item in enumerate(distractors[:4]):
            item.status = "UNDER_VERIFICATION"
            item.supporting_evidence_fact_ids = [
                f"SYNTH-DISTRACTOR-{index}-A",
                f"SYNTH-DISTRACTOR-{index}-B",
            ]
            item.contradicting_evidence_fact_ids = []
    elif perturbation == "unit_state_mismatch":
        for index, item in enumerate(distractors[:2]):
            item.status = "INSUFFICIENT_EVIDENCE"
            item.supporting_evidence_fact_ids = [f"SYNTH-DISTRACTOR-{index}"]
            item.missing_evidence_definition_ids = [f"SYNTH-DISTRACTOR-MISSING-{index}"]
    return items


def run_topk_robustness(
    documents: list[tuple[str, dict[str, Any]]],
    *,
    top_k_values: Iterable[int] = (2, 4, 6),
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for scenario, document in documents:
        base = signals_from_hypothesis_set(document)
        registered = sorted(item.fault_id for item in base)
        expected = expected_targets(base)
        for perturbation in PERTURBATIONS:
            perturbed = perturb_signals(
                base,
                expected_fault_ids=expected,
                perturbation=perturbation,
            )
            for top_k in top_k_values:
                result = select_candidates(CandidateSelectionInput(
                    registered_fault_ids=registered,
                    candidates=perturbed,
                    top_k=top_k,
                ))
                selected = set(result.selected_fault_ids)
                hits = len(selected.intersection(expected))
                details.append({
                    "scenario": scenario,
                    "perturbation": perturbation,
                    "top_k": top_k,
                    "expected_fault_ids": expected,
                    "selected_fault_ids": result.selected_fault_ids,
                    "target_member_count": len(expected),
                    "target_member_hits": hits,
                    "all_targets_recalled": hits == len(expected),
                    "nominal_short_circuit": result.nominal_short_circuit,
                    "forced_candidate_count": result.forced_candidate_count,
                })

    # One explicit unregistered-anomaly/open-set proxy based on the nominal registered
    # catalogue. It evaluates safe refusal only; it is not a formal unknown-fault claim.
    nominal = next(((scenario, doc) for scenario, doc in documents if scenario == "nominal"), None)
    if nominal:
        scenario, document = nominal
        base = signals_from_hypothesis_set(document)
        registered = sorted(item.fault_id for item in base)
        for top_k in top_k_values:
            result = select_candidates(CandidateSelectionInput(
                registered_fault_ids=registered,
                candidates=base,
                top_k=top_k,
            ))
            details.append({
                "scenario": "synthetic_unregistered_anomaly",
                "perturbation": "unregistered_anomaly_proxy",
                "top_k": top_k,
                "expected_fault_ids": [],
                "selected_fault_ids": result.selected_fault_ids,
                "target_member_count": 0,
                "target_member_hits": 0,
                "all_targets_recalled": len(result.selected_fault_ids) == 0,
                "nominal_short_circuit": result.nominal_short_circuit,
                "forced_candidate_count": result.forced_candidate_count,
            })

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[int(row["top_k"])].append(row)
    summary = []
    for top_k in sorted(grouped):
        rows = grouped[top_k]
        target_rows = [row for row in rows if row["target_member_count"] > 0]
        member_total = sum(row["target_member_count"] for row in target_rows)
        member_hits = sum(row["target_member_hits"] for row in target_rows)
        summary.append({
            "top_k": top_k,
            "case_count": len(rows),
            "target_case_count": len(target_rows),
            "member_recall": round(member_hits / member_total, 6) if member_total else 1.0,
            "all_targets_case_recall": round(
                sum(bool(row["all_targets_recalled"]) for row in target_rows) / len(target_rows), 6
            ) if target_rows else 1.0,
            "safe_empty_unknown_proxy": all(
                not row["selected_fault_ids"]
                for row in rows
                if row["perturbation"] == "unregistered_anomaly_proxy"
            ),
        })
    return {
        "schema_version": "astrograph-topk-robustness-report.v1",
        "benchmark_scope": "deterministic candidate-layer perturbation; not an online Qwen accuracy claim",
        "source_scenario_count": len(documents),
        "perturbations": list(PERTURBATIONS) + ["unregistered_anomaly_proxy"],
        "summary": summary,
        "details": details,
    }

def run_adaptive_topk_robustness(
    documents: list[tuple[str, dict[str, Any]]],
    *,
    min_top_k: int = 2,
    max_top_k: int = 6,
) -> dict[str, Any]:
    """Evaluate the V0.5.11 adaptive forced-candidate-count policy."""

    details: list[dict[str, Any]] = []
    for scenario, document in documents:
        base = signals_from_hypothesis_set(document)
        registered = sorted(item.fault_id for item in base)
        expected = expected_targets(base)
        for perturbation in PERTURBATIONS:
            perturbed = perturb_signals(
                base,
                expected_fault_ids=expected,
                perturbation=perturbation,
            )
            effective_top_k = resolve_candidate_top_k(
                perturbed,
                mode="adaptive_forced_count",
                configured_top_k=min_top_k,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
            )
            result = select_candidates(CandidateSelectionInput(
                registered_fault_ids=registered,
                candidates=perturbed,
                top_k=effective_top_k,
            ))
            selected = set(result.selected_fault_ids)
            hits = len(selected.intersection(expected))
            details.append({
                "scenario": scenario,
                "perturbation": perturbation,
                "selection_mode": "adaptive_forced_count",
                "effective_top_k": effective_top_k,
                "expected_fault_ids": expected,
                "selected_fault_ids": result.selected_fault_ids,
                "target_member_count": len(expected),
                "target_member_hits": hits,
                "all_targets_recalled": hits == len(expected),
                "nominal_short_circuit": result.nominal_short_circuit,
                "forced_candidate_count": result.forced_candidate_count,
            })

    nominal = next(((scenario, doc) for scenario, doc in documents if scenario == "nominal"), None)
    if nominal:
        _, document = nominal
        base = signals_from_hypothesis_set(document)
        registered = sorted(item.fault_id for item in base)
        effective_top_k = resolve_candidate_top_k(
            base,
            mode="adaptive_forced_count",
            configured_top_k=min_top_k,
            min_top_k=min_top_k,
            max_top_k=max_top_k,
        )
        result = select_candidates(CandidateSelectionInput(
            registered_fault_ids=registered,
            candidates=base,
            top_k=effective_top_k,
        ))
        details.append({
            "scenario": "synthetic_unregistered_anomaly",
            "perturbation": "unregistered_anomaly_proxy",
            "selection_mode": "adaptive_forced_count",
            "effective_top_k": effective_top_k,
            "expected_fault_ids": [],
            "selected_fault_ids": result.selected_fault_ids,
            "target_member_count": 0,
            "target_member_hits": 0,
            "all_targets_recalled": len(result.selected_fault_ids) == 0,
            "nominal_short_circuit": result.nominal_short_circuit,
            "forced_candidate_count": result.forced_candidate_count,
        })

    target_rows = [row for row in details if row["target_member_count"] > 0]
    member_total = sum(row["target_member_count"] for row in target_rows)
    member_hits = sum(row["target_member_hits"] for row in target_rows)
    top_k_distribution: dict[str, int] = {}
    for row in details:
        key = str(row["effective_top_k"])
        top_k_distribution[key] = top_k_distribution.get(key, 0) + 1
    summary = {
        "selection_mode": "adaptive_forced_count",
        "min_top_k": min_top_k,
        "max_top_k": max_top_k,
        "case_count": len(details),
        "target_case_count": len(target_rows),
        "member_recall": round(member_hits / member_total, 6) if member_total else 1.0,
        "all_targets_case_recall": round(
            sum(bool(row["all_targets_recalled"]) for row in target_rows) / len(target_rows), 6
        ) if target_rows else 1.0,
        "safe_empty_unknown_proxy": all(
            not row["selected_fault_ids"]
            for row in details
            if row["perturbation"] == "unregistered_anomaly_proxy"
        ),
        "effective_top_k_distribution": top_k_distribution,
    }
    return {
        "schema_version": "astrograph-adaptive-topk-robustness-report.v1",
        "benchmark_scope": "deterministic candidate-layer perturbation; not an online Qwen accuracy claim",
        "source_scenario_count": len(documents),
        "perturbations": list(PERTURBATIONS) + ["unregistered_anomaly_proxy"],
        "summary": summary,
        "details": details,
    }

