from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evaluation.llm_method_campaign import paired_arm_effect


class LLMMethodAblationRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    case_id: str
    dataset_id: str
    group_id: str
    arm_id: str
    true_fault_id: str
    predicted_fault_id: str | None = None
    abstained: bool
    latency_s: float = Field(ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    valid_tool_call_count: int = Field(default=0, ge=0)
    rejected_tool_call_count: int = Field(default=0, ge=0)
    observation_conditioned_replan_count: int = Field(default=0, ge=0)
    evidence_citation_valid: bool = True
    substance_status: str = "NOT_EXECUTED"

    # V2 contribution metrology. ``None`` means the arm did not expose an LLM
    # compact-candidate set (for example A0 deterministic-only), not that recall failed.
    true_fault_in_candidate_set: bool | None = None
    probe_execution_count: int = Field(default=0, ge=0)
    new_observation_count: int = Field(default=0, ge=0)
    new_raw_observation_count: int = Field(default=0, ge=0)
    formal_evidence_generated_count: int = Field(default=0, ge=0)
    resolved_gap_count: int = Field(default=0, ge=0)
    diagnostic_contribution_status: str = "NOT_EXECUTED"
    shadow_revision_available: bool = False
    shadow_candidate_set_changed: bool = False
    shadow_abstain_recommended: bool | None = None
    active_telemetry_attributable: bool = False
    shadow_revision_reason_code: str = "NOT_EXECUTED"
    true_fault_in_shadow_candidate_set: bool | None = None
    active_probe_available_count: int = Field(default=0, ge=0)
    active_probe_admission_status: str = "NOT_CONFIGURED"
    corrected_case: bool = False
    induced_error: bool = False

    @model_validator(mode="after")
    def abstention_prediction_consistency(self) -> "LLMMethodAblationRow":
        if self.abstained == (self.predicted_fault_id is not None):
            raise ValueError("abstained must be true exactly when predicted_fault_id is absent")
        if self.corrected_case and self.induced_error:
            raise ValueError("a row cannot be both corrected_case and induced_error")
        return self


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else round(float(numerator) / float(denominator), 6)


def _macro_metrics(rows: list[LLMMethodAblationRow]) -> tuple[float, float]:
    labels = sorted({item.true_fault_id for item in rows})
    f1_values: list[float] = []
    recalls: list[float] = []
    for label in labels:
        tp = sum(item.true_fault_id == label and item.predicted_fault_id == label for item in rows)
        fp = sum(item.true_fault_id != label and item.predicted_fault_id == label for item in rows)
        fn = sum(item.true_fault_id == label and item.predicted_fault_id != label for item in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls.append(recall)
    return (
        round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
    )


def annotate_net_diagnostic_effect(
    rows: list[LLMMethodAblationRow],
    *,
    baseline_arm_id: str = "A0_ALGORITHM_ONLY",
) -> list[LLMMethodAblationRow]:
    """Annotate paired corrections and regressions against one immutable baseline arm.

    A correction means the baseline did not produce the ground-truth fault while the
    compared arm did. An induced error means the baseline was correct and the compared
    arm was not (including abstention). This is paired metrology only; it is not a
    statistical superiority claim.
    """

    baseline: dict[str, LLMMethodAblationRow] = {}
    for row in rows:
        if row.arm_id != baseline_arm_id:
            continue
        if row.case_id in baseline:
            raise ValueError("duplicate baseline case")
        baseline[row.case_id] = row
    if not baseline:
        raise ValueError(f"baseline arm not found: {baseline_arm_id}")

    annotated: list[LLMMethodAblationRow] = []
    for row in rows:
        reference = baseline.get(row.case_id)
        if reference is None:
            raise ValueError(f"baseline case missing for {row.case_id}")
        baseline_correct = reference.predicted_fault_id == reference.true_fault_id
        row_correct = row.predicted_fault_id == row.true_fault_id
        corrected = row.arm_id != baseline_arm_id and (not baseline_correct and row_correct)
        induced = row.arm_id != baseline_arm_id and (baseline_correct and not row_correct)
        annotated.append(row.model_copy(update={"corrected_case": corrected, "induced_error": induced}))
    return annotated


def summarize_llm_method_ablation(
    rows: list[LLMMethodAblationRow],
    *,
    expected_arm_ids: list[str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("ablation rows cannot be empty")
    if not expected_arm_ids or len(expected_arm_ids) != len(set(expected_arm_ids)):
        raise ValueError("expected arm IDs must be nonempty and unique")
    # Each case has one immutable dataset/group/label identity across all arms.
    # A repeated easy case must not inflate an arm's denominator or accuracy.
    seen: set[tuple[str, str]] = set()
    identities: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if not all(value.strip() for value in (row.case_id, row.dataset_id, row.group_id, row.arm_id, row.true_fault_id)):
            raise ValueError("ablation identifiers must be nonblank")
        key = (row.arm_id, row.case_id)
        if key in seen:
            raise ValueError("duplicate case in an ablation arm")
        seen.add(key)
        identity = (row.dataset_id, row.group_id, row.true_fault_id)
        previous = identities.setdefault(row.case_id, identity)
        if previous != identity:
            raise ValueError("case dataset/group/ground-truth identity differs between arms")
    by_arm: dict[str, list[LLMMethodAblationRow]] = defaultdict(list)
    for row in rows:
        by_arm[row.arm_id].append(row)
    if set(by_arm) != set(expected_arm_ids):
        raise ValueError("ablation arms do not match preregistered arm IDs")
    case_sets = {arm: {item.case_id for item in items} for arm, items in by_arm.items()}
    if len({frozenset(value) for value in case_sets.values()}) != 1:
        raise ValueError("every ablation arm must evaluate the same case IDs")

    arm_metrics: dict[str, Any] = {}
    for arm in expected_arm_ids:
        items = by_arm[arm]
        macro_f1, balanced_accuracy = _macro_metrics(items)
        correct = sum(item.predicted_fault_id == item.true_fault_id for item in items)
        non_abstained = [item for item in items if not item.abstained]
        incorrect_non_abstained = sum(
            item.predicted_fault_id != item.true_fault_id for item in non_abstained
        )
        valid_calls = sum(item.valid_tool_call_count for item in items)
        rejected_calls = sum(item.rejected_tool_call_count for item in items)
        candidate_measured = [item for item in items if item.true_fault_in_candidate_set is not None]
        shadow_measured = [item for item in items if item.true_fault_in_shadow_candidate_set is not None]
        contribution_counts: dict[str, int] = {}
        for item in items:
            contribution_counts[item.diagnostic_contribution_status] = (
                contribution_counts.get(item.diagnostic_contribution_status, 0) + 1
            )
        corrected = sum(item.corrected_case for item in items)
        induced = sum(item.induced_error for item in items)
        arm_metrics[arm] = {
            "case_count": len(items),
            "accuracy": _safe_ratio(correct, len(items)),
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "coverage": _safe_ratio(len(non_abstained), len(items)),
            "selective_risk": _safe_ratio(incorrect_non_abstained, len(non_abstained)),
            "failure_to_abstain_rate": _safe_ratio(incorrect_non_abstained, len(items)),
            "mean_latency_s": round(sum(item.latency_s for item in items) / len(items), 6),
            "prompt_tokens": sum(item.prompt_tokens for item in items),
            "completion_tokens": sum(item.completion_tokens for item in items),
            "valid_tool_call_rate": _safe_ratio(valid_calls, valid_calls + rejected_calls),
            "observation_conditioned_replan_count": sum(
                item.observation_conditioned_replan_count for item in items
            ),
            "evidence_citation_valid_rate": _safe_ratio(
                sum(item.evidence_citation_valid for item in items), len(items)
            ),
            "substantive_case_count": sum(
                item.substance_status == "SUBSTANTIVE_METHOD_PATH" for item in items
            ),
            "candidate_recall_case_count": len(candidate_measured),
            "true_fault_in_candidate_set_rate": _safe_ratio(
                sum(bool(item.true_fault_in_candidate_set) for item in candidate_measured),
                len(candidate_measured),
            ),
            "probe_execution_count": sum(item.probe_execution_count for item in items),
            "probe_execution_case_count": sum(item.probe_execution_count > 0 for item in items),
            "new_observation_count": sum(item.new_observation_count for item in items),
            "new_raw_observation_count": sum(item.new_raw_observation_count for item in items),
            "shadow_revision_available_case_count": sum(item.shadow_revision_available for item in items),
            "shadow_candidate_set_changed_case_count": sum(item.shadow_candidate_set_changed for item in items),
            "active_telemetry_attributable_case_count": sum(item.active_telemetry_attributable for item in items),
            "active_probe_available_case_count": sum(item.active_probe_available_count > 0 for item in items),
            "shadow_abstain_recommendation_count": sum(item.shadow_abstain_recommended is True for item in items),
            "shadow_candidate_recall_case_count": len(shadow_measured),
            "true_fault_in_shadow_candidate_set_rate": _safe_ratio(
                sum(bool(item.true_fault_in_shadow_candidate_set) for item in shadow_measured),
                len(shadow_measured),
            ),
            "formal_evidence_generated_count": sum(item.formal_evidence_generated_count for item in items),
            "resolved_gap_count": sum(item.resolved_gap_count for item in items),
            "diagnostic_contribution_status_counts": contribution_counts,
            "corrected_case_count": corrected,
            "induced_error_count": induced,
            "net_diagnostic_gain": corrected - induced,
        }
    paired_comparisons = {}
    baseline = expected_arm_ids[0]
    for challenger in expected_arm_ids[1:]:
        key = f"{baseline}__vs__{challenger}"
        paired_comparisons[key] = paired_arm_effect(
            rows, baseline_arm_id=baseline, challenger_arm_id=challenger
        )
    # Probe promotion is a distinct estimand: compare the full controlled arm
    # directly against the otherwise-identical arm with active Probe disabled.
    # It must never be inferred from the A2-vs-A0 whole-stack comparison.
    if "A7_NO_ACTIVE_PROBE" in by_arm and "A2_CONTROLLED_FULL" in by_arm:
        probe_key = "A7_NO_ACTIVE_PROBE__vs__A2_CONTROLLED_FULL"
        paired_comparisons[probe_key] = paired_arm_effect(
            rows, baseline_arm_id="A7_NO_ACTIVE_PROBE", challenger_arm_id="A2_CONTROLLED_FULL"
        )
    if "A8_ACTIVE_TELEMETRY" in by_arm and "A2_CONTROLLED_FULL" in by_arm:
        active_key = "A2_CONTROLLED_FULL__vs__A8_ACTIVE_TELEMETRY"
        active_effect = paired_arm_effect(
            rows, baseline_arm_id="A2_CONTROLLED_FULL", challenger_arm_id="A8_ACTIVE_TELEMETRY"
        )
        exposed = sum(item.new_raw_observation_count > 0 for item in by_arm["A8_ACTIVE_TELEMETRY"])
        paired_comparisons[active_key] = {
            **active_effect,
            "mean_latency_delta_s": round(
                arm_metrics["A8_ACTIVE_TELEMETRY"]["mean_latency_s"]
                - arm_metrics["A2_CONTROLLED_FULL"]["mean_latency_s"], 6
            ),
            "prompt_token_delta": (
                arm_metrics["A8_ACTIVE_TELEMETRY"]["prompt_tokens"]
                - arm_metrics["A2_CONTROLLED_FULL"]["prompt_tokens"]
            ),
            "completion_token_delta": (
                arm_metrics["A8_ACTIVE_TELEMETRY"]["completion_tokens"]
                - arm_metrics["A2_CONTROLLED_FULL"]["completion_tokens"]
            ),
            "valid_tool_call_delta": (
                sum(item.valid_tool_call_count for item in by_arm["A8_ACTIVE_TELEMETRY"])
                - sum(item.valid_tool_call_count for item in by_arm["A2_CONTROLLED_FULL"])
            ),
            "active_telemetry_exposed_case_count": exposed,
            "active_telemetry_attributable_case_count": sum(
                item.active_telemetry_attributable for item in by_arm["A8_ACTIVE_TELEMETRY"]
            ),
            "shadow_candidate_recall_delta": (
                round(
                    arm_metrics["A8_ACTIVE_TELEMETRY"]["true_fault_in_shadow_candidate_set_rate"]
                    - arm_metrics["A2_CONTROLLED_FULL"]["true_fault_in_shadow_candidate_set_rate"], 6
                )
                if all(
                    arm_metrics[arm]["true_fault_in_shadow_candidate_set_rate"] is not None
                    for arm in ("A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY")
                ) else None
            ),
            "effect_identifiability": (
                "DESCRIPTIVE_ACTIVE_EXPOSURE_ONLY" if exposed else "NO_ACTIVE_TELEMETRY_EXPOSURE"
            ),
        }
    return {
        "schema_version": "astrograph-llm-method-ablation-report.v2",
        "arm_order": expected_arm_ids,
        "case_ids": sorted(next(iter(case_sets.values()))),
        "rows": [item.model_dump(mode="json") for item in rows],
        "arm_metrics": arm_metrics,
        "paired_comparisons": paired_comparisons,
        "metric_semantics": {
            "true_fault_in_candidate_set_rate": (
                "Measured only where the LLM backend recorded its compact candidate set; null-denominator arms are reported as null."
            ),
            "net_diagnostic_gain": (
                "Paired corrected_case_count minus induced_error_count relative to a frozen baseline; not a statistical superiority claim."
            ),
            "new_raw_observation_count": (
                "Counts newly acquired/derived-from-admitted-raw-input observations, not re-statements of existing formal evidence."
            ),
        },
        "claim_boundary": (
            "Same-case research ablation. Structural, contribution, and paired net-gain metrics do not establish statistical "
            "superiority, hard-real-time behavior, safety, HIL, flight, or cross-domain generalization."
        ),
    }


__all__ = [
    "LLMMethodAblationRow",
    "annotate_net_diagnostic_effect",
    "summarize_llm_method_ablation",
]
