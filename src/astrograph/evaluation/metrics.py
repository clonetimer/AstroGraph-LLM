from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from astrograph.investigation import HypothesisSet


class SetMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    @classmethod
    def from_sets(cls, predicted: Iterable[str], expected: Iterable[str]) -> "SetMetric":
        pred = set(predicted)
        exp = set(expected)
        tp = len(pred & exp)
        fp = len(pred - exp)
        fn = len(exp - pred)
        precision = tp / (tp + fp) if tp + fp else (1.0 if not exp else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class HypothesisEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-hypothesis-evaluation.v1"
    episode_id: str
    hypothesis_key_count: int = 0
    status_match_count: int = 0
    status_match_rate: float = 0.0
    supporting_evidence: SetMetric = Field(default_factory=SetMetric)
    contradicting_evidence: SetMetric = Field(default_factory=SetMetric)
    missing_evidence: SetMetric = Field(default_factory=SetMetric)
    no_fault_semantic_match: bool = False
    exact_structural_match: bool = False
    reasoning_summary_coverage: float = 0.0
    claim_boundary: str = (
        "Metrics compare model hypotheses against the deterministic executable-evidence reference; "
        "they do not establish flight diagnostic accuracy."
    )


def _key(item: object) -> tuple[str, str | None]:
    return (str(getattr(item, "hypothesis_type").value), getattr(item, "target_fault_id"))


def evaluate_hypothesis_set(candidate: HypothesisSet, reference: HypothesisSet) -> HypothesisEvaluation:
    cand = {_key(item): item for item in candidate.hypotheses}
    ref = {_key(item): item for item in reference.hypotheses}
    keys = sorted(set(cand) | set(ref), key=str)
    status_match = sum(
        1 for key in keys
        if key in cand and key in ref and cand[key].status == ref[key].status
    )

    pred_support = {
        f"{key}|{fact_id}"
        for key, item in cand.items()
        for fact_id in item.supporting_evidence_fact_ids
    }
    ref_support = {
        f"{key}|{fact_id}"
        for key, item in ref.items()
        for fact_id in item.supporting_evidence_fact_ids
    }
    pred_contra = {
        f"{key}|{fact_id}"
        for key, item in cand.items()
        for fact_id in item.contradicting_evidence_fact_ids
    }
    ref_contra = {
        f"{key}|{fact_id}"
        for key, item in ref.items()
        for fact_id in item.contradicting_evidence_fact_ids
    }
    pred_missing = {
        f"{key}|{definition_id}"
        for key, item in cand.items()
        for definition_id in item.missing_evidence_definition_ids
    }
    ref_missing = {
        f"{key}|{definition_id}"
        for key, item in ref.items()
        for definition_id in item.missing_evidence_definition_ids
    }
    no_fault_key = ("NO_REGISTERED_FAULT", None)
    no_fault_match = (
        no_fault_key in cand
        and no_fault_key in ref
        and cand[no_fault_key].status == ref[no_fault_key].status
    )
    reasoning_nonempty = sum(1 for item in candidate.hypotheses if item.reasoning_summary.strip())
    def structural_signature(items: dict[tuple[str, str | None], object]) -> dict:
        return {
            str(key): {
                "status": item.status.value,
                "supporting": sorted(item.supporting_evidence_fact_ids),
                "contradicting": sorted(item.contradicting_evidence_fact_ids),
                "missing": sorted(item.missing_evidence_definition_ids),
            }
            for key, item in sorted(items.items(), key=lambda pair: str(pair[0]))
        }

    exact = structural_signature(cand) == structural_signature(ref)
    return HypothesisEvaluation(
        episode_id=candidate.episode_id,
        hypothesis_key_count=len(keys),
        status_match_count=status_match,
        status_match_rate=(status_match / len(keys) if keys else 1.0),
        supporting_evidence=SetMetric.from_sets(pred_support, ref_support),
        contradicting_evidence=SetMetric.from_sets(pred_contra, ref_contra),
        missing_evidence=SetMetric.from_sets(pred_missing, ref_missing),
        no_fault_semantic_match=no_fault_match,
        exact_structural_match=exact,
        reasoning_summary_coverage=(reasoning_nonempty / len(candidate.hypotheses) if candidate.hypotheses else 1.0),
    )
