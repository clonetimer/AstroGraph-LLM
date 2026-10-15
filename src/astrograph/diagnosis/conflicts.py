from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.schemas import DiagnosisResult, DiagnosisStatus


class ConflictClassification(str, Enum):
    NONE = "NONE"
    NO_CANDIDATE = "NO_CANDIDATE"
    MISSING_CRITICAL_EVIDENCE = "MISSING_CRITICAL_EVIDENCE"
    COUNTER_EVIDENCE_PRESENT = "COUNTER_EVIDENCE_PRESENT"
    COMPETING_CANDIDATES = "COMPETING_CANDIDATES"
    MIXED_CONFLICT = "MIXED_CONFLICT"


class CandidateConflictExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    fault_name: str
    support_score: float
    final_score: float
    penalty: float
    missing_critical_requirement_ids: list[str] = Field(default_factory=list)
    counter_evidence_rule_ids: list[str] = Field(default_factory=list)
    eligible_for_diagnosis: bool
    explanation: list[str] = Field(default_factory=list)


class DiagnosticConflictExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-diagnostic-conflict-explanation.v1"
    classification: ConflictClassification
    summary: str
    candidate_explanations: list[CandidateConflictExplanation] = Field(default_factory=list)
    competing_fault_ids: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "Deterministic explanation of configured support, missing-critical-evidence and counter-evidence rules; "
        "not a probabilistic causal explanation."
    )


def explain_diagnostic_conflicts(result: DiagnosisResult) -> DiagnosticConflictExplanation:
    candidate_explanations: list[CandidateConflictExplanation] = []
    has_missing = False
    has_counter = False
    for candidate in result.candidates:
        missing_ids = [str(item.get("requirement_id", "")) for item in candidate.missing_critical_evidence]
        counter_ids = [str(item.get("rule_id", "")) for item in candidate.counter_evidence]
        has_missing = has_missing or bool(missing_ids)
        has_counter = has_counter or bool(counter_ids)
        notes: list[str] = []
        if candidate.support_score > 0:
            notes.append(f"Configured support score before penalties: {candidate.support_score:.6f}.")
        if missing_ids:
            notes.append(f"Missing critical evidence requirements: {', '.join(missing_ids)}.")
        if counter_ids:
            notes.append(
                f"Counter-evidence rules applied: {', '.join(counter_ids)}; total configured penalty "
                f"{candidate.counter_evidence_penalty:.6f}."
            )
        if not candidate.eligible_for_diagnosis:
            notes.append("Candidate is blocked from deterministic diagnosis by configured constraints.")
        candidate_explanations.append(
            CandidateConflictExplanation(
                fault_id=candidate.fault_id,
                fault_name=candidate.fault_name,
                support_score=candidate.support_score,
                final_score=candidate.score,
                penalty=candidate.counter_evidence_penalty,
                missing_critical_requirement_ids=[item for item in missing_ids if item],
                counter_evidence_rule_ids=[item for item in counter_ids if item],
                eligible_for_diagnosis=candidate.eligible_for_diagnosis,
                explanation=notes,
            )
        )

    competing = []
    if result.status == DiagnosisStatus.AMBIGUOUS:
        eligible = [item for item in result.candidates if item.eligible_for_diagnosis]
        competing = [item.fault_id for item in eligible[:2]]

    if not result.candidates:
        classification = ConflictClassification.NO_CANDIDATE
        summary = "No candidate fault matched the supplied anomalies and active knowledge release."
    elif result.status == DiagnosisStatus.AMBIGUOUS:
        classification = ConflictClassification.MIXED_CONFLICT if (has_missing or has_counter) else ConflictClassification.COMPETING_CANDIDATES
        summary = "Multiple eligible candidates remain too close under the configured deterministic margin."
    elif has_missing and has_counter:
        classification = ConflictClassification.MIXED_CONFLICT
        summary = "The candidate set contains both missing critical evidence and configured counter-evidence."
    elif has_missing:
        classification = ConflictClassification.MISSING_CRITICAL_EVIDENCE
        summary = "At least one candidate lacks configured critical evidence."
    elif has_counter:
        classification = ConflictClassification.COUNTER_EVIDENCE_PRESENT
        summary = "Configured counter-evidence reduced one or more candidate scores."
    else:
        classification = ConflictClassification.NONE
        summary = "No configured evidence conflict was detected."

    return DiagnosticConflictExplanation(
        classification=classification,
        summary=summary,
        candidate_explanations=candidate_explanations,
        competing_fault_ids=competing,
    )
