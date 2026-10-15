"""Evidence-definition suggestions from high-confidence diagnosis episodes.

Astrograph AI-Native roadmap Phase 4 (data-knowledge double loop): a
deterministic generator that turns episode evidence into *suggestions* for new
or recalibrated EvidenceDefinition entries. Outputs are always
READY_FOR_HUMAN_REVIEW with publication_allowed=False; they never modify the
active runtime-evidence configuration or the Active KG.

Two suggestion kinds:
- THRESHOLD_RECALIBRATION: a registered evidence definition has no satisfied
  fact, yet the observed feature value exceeds a derived observation band.
- NEW_EVIDENCE_DEFINITION: a feature shows high observed activity but has no
  evidence definition referencing it (candidate for an unregistered anomaly).
"""

from __future__ import annotations

import statistics
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus
from astrograph.evidence.executable import EvidenceRole, stable_id


class SuggestionKind(str, Enum):
    THRESHOLD_RECALIBRATION = "THRESHOLD_RECALIBRATION"
    NEW_EVIDENCE_DEFINITION = "NEW_EVIDENCE_DEFINITION"


class EvidenceDefinitionSuggestion(BaseModel):
    """One governed, human-review-only evidence-definition suggestion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-definition-suggestion.v1"
    suggestion_id: str = Field(min_length=1)
    kind: SuggestionKind
    episode_id: str = Field(min_length=1)
    target_fault_id: str | None = None
    feature_definition_id: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    role: str = Field(default="SUPPORTING")
    evidence_level: str = Field(default="E0")
    suggested_threshold: float | None = None
    threshold_basis: str = ""
    observed_value: float | None = None
    observed_values: list[float] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    review_status: str = "READY_FOR_HUMAN_REVIEW"
    publication_allowed: bool = False
    active_config_modified: bool = False
    source_hashes: list[str] = Field(default_factory=list)


class EvidenceSuggestionReport(BaseModel):
    """Deterministic suggestion report for one episode."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-suggestion-report.v1"
    report_id: str = Field(min_length=1)
    source_episode_id: str = Field(min_length=1)
    suggestions: list[EvidenceDefinitionSuggestion] = Field(default_factory=list)
    claim_boundary: str = Field(
        default="Suggestions are machine drafts requiring human review; they never modify the active configuration."
    )


def _observed_values(feature_id: str, bundle: EpisodeEvidenceBundle) -> list[float]:
    return [
        float(result.value)
        for result in bundle.feature_results
        if result.feature_definition_id == feature_id and result.value is not None
    ]


def suggest_evidence_definitions(
    bundle: EpisodeEvidenceBundle,
    *,
    activity_multiplier: float = 3.0,
    minimum_activity: float = 1e-6,
) -> EvidenceSuggestionReport:
    """Generate governed evidence-definition suggestions for one episode.

    Heuristics (deterministic, observation-derived):
    - For each evidence definition whose facts are all NOT_SATISFIED or MISSING,
      if the observed feature activity exceeds ``activity_multiplier`` times the
      feature scale, suggest a threshold recalibration.
    - For each feature without any referencing evidence definition, if its
      observed activity exceeds ``minimum_activity`` and the feature is not a
      status/validity flag, suggest a new evidence definition.
    """
    feature_names = {item.feature_definition_id: item.name for item in bundle.feature_definitions}
    facts_by_definition: dict[str, list] = {}
    for fact in bundle.evidence_facts:
        facts_by_definition.setdefault(fact.evidence_definition_id, []).append(fact)
    referenced_features = {item.feature_definition_id for item in bundle.evidence_definitions}
    suggestions: list[EvidenceDefinitionSuggestion] = []

    # Kind A: threshold recalibration for registered definitions without a satisfied fact.
    for definition in bundle.evidence_definitions:
        facts = facts_by_definition.get(definition.evidence_definition_id, [])
        satisfied = any(fact.status == EvidenceFactStatus.SATISFIED for fact in facts)
        if satisfied:
            continue
        values = _observed_values(definition.feature_definition_id, bundle)
        if not values:
            continue
        magnitude = max(abs(value) for value in values)
        if magnitude < minimum_activity:
            continue
        # Derived threshold: activity_multiplier * feature scale (max of mean abs / std).
        mean_abs = statistics.fmean(abs(value) for value in values)
        scale = statistics.pstdev(values) if len(values) > 1 else mean_abs
        scale = scale if scale > 0.0 else mean_abs
        suggested = max(minimum_activity, activity_multiplier * scale)
        suggestions.append(
            EvidenceDefinitionSuggestion(
                suggestion_id=stable_id(
                    "SUG",
                    {"episode": bundle.episode.episode_id,
                     "kind": SuggestionKind.THRESHOLD_RECALIBRATION.value,
                     "definition": definition.evidence_definition_id},
                ),
                kind=SuggestionKind.THRESHOLD_RECALIBRATION,
                episode_id=bundle.episode.episode_id,
                target_fault_id=definition.target_fault_id,
                feature_definition_id=definition.feature_definition_id,
                feature_name=feature_names.get(definition.feature_definition_id, definition.feature_definition_id),
                role=definition.role.value,
                evidence_level=definition.evidence_level,
                suggested_threshold=round(suggested, 6),
                threshold_basis=f"activity_multiplier={activity_multiplier} x observed scale on unsatisfied definition",
                observed_value=max(values),
                observed_values=values,
                confidence=0.5,
                rationale=(
                    f"Registered definition {definition.evidence_definition_id} for {definition.target_fault_id} "
                    f"has no satisfied fact yet observed activity is {magnitude:g}; suggest threshold recalibration."
                ),
                source_hashes=[result.input_data_sha256 for result in bundle.feature_results],
            )
        )

    # Kind B: high-activity feature with no referencing evidence definition.
    for feature in bundle.feature_definitions:
        if feature.feature_definition_id in referenced_features:
            continue
        if any(token in feature.name.lower() for token in ("valid", "fresh", "status")):
            continue
        values = _observed_values(feature.feature_definition_id, bundle)
        if not values:
            continue
        magnitude = max(abs(value) for value in values)
        if magnitude < minimum_activity:
            continue
        mean_abs = statistics.fmean(abs(value) for value in values)
        scale = statistics.pstdev(values) if len(values) > 1 else mean_abs
        scale = scale if scale > 0.0 else mean_abs
        suggested = max(minimum_activity, activity_multiplier * scale)
        suggestions.append(
            EvidenceDefinitionSuggestion(
                suggestion_id=stable_id(
                    "SUG",
                    {"episode": bundle.episode.episode_id,
                     "kind": SuggestionKind.NEW_EVIDENCE_DEFINITION.value,
                     "feature": feature.feature_definition_id},
                ),
                kind=SuggestionKind.NEW_EVIDENCE_DEFINITION,
                episode_id=bundle.episode.episode_id,
                target_fault_id=None,
                feature_definition_id=feature.feature_definition_id,
                feature_name=feature.name,
                role=EvidenceRole.SUPPORTING.value,
                evidence_level="E0",
                suggested_threshold=round(suggested, 6),
                threshold_basis=f"activity_multiplier={activity_multiplier} x observed scale on unregistered feature",
                observed_value=max(values),
                observed_values=values,
                confidence=0.4,
                rationale=(
                    f"Feature {feature.name} shows observed activity {magnitude:g} but has no "
                    "evidence definition; candidate for an unregistered anomaly."
                ),
                source_hashes=[result.input_data_sha256 for result in bundle.feature_results],
            )
        )

    return EvidenceSuggestionReport(
        report_id=stable_id(
            "SUG-REPORT", {"episode": bundle.episode.episode_id, "suggestions": len(suggestions)},
        ),
        source_episode_id=bundle.episode.episode_id,
        suggestions=suggestions,
    )


__all__ = ["EvidenceDefinitionSuggestion", "EvidenceSuggestionReport", "SuggestionKind", "suggest_evidence_definitions"]
