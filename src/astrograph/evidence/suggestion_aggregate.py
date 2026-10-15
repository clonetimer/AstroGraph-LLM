from __future__ import annotations
import statistics
import uuid
from collections import defaultdict
from enum import Enum
from typing import Sequence
from pydantic import BaseModel, ConfigDict, Field
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus
from astrograph.evidence.executable import EvidenceRole, ComparisonOperator
from astrograph.evidence.suggestion import EvidenceDefinitionSuggestion, SuggestionKind


class AggregationMode(str, Enum):
    TARGET_CLASS_SEPARATION = 'TARGET_CLASS_SEPARATION'
    MAX_OBSERVED_TIMES_3 = 'MAX_OBSERVED_TIMES_3'


class AggregatedSuggestionReport(BaseModel):
    """Multi-episode aggregated evidence-definition suggestion report."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-evidence-suggestion-aggregate.v1'
    report_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    episode_count: int = Field(ge=1)
    source_episode_ids: list[str] = Field(min_length=1)
    aggregation_mode: AggregationMode
    minimum_episode_agreement: int = Field(ge=2)
    suggestions: list[EvidenceDefinitionSuggestion] = Field(default_factory=list)
    skipped_definitions: list[dict] = Field(default_factory=list)
    claim_boundary: str = Field(
        default='Aggregated suggestions are machine drafts requiring human review; they never modify the active configuration.'
    )


def _observed_values(bundle: EpisodeEvidenceBundle, feature_definition_id: str) -> list[float]:
    return [
        float(result.value)
        for result in bundle.feature_results
        if result.feature_definition_id == feature_definition_id and result.value is not None
    ]


def _quantile(values: Sequence[float], q: float) -> float:
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)
    if n == 0:
        raise ValueError('empty values')
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * q
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (pos - lower)


def _separation_threshold(target: Sequence[float], other: Sequence[float], comparison: str) -> float | None:
    """Threshold separating target-class observations from other-class observations.

    For GREATER_OR_EQUAL (feature rises in the target fault), the threshold must
    sit below the target-class lower tail and above the other-class upper tail.
    For LESS_OR_EQUAL (feature stays low in the target fault), the threshold
    must sit above the target-class upper tail and below the other-class lower
    tail. Returns None when distributions overlap (no single-value split), in
    which case a recalibration would not be separable and is not suggested.
    """
    if not target or not other:
        return None
    if comparison == 'GREATER_OR_EQUAL':
        # Target fires when the feature is HIGH; other classes stay LOW.
        # The threshold must sit below the target lower tail and above the
        # other upper tail.
        target_p10 = _quantile(target, 0.10)
        other_p90 = _quantile(other, 0.90)
        if target_p10 <= other_p90:
            return None
        return round(0.5 * (target_p10 + other_p90), 6)
    if comparison == 'LESS_OR_EQUAL':
        # Target fires when the feature stays LOW; other classes go HIGH.
        # The threshold must sit above the target upper tail and below the
        # other lower tail.
        target_p90 = _quantile(target, 0.90)
        other_p10 = _quantile(other, 0.10)
        if target_p90 >= other_p10:
            return None
        return round(0.5 * (target_p90 + other_p10), 6)
    return None


def aggregate_suggestions(
    bundles: Sequence[EpisodeEvidenceBundle],
    *,
    dataset_id: str,
    bundle_fault_classes: Sequence[str] | None = None,
    mode: AggregationMode = AggregationMode.TARGET_CLASS_SEPARATION,
    minimum_episode_agreement: int = 3,
    activity_multiplier: float = 3.0,
    minimum_activity: float = 1e-6,
) -> AggregatedSuggestionReport:
    """Generate threshold-recalibration suggestions from multi-episode agreement.

    For each evidence definition, bundles are split into the group whose
    episode's *target fault class* equals the definition's target (the class
    the definition is meant to fire for) and the rest. A suggestion is emitted
    only when:
      1) >= minimum_episode_agreement bundles belong to the definition's target class,
      2) none of those target-class bundles had a SATISFIED fact,
      3) the definition's threshold is not currently satisfied for the target class,
      4) a separation threshold exists between target-class and other-class
         observations (distributions do not overlap), so recalibration is
         both necessary and non-harmful.
    The suggested threshold is the class-separation midpoint, not a
    single-episode point value.
    """
    if not bundles:
        raise ValueError('at least one episode bundle is required')
    if minimum_episode_agreement < 2:
        raise ValueError('minimum_episode_agreement must be >= 2')

    feature_names: dict[str, str] = {}
    # Group observations by evidence definition, then by target-class vs other.
    target_obs: defaultdict = defaultdict(list)
    other_obs: defaultdict = defaultdict(list)
    target_episodes: defaultdict = defaultdict(set)
    satisfied_target: defaultdict = defaultdict(bool)
    def_by_id: dict = {}
    if bundle_fault_classes is not None and len(bundle_fault_classes) != len(bundles):
        raise ValueError('bundle_fault_classes must match bundles length')
    for index, bundle in enumerate(bundles):
        bundle_class = None if bundle_fault_classes is None else bundle_fault_classes[index]
        for item in bundle.feature_definitions:
            feature_names.setdefault(item.feature_definition_id, item.name)
        for definition in bundle.evidence_definitions:
            def_by_id.setdefault(definition.evidence_definition_id, definition)
            values = _observed_values(bundle, definition.feature_definition_id)
            if not values:
                continue
            facts = [f for f in bundle.evidence_facts if f.evidence_definition_id == definition.evidence_definition_id]
            is_target = bundle_class is not None and bundle_class == definition.target_fault_id
            bucket = target_obs if is_target else other_obs
            bucket[definition.evidence_definition_id].extend(values)
            if is_target:
                target_episodes[definition.evidence_definition_id].add(bundle.episode.episode_id)
                if any(fact.status == EvidenceFactStatus.SATISFIED for fact in facts):
                    satisfied_target[definition.evidence_definition_id] = True

    suggestions: list[EvidenceDefinitionSuggestion] = []
    skipped: list[dict] = []
    for evidence_definition_id, definition in def_by_id.items():
        target_values = target_obs.get(evidence_definition_id, [])
        other_values = other_obs.get(evidence_definition_id, [])
        agreement = len(target_episodes.get(evidence_definition_id, []))
        if satisfied_target.get(evidence_definition_id):
            skipped.append({'evidence_definition_id': evidence_definition_id, 'reason': 'satisfied_target_fact_exists'})
            continue
        if agreement < minimum_episode_agreement:
            skipped.append({'evidence_definition_id': evidence_definition_id, 'reason': f'target_agreement_{agreement}<{minimum_episode_agreement}'})
            continue
        if not target_values:
            skipped.append({'evidence_definition_id': evidence_definition_id, 'reason': 'no_target_observations'})
            continue
        peak = max(abs(v) for v in target_values)
        if peak < minimum_activity:
            skipped.append({'evidence_definition_id': evidence_definition_id, 'reason': 'below_minimum_activity'})
            continue
        comparison = definition.comparison.value if isinstance(definition.comparison, ComparisonOperator) else str(definition.comparison)
        if mode == AggregationMode.TARGET_CLASS_SEPARATION:
            suggested = _separation_threshold(target_values, other_values, comparison)
            basis = f'class-separation midpoint across {agreement} target episodes (+{len(other_values)} other-class observations)'
            if suggested is None:
                skipped.append({'evidence_definition_id': evidence_definition_id, 'reason': 'overlapping_distributions_no_separation'})
                continue
        else:
            spread = statistics.fmean(abs(v) for v in target_values)
            suggested = max(minimum_activity, activity_multiplier * spread)
            basis = f'{activity_multiplier}x mean target magnitude across {agreement} episodes'
        episode_ids = sorted(target_episodes.get(evidence_definition_id, []))
        suggestions.append(
            EvidenceDefinitionSuggestion(
                suggestion_id='SUG-AGG-' + uuid.uuid4().hex[:12].upper(),
                kind=SuggestionKind.THRESHOLD_RECALIBRATION,
                episode_id=episode_ids[0],
                target_fault_id=definition.target_fault_id,
                feature_definition_id=definition.feature_definition_id,
                feature_name=feature_names.get(definition.feature_definition_id, definition.feature_definition_id),
                role=definition.role.value,
                evidence_level=definition.evidence_level,
                suggested_threshold=round(suggested, 6),
                threshold_basis=basis,
                observed_value=round(peak, 6),
                observed_values=[round(v, 6) for v in target_values],
                confidence=round(min(0.9, 0.5 + 0.08 * (agreement - minimum_episode_agreement + 1)), 2),
                rationale=(
                    f'Across {agreement} target-class episodes of {definition.target_fault_id}, definition {evidence_definition_id} '
                    f'never satisfied while feature activity reached {peak:g}; the class-separation midpoint is a non-harmful recalibration candidate.'
                ),
                source_hashes=[r.input_data_sha256 for b in bundles for r in b.feature_results if r.feature_definition_id == definition.feature_definition_id],
            )
        )

    return AggregatedSuggestionReport(
        report_id='SUG-AGG-REPORT-' + uuid.uuid4().hex[:8].upper(),
        dataset_id=dataset_id,
        episode_count=len(bundles),
        source_episode_ids=sorted({b.episode.episode_id for b in bundles}),
        aggregation_mode=mode,
        minimum_episode_agreement=minimum_episode_agreement,
        suggestions=suggestions,
        skipped_definitions=skipped,
    )


__all__ = ['AggregatedSuggestionReport', 'AggregationMode', 'aggregate_suggestions']