"""AI-Native Phase 4: multi-episode aggregated evidence-suggestion tests."""
from __future__ import annotations

import pytest

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.evidence.executable import (
    EpisodeWindow, EvidenceDefinition, EvidenceFact, EvidenceFactStatus, FeatureDefinition,
    FeatureResult, FeatureAggregation, EvidenceRole, ComparisonOperator,
)
from astrograph.evidence.suggestion_aggregate import (
    AggregationMode, aggregate_suggestions,
)


def _bundle(
    episode_id: str, *, value: float, target_fault_id: str, threshold: float = 0.5,
    comparison: ComparisonOperator = ComparisonOperator.GREATER_OR_EQUAL,
    satisfied: bool = False,
) -> EpisodeEvidenceBundle:
    feature_def = FeatureDefinition(
        feature_definition_id='FD-1', name='residual peak',
        source_field_template='asset.rw.residual', aggregation=FeatureAggregation.MAX_VALUE,
        unit='N*m', window_scope='full_run', algorithm_version='v1',
    )
    evidence_def = EvidenceDefinition(
        evidence_definition_id='ED-1', target_fault_id=target_fault_id,
        name='residual gate', feature_definition_id='FD-1', role=EvidenceRole.REQUIRED,
        comparison=comparison, threshold=threshold,
        threshold_unit='N*m',
    )
    result = FeatureResult(
        feature_result_id=f'FR-{episode_id}', feature_definition_id='FD-1',
        run_id='run', paired_nominal_run_id='nom', episode_id=episode_id,
        resolved_field='asset.rw.residual', component_instance='RW_1', value=value,
        unit='N*m', window_start_s=0.0, window_end_s=10.0, sample_count=100,
        algorithm_version='v1', input_data_sha256='0' * 64,
    )
    fact = EvidenceFact(
        evidence_fact_id=f'EF-{episode_id}', evidence_definition_id='ED-1',
        target_fault_id=target_fault_id, feature_result_id=f'FR-{episode_id}',
        episode_id=episode_id, role=EvidenceRole.REQUIRED,
        status=EvidenceFactStatus.SATISFIED if satisfied else EvidenceFactStatus.NOT_SATISFIED,
        observed_value=value, threshold_value=threshold, unit='N*m',
        comparison=comparison,
    )
    window = EpisodeWindow(
        episode_id=episode_id, run_id='run', paired_nominal_run_id='nom',
        start_s=0.0, end_s=10.0, source='synthetic',
    )
    return EpisodeEvidenceBundle(
        episode=window, domain_id='D3_RW', analysis_component_instance='RW_1',
        feature_definitions=[feature_def], feature_results=[result],
        evidence_definitions=[evidence_def], evidence_facts=[fact],
    )


def test_separation_emits_for_greater_or_equal():
    # Target class (FAULT_A) observes high values; other class observes low values.
    bundles = [_bundle(f'ep-{i}', value=5.0, target_fault_id='FAULT_A') for i in range(1, 4)]
    bundles += [_bundle(f'ep-{i}', value=0.1, target_fault_id='FAULT_A') for i in range(4, 8)]
    classes = ['FAULT_A'] * 3 + ['FAULT_B'] * 4
    report = aggregate_suggestions(bundles, dataset_id='adcs', bundle_fault_classes=classes, minimum_episode_agreement=3)
    assert len(report.suggestions) == 1
    item = report.suggestions[0]
    assert item.target_fault_id == 'FAULT_A'
    # Separation midpoint between FAULT_A p90 (~5) and FAULT_B p10 (~0.1).
    assert 0.1 < item.suggested_threshold < 5.0
    assert item.kind.value == 'THRESHOLD_RECALIBRATION'


def test_separation_emits_for_less_or_equal():
    # Target class stays LOW; other class is HIGH -> upper band can separate.
    bundles = [_bundle(f'ep-{i}', value=0.005, target_fault_id='FAULT_A', comparison=ComparisonOperator.LESS_OR_EQUAL, threshold=0.001) for i in range(1, 4)]
    bundles += [_bundle(f'ep-{i}', value=0.03, target_fault_id='FAULT_A', comparison=ComparisonOperator.LESS_OR_EQUAL, threshold=0.001) for i in range(4, 8)]
    classes = ['FAULT_A'] * 3 + ['FAULT_B'] * 4
    report = aggregate_suggestions(bundles, dataset_id='adcs', bundle_fault_classes=classes, minimum_episode_agreement=3)
    assert len(report.suggestions) == 1
    item = report.suggestions[0]
    assert 0.005 < item.suggested_threshold < 0.03


def test_below_agreement_skips():
    bundles = [_bundle('ep-1', value=5.0, target_fault_id='FAULT_A'), _bundle('ep-2', value=5.0, target_fault_id='FAULT_A')]
    bundles += [_bundle(f'ep-{i}', value=0.1, target_fault_id='FAULT_A') for i in range(3, 8)]
    classes = ['FAULT_A'] * 2 + ['FAULT_B'] * 5
    report = aggregate_suggestions(bundles, dataset_id='adcs', bundle_fault_classes=classes, minimum_episode_agreement=3)
    assert report.suggestions == []
    assert any('target_agreement' in item['reason'] for item in report.skipped_definitions)


def test_satisfied_target_skips():
    bundles = [_bundle(f'ep-{i}', value=5.0, target_fault_id='FAULT_A', satisfied=True) for i in range(1, 4)]
    bundles += [_bundle(f'ep-{i}', value=0.1, target_fault_id='FAULT_A') for i in range(4, 8)]
    classes = ['FAULT_A'] * 3 + ['FAULT_B'] * 4
    report = aggregate_suggestions(bundles, dataset_id='adcs', bundle_fault_classes=classes, minimum_episode_agreement=3)
    assert report.suggestions == []
    assert any(item['reason'] == 'satisfied_target_fact_exists' for item in report.skipped_definitions)


def test_overlapping_distributions_skips():
    # Both classes observe ~same values -> no separation possible.
    bundles = [_bundle(f'ep-{i}', value=0.5, target_fault_id='FAULT_A') for i in range(1, 4)]
    bundles += [_bundle(f'ep-{i}', value=0.6, target_fault_id='FAULT_A') for i in range(4, 8)]
    classes = ['FAULT_A'] * 3 + ['FAULT_B'] * 4
    report = aggregate_suggestions(bundles, dataset_id='adcs', bundle_fault_classes=classes, minimum_episode_agreement=3)
    assert report.suggestions == []
    assert any(item['reason'] == 'overlapping_distributions_no_separation' for item in report.skipped_definitions)


def test_max_observed_mode():
    bundles = [_bundle(f'ep-{i}', value=5.0, target_fault_id='FAULT_A') for i in range(1, 4)]
    bundles += [_bundle(f'ep-{i}', value=0.1, target_fault_id='FAULT_A') for i in range(4, 8)]
    classes = ['FAULT_A'] * 3 + ['FAULT_B'] * 4
    report = aggregate_suggestions(
        bundles, dataset_id='adcs', bundle_fault_classes=classes,
        mode=AggregationMode.MAX_OBSERVED_TIMES_3, minimum_episode_agreement=3,
    )
    assert len(report.suggestions) == 1
    assert report.suggestions[0].suggested_threshold > 5.0


def test_requires_nonempty():
    with pytest.raises(ValueError):
        aggregate_suggestions([], dataset_id='adcs')