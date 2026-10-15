"""Evidence conjunctive (AND) condition tests for GYRO/RW disambiguation support."""
from __future__ import annotations

import pytest

from astrograph.evidence.executable import (
    ComparisonOperator, ConjunctiveFeatureCondition, EpisodeEvidenceBundle, EpisodeWindow,
    EvidenceDefinition, EvidenceFactStatus, FeatureResult, evaluate_evidence,
)
from astrograph.evidence.executable import EvidenceRole, FeatureAggregation, FeatureDefinition
from astrograph.services import DeterministicReferenceReasoner, SafetyValidationService


def _feature_result(feature_definition_id: str, value: float, episode_id: str = 'ep-1') -> FeatureResult:
    return FeatureResult(
        feature_result_id=f'FR-{feature_definition_id}',
        feature_definition_id=feature_definition_id,
        run_id='run', paired_nominal_run_id='nom', episode_id=episode_id,
        resolved_field='x', component_instance='RW_1', value=value,
        unit='1', window_start_s=0.0, window_end_s=10.0, sample_count=100,
        algorithm_version='v1', input_data_sha256='0' * 64,
    )


def _definition(*, with_conjunct: bool = False) -> EvidenceDefinition:
    conjuncts = []
    if with_conjunct:
        conjuncts.append(ConjunctiveFeatureCondition(
            feature_definition_id='FD-RW-COMMAND-RESIDUAL',
            comparison=ComparisonOperator.LESS_OR_EQUAL, threshold=0.33, threshold_unit='N*m',
        ))
    return EvidenceDefinition(
        evidence_definition_id='ED-GYRO-UPPER', target_fault_id='ADCS_GYRO_BIAS_DRIFT',
        name='gyro innovation upper', feature_definition_id='FD-STAR-GYRO-INNOVATION',
        role=EvidenceRole.REQUIRED, comparison=ComparisonOperator.LESS_OR_EQUAL,
        threshold=0.01, threshold_unit='rad/s', critical=True,
        and_feature_conditions=conjuncts,
    )


def test_single_feature_behaviour_unchanged():
    # No conjuncts: historical behaviour (innovation <= 0.01 satisfied).
    definition = _definition()
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
    )
    assert fact.status == EvidenceFactStatus.SATISFIED


def test_conjunct_satisfied_fires():
    definition = _definition(with_conjunct=True)
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
        all_feature_results=[_feature_result('FD-RW-COMMAND-RESIDUAL', 0.1)],
    )
    assert fact.status == EvidenceFactStatus.SATISFIED


def test_conjunct_not_met_blocks_firing():
    # GYRO-like innovation but RW command residual high (wheel-jam signature).
    definition = _definition(with_conjunct=True)
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
        all_feature_results=[_feature_result('FD-RW-COMMAND-RESIDUAL', 0.5)],
    )
    assert fact.status == EvidenceFactStatus.NOT_SATISFIED
    assert 'FD-RW-COMMAND-RESIDUAL' in fact.explanation
    assert 'not met' in fact.explanation


def test_conjunct_missing_is_fail_closed():
    definition = _definition(with_conjunct=True)
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
        all_feature_results=[],
    )
    assert fact.status == EvidenceFactStatus.MISSING_DATA


def test_conjunct_from_another_episode_is_not_reused():
    definition = _definition(with_conjunct=True)
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
        all_feature_results=[
            _feature_result('FD-RW-COMMAND-RESIDUAL', 0.1, episode_id='other-episode')
        ],
    )
    assert fact.status == EvidenceFactStatus.MISSING_DATA


def test_conjunct_requires_main_condition_first():
    # Main condition not met (innovation too high) -> NOT_SATISFIED regardless of conjunct.
    definition = _definition(with_conjunct=True)
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.5),
        all_feature_results=[_feature_result('FD-RW-COMMAND-RESIDUAL', 0.1)],
    )
    assert fact.status == EvidenceFactStatus.NOT_SATISFIED


def test_multiple_conjuncts_all_required():
    definition = _definition(with_conjunct=True)
    definition.and_feature_conditions.append(ConjunctiveFeatureCondition(
        feature_definition_id='FD-RW-SPEED-RESIDUAL',
        comparison=ComparisonOperator.LESS_OR_EQUAL, threshold=0.2, threshold_unit='N*m',
    ))
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
        all_feature_results=[
            _feature_result('FD-RW-COMMAND-RESIDUAL', 0.1),
            _feature_result('FD-RW-SPEED-RESIDUAL', 0.15),
        ],
    )
    assert fact.status == EvidenceFactStatus.SATISFIED
    # Second conjunct violated.
    fact = evaluate_evidence(
        definition, _feature_result('FD-STAR-GYRO-INNOVATION', 0.005),
        all_feature_results=[
            _feature_result('FD-RW-COMMAND-RESIDUAL', 0.1),
            _feature_result('FD-RW-SPEED-RESIDUAL', 0.5),
        ],
    )
    assert fact.status == EvidenceFactStatus.NOT_SATISFIED


def test_safety_critic_replays_conjunctive_status_with_the_same_evaluator():
    definition = _definition(with_conjunct=True)
    primary = _feature_result('FD-STAR-GYRO-INNOVATION', 0.005).model_copy(
        update={'unit': 'rad/s'}
    )
    conjunct = _feature_result('FD-RW-COMMAND-RESIDUAL', 0.5).model_copy(
        update={'unit': 'N*m'}
    )
    fact = evaluate_evidence(
        definition, primary, all_feature_results=[primary, conjunct]
    )
    bundle = EpisodeEvidenceBundle(
        episode=EpisodeWindow(
            episode_id='ep-1', run_id='run', paired_nominal_run_id='nom',
            start_s=0.0, end_s=10.0, source='test',
        ),
        analysis_component_instance='ADCS_ESTIMATOR',
        feature_definitions=[
            FeatureDefinition(
                feature_definition_id='FD-STAR-GYRO-INNOVATION', name='innovation',
                source_field_template='innovation', aggregation=FeatureAggregation.MAX_VALUE,
                unit='rad/s', algorithm_version='v1',
            ),
            FeatureDefinition(
                feature_definition_id='FD-RW-COMMAND-RESIDUAL', name='residual',
                source_field_template='residual', aggregation=FeatureAggregation.MAX_VALUE,
                unit='N*m', algorithm_version='v1',
            ),
        ],
        feature_results=[primary, conjunct],
        evidence_definitions=[definition], evidence_facts=[fact],
    )
    hypotheses = DeterministicReferenceReasoner.build(bundle)
    critique = SafetyValidationService().critique(hypotheses, bundle)
    assert 'EVIDENCE_STATUS_INCONSISTENT' not in {item.code for item in critique.findings}
