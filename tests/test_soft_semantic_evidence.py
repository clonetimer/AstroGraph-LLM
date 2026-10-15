from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from astrograph.evidence import (
    ComparisonOperator,
    ConjunctiveFeatureCondition,
    EpisodeEvidenceBundle,
    EpisodeWindow,
    EvidenceDefinition,
    EvidenceFactStatus,
    EvidenceRole,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
    SoftConditionCalibration,
    SoftCalibrationEpisode,
    SoftEvidenceBundle,
    SoftEvidenceCalibrationEntry,
    SoftEvidenceCalibrationPolicy,
    SoftEvidencePolicyStatus,
    SoftQuantificationMode,
    RuntimeEvidenceBuildService,
    build_soft_evidence_bundle,
    evaluate_evidence,
    load_frozen_soft_evidence_policy,
    select_development_soft_calibration,
)
from astrograph.simulation import OnlineDataGateReport, SimulationRunCompleted


def _feature_definition(feature_id: str, unit: str = "W") -> FeatureDefinition:
    return FeatureDefinition(
        feature_definition_id=feature_id,
        name=feature_id,
        source_field_template=feature_id.lower(),
        aggregation=FeatureAggregation.MEAN_VALUE,
        unit=unit,
        algorithm_version="test.v1",
    )


def _feature_result(feature_id: str, value: float | None, unit: str = "W") -> FeatureResult:
    return FeatureResult(
        feature_result_id=f"FR-{feature_id}",
        feature_definition_id=feature_id,
        run_id="RUN-1",
        paired_nominal_run_id="NOM-1",
        episode_id="EP-1",
        resolved_field=feature_id.lower(),
        component_instance="COMPONENT-1",
        value=value,
        unit=unit,
        window_start_s=0.0,
        window_end_s=10.0,
        sample_count=11,
        algorithm_version="test.v1",
        input_data_sha256="a" * 64,
    )


def _bundle(
    value: float | None,
    *,
    comparison: ComparisonOperator = ComparisonOperator.GREATER_THAN,
    conjunct_value: float | None | object = ...,
) -> EpisodeEvidenceBundle:
    primary = _feature_definition("FD-PRIMARY")
    definitions = [primary]
    results = [_feature_result(primary.feature_definition_id, value)]
    conjuncts: list[ConjunctiveFeatureCondition] = []
    if conjunct_value is not ...:
        secondary = _feature_definition("FD-SECONDARY", "A")
        definitions.append(secondary)
        results.append(_feature_result(secondary.feature_definition_id, conjunct_value, "A"))
        conjuncts.append(
            ConjunctiveFeatureCondition(
                feature_definition_id=secondary.feature_definition_id,
                comparison=ComparisonOperator.GREATER_THAN,
                threshold=2.0,
                threshold_unit="A",
            )
        )
    evidence_definition = EvidenceDefinition(
        evidence_definition_id="ED-1",
        target_fault_id="FAULT-1",
        name="test evidence",
        feature_definition_id=primary.feature_definition_id,
        role=EvidenceRole.REQUIRED,
        comparison=comparison,
        threshold=10.0,
        threshold_unit="W",
        and_feature_conditions=conjuncts,
    )
    by_feature = {item.feature_definition_id: item for item in results}
    fact = evaluate_evidence(
        evidence_definition,
        by_feature.get(primary.feature_definition_id),
        all_feature_results=results,
    )
    return EpisodeEvidenceBundle(
        episode=EpisodeWindow(
            episode_id="EP-1",
            run_id="RUN-1",
            paired_nominal_run_id="NOM-1",
            start_s=0.0,
            end_s=10.0,
            source="test",
        ),
        domain_id="TEST",
        analysis_component_instance="COMPONENT-1",
        feature_definitions=definitions,
        feature_results=results,
        evidence_definitions=[evidence_definition],
        evidence_facts=[fact],
        evaluation_metadata={"ground_truth_for_offline_evaluation_only": "FAULT-1"},
    )


def _policy(*, include_conjunct: bool = False) -> SoftEvidenceCalibrationPolicy:
    conditions = [
        SoftConditionCalibration(
            feature_definition_id="FD-PRIMARY",
            unit="W",
            transition_width=1.0,
            observation_uncertainty=0.25,
            reference_uncertainty=0.40,
            threshold_uncertainty=0.10,
        )
    ]
    if include_conjunct:
        conditions.append(
            SoftConditionCalibration(
                feature_definition_id="FD-SECONDARY",
                unit="A",
                transition_width=0.5,
                observation_uncertainty=0.1,
            )
        )
    return SoftEvidenceCalibrationPolicy(
        policy_id="soft-policy-test",
        version="1.0.0",
        entries=[
            SoftEvidenceCalibrationEntry(
                evidence_definition_id="ED-1",
                conditions=conditions,
            )
        ],
    )


@pytest.mark.parametrize(
    ("value", "hard_status", "degree"),
    [
        (11.0, EvidenceFactStatus.SATISFIED, 1.0),
        (9.0, EvidenceFactStatus.NOT_SATISFIED, 0.0),
        (None, EvidenceFactStatus.MISSING_DATA, None),
    ],
)
def test_uncalibrated_projection_is_exact_hard_compatibility(
    value: float | None,
    hard_status: EvidenceFactStatus,
    degree: float | None,
) -> None:
    soft = build_soft_evidence_bundle(_bundle(value))
    fact = soft.facts[0]
    assert fact.hard_status is hard_status
    assert fact.predicate_truth_degree == degree
    assert fact.uncertainty_quantified is False
    assert fact.quantification_mode is (
        SoftQuantificationMode.UNAVAILABLE
        if degree is None
        else SoftQuantificationMode.BINARY_COMPATIBILITY
    )


def test_counter_predicate_match_is_truth_one_while_preserving_hard_semantics() -> None:
    bundle = _bundle(11.0)
    definition = bundle.evidence_definitions[0].model_copy(update={"role": EvidenceRole.COUNTER})
    fact = evaluate_evidence(definition, bundle.feature_results[0])
    counter_bundle = bundle.model_copy(
        update={"evidence_definitions": [definition], "evidence_facts": [fact]}
    )
    soft = build_soft_evidence_bundle(counter_bundle).facts[0]
    assert soft.hard_status is EvidenceFactStatus.CONTRADICTED
    assert soft.predicate_truth_degree == 1.0
    assert soft.role == EvidenceRole.COUNTER.value


def test_calibrated_projection_is_monotonic_and_reports_uncertainty_interval() -> None:
    policy = _policy()
    low = build_soft_evidence_bundle(_bundle(9.0), policy).facts[0]
    boundary = build_soft_evidence_bundle(_bundle(10.0), policy).facts[0]
    high = build_soft_evidence_bundle(_bundle(11.0), policy).facts[0]

    assert low.predicate_truth_degree < boundary.predicate_truth_degree < high.predicate_truth_degree
    assert boundary.predicate_truth_degree == pytest.approx(0.5)
    assert boundary.predicate_truth_lower < 0.5 < boundary.predicate_truth_upper
    assert high.quantification_mode is SoftQuantificationMode.LOGISTIC_INTERVAL
    assert high.uncertainty_quantified is True
    assert high.calibration_policy_sha256 == policy.compute_sha256()
    assert high.conditions[0].normalized_margin == pytest.approx(1.0)


def test_conjunctive_evidence_requires_complete_calibration_and_uses_bottleneck() -> None:
    bundle = _bundle(12.0, conjunct_value=2.25)
    with pytest.raises(ValueError, match="primary and every conjunct"):
        build_soft_evidence_bundle(bundle, _policy())

    soft = build_soft_evidence_bundle(bundle, _policy(include_conjunct=True))
    fact = soft.facts[0]
    assert fact.quantification_mode is SoftQuantificationMode.LOGISTIC_INTERVAL
    assert fact.predicate_truth_degree == min(
        item.truth_degree for item in fact.conditions if item.truth_degree is not None
    )


def test_missing_calibrated_conjunct_is_unavailable_not_false_confidence() -> None:
    soft = build_soft_evidence_bundle(
        _bundle(12.0, conjunct_value=None),
        _policy(include_conjunct=True),
    )
    fact = soft.facts[0]
    assert fact.quantification_mode is SoftQuantificationMode.UNAVAILABLE
    assert fact.predicate_truth_degree is None
    assert soft.unavailable_fact_count == 1


def test_posthoc_groups_cannot_select_soft_evidence_policy() -> None:
    with pytest.raises(ValidationError, match="forbidden post-hoc groups"):
        SoftEvidenceCalibrationPolicy(
            policy_id="invalid",
            version="1",
            status=SoftEvidencePolicyStatus.DEVELOPMENT_SELECTED,
            selected_on_groups=[4],
            selection_receipt_sha256="b" * 64,
        )


def test_calibration_unit_mismatch_and_out_of_bundle_definition_fail_closed() -> None:
    wrong_unit = _policy().model_copy(deep=True)
    wrong_unit.entries[0].conditions[0].unit = "Wh"
    with pytest.raises(ValueError, match="calibration unit mismatch"):
        build_soft_evidence_bundle(_bundle(11.0), wrong_unit)

    outside = _policy().model_copy(deep=True)
    outside.entries[0].evidence_definition_id = "ED-OUTSIDE"
    with pytest.raises(ValueError, match="definitions outside the source bundle"):
        build_soft_evidence_bundle(_bundle(11.0), outside)


def test_soft_bundle_is_content_addressed_and_truth_metadata_blind() -> None:
    bundle = _bundle(11.0)
    first = build_soft_evidence_bundle(bundle)
    second = build_soft_evidence_bundle(
        bundle.model_copy(update={"evaluation_metadata": {"different_truth": "FAULT-X"}})
    )
    assert first.source_evidence_sha256 == second.source_evidence_sha256
    assert first.bundle_sha256 == first.compute_sha256()
    with pytest.raises(ValidationError, match="bundle_sha256 mismatch"):
        SoftEvidenceBundle.model_validate(
            {**first.model_dump(mode="json"), "bundle_sha256": "0" * 64}
        )


def test_equality_predicate_does_not_silently_claim_fuzzy_calibration() -> None:
    bundle = _bundle(10.0, comparison=ComparisonOperator.EQUAL)
    fact = build_soft_evidence_bundle(bundle, _policy()).facts[0]
    assert fact.quantification_mode is SoftQuantificationMode.BINARY_COMPATIBILITY
    assert fact.uncertainty_quantified is False


def test_development_selector_uses_only_declared_groups_and_never_freezes() -> None:
    episodes = [
        SoftCalibrationEpisode(group_id=3, true_fault_id="FAULT-1", bundle=_bundle(12.0)),
        SoftCalibrationEpisode(group_id=3, true_fault_id="FAULT-X", bundle=_bundle(8.0)),
        SoftCalibrationEpisode(group_id=8, true_fault_id="FAULT-1", bundle=_bundle(11.0)),
        SoftCalibrationEpisode(group_id=8, true_fault_id="FAULT-X", bundle=_bundle(9.0)),
    ]
    policy, receipt = select_development_soft_calibration(
        episodes,
        dataset_id="test-dataset",
        runtime_evidence_package_sha256="a" * 64,
        source_hashes={"sample_index.csv": "b" * 64},
        selected_evidence_definition_ids=["ED-1"],
        policy_id="test-development-policy",
        policy_version="0.1.0",
        receipt_id="test-selection-receipt",
    )

    assert policy.status is SoftEvidencePolicyStatus.DEVELOPMENT_SELECTED
    assert policy.selected_on_groups == [3, 8]
    assert policy.selection_receipt_sha256 == receipt.receipt_sha256
    assert policy.policy_sha256 == policy.compute_sha256()
    assert receipt.decision == "DEVELOPMENT_SELECTED_NOT_FROZEN"
    assert receipt.freeze_eligible is False
    assert receipt.group_episode_counts == {"3": 2, "8": 2}
    assert all(item.observation_count == 4 for item in receipt.candidate_metrics)
    assert receipt.receipt_sha256 == receipt.compute_sha256()

    with pytest.raises(ValueError, match="non-development groups"):
        select_development_soft_calibration(
            [*episodes, SoftCalibrationEpisode(group_id=4, true_fault_id="FAULT-1", bundle=_bundle(12.0))],
            dataset_id="test-dataset",
            runtime_evidence_package_sha256="a" * 64,
            source_hashes={"sample_index.csv": "b" * 64},
            selected_evidence_definition_ids=["ED-1"],
            policy_id="test-development-policy",
            policy_version="0.1.0",
            receipt_id="test-selection-receipt",
        )


def test_runtime_loader_rejects_development_policy_and_accepts_hash_bound_frozen(
    tmp_path,
) -> None:
    development = _policy().model_copy(
        update={
            "status": SoftEvidencePolicyStatus.DEVELOPMENT_SELECTED,
            "selected_on_groups": [3, 8],
            "selection_receipt_sha256": "c" * 64,
        }
    )
    development = development.model_copy(
        update={"policy_sha256": development.compute_sha256()}
    )
    development_path = tmp_path / "development.json"
    development_path.write_text(
        json.dumps(development.model_dump(mode="json")), encoding="utf-8"
    )
    development_asset_sha = hashlib.sha256(development_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="must have FROZEN status"):
        load_frozen_soft_evidence_policy(
            development_path, expected_sha256=development_asset_sha
        )

    frozen = _policy().model_copy(
        update={
            "status": SoftEvidencePolicyStatus.FROZEN,
            "selected_on_groups": [3, 8],
            "selection_receipt_sha256": "d" * 64,
        }
    )
    frozen = frozen.model_copy(update={"policy_sha256": frozen.compute_sha256()})
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen.model_dump(mode="json")), encoding="utf-8")
    frozen_asset_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    loaded = load_frozen_soft_evidence_policy(
        frozen_path, expected_sha256=frozen_asset_sha
    )
    assert loaded.status is SoftEvidencePolicyStatus.FROZEN
    with pytest.raises(ValueError, match="asset SHA-256 mismatch"):
        load_frozen_soft_evidence_policy(frozen_path, expected_sha256="0" * 64)


def test_frozen_policy_requires_identified_uncertainty() -> None:
    with pytest.raises(ValidationError, match="non-zero identified uncertainty"):
        SoftEvidenceCalibrationPolicy(
            policy_id="invalid-frozen",
            version="1",
            status=SoftEvidencePolicyStatus.FROZEN,
            entries=[
                SoftEvidenceCalibrationEntry(
                    evidence_definition_id="ED-1",
                    conditions=[
                        SoftConditionCalibration(
                            feature_definition_id="FD-PRIMARY",
                            unit="W",
                            transition_width=1.0,
                        )
                    ],
                )
            ],
            selected_on_groups=[3, 8],
            selection_receipt_sha256="e" * 64,
        )


def test_runtime_service_applies_only_hash_bound_frozen_policy(tmp_path) -> None:
    frozen = _policy().model_copy(
        update={
            "status": SoftEvidencePolicyStatus.FROZEN,
            "selected_on_groups": [3, 8],
            "selection_receipt_sha256": "f" * 64,
        }
    )
    frozen = frozen.model_copy(update={"policy_sha256": frozen.compute_sha256()})
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen.model_dump(mode="json")), encoding="utf-8")
    asset_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    class _Builder:
        builder_version = "test-builder.v1"

        def build(self, event):
            return _bundle(11.0), OnlineDataGateReport(passed=True, run_id=event.run_id)

    service = RuntimeEvidenceBuildService()
    service.resolve = lambda payload: _Builder()  # type: ignore[method-assign]
    event = SimulationRunCompleted(
        event_id="soft-runtime",
        run_id="soft-runtime",
        run_dir=str(tmp_path / "run"),
        nominal_run_dir=str(tmp_path / "nominal"),
    )
    built = service.build(
        {
            "simulation_event": event.model_dump(mode="json"),
            "soft_evidence_calibration_policy_path": str(path),
            "soft_evidence_calibration_policy_sha256": asset_sha,
        }
    )
    soft = SoftEvidenceBundle.model_validate(built["soft_evidence"])
    assert soft.calibrated_fact_count == 1
    assert soft.calibration_policy_id == frozen.policy_id
    assert soft.calibration_policy_sha256 == frozen.policy_sha256
