"""AI-Native Phase 4: evidence-definition suggestion generator tests.
Verifies the deterministic, governed suggestions (threshold recalibration and
new evidence definition) and the human-review-only contract.
"""

from __future__ import annotations

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus
from astrograph.evidence.executable import (
    EpisodeWindow,
    EvidenceDefinition,
    EvidenceFact,
    EvidenceRole,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
    stable_id,
)
from astrograph.evidence.suggestion import (
    SuggestionKind,
    suggest_evidence_definitions,
)


def _bundle(
    *,
    observed_values: list[float],
    has_definition: bool = True,
    satisfied: bool = False,
) -> EpisodeEvidenceBundle:
    feature = FeatureDefinition(
        feature_definition_id="FD-TEST-RESIDUAL",
        name="test residual feature",
        source_field_template="asset.test.residual",
        aggregation=FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL,
        unit="W",
        algorithm_version="test@1",
    )
    results = [
        FeatureResult(
            feature_result_id=f"FR-TEST-{index}",
            feature_definition_id="FD-TEST-RESIDUAL",
            run_id="RUN",
            paired_nominal_run_id="NOM",
            episode_id="EP-1",
            resolved_field="asset.test.residual",
            component_instance="TEST",
            value=value,
            unit="W",
            window_start_s=0.0,
            window_end_s=1.0,
            sample_count=1,
            algorithm_version="test@1",
            input_data_sha256="a" * 64,
        )
        for index, value in enumerate(observed_values)
    ]
    definitions: list[EvidenceDefinition] = []
    facts: list[EvidenceFact] = []
    if has_definition:
        definition = EvidenceDefinition(
            evidence_definition_id="ED-TEST-RESIDUAL",
            target_fault_id="FAULT_A",
            name="residual gate",
            feature_definition_id="FD-TEST-RESIDUAL",
            role=EvidenceRole.REQUIRED,
            comparison="GREATER_OR_EQUAL",
            threshold=0.001,
            threshold_unit="W",
        )
        definitions.append(definition)
        facts.append(
            EvidenceFact(
                evidence_fact_id="EF-TEST-1",
                evidence_definition_id="ED-TEST-RESIDUAL",
                target_fault_id="FAULT_A",
                feature_result_id="FR-TEST-0",
                episode_id="EP-1",
                role=EvidenceRole.REQUIRED,
                status=EvidenceFactStatus.SATISFIED if satisfied else EvidenceFactStatus.NOT_SATISFIED,
                observed_value=observed_values[0] if observed_values else None,
                threshold_value=0.001,
                unit="W",
                comparison="GREATER_OR_EQUAL",
            )
        )
    return EpisodeEvidenceBundle(
        episode=EpisodeWindow(
            episode_id="EP-1", run_id="RUN", paired_nominal_run_id="NOM",
            start_s=0.0, end_s=1.0, source="test",
        ),
        analysis_component_instance="TEST",
        feature_definitions=[feature],
        feature_results=results,
        evidence_definitions=definitions,
        evidence_facts=facts,
    )


def test_suggests_new_evidence_definition_for_unregistered_high_activity_feature():
    bundle = _bundle(observed_values=[10.0, 12.0, 11.0], has_definition=False)
    report = suggest_evidence_definitions(bundle)
    kinds = {item.kind for item in report.suggestions}
    assert SuggestionKind.NEW_EVIDENCE_DEFINITION in kinds
    suggestion = next(
        item for item in report.suggestions if item.kind == SuggestionKind.NEW_EVIDENCE_DEFINITION
    )
    assert suggestion.target_fault_id is None
    assert suggestion.feature_definition_id == "FD-TEST-RESIDUAL"
    assert suggestion.review_status == "READY_FOR_HUMAN_REVIEW"
    assert suggestion.publication_allowed is False
    assert suggestion.active_config_modified is False


def test_suggests_threshold_recalibration_for_unsatisfied_definition():
    bundle = _bundle(observed_values=[10.0, 12.0, 11.0], has_definition=True, satisfied=False)
    report = suggest_evidence_definitions(bundle)
    kinds = {item.kind for item in report.suggestions}
    assert SuggestionKind.THRESHOLD_RECALIBRATION in kinds
    suggestion = next(
        item for item in report.suggestions if item.kind == SuggestionKind.THRESHOLD_RECALIBRATION
    )
    assert suggestion.target_fault_id == "FAULT_A"
    assert suggestion.suggested_threshold is not None
    assert suggestion.suggested_threshold > 0.0


def test_satisfied_definition_yields_no_suggestion():
    bundle = _bundle(observed_values=[10.0, 12.0, 11.0], has_definition=True, satisfied=True)
    report = suggest_evidence_definitions(bundle)
    assert report.suggestions == []


def test_low_activity_feature_yields_no_suggestion():
    bundle = _bundle(observed_values=[1e-9, 2e-9, 1.5e-9], has_definition=False)
    report = suggest_evidence_definitions(bundle)
    assert report.suggestions == []


def test_suggestion_ids_are_stable_and_unique():
    bundle = _bundle(observed_values=[10.0, 12.0, 11.0], has_definition=False)
    first = suggest_evidence_definitions(bundle)
    second = suggest_evidence_definitions(bundle)
    first_ids = [item.suggestion_id for item in first.suggestions]
    second_ids = [item.suggestion_id for item in second.suggestions]
    assert first_ids == second_ids
    assert len(set(first_ids)) == len(first_ids)


def test_status_flags_are_excluded_from_new_definitions():
    bundle = _bundle(observed_values=[10.0, 12.0, 11.0], has_definition=False)
    report = suggest_evidence_definitions(bundle)
    names = [item.feature_name for item in report.suggestions]
    assert not any(any(token in name.lower() for token in ("valid", "fresh", "status")) for name in names)
