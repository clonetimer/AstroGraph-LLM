from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import pandas as pd
from pydantic import ValidationError

from astrograph.evidence import (
    ComparisonOperator,
    EpisodeEvidenceBundle,
    EpisodeWindow,
    EvidenceDefinition,
    EvidenceFact,
    EvidenceFactStatus,
    EvidenceRole,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
)
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeExecutor,
    DiagnosticProbeKind,
    DiagnosticProbeCatalog,
    build_evidence_margin_probe_catalog,
)
from astrograph.investigation.active_telemetry_probes import (
    ActiveTelemetryProbeExecutor,
    build_admitted_telemetry_change_probe_catalog,
    is_admitted_telemetry_probe,
)
from astrograph.investigation.shadow_candidate_revision import validate_shadow_candidate_revision
from astrograph.evidence.configured_runtime import (
    ConfiguredChannelMapEntry,
    ConfiguredRuntimeEvidenceBuilder,
    ConfiguredRuntimeEvidencePackage,
    column_schema_sha256,
)
from astrograph.orchestration.llm_diagnostic_loop import (
    ControlledDiagnosticDeliberationService,
    DiagnosticContributionStatus,
)
from astrograph.llm.task_prompts import structured_task_messages
from astrograph.orchestration.whole_spacecraft_stage_executor import (
    _revalidated_telemetry_probe_catalog,
    _stable_evidence_payload,
)
from astrograph.simulation import SimulationRunCompleted


def _bundle() -> EpisodeEvidenceBundle:
    episode = EpisodeWindow(
        episode_id="EP-PROBE", run_id="RUN-1", paired_nominal_run_id="RUN-N",
        start_s=0.0, end_s=10.0, source="unit-test",
    )
    feature_def = FeatureDefinition(
        feature_definition_id="FD-1", name="current residual", source_field_template="x",
        aggregation=FeatureAggregation.MAX_VALUE, unit="A", algorithm_version="test-v1",
    )
    feature = FeatureResult(
        feature_result_id="FR-1", feature_definition_id="FD-1", run_id="RUN-1",
        paired_nominal_run_id="RUN-N", episode_id="EP-PROBE", resolved_field="x",
        component_instance="rw-2", value=3.0, unit="A", window_start_s=0.0, window_end_s=10.0,
        sample_count=10, algorithm_version="test-v1", input_data_sha256="1" * 64,
    )
    definition = EvidenceDefinition(
        evidence_definition_id="ED-1", target_fault_id="FAULT-A", name="current above limit",
        feature_definition_id="FD-1", role=EvidenceRole.REQUIRED,
        comparison=ComparisonOperator.GREATER_OR_EQUAL, threshold=2.0, threshold_unit="A", critical=True,
    )
    fact = EvidenceFact(
        evidence_fact_id="EF-1", evidence_definition_id="ED-1", target_fault_id="FAULT-A",
        feature_result_id="FR-1", episode_id="EP-PROBE", role=EvidenceRole.REQUIRED,
        status=EvidenceFactStatus.SATISFIED, observed_value=3.0, threshold_value=2.0,
        unit="A", comparison=ComparisonOperator.GREATER_OR_EQUAL,
    )
    return EpisodeEvidenceBundle(
        episode=episode, analysis_component_instance="rw-2",
        feature_definitions=[feature_def], feature_results=[feature],
        evidence_definitions=[definition], evidence_facts=[fact],
    )


def test_margin_probe_catalog_is_host_bound_and_shadow_only():
    catalog = build_evidence_margin_probe_catalog(_bundle())
    assert len(catalog.probes) == 1
    spec = catalog.probes[0]
    assert spec.kind == DiagnosticProbeKind.EVIDENCE_THRESHOLD_MARGIN
    assert spec.formal_evidence_eligible is False
    assert spec.evidence_definition_id == "ED-1"
    result = DiagnosticProbeExecutor().execute(catalog=catalog, probe_id=spec.probe_id)
    assert result.metrics["signed_margin"] == 1.0
    assert result.formal_evidence_generated is False
    assert result.creates_new_raw_observation is False
    assert result.evidence_definition_id == "ED-1"
    assert result.feature_result_id == "FR-1"
    assert result.target_fault_ids == ("FAULT-A",)


def test_probe_executor_rejects_model_invented_probe_id():
    catalog = build_evidence_margin_probe_catalog(_bundle())
    with pytest.raises(KeyError, match="unknown diagnostic probe_id"):
        DiagnosticProbeExecutor().execute(catalog=catalog, probe_id="PROBE-INVENTED")


class ProbeBackend:
    model_id = "probe-test"
    model_version = "v1"
    supports_diagnostic_deliberation = True

    def __init__(self, probe_id: str):
        self.probe_id = probe_id
        self.payloads: list[dict] = []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.payloads.append(payload)
        observations = payload["observations"]
        if not observations:
            body = {"capability_id": "inspect_probe_catalog", "objective": "Inspect approved probes"}
        elif len(observations) == 1:
            body = {
                "capability_id": "execute_diagnostic_probe", "probe_id": self.probe_id,
                "objective": "Recheck frozen threshold margin",
                "cited_observation_ids": [observations[-1]["observation_id"]],
            }
        else:
            body = {
                "capability_id": "synthesize_hypotheses", "objective": "Synthesize",
                "cited_observation_ids": [x["observation_id"] for x in observations],
            }
        return output_model.model_validate(body)


def test_controlled_loop_executes_only_catalog_probe_and_marks_derived_contribution():
    bundle = _bundle()
    catalog = build_evidence_margin_probe_catalog(bundle)
    result = ControlledDiagnosticDeliberationService().run(
        backend=ProbeBackend(catalog.probes[0].probe_id),
        episode_id=bundle.episode.episode_id,
        artifacts={"evidence": bundle.model_dump(mode="json")},
        probe_catalog=catalog,
        allowed_capability_ids=["inspect_probe_catalog", "execute_diagnostic_probe"],
        max_steps=3,
    )
    assert result.status.value == "COMPLETED"
    assert result.steps[1].policy_status == "EXECUTED_SHADOW_PROBE"
    assert result.steps[0].observation.content["probes"][0]["evidence_definition_id"] == "ED-1"
    assert result.steps[1].observation.content["evidence_definition_id"] == "ED-1"
    assert result.contribution_audit.status == DiagnosticContributionStatus.DERIVED_OBSERVATION_ONLY
    assert result.contribution_audit.new_raw_observation_count == 0
    assert result.contribution_audit.formal_evidence_generated_count == 0


def test_contribution_audit_is_inside_trajectory_hash():
    bundle = _bundle()
    catalog = build_evidence_margin_probe_catalog(bundle)
    result = ControlledDiagnosticDeliberationService().run(
        backend=ProbeBackend(catalog.probes[0].probe_id), episode_id="EP-PROBE",
        artifacts={"evidence": bundle.model_dump(mode="json")}, probe_catalog=catalog,
        allowed_capability_ids=["inspect_probe_catalog", "execute_diagnostic_probe"], max_steps=3,
    )
    tampered = result.model_dump(mode="json")
    tampered["contribution_audit"]["probe_execution_count"] = 0
    with pytest.raises(ValidationError, match="trajectory_sha256 mismatch"):
        type(result).model_validate(tampered)


def test_telemetry_probe_budget_prioritizes_candidate_discrimination():
    bundle = _bundle()
    feature_b = bundle.feature_definitions[0].model_copy(update={
        "feature_definition_id": "FD-2", "source_field_template": "z",
    })
    definition_b = bundle.evidence_definitions[0].model_copy(update={
        "evidence_definition_id": "ED-2", "feature_definition_id": "FD-2",
        "target_fault_id": "FAULT-B",
    })
    bundle = bundle.model_copy(update={
        "feature_definitions": [*bundle.feature_definitions, feature_b],
        "evidence_definitions": [*bundle.evidence_definitions, definition_b],
    })
    frame = pd.DataFrame({
        "time_s": [i / 2 for i in range(20)],
        "x": [1.0] * 20, "z": [1.0] * 10 + [3.0] * 10,
    })
    channels = [
        ConfiguredChannelMapEntry(
            source_field=source, semantic_channel_id=channel,
            component_instance="rw-2", source_unit="A", canonical_unit="A",
        ) for source, channel in (("current_a", "x"), ("current_b", "z"))
    ]
    common = dict(
        bundle=bundle, mapped_frame=frame, approved_channels=channels,
        source_sha256="a" * 64, mapping_sha256="b" * 64, max_probes=1,
    )
    alphabetical = build_admitted_telemetry_change_probe_catalog(**common)
    prioritized = build_admitted_telemetry_change_probe_catalog(
        **common, priority_fault_ids=("FAULT-B", "UNKNOWN"),
    )
    assert alphabetical.probes[0].bound_inputs["semantic_channel_id"] == "x"
    assert prioritized.probes[0].bound_inputs["semantic_channel_id"] == "z"
    assert is_admitted_telemetry_probe(prioritized.probes[0], prioritized)
    assert ActiveTelemetryProbeExecutor().execute(
        catalog=prioritized, probe_id=prioritized.probes[0].probe_id,
    ).metrics["delta"] == 2.0


def test_paired_residual_probe_distinguishes_changed_and_healthy_samples():
    bundle = _bundle()
    nominal = pd.DataFrame({"time_s": [i / 2 for i in range(20)], "x": [1.0] * 20})
    channel = ConfiguredChannelMapEntry(
        source_field="rw_current", semantic_channel_id="x", component_instance="rw-2",
        source_unit="A", canonical_unit="A",
    )
    common = dict(
        bundle=bundle, approved_channels=[channel], source_sha256="a" * 64,
        mapping_sha256="b" * 64, paired_nominal_mapped_frame=nominal,
        nominal_sha256="c" * 64,
    )
    changed = build_admitted_telemetry_change_probe_catalog(
        **common, mapped_frame=nominal.assign(x=[1.0] * 10 + [3.0] * 10),
    )
    spec = changed.probes[0]
    assert spec.bound_inputs["measurement_basis"] == "PAIRED_NOMINAL_RESIDUAL_V1"
    assert spec.bound_inputs["before"] == [0.0] * 8
    assert spec.bound_inputs["after"] == [2.0] * 8
    assert is_admitted_telemetry_probe(spec, changed)
    assert ActiveTelemetryProbeExecutor().execute(
        catalog=changed, probe_id=spec.probe_id,
    ).metrics["delta"] == 2.0
    healthy = build_admitted_telemetry_change_probe_catalog(**common, mapped_frame=nominal)
    assert ActiveTelemetryProbeExecutor().execute(
        catalog=healthy, probe_id=healthy.probes[0].probe_id,
    ).metrics["delta"] == 0.0

    altered = changed.model_copy(update={"probes": (spec.model_copy(update={
        "bound_inputs": {**spec.bound_inputs, "nominal_sha256": "d" * 64},
    }),)})
    with pytest.raises(ValueError, match="requires an admitted runtime telemetry catalog"):
        ActiveTelemetryProbeExecutor().execute(catalog=altered, probe_id=spec.probe_id)
    shifted = nominal.assign(time_s=[i / 2 + 0.1 for i in range(20)])
    with pytest.raises(ValueError, match="time axes must align exactly"):
        build_admitted_telemetry_change_probe_catalog(
            **{**common, "paired_nominal_mapped_frame": shifted}, mapped_frame=nominal,
        )


def test_fixed_probe_policy_executes_admitted_probe_inside_host_loop():
    from scripts.experiments.fixed_probe_action_policy import FixedProbeActionBackend

    bundle = _bundle()
    nominal = pd.DataFrame({"time_s": [i / 2 for i in range(20)], "x": [1.0] * 20})
    source = nominal.assign(x=[1.0] * 10 + [3.0] * 10)
    channel = ConfiguredChannelMapEntry(
        source_field="rw_current", semantic_channel_id="x", component_instance="rw-2",
        source_unit="A", canonical_unit="A",
    )
    catalog = build_admitted_telemetry_change_probe_catalog(
        bundle=bundle, mapped_frame=source, paired_nominal_mapped_frame=nominal,
        approved_channels=[channel], source_sha256="a" * 64,
        nominal_sha256="c" * 64, mapping_sha256="b" * 64,
    )

    class NoActionLLM:
        supports_diagnostic_deliberation = True
        supports_dynamic_diagnostic_action_schema = True

        def generate_structured(self, **kwargs):
            raise AssertionError("fixed action selector must not call the LLM")

    backend = FixedProbeActionBackend(NoActionLLM())
    trajectory = ControlledDiagnosticDeliberationService().run(
        backend=backend, episode_id=bundle.episode.episode_id,
        artifacts={"evidence": bundle.model_dump(mode="json")}, probe_catalog=catalog,
        probe_executor=ActiveTelemetryProbeExecutor(), max_steps=4,
        allowed_capability_ids=[
            "inspect_candidate_fusion", "execute_diagnostic_probe",
            "inspect_deterministic_core", "synthesize_hypotheses",
        ],
    )
    assert [step.proposal.capability_id for step in trajectory.steps] == [
        "inspect_candidate_fusion", "execute_diagnostic_probe",
        "inspect_deterministic_core", "synthesize_hypotheses",
    ]
    assert trajectory.contribution_audit.new_raw_observation_count == 1
    assert trajectory.contribution_audit.formal_evidence_generated_count == 0
    assert trajectory.steps[1].observation.content["metrics"]["delta"] == 2.0


def test_agent_can_select_newly_inspected_admitted_telemetry_without_formal_vote():
    bundle = _bundle()
    frame = pd.DataFrame({"time_s": [i / 2 for i in range(20)], "x": [1.0] * 10 + [3.0] * 10})
    channel = ConfiguredChannelMapEntry(
        source_field="rw_current", semantic_channel_id="x", component_instance="rw-2",
        source_unit="A", canonical_unit="A",
    )
    catalog = build_admitted_telemetry_change_probe_catalog(
        bundle=bundle, mapped_frame=frame, approved_channels=[channel],
        source_sha256="a" * 64, mapping_sha256="b" * 64,
    )
    assert len(catalog.probes) == 1
    spec = catalog.probes[0]
    assert is_admitted_telemetry_probe(spec, catalog)
    assert spec.target_fault_ids == ("FAULT-A",)
    backend = ProbeBackend(spec.probe_id)
    result = ControlledDiagnosticDeliberationService().run(
        backend=backend, episode_id=bundle.episode.episode_id,
        artifacts={"evidence": bundle.model_dump(mode="json")}, probe_catalog=catalog,
        probe_executor=ActiveTelemetryProbeExecutor(),
        allowed_capability_ids=["inspect_probe_catalog", "execute_diagnostic_probe"], max_steps=3,
    )
    assert result.status.value == "COMPLETED"
    assert backend.payloads[0]["probe_index"][0]["diagnostic_question"] == spec.objective
    assert backend.payloads[0]["probe_index"][0]["creates_new_raw_observation"] is True
    assert result.steps[0].observation.observation_id in backend.payloads[1]["citation_requirement"]
    assert "do not run one just to satisfy a quota" in backend.payloads[1]["instruction"]
    assert result.steps[0].observation.content["probes"][0]["creates_new_raw_observation"] is True
    assert result.steps[1].observation.content["metrics"]["delta"] == 2.0
    assert result.contribution_audit.status == DiagnosticContributionStatus.NEW_INFORMATION_NO_DIAGNOSTIC_CHANGE
    assert result.contribution_audit.new_raw_observation_count == 1
    assert result.contribution_audit.formal_evidence_generated_count == 0
    active_observation_id = result.steps[1].observation.observation_id
    revision = validate_shadow_candidate_revision(
        raw_proposal={
            "candidate_fault_ids": ["FAULT-A"], "abstain_recommended": False,
            "cited_observation_ids": [active_observation_id],
        },
        trajectory=result, registered_fault_ids={"FAULT-A"},
        baseline_candidate_fault_ids=["FAULT-B"],
    )
    # A baseline outside the registered catalog cannot establish a candidate delta.
    assert revision.status == "UNAVAILABLE"
    unrelated = validate_shadow_candidate_revision(
        raw_proposal={
            "candidate_fault_ids": ["FAULT-A"], "abstain_recommended": False,
            "cited_observation_ids": [active_observation_id],
        },
        trajectory=result, registered_fault_ids={"FAULT-A", "FAULT-B"},
        baseline_candidate_fault_ids=["FAULT-B"],
    )
    assert unrelated.status == "AVAILABLE"
    assert unrelated.candidate_set_changed is True
    assert unrelated.active_telemetry_observation_ids == [active_observation_id]
    assert unrelated.candidate_relevant_telemetry_observation_ids == []
    assert unrelated.active_telemetry_attributable is False
    assert unrelated.reason_code == "ACTIVE_PROBE_OUTSIDE_BASELINE_CANDIDATES"
    revision = validate_shadow_candidate_revision(
        raw_proposal={
            "candidate_fault_ids": [], "abstain_recommended": True,
            "cited_observation_ids": [active_observation_id],
        },
        trajectory=result, registered_fault_ids={"FAULT-A"},
        baseline_candidate_fault_ids=["FAULT-A"],
    )
    assert revision.active_telemetry_attributable is True
    assert revision.candidate_relevant_telemetry_observation_ids == [active_observation_id]
    revised_trajectory = ControlledDiagnosticDeliberationService().record_shadow_revision(result, revision)
    assert revised_trajectory.contribution_audit.status == DiagnosticContributionStatus.CANDIDATE_SET_REFINED
    assert revised_trajectory.contribution_audit.formal_evidence_generated_count == 0
    assert revised_trajectory.contribution_audit.decision_changed is False
    assert revised_trajectory.trajectory_sha256 != result.trajectory_sha256
    assert type(result).model_validate(revised_trajectory.model_dump(mode="json")) == revised_trajectory


def test_action_prompt_preserves_optional_probe_choice_and_citation_boundary():
    action_model = ControlledDiagnosticDeliberationService._constrained_proposal_model(
        available_ids=["inspect_probe_catalog", "execute_diagnostic_probe", "synthesize_hypotheses"],
        observation_ids=["OBS-01-valid"], citation_required=True, probe_ids=["PROBE-valid"],
    )
    messages = structured_task_messages(action_model, {"observation_index": [
        {"observation_id": "OBS-01-valid", "capability_id": "inspect_probe_catalog"},
    ]})
    assert "Prefer a relevant raw probe" in messages[0]["content"]
    assert "do not probe merely to increase tool use" in messages[0]["content"]
    assert "cite at least one actual ID" in messages[0]["content"]


def test_replan_schema_requires_an_actual_prior_observation_id():
    action_model = ControlledDiagnosticDeliberationService._constrained_proposal_model(
        available_ids=["execute_diagnostic_probe"], observation_ids=["OBS-01-valid"],
        citation_required=True, probe_ids=["PROBE-valid"],
    )
    base = {"capability_id": "execute_diagnostic_probe", "objective": "Distinguish faults",
            "probe_id": "PROBE-valid"}
    assert "cited_observation_ids" in action_model.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        action_model.model_validate(base)
    with pytest.raises(ValidationError):
        action_model.model_validate({**base, "cited_observation_ids": []})
    assert action_model.model_validate({
        **base, "cited_observation_ids": ["OBS-01-valid"],
    }).cited_observation_ids == ["OBS-01-valid"]


def test_shadow_revision_invalid_output_and_missing_exposure_do_not_change_diagnosis():
    bundle = _bundle()
    catalog = build_evidence_margin_probe_catalog(bundle)
    trajectory = ControlledDiagnosticDeliberationService().run(
        backend=ProbeBackend(catalog.probes[0].probe_id), episode_id=bundle.episode.episode_id,
        artifacts={"evidence": bundle.model_dump(mode="json")}, probe_catalog=catalog,
        allowed_capability_ids=["inspect_probe_catalog", "execute_diagnostic_probe"], max_steps=3,
    )
    observation_id = trajectory.steps[1].observation.observation_id
    for raw, reason in [
        ({"candidate_fault_ids": ["INVENTED"], "abstain_recommended": False,
          "cited_observation_ids": [observation_id]}, "UNREGISTERED_SHADOW_FAULT_ID"),
        ({"candidate_fault_ids": [], "abstain_recommended": True,
          "cited_observation_ids": ["OBS-INVENTED"]}, "UNKNOWN_SHADOW_OBSERVATION_ID"),
        ({"candidate_fault_ids": "FAULT-A", "abstain_recommended": False}, "INVALID_SHADOW_PROPOSAL"),
    ]:
        revision = validate_shadow_candidate_revision(
            raw_proposal=raw, trajectory=trajectory, registered_fault_ids={"FAULT-A"},
            baseline_candidate_fault_ids=["FAULT-A"],
        )
        assert revision.status == "UNAVAILABLE"
        assert revision.reason_code == reason
        assert ControlledDiagnosticDeliberationService().record_shadow_revision(trajectory, revision) == trajectory
    valid_derived = validate_shadow_candidate_revision(
        raw_proposal={"candidate_fault_ids": [], "abstain_recommended": True,
                      "cited_observation_ids": [observation_id]},
        trajectory=trajectory, registered_fault_ids={"FAULT-A"},
        baseline_candidate_fault_ids=["FAULT-A"],
    )
    assert valid_derived.status == "AVAILABLE"
    assert valid_derived.candidate_set_changed is True
    assert valid_derived.active_telemetry_attributable is False
    assert ControlledDiagnosticDeliberationService().record_shadow_revision(trajectory, valid_derived) == trajectory


def test_raw_probe_rejects_catalog_without_telemetry_admission():
    bundle = _bundle()
    frame = pd.DataFrame({"time_s": [i / 2 for i in range(20)], "current": [1.0] * 20})
    channel = ConfiguredChannelMapEntry(
        source_field="rw_current", semantic_channel_id="current", component_instance="rw-2",
        source_unit="A", canonical_unit="A",
    )
    admitted = build_admitted_telemetry_change_probe_catalog(
        bundle=bundle, mapped_frame=frame, approved_channels=[channel],
        source_sha256="a" * 64, mapping_sha256="b" * 64,
    )
    unadmitted = DiagnosticProbeCatalog(episode_id=bundle.episode.episode_id, probes=admitted.probes)
    with pytest.raises(ValueError, match="requires an admitted runtime telemetry catalog"):
        ActiveTelemetryProbeExecutor().execute(catalog=unadmitted, probe_id=admitted.probes[0].probe_id)
    altered = admitted.model_copy(update={
        "probes": (admitted.probes[0].model_copy(update={
            "bound_inputs": {**admitted.probes[0].bound_inputs, "after": [99.0] * 8}
        }),)
    })
    with pytest.raises(ValueError, match="requires an admitted runtime telemetry catalog"):
        ActiveTelemetryProbeExecutor().execute(catalog=altered, probe_id=altered.probes[0].probe_id)
    removed_marker = admitted.model_copy(update={
        "probes": (admitted.probes[0].model_copy(update={
            "bound_inputs": {
                key: value for key, value in admitted.probes[0].bound_inputs.items()
                if key != "admission_kind"
            }
        }),)
    })
    with pytest.raises(ValueError, match="requires an admitted runtime telemetry catalog"):
        ActiveTelemetryProbeExecutor().execute(
            catalog=removed_marker, probe_id=removed_marker.probes[0].probe_id
        )
    retargeted = admitted.model_copy(update={
        "probes": (admitted.probes[0].model_copy(update={"target_fault_ids": ("FAULT-OTHER",)}),)
    })
    with pytest.raises(ValueError, match="requires an admitted runtime telemetry catalog"):
        ActiveTelemetryProbeExecutor().execute(catalog=retargeted, probe_id=retargeted.probes[0].probe_id)
    with pytest.raises(ValueError, match="requires the admission-aware executor"):
        ControlledDiagnosticDeliberationService().run(
            backend=ProbeBackend(admitted.probes[0].probe_id), episode_id=bundle.episode.episode_id,
            artifacts={"evidence": bundle.model_dump(mode="json")}, probe_catalog=admitted,
            allowed_capability_ids=["inspect_probe_catalog", "execute_diagnostic_probe"], max_steps=2,
        )


def test_revalidation_ignores_only_construction_time_not_evidence_values():
    original = _bundle()
    rebuilt = original.model_copy(update={
        "feature_results": [original.feature_results[0].model_copy(update={"created_at": "later"})],
        "evidence_facts": [original.evidence_facts[0].model_copy(update={"created_at": "later"})],
    })
    assert _stable_evidence_payload(original) == _stable_evidence_payload(rebuilt)
    altered = rebuilt.model_copy(update={
        "feature_results": [rebuilt.feature_results[0].model_copy(update={"value": 99.0})],
    })
    assert _stable_evidence_payload(original) != _stable_evidence_payload(altered)


@pytest.mark.parametrize("late_current, expected_delta", [(3.0, 2.0), (1.0, 0.0)])
def test_runtime_telemetry_probe_revalidates_source_and_fails_closed_on_change(
    tmp_path, late_current, expected_delta,
):
    source_dir = tmp_path / "source"
    nominal_dir = tmp_path / "nominal"
    source_dir.mkdir()
    nominal_dir.mkdir()
    source = pd.DataFrame({
        "time_s": [i / 2 for i in range(20)],
        "rw_current": [1.0] * 10 + [late_current] * 10,
    })
    nominal = source.assign(rw_current=1.0)
    source_path = source_dir / "telemetry.csv"
    source.to_csv(source_path, index=False)
    nominal.to_csv(nominal_dir / "telemetry.csv", index=False)
    template = _bundle()
    package = ConfiguredRuntimeEvidencePackage(
        package_id="probe-test", version="1", domain_id="D3_RW",
        analysis_component_instance="rw-2",
        source_schema_sha256=column_schema_sha256(list(source.columns)),
        time_axis={"source_field": "time_s", "source_unit": "s"},
        channels=[{
            "source_field": "rw_current", "semantic_channel_id": "x",
            "component_instance": "rw-2", "source_unit": "A", "canonical_unit": "A",
        }],
        feature_definitions=template.feature_definitions,
        evidence_definitions=template.evidence_definitions,
        claim_boundary="Synthetic probe integration test only.",
    )
    package_path = tmp_path / "runtime_evidence.json"
    package_path.write_text(json.dumps(package.model_dump(mode="json")), encoding="utf-8")
    event = SimulationRunCompleted(
        event_id="probe-test", run_id="run", run_dir=str(source_dir), nominal_run_dir=str(nominal_dir),
    )
    builder = ConfiguredRuntimeEvidenceBuilder(package_path)
    bundle, gate = builder.build(event)
    assert gate.passed
    base = build_evidence_margin_probe_catalog(bundle)
    request = SimpleNamespace(
        runtime_evidence_builder_id="configured.package.v1",
        evidence_package_path=str(package_path), simulation_event=event,
    )
    admitted = _revalidated_telemetry_probe_catalog(
        request=request, bundle=bundle, gate=gate, base_catalog=base,
    )
    assert admitted.telemetry_admission_status == "ADMITTED_RUNTIME_TELEMETRY"
    active = [item for item in admitted.probes if is_admitted_telemetry_probe(item, admitted)]
    assert len(active) == 1
    assert active[0].bound_inputs["measurement_basis"] == "PAIRED_NOMINAL_RESIDUAL_V1"
    observation = ActiveTelemetryProbeExecutor().execute(catalog=admitted, probe_id=active[0].probe_id)
    assert observation.metrics["delta"] == expected_delta
    assert observation.formal_evidence_generated is False

    source.loc[0, "rw_current"] = 1.5
    source.to_csv(source_path, index=False)
    rejected = _revalidated_telemetry_probe_catalog(
        request=request, bundle=bundle, gate=gate, base_catalog=base,
    )
    assert rejected.telemetry_admission_status == "REJECTED_TELEMETRY_REVALIDATION"
    assert rejected.probes == base.probes

    source.to_csv(source_path, index=False)
    nominal_path = nominal_dir / "telemetry.csv"
    nominal.loc[0, "rw_current"] = 1.5
    nominal.to_csv(nominal_path, index=False)
    rejected_nominal = _revalidated_telemetry_probe_catalog(
        request=request, bundle=bundle, gate=gate, base_catalog=base,
    )
    assert rejected_nominal.telemetry_admission_status == "REJECTED_TELEMETRY_REVALIDATION"
    assert rejected_nominal.probes == base.probes


def test_r23_minimal_diagnosis_probe_is_host_bound_shadow_only():
    from astrograph.investigation.diagnostic_probes import DiagnosticProbeCatalog, DiagnosticProbeSpec

    spec = DiagnosticProbeSpec(
        probe_id="PROBE-MBD-1", episode_id="EP-PROBE", kind=DiagnosticProbeKind.MINIMAL_DIAGNOSIS_REVIEW,
        title="Review conflicts", objective="Enumerate bounded minimal diagnoses",
        bound_inputs={"conflicts": [["A", "B"], ["B", "C"]], "max_cardinality": 2},
        source_artifact_sha256="a" * 64,
    )
    result = DiagnosticProbeExecutor().execute(
        catalog=DiagnosticProbeCatalog(episode_id="EP-PROBE", probes=(spec,)), probe_id=spec.probe_id
    )
    assert [row["fault_ids"] for row in result.metrics["subset_minimal_diagnoses"]] == [["B"], ["A", "C"]]
    assert result.formal_evidence_generated is False


def test_r23_simulation_counterfactual_probe_executes_only_bound_traces():
    from astrograph.investigation.diagnostic_probes import DiagnosticProbeCatalog, DiagnosticProbeSpec
    from astrograph.simulation.diagnostic_provider import build_simulation_trace

    observed = build_simulation_trace(
        trace_id="OBS", time_s=[0.0, 1.0], channels={"x": [0.0, 1.0]}, source_artifact_sha256="a" * 64,
    )
    near = build_simulation_trace(
        trace_id="NEAR", time_s=[0.0, 1.0], channels={"x": [0.0, 1.1]}, source_artifact_sha256="b" * 64,
    )
    far = build_simulation_trace(
        trace_id="FAR", time_s=[0.0, 1.0], channels={"x": [0.0, 5.0]}, source_artifact_sha256="c" * 64,
    )
    spec = DiagnosticProbeSpec(
        probe_id="PROBE-SIM-1", episode_id="EP-PROBE", kind=DiagnosticProbeKind.SIMULATION_COUNTERFACTUAL_REVIEW,
        title="Counterfactual trace review", objective="Compare admitted counterfactual traces",
        bound_inputs={
            "observed_trace": observed.model_dump(mode="json"),
            "candidate_traces": [
                {"diagnosis_id": "D-A", "fault_ids": ["A"], "minimum_cardinality": True, "trace": near.model_dump(mode="json")},
                {"diagnosis_id": "D-B", "fault_ids": ["B"], "minimum_cardinality": True, "trace": far.model_dump(mode="json")},
            ],
            "channel_scales": {"x": 1.0},
        },
        source_artifact_sha256="d" * 64,
    )
    result = DiagnosticProbeExecutor().execute(
        catalog=DiagnosticProbeCatalog(episode_id="EP-PROBE", probes=(spec,)), probe_id=spec.probe_id
    )
    assert result.metrics["best_diagnosis_id"] == "D-A"
    assert result.metrics["score_semantics"] == "DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY"
    assert result.formal_evidence_eligible is False


def test_r23_model_based_catalog_builder_keeps_agent_on_frozen_probe_id():
    from astrograph.investigation.diagnostic_probes import build_model_based_probe_catalog

    catalog = build_model_based_probe_catalog(
        episode_id="EP-PROBE", conflicts=[["A", "B"], ["B", "C"]], source_artifact_sha256="e" * 64,
        max_cardinality=2,
    )
    assert catalog.telemetry_admission_status == "MODEL_CONFLICTS_HOST_BOUND"
    assert len(catalog.probes) == 1
    assert catalog.probes[0].formal_evidence_eligible is False
    result = DiagnosticProbeExecutor().execute(catalog=catalog, probe_id=catalog.probes[0].probe_id)
    assert result.metrics["search_complete"] is True
