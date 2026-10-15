from __future__ import annotations

from astrograph.data_driven import (
    ActivityAxis,
    ActivityAxisGate,
    ActivityAxisGateStatus,
    CampaignScenario,
    CampaignSplitCounts,
    CampaignVariant,
    IndependentRunCampaignSpec,
    decide_campaign_activity_axes,
    default_v0522_activity_axis_manifest,
    requested_axes_from_campaign_spec,
)


def test_default_activity_axis_manifest_accepts_seed_only_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(manifest, requested_axes=[ActivityAxis.SEED])

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_fault_timing_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.FAULT_TIMING],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.FAULT_TIMING, ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_mapped_severity_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.SEVERITY],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.SEED, ActivityAxis.SEVERITY]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_sensor_noise_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.NOISE],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.NOISE, ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_sensor_dropout_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.DROPOUT],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.DROPOUT, ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_operating_condition_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.OPERATING_CONDITION],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.OPERATING_CONDITION, ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_initial_state_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.INITIAL_STATE],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.INITIAL_STATE, ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_satellite_instance_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[ActivityAxis.SEED, ActivityAxis.SATELLITE_INSTANCE],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [ActivityAxis.SATELLITE_INSTANCE, ActivityAxis.SEED]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_accepts_flight_data_intake_request() -> None:
    manifest = default_v0522_activity_axis_manifest()
    decision = ActivityAxisGate().decide(
        manifest,
        requested_axes=[
            ActivityAxis.FLIGHT_DATA,
        ],
    )

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.accepted_axes == [
        ActivityAxis.FLIGHT_DATA,
    ]
    assert decision.blocked_axes == []


def test_default_activity_axis_manifest_has_no_unsupported_axes_after_flight_data_intake() -> None:
    manifest = default_v0522_activity_axis_manifest()

    assert manifest.unsupported_axes == []


def test_default_activity_axis_manifest_separates_simulation_and_dataset_intake_domains() -> None:
    manifest = default_v0522_activity_axis_manifest()
    domains = {item.axis: item.capability_domain for item in manifest.capabilities}

    assert domains[ActivityAxis.FLIGHT_DATA] == "dataset_intake"
    assert all(
        domain == "simulation_runtime"
        for axis, domain in domains.items()
        if axis != ActivityAxis.FLIGHT_DATA
    )


def test_campaign_spec_requested_axes_are_inferred_from_variants_and_cli() -> None:
    spec = IndependentRunCampaignSpec(
        campaign_id="CAMP-AXES",
        scenarios=[CampaignScenario(scenario_id="fault_a", class_ids=["F-A"])],
        seeds=[1, 2, 3],
        variants=[
            CampaignVariant(variant_id="A", fault_severity_id="LOW", cli_args=["--noise-profile", "quiet"]),
            CampaignVariant(
                variant_id="B",
                fault_severity_id="HIGH",
                cli_args=["--fault-start-s", "20", "--dataset-root", "flight/telemetry.csv"],
            ),
        ],
        split_counts=CampaignSplitCounts(train=1, validation=1, test=1),
    )

    assert requested_axes_from_campaign_spec(spec) == [
        ActivityAxis.FAULT_TIMING,
        ActivityAxis.FLIGHT_DATA,
        ActivityAxis.NOISE,
        ActivityAxis.SEED,
        ActivityAxis.SEVERITY,
    ]


def test_campaign_spec_duration_s_does_not_imply_fault_timing_axis() -> None:
    spec = IndependentRunCampaignSpec(
        campaign_id="CAMP-DURATION-ONLY",
        scenarios=[CampaignScenario(scenario_id="fault_a", class_ids=["F-A"])],
        seeds=[1],
        variants=[
            CampaignVariant(variant_id="A", cli_args=["--window-duration-s", "120"]),
        ],
        split_counts=CampaignSplitCounts(train=1, validation=1, test=1),
    )

    assert ActivityAxis.FAULT_TIMING not in requested_axes_from_campaign_spec(spec)


def test_campaign_activity_gate_accepts_cross_operating_condition_request() -> None:
    spec = IndependentRunCampaignSpec(
        campaign_id="CAMP-BLOCK",
        scenarios=[CampaignScenario(scenario_id="fault_a", class_ids=["F-A"])],
        seeds=[1, 2, 3],
        variants=[
            CampaignVariant(variant_id="A", operating_condition_id="OC-A"),
            CampaignVariant(variant_id="B", operating_condition_id="OC-B"),
            CampaignVariant(variant_id="C", operating_condition_id="OC-C"),
        ],
        split_counts=CampaignSplitCounts(train=1, validation=1, test=1),
    )

    decision = decide_campaign_activity_axes(spec, default_v0522_activity_axis_manifest())

    assert decision.status == ActivityAxisGateStatus.ACCEPTED
    assert decision.blocked_axes == []
