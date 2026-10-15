from __future__ import annotations

import json
from pathlib import Path

from astrograph.diagnosis.dxc_expected_response import DXCExpectedResponseConfig, rank_dxc_expected_response
from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.dxc_discrete_consistency import DXCDiscreteConsistencyConfig, diagnose_discrete_state_consistency
from astrograph.evaluation.dxc_hybrid_model_based import DXCHybridPrediction
from astrograph.evaluation.dxc_r27_expected_response import assemble_dxc_r27_prediction
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeExecutor,
    DiagnosticProbeKind,
    build_dxc_expected_response_probe_catalog,
)


XML = b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<modeCatalog>
  <modeGroup><name>RelayMode</name><mode xsi:type="mode"><name>NominalClosed</name></mode><mode xsi:type="mode"><name>NominalOpen</name></mode><mode xsi:type="faultMode"><name>StuckOpen</name></mode></modeGroup>
  <modeGroup><name>PositionMode</name><mode xsi:type="mode"><name>Nominal</name></mode><mode xsi:type="faultMode"><name>Stuck</name></mode></modeGroup>
</modeCatalog>
<componentTypeCatalog>
  <componentType xsi:type=""><name>Relay</name><modesRef>RelayMode</modesRef></componentType>
  <componentType xsi:type="sensor"><name>PositionSensor</name><modesRef>PositionMode</modesRef><sensorValue><dataType>bool</dataType></sensorValue></componentType>
</componentTypeCatalog>
<systems><system><components>
  <component><name>R1</name><componentType>Relay</componentType><command><name>R1_CL</name><commandValue><dataType>bool</dataType></commandValue></command></component>
  <component><name>P1</name><componentType>PositionSensor</componentType></component>
</components><connections><connection><c1>R1</c1><c2>P1</c2></connection></connections></system></systems>
</root>'''


def test_expected_response_prefers_physical_fan_mode_over_sensor_offset():
    catalog = {
        "components": [
            {"component_id": "S1", "kind": "sensor", "component_type": "SpeedTransmitter", "unit_status": "VERIFIED_FROM_XML", "fault_modes": ["Offset", "Stuck"]},
            {"component_id": "FAN1", "kind": "component", "component_type": "LargeFan", "fault_modes": ["OverSpeed", "UnderSpeed", "FailedOff"]},
        ],
        "connections": [{"c1": "S1", "c2": "FAN1"}],
    }
    records = [
        {"kind": "sensors", "time_tick": i * 1000, "values": {"S1": 100.0 if i < 40 else 120.0}}
        for i in range(60)
    ]
    decision = rank_dxc_expected_response(
        records=records,
        catalog=catalog,
        continuous_fault_ids=("S1:Offset",),
        discrete_events=(),
        discrete_fault_ids=(),
        discrete_first_event_time_s=None,
        config=DXCExpectedResponseConfig(startup_s=20.0),
    )
    assert decision.selected_fault_ids == ("FAN1:OverSpeed",)
    assert decision.candidates[0].source == "DIRECT_EXPECTED_RESPONSE"
    assert decision.score_semantics.endswith("NOT_PROBABILITY")
    assert decision.formal_evidence_eligible is False


def test_agent_expected_response_probe_is_read_only_mirror_of_core_decision():
    records, _ = parse_scenario("\n".join([
        "sensors @0 {P1=true};",
        "sensors @1000 {P1=false};",
    ]))
    catalog = system_catalog(XML)
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["r27-probe"]):
        discrete = diagnose_discrete_state_consistency(records, catalog, DXCDiscreteConsistencyConfig(startup_s=0.0))
    continuous = DXCHybridPrediction(
        predicted_fault_ids=("P1:Stuck",),
        first_alarm_time_s=0.5,
        final_alarm_time_s=1.0,
        final_component_ids=("P1",),
        model_sha256="1" * 64,
        graph_sha256="2" * 64,
    )
    prediction = assemble_dxc_r27_prediction(
        records=records,
        catalog=catalog,
        continuous=continuous,
        discrete=discrete,
        expected_response_config=DXCExpectedResponseConfig(startup_s=0.1),
    )
    assert prediction.agent_required is False
    before = prediction.predicted_fault_ids

    catalog_probe = build_dxc_expected_response_probe_catalog(
        episode_id="EP-R27",
        decision=prediction.expected_response,
        source_artifact_sha256="a" * 64,
    )
    spec = catalog_probe.probes[0]
    assert spec.kind == DiagnosticProbeKind.DXC_EXPECTED_RESPONSE_REVIEW
    assert spec.formal_evidence_eligible is False
    result = DiagnosticProbeExecutor().execute(catalog=catalog_probe, probe_id=spec.probe_id)
    assert tuple(result.metrics["selected_fault_ids"]) == before
    assert result.metrics["review_semantics"] == "READ_ONLY_MIRROR_OF_DETERMINISTIC_CORE_DECISION"
    assert result.formal_evidence_generated is False
    assert result.formal_evidence_eligible is False
    assert prediction.predicted_fault_ids == before


def test_r27_algorithm_policy_retains_legacy_and_does_not_delegate_core_authority_to_agent():
    path = Path(__file__).resolve().parents[1] / "configs" / "evaluation" / "r27_dxc_algorithm_policy.v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    assert policy["legacy_branch"]["retained"] is True
    assert policy["legacy_branch"]["removed"] is False
    assert policy["candidate_branch"]["role"] == "PRIMARY_CAPABLE_DEVELOPMENT_CANDIDATE"
    assert policy["candidate_branch"]["agent_required"] is False
    assert policy["agent_layer"]["diagnostic_authority"] is False
    assert policy["agent_layer"]["may_change_thresholds_or_model"] is False


def test_r27_architecture_keeps_legacy_and_does_not_require_agent():
    import json
    from pathlib import Path

    policy = json.loads(Path("configs/evaluation/r27_dxc_algorithm_policy.v1.json").read_text(encoding="utf-8"))
    assert policy["architecture"] == "LEGACY_RETAINED_PLUS_DETERMINISTIC_PRIMARY_CANDIDATE_PLUS_OPTIONAL_AGENT_SHADOW_REVIEW"
    assert policy["legacy_branch"]["retained"] is True
    assert policy["legacy_branch"]["removed"] is False
    assert policy["legacy_branch"]["runtime_default_changed_by_r27"] is False
    assert policy["candidate_branch"]["agent_required"] is False
    assert policy["candidate_branch"]["runtime_default_changed_by_r27"] is False
    assert policy["candidate_branch"]["legacy_fallback_available"] is True
    assert "DXC_EXPECTED_RESPONSE_REVIEW" in policy["agent_layer"]["existing_tools"]
    assert policy["agent_layer"]["diagnostic_authority"] is False
    assert policy["agent_layer"]["performance_dependency"] == "NONE_CORE_EXECUTES_WITHOUT_AGENT"
