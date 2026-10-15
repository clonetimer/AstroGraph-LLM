from __future__ import annotations

from astrograph.diagnosis.model_simulation import run_model_simulation_diagnosis
from astrograph.evaluation.nasa_adapt_adapter import system_catalog
from astrograph.knowledge.system_model_graph import ObservationMismatch, build_system_model_graph
from astrograph.simulation.diagnostic_provider import PrecomputedTraceSimulationProvider, build_simulation_trace

XML = b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<modeCatalog>
 <modeGroup><name>A</name><mode xsi:type="faultMode"><name>F1</name></mode></modeGroup>
 <modeGroup><name>B</name><mode xsi:type="faultMode"><name>F2</name></mode></modeGroup>
</modeCatalog>
<componentTypeCatalog>
 <componentType xsi:type="switch"><name>TA</name><modesRef>A</modesRef></componentType>
 <componentType xsi:type="sensor"><name>TB</name><modesRef>B</modesRef><engUnits>V</engUnits></componentType>
</componentTypeCatalog>
<systems><system><components>
 <component><name>A1</name><componentType>TA</componentType><command><name>A1_CMD</name><commandValue><dataType>bool</dataType></commandValue></command></component>
 <component><name>B1</name><componentType>TB</componentType></component>
</components><connections><connection><c1>A1</c1><c2>B1</c2></connection></connections></system></systems></root>'''


def test_integrated_service_can_stop_at_model_diagnosis_without_simulator():
    graph = build_system_model_graph(system_catalog(XML))
    result = run_model_simulation_diagnosis(
        graph=graph,
        mismatches=[ObservationMismatch(mismatch_id="M1", sensor_component_id="B1", command_id="A1_CMD")],
        max_topology_hops=1,
    )
    assert result.status == "MODEL_DIAGNOSIS_COMPLETE_SIMULATION_NOT_ADMITTED"
    assert result.counterfactual_review is None
    assert {item.fault_ids for item in result.diagnosis.subset_minimal_diagnoses} == {("A1:F1",), ("B1:F2",)}


def test_integrated_service_adds_counterfactual_ranking_without_formal_promotion():
    graph = build_system_model_graph(system_catalog(XML))
    model_only = run_model_simulation_diagnosis(
        graph=graph,
        mismatches=[ObservationMismatch(mismatch_id="M1", sensor_component_id="B1", command_id="A1_CMD")],
    )
    observed = build_simulation_trace(
        trace_id="OBS", time_s=[0, 1, 2], channels={"v": [0, 1, 2]}, source_artifact_sha256="a"*64,
    )
    traces = {}
    for candidate in model_only.diagnosis.subset_minimal_diagnoses:
        values = [0, 1.05, 2.05] if candidate.fault_ids == ("A1:F1",) else [0, 4, 5]
        traces[candidate.diagnosis_id] = build_simulation_trace(
            trace_id=candidate.diagnosis_id, time_s=[0, 1, 2], channels={"v": values}, source_artifact_sha256="b"*64,
        )
    result = run_model_simulation_diagnosis(
        graph=graph,
        mismatches=[ObservationMismatch(mismatch_id="M1", sensor_component_id="B1", command_id="A1_CMD")],
        observed_trace=observed,
        simulation_provider=PrecomputedTraceSimulationProvider(traces),
        channel_scales={"v": 1.0},
    )
    assert result.status == "MODEL_DIAGNOSIS_COMPLETE_WITH_COUNTERFACTUAL_REVIEW"
    best = next(item for item in result.diagnosis.subset_minimal_diagnoses if item.fault_ids == ("A1:F1",))
    assert result.counterfactual_review is not None
    assert result.counterfactual_review.best_diagnosis_id == best.diagnosis_id
    assert result.formal_evidence_eligible is False
