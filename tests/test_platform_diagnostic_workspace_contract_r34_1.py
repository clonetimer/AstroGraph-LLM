from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.diagnosis.dxc_pump_reduced_order import (
    DXCWaterPumpCandidateFit,
    DXCWaterPumpPhysicsDecision,
)
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.integrations.diagnostic_workspace_presentation import (
    SIMULATION_SEMANTICS,
    DiagnosticWorkspaceSourceBundle,
    WorkspaceSimulationSource,
    authority_snapshot,
    build_diagnostic_workspace_presentation,
    build_source_bundle,
    candidate_context_identity,
)
from astrograph.knowledge.system_model_graph import build_system_model_graph
from astrograph.simulation.adapt_pump_reduced_order import (
    ADAPTWaterPumpFaultSpec,
    ADAPTWaterPumpReducedOrderProvider,
    build_pump_boundary_condition,
)
from astrograph.simulation.diagnostic_provider import (
    SimulationHypothesisCandidate,
    build_counterfactual_request,
)


JOB_ID = "JOB-0123456789ABCDEF01234567"


def _digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _graph():
    return build_system_model_graph({
        "system_universe": "ADAPT",
        "xml_sha256": "a" * 64,
        "component_types": {
            "WaterPump": {"kind": "load", "unit": None, "unit_status": None, "fault_modes": ["FailedOff", "FlowBlocked"], "nominal_modes": ["Nominal"]},
            "FlowTransmitter": {"kind": "sensor", "unit": "GPH", "unit_status": "KNOWN", "fault_modes": [], "nominal_modes": []},
        },
        "components": [
            {"component_id": "PUMP", "component_type": "WaterPump", "kind": "load", "unit": None, "unit_status": None, "data_type": None, "nominal_modes": ["Nominal"], "fault_modes": ["FailedOff", "FlowBlocked"]},
            {"component_id": "FT", "component_type": "FlowTransmitter", "kind": "sensor", "unit": "GPH", "unit_status": "KNOWN", "data_type": "float", "nominal_modes": [], "fault_modes": []},
        ],
        "connections": [{"c1": "PUMP", "c2": "FT"}],
        "command_bindings": [],
    })


def _simulation_source() -> WorkspaceSimulationSource:
    boundary = build_pump_boundary_condition(
        branch_id="PUMP", current_sensor_id="I", voltage_sensor_id="V", flow_sensor_id="FT",
        transition_time_s=0.0, time_s=(0.0, 1.0, 2.0), observed_voltage_v=(120.0, 120.0, 120.0),
        pre_current_a=1.3, pre_voltage_v=120.0, pre_flow_gph=250.0,
        source_artifact_sha256="b" * 64,
    )
    spec = ADAPTWaterPumpFaultSpec(
        fault_id="PUMP:FlowBlocked", component_id="PUMP", fault_mode="FlowBlocked",
        rated_power_w=35.0, nominal_flow_gph=250.0,
        current_sensor_id="I", voltage_sensor_id="V", flow_sensor_id="FT",
        flow_asymptote_ratio=0.4, flow_tau_s=2.0,
    )
    provider = ADAPTWaterPumpReducedOrderProvider(boundary=boundary, fault_specs=(spec,))
    candidate = SimulationHypothesisCandidate(
        diagnosis_id="R31-PUMP-FlowBlocked", fault_ids=("PUMP:FlowBlocked",),
    )
    request = build_counterfactual_request(candidate, required_channels=("FT", "I"), time_s=boundary.time_s)
    trace = provider.predict(request)
    fit = DXCWaterPumpCandidateFit(
        diagnosis_id="R31-PUMP-FlowBlocked", fault_id="PUMP:FlowBlocked", component_id="PUMP",
        fault_mode="FlowBlocked", current_sensor_id="I", voltage_sensor_id="V", flow_sensor_id="FT",
        fitted_flow_tau_s=2.0, fitted_flow_asymptote_ratio=0.4,
        normalized_current_rmse=0.1, normalized_flow_rmse=0.2, aggregate_normalized_rmse=0.15,
        simulation_trace_sha256=trace.trace_sha256, fit_status="BEST_PHYSICAL_FIT",
    )
    decision_payload = {
        "selected_fault_ids": ("PUMP:FlowBlocked",),
        "base_fault_ids": ("PUMP:FlowBlocked",),
        "physics_preferred_fault_ids": ("PUMP:FlowBlocked",),
        "candidate_fits": [fit.model_dump(mode="json")],
        "application_status": "PUMP_PHYSICS_CORROBORATES_BASE_MODE",
        "reason_codes": ("R31_CURRENT_FLOW_REDUCED_ORDER_BEST_FIT_MATCHES_BASE",),
        "supports_base_mode": True,
    }
    decision = DXCWaterPumpPhysicsDecision(**decision_payload, result_sha256=_digest(decision_payload))
    return WorkspaceSimulationSource(
        provider_id="adapt-waterpump-reduced-order-v1",
        diagnosis_id=fit.diagnosis_id,
        candidate_id=fit.fault_id,
        channel_units={"FT": "GPH", "I": "Amps"},
        trace=trace,
        decision=decision.model_dump(mode="json"),
    )


def _inputs():
    report = {"job_id": JOB_ID, "formal_diagnosis_publishable": False, "analysis": {}}
    maintenance = {
        "diagnosis_disposition": "SUPPORTED",
        "release_status": "FORMAL_RELEASE_BLOCKED",
        "fault_id": "PUMP:FlowBlocked",
        "component_id": "PUMP",
        "evidence_summary": {"status": "UNAVAILABLE"},
    }
    visualization = {"channels": [
        {"name": "FT", "label": "Flow", "unit": "GPH", "points": [{"t": 0, "v": 250}, {"t": 1, "v": 170}]},
        {"name": "I", "label": "Current", "unit": "Amps", "points": [{"t": 0, "v": 1.3}, {"t": 1, "v": 1.3}]},
    ]}
    return report, maintenance, visualization


def test_real_graph_and_r31_trace_reach_workspace_without_changing_authority() -> None:
    report, maintenance, visualization = _inputs()
    authority = authority_snapshot(report, maintenance)
    bundle = build_source_bundle(
        job_id=JOB_ID, authority=authority, system_model_graph=_graph(), simulation_sources=(_simulation_source(),),
        candidate_context_sha256=_digest(candidate_context_identity(maintenance)),
    )
    before = authority.model_dump(mode="json")
    presentation = build_diagnostic_workspace_presentation(
        report=report, maintenance=maintenance, visualization=visualization, source_bundle=bundle,
    )
    assert presentation["schema_version"] == "astrograph-diagnostic-workspace-presentation.v1"
    assert presentation["mechanism_graph"]["status"] == "AVAILABLE"
    assert {node["id"] for node in presentation["mechanism_graph"]["nodes"]} >= {"component:PUMP", "component:FT", "fault:PUMP:FlowBlocked"}
    assert {item["channel"] for item in presentation["simulation_traces"]} == {"FT", "I"}
    assert all(item["semantics"] == SIMULATION_SEMANTICS for item in presentation["simulation_traces"])
    assert presentation["authority_snapshot"] == before
    assert authority_snapshot(report, maintenance).model_dump(mode="json") == before


def test_projection_rejects_authority_drift_and_tampered_retained_bytes() -> None:
    report, maintenance, visualization = _inputs()
    bundle = build_source_bundle(
        job_id=JOB_ID, authority=authority_snapshot(report, maintenance),
        candidate_context_sha256=_digest(candidate_context_identity(maintenance)),
        system_model_graph=_graph(), simulation_sources=(_simulation_source(),),
    )
    changed = dict(maintenance, release_status="FORMAL_RELEASE_ALLOWED")
    with pytest.raises(ValueError, match="authority snapshot drift"):
        build_diagnostic_workspace_presentation(
            report=report, maintenance=changed, visualization=visualization, source_bundle=bundle,
        )
    changed_candidate = dict(maintenance, fault_id="PUMP:FailedOff")
    with pytest.raises(ValueError, match="candidate context drift"):
        build_diagnostic_workspace_presentation(
            report=report, maintenance=changed_candidate, visualization=visualization, source_bundle=bundle,
        )
    payload = bundle.model_dump(mode="json")
    payload["system_model_graph"]["nodes"][0]["properties"]["name"] = "TAMPERED"
    with pytest.raises(ValueError):
        DiagnosticWorkspaceSourceBundle.model_validate(payload)


def test_hash_only_historical_receipt_does_not_fabricate_numeric_trace() -> None:
    report, maintenance, visualization = _inputs()
    presentation = build_diagnostic_workspace_presentation(
        report=report, maintenance=maintenance, visualization=visualization, source_bundle=None,
    )
    assert presentation["source_bundle_status"] == "NOT_RETAINED"
    assert presentation["simulation_traces"] == []
    assert presentation["mechanism_graph"]["status"] == "UNAVAILABLE"


def test_store_retains_exact_producer_bytes_and_serves_the_read_only_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, maintenance, visualization = _inputs()
    store = UploadReportJobStore(tmp_path / "jobs")
    (tmp_path / "jobs" / JOB_ID / "output").mkdir(parents=True)
    monkeypatch.setattr(store, "get", lambda job_id: {"job_id": job_id, "status": "SUCCEEDED"})
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "maintenance", lambda _job_id: maintenance)
    monkeypatch.setattr(store, "visualization", lambda _job_id: visualization)

    source = _simulation_source()
    retained = store.retain_diagnostic_workspace_sources(
        JOB_ID, system_model_graph=_graph(), simulation_sources=(source,),
    )
    result = store.diagnostic_workspace(JOB_ID)

    assert result["source_bundle_status"] == "VERIFIED"
    assert result["source_bundle_sha256"] == retained["bundle_sha256"]
    assert result["simulation_traces"][0]["source_trace_sha256"] == source.trace.trace_sha256
    assert result["mechanism_graph"]["provenance_sha256"] == _graph().graph_sha256
    assert store.retain_diagnostic_workspace_sources(
        JOB_ID, system_model_graph=_graph(), simulation_sources=(source,),
    ) == retained
    with pytest.raises(ValueError, match="immutable after retention"):
        store.retain_diagnostic_workspace_sources(JOB_ID, system_model_graph=None, simulation_sources=(source,))
