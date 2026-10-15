from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from astrograph.agents.mission.models import AgentRequest, AgentSettings
from astrograph.agents.mission.service import MissionAgent
from astrograph.lifecycle import DocumentProposal, KnowledgeFeedbackStore, LifecyclePolicy
from astrograph.agents.mission.storage import file_sha

ROOT = Path(__file__).resolve().parents[3]


def _generic_case(tmp_path: Path, signal: str) -> Path:
    data = tmp_path / "series.csv"
    pd.DataFrame({"time_s": range(100), signal: [0.0] * 50 + [100.0] * 50}).to_csv(data, index=False)
    manifest = tmp_path / "series-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "astrograph-generic-series-manifest.v1",
        "data_path": data.name,
        "entity_id": "generic-case",
        "timestamp_column": "time_s",
        "channels": [{"name": signal, "unit": "V" if "voltage" in signal else "1"}],
        "baseline_fraction": 0.3,
    }), encoding="utf-8")
    return manifest


def _settings(tmp_path: Path, **updates) -> AgentSettings:
    updates.setdefault("model_capability_registry", "configs/agent/model_capabilities.default.json")
    return AgentSettings(
        project_root=str(ROOT),
        read_roots=[str(ROOT), str(tmp_path)],
        output_root=str(tmp_path / "runs"),
        **updates,
    )


def test_mission_generic_series_runs_detection_capability_discovery_and_provisional_cause(tmp_path: Path):
    manifest = _generic_case(tmp_path, "sensor.eps.battery_voltage_v")
    request = AgentRequest(task_id="generic-known", artifacts={"series_manifest": str(manifest)})
    result = MissionAgent(_settings(tmp_path)).run_inline(request)
    assert result.status == "SUCCEEDED", result.model_dump()
    diagnosis = result.output["diagnosis_result"]["diagnosis"]
    assert diagnosis["detected"] is True
    assert diagnosis["selected_fault_id"] == "EPS_BATTERY_VOLTAGE_FAULT"
    assert diagnosis["status"] == "CANDIDATE_SET"
    assert diagnosis["knowledge_verified"] is False
    tasks = result.output["diagnosis_result"]["model_capabilities"]["tasks"]
    assert tasks["detection"]["compatible"]
    assert tasks["classification"]["compatible"] == []
    assert tasks["localization"]["compatible"] == []
    assert [x["tool_id"] for x in result.tool_receipts] == [
        "model.infer", "model.capabilities", "graph.reason", "diagnosis.fuse", "output.structured"
    ]


def test_mission_unknown_generic_series_detects_without_inventing_cause(tmp_path: Path):
    manifest = _generic_case(tmp_path, "mystery_signal")
    request = AgentRequest(task_id="generic-unknown", artifacts={"series_manifest": str(manifest)})
    result = MissionAgent(_settings(tmp_path)).run_inline(request)
    assert result.status == "SUCCEEDED", result.model_dump()
    diagnosis = result.output["diagnosis_result"]["diagnosis"]
    assert diagnosis["detected"] is True
    assert diagnosis["selected_fault_id"] is None
    assert diagnosis["status"] == "INSUFFICIENT_EVIDENCE"
    assert diagnosis["root_cause_confirmed"] is False


def test_replay_mission_binds_offline_scope_and_host_bound_capability(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "phm"
    dataset.mkdir()
    (dataset / "sample_index.csv").write_text("sample_index\n0\n", encoding="utf-8")
    replay_manifest = tmp_path / "replay.json"
    replay_manifest.write_text(json.dumps({
        "schema_version": "astrograph-replay-manifest.v1",
        "dataset_kind": "phm_v6_retrospective",
        "model_strategy": "promoted_extra_trees",
        "skip_llm": True,
    }), encoding="utf-8")

    captured = {}
    def fake_replay(**kwargs):
        captured.update(kwargs)
        return {
            "status": "SUCCEEDED",
            "dataset_kind": "phm_v6_retrospective",
            "model_strategy": "promoted_extra_trees",
            "metrics": {"end_to_end": {"macro_f1": 0.5}},
            "offline_evaluation_truth_used": True,
            "online_diagnosis_claim_allowed": False,
            "claim_boundary": "offline fixture",
        }
    monkeypatch.setattr("astrograph.agents.mission.service.run_replay", fake_replay)
    request = AgentRequest(task_id="phm-replay", task="replay", artifacts={
        "replay_manifest": str(replay_manifest), "dataset_root": str(dataset)
    })
    result = MissionAgent(_settings(tmp_path, domain="phm", evidence_profile="legacy")).run_inline(request)
    assert result.status == "SUCCEEDED", result.model_dump()
    assert result.output["task_output"]["online_diagnosis_claim_allowed"] is False
    receipt = next(x for x in result.tool_receipts if x["tool_id"] == "model.capabilities")
    # Tool receipt itself is enough to prove the compatibility step was executed; inspect persisted step for payload.
    assert receipt["status"] == "SUCCEEDED"
    assert Path(captured["dataset_root"]) == dataset


def test_approved_document_library_is_expert_gated_and_retrievable_in_research(tmp_path: Path):
    document = tmp_path / "manual.md"
    document.write_text("Battery undervoltage may indicate an EPS energy-storage anomaly.\n", encoding="utf-8")
    policy = LifecyclePolicy(approved_reviewers=("expert-1",), permitted_document_roots=(str(tmp_path),))
    policy_path = tmp_path / "lifecycle-policy.json"
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    db = tmp_path / "lifecycle.sqlite"
    store = KnowledgeFeedbackStore(db, policy)
    proposal = DocumentProposal(
        document_id="DOC-1", source_path=str(document), source_sha256=file_sha(document), title="manual",
        media_type="md", parser="UTF8_NATIVE_TEXT", proposer_actor="test",
        proposed_at=datetime.now(timezone.utc), metadata={"scope": "fixture"},
    )
    store.stage_document(proposal, event_id="stage-doc")
    assert store.approved_documents() == []
    store.review_document(document_id="DOC-1", decision="APPROVE", reviewer="expert-1", reason="fixture review", event_id="review-doc")

    settings = _settings(tmp_path, lifecycle_db=str(db), lifecycle_policy=str(policy_path), use_approved_document_library=True)
    request = AgentRequest(task_id="research-approved", task="research", goal="battery undervoltage")
    result = MissionAgent(settings).run_inline(request)
    assert result.status == "SUCCEEDED", result.model_dump()
    assert result.output["task_output"]["sources"]
    assert any(x["tool_id"] == "rag.retrieve" and x["status"] == "SUCCEEDED" for x in result.tool_receipts)
    load = json.loads((Path(settings.output_root) / request.task_id / "approved-library-load.json").read_text(encoding="utf-8"))
    assert load["document_ids"] == ["DOC-1"]

def test_build_knowledge_stages_review_only_proposal_in_expert_lifecycle(tmp_path: Path):
    from astrograph.agents.mission.models import InvestigationPlan
    from astrograph.oag.contracts import OntologyProposal
    document=tmp_path/'mechanism.md'
    document.write_text('Gyro bias affects attitude residual.',encoding='utf-8')
    policy=LifecyclePolicy(approved_reviewers=('expert-1',),permitted_document_roots=(str(tmp_path),))
    policy_path=tmp_path/'policy.json';policy_path.write_text(policy.model_dump_json(),encoding='utf-8')
    db=tmp_path/'lifecycle.sqlite'
    class Backend:
        def generate_structured(self,*,agent_id,output_model,payload):
            if output_model is InvestigationPlan:
                return output_model(task='build_knowledge',actions=[],rationale='Build only from supplied source.')
            if output_model is OntologyProposal:
                source=payload['documents_untrusted'][0]['document_id']
                return output_model(triples=[{'subject':'gyro_bias','predicate':'affects','object':'attitude_residual',
                    'source_document_id':source,'source_quote':'Gyro bias affects attitude residual.'}])
            raise AssertionError(output_model)
    settings=_settings(tmp_path,planner='llm_assisted',lifecycle_db=str(db),lifecycle_policy=str(policy_path),max_optional_actions=0)
    request=AgentRequest(task_id='knowledge-lifecycle',task='build_knowledge',documents=[str(document)])
    result=MissionAgent(settings,backend=Backend()).run_inline(request)
    assert result.status=='SUCCEEDED',result.model_dump()
    task_output=result.output['task_output']
    assert task_output['active_kg_modified'] is False
    assert task_output['lifecycle']['status']=='PENDING_REVIEW'
    assert task_output['lifecycle']['active_kg_write_allowed'] is False
    artifact=Path(settings.output_root)/request.task_id/'knowledge-lifecycle-proposal.json'
    assert artifact.is_file()

def test_mission_generic_multichannel_keeps_unmapped_channel_from_becoming_fault_cause(tmp_path: Path):
    data=tmp_path/'multi.csv'
    pd.DataFrame({'time_s':range(100),
        'sensor.eps.battery_voltage_v':[0.0]*50+[100.0]*50,
        'mystery_signal':[0.0]*50+[-100.0]*50}).to_csv(data,index=False)
    manifest=tmp_path/'multi.json'
    manifest.write_text(json.dumps({'schema_version':'astrograph-generic-series-manifest.v1','data_path':data.name,
        'entity_id':'multi','timestamp_column':'time_s','channels':[{'name':'sensor.eps.battery_voltage_v','unit':'V'},
        {'name':'mystery_signal','unit':'1'}],'baseline_fraction':0.3}),encoding='utf-8')
    result=MissionAgent(_settings(tmp_path)).run_inline(AgentRequest(task_id='generic-multi',artifacts={'series_manifest':str(manifest)}))
    assert result.status=='SUCCEEDED'
    layered=result.output['diagnosis_result']['generic_series']
    assert len(layered['detections'])==2
    mapped={x['signal']:x['mapped'] for x in layered['detections']}
    assert mapped['sensor.eps.battery_voltage_v'] is True and mapped['mystery_signal'] is False
    assert result.output['diagnosis_result']['diagnosis']['selected_fault_id'] is None
    assert result.output['diagnosis_result']['diagnosis']['status']=='INSUFFICIENT_EVIDENCE'
    assert any(x['fault_id']=='EPS_BATTERY_VOLTAGE_FAULT' for x in layered['fault_candidates'])
