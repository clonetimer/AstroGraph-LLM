from pathlib import Path
import json
import hashlib
import pytest
from astrograph.data_driven.models import DataDrivenPrediction, ClassProbability
from astrograph.oag.cli import demo_input
from astrograph.oag.contracts import OAGInput, OAGExplanation, LiteratureDocument
from astrograph.oag.assessment import DiagnosticAssessment
from astrograph.oag.harness import execute_harness, verify_run, HarnessPolicy, AgentProfile
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.integrations.platform_knowledge import PlatformKnowledgeEnrichmentService, PlatformKnowledgeConfig

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / 'benchmarks/knowledge_graph/online_extended'


def prediction_input(detected=True):
    bundle = demo_input(ROOT).bundle
    faults = sorted({d.target_fault_id for d in bundle.evidence_definitions})
    prediction = DataDrivenPrediction(
        prediction_id='test-provider-output', episode_id=bundle.episode.episode_id,
        model_id='test-three-task', model_version='1', feature_schema_id='test-features',
        input_data_sha256=bundle.feature_results[0].input_data_sha256,
        classes=[ClassProbability(class_id=f, rank=i+1, probability=1/len(faults)) for i, f in enumerate(faults)],
        metadata={'raw_output_metadata': {'detected': detected, 'detection_probability': .9 if detected else .1,
                  'detection_threshold': .5, 'predicted_component': bundle.analysis_component_instance,
                  'component_probabilities': {bundle.analysis_component_instance: 1.0},
                  'private_evaluation_label': 'MUST_NOT_REACH_PROMPT'}})
    return OAGInput.from_prediction(bundle, prediction, review_evidence=True)


class Backend:
    def __init__(self):
        self.calls = []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append(payload)
        selected = payload['summary']['suspected_fault_id']
        return OAGExplanation(candidate_fault_id=selected, abstained=selected is None,
                              assessment=DiagnosticAssessment.model_validate(payload['required_assessment']))


def execute(tmp_path, data=None, **kwargs):
    return execute_harness(data or prediction_input(), graph_dir=GRAPH,
                           output_root=tmp_path, request_id='review-1', **kwargs)


def test_three_task_bridge_preserves_all_heads_without_extra_votes(tmp_path):
    backend = Backend()
    result = execute(tmp_path, llm_backend=backend)
    assert result['status'] == 'SUCCEEDED', result.get('error')
    summary = result['output']['summary']
    assert summary['three_task_results']['additional_fusion_votes'] == 0
    assert summary['three_task_results']['detection']['detected'] is True
    assert summary['three_task_results']['classification']
    assert result['output']['assessment']['causal_identification'] == 'NOT_ESTABLISHED'
    assert 'MUST_NOT_REACH_PROMPT' not in json.dumps(backend.calls)
    assert 'run_id' not in backend.calls[0]['summary']
    assert verify_run(Path(result['harness']['run_directory']))['status'] == 'PASS'


@pytest.mark.parametrize('change', ['episode', 'window', 'hash', 'ranking', 'decision', 'localization'])
def test_three_task_binding_rejects_inconsistent_input(change):
    x = prediction_input().model_dump(mode='json')
    h = x['three_task_result']
    if change == 'episode': h['episode_id'] = 'other'
    if change == 'window': h['window_end_s'] += 1
    if change == 'hash': h['input_data_sha256'] = '0' * 64
    if change == 'ranking': x['diagnostic_ranking']['producer_version'] = 'changed'
    if change == 'decision': h['detected'] = False
    if change == 'localization': h['component_probabilities'] = {'a': .4}
    with pytest.raises(ValueError): OAGInput.model_validate(x)


def test_non_triggered_detector_cannot_yield_explainable_fault(tmp_path):
    r = execute(tmp_path, prediction_input(False))
    assert r['status'] == 'SUCCEEDED'
    assert r['output']['summary']['suspected_fault_id'] is None
    assert r['output']['assessment']['conclusion_status'] == 'UNKNOWN'


def test_harness_replay_does_not_repeat_model_calls_and_detects_tampering(tmp_path):
    backend = Backend()
    a = execute(tmp_path, llm_backend=backend)
    b = execute(tmp_path, llm_backend=backend)
    assert a == b and len(backend.calls) == 1
    run = Path(a['harness']['run_directory'])
    result = json.loads((run / 'result.json').read_text(encoding="utf-8"))
    result['status'] = 'FORGED'
    (run / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='integrity'): execute(tmp_path, llm_backend=backend)


def test_harness_id_is_bound_to_input_and_policy(tmp_path):
    execute(tmp_path)
    with pytest.raises(ValueError, match='different inputs or configuration'):
        execute(tmp_path, policy=HarnessPolicy(max_model_calls=0))


@pytest.mark.parametrize('kind', ['erase', 'mutate', 'budget', 'role', 'missing'])
def test_harness_rejects_authority_or_fidelity_failures(tmp_path, kind):
    class Invalid(Backend):
        def generate_structured(self, **kwargs):
            x = super().generate_structured(**kwargs)
            if kind == 'erase':
                x = x.model_copy(update={'assessment': x.assessment.model_copy(update={'hypotheses': []})})
            if kind == 'mutate': kwargs['payload']['summary']['candidates'] = []
            if kind == 'missing': x = x.model_copy(update={'assessment': None})
            return x
    policy = HarnessPolicy(max_model_calls=0) if kind == 'budget' else HarnessPolicy()
    if kind == 'role':
        policy = HarnessPolicy(profiles=(AgentProfile(role_id='invalid', mission='test', allowed_stages=('validate',), may_execute_control=True),))
    result = execute(tmp_path, llm_backend=Invalid(), policy=policy)
    assert result['status'] == 'FAILED'
    assert verify_run(Path(result['harness']['run_directory']))['status'] == 'PASS'


def test_prompt_over_budget_is_not_silently_truncated(tmp_path):
    b = Backend()
    result = execute(tmp_path, llm_backend=b, policy=HarnessPolicy(max_prompt_bytes=1024))
    assert result['status'] == 'FAILED' and not b.calls
    assert 'budget' in result['error']['message']


def test_upload_oag_uses_existing_job_route_and_local_documents(tmp_path):
    service = PlatformKnowledgeEnrichmentService(config=PlatformKnowledgeConfig(), llm_backend=Backend())
    store = UploadReportJobStore(tmp_path/'jobs', knowledge_service=service)
    job = store.create('This task text cannot alter the evidence')
    store.put_file(job['job_id'], 'diagnostic.json', prediction_input().model_dump_json().encode())
    store.put_file(job['job_id'], 'manual.md', b'Reaction wheel fault diagnosis evidence mechanisms bearing friction torque speed.')
    result = store.run(job['job_id'])
    assert result['status'] == 'SUCCEEDED', result.get('error')
    oag = result['analysis']['oag_result']['output']
    assert result['artifact_kind'] == 'OAG_DIAGNOSTIC'
    assert oag['literature']['paperseek_status'] == 'NOT_CONFIGURED'
    assert oag['literature']['documents']
    assert oag['literature']['documents'][0]['source_url'].startswith('urn:sha256:')
    assert store.report_markdown_path(job['job_id']).is_file()


def test_local_document_uri_must_bind_exact_content():
    with pytest.raises(ValueError):
        LiteratureDocument(document_id='a', title='a', text='a', retrieved_at='now',
                           source_content_sha256=hashlib.sha256(b'a').hexdigest(),
                           source_url='urn:sha256:' + '0'*64)


def test_async_start_queues_only_one_worker_and_freezes_inputs(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_JOB_EXECUTOR", "process")
    calls = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: calls.append((a, kw)))
    store = UploadReportJobStore(tmp_path/'jobs')
    job = store.create('async review')
    store.put_file(job['job_id'], 'input.json', prediction_input().model_dump_json().encode())
    assert store.start(job['job_id'])['status'] == 'QUEUED'
    assert store.start(job['job_id'])['status'] == 'QUEUED'
    assert len(calls) == 1 and calls[0][1]['start_new_session'] is True
    with pytest.raises(ValueError, match='immutable'):
        store.put_file(job['job_id'], 'changed.json', b'{}')
    # The worker consumes the same canonical store, including queued inputs.
    assert store.run(job['job_id'])['status'] == 'SUCCEEDED'


def test_run_lock_rejects_concurrent_job_writers(tmp_path):
    store = UploadReportJobStore(tmp_path/'jobs')
    job = store.create('run lock')
    store.put_file(job['job_id'], 'input.json', prediction_input().model_dump_json().encode())
    (store._job_root(job['job_id'])/'.run.lock').touch()
    with pytest.raises(ValueError, match='already running'): store.run(job['job_id'])


def test_render_oag_report_accepts_evidence_aware_fusion_decision_schema(tmp_path):
    from astrograph.oag.platform import render_oag_report
    data = prediction_input().model_copy(update={"decision_policy": "evidence_aware"})
    result = execute(tmp_path, data=data)
    assert result["status"] == "SUCCEEDED", result.get("error")
    decision = result["output"]["summary"]["diagnostic_decision"]
    assert "additional_observations" not in decision
    rendered = render_oag_report(result)
    assert "## 有限候选与区分观测" in rendered
    assert decision["status"] in rendered
