"""Real task artifacts and HTTP API for source-bound follow-up analysis."""
from copy import deepcopy
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading

from fastapi.testclient import TestClient
import pytest

from astrograph.api import create_app
from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


def conversation():
    from astrograph.integrations import external_diagnosis_conversation
    return external_diagnosis_conversation


def output_for(payload, *, initial=False):
    sid = payload['sources'][0]['source_id']
    cited = {'text': '报告描述响应延迟，仍需核对独立读数。', 'citation_ids': [sid]}
    if initial:
        return {'summary': '响应延迟需要复核。', 'findings': [cited], 'hypotheses': [],
                'information_gaps': ['独立读数'], 'recommended_review_actions': ['比对两路读数']}
    return {'answer': [cited], 'information_gaps': ['独立读数的时间戳']}


class Backend:
    def __init__(self, change=None):
        self.calls = []
        self.change = change

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, deepcopy(payload)))
        data = output_for(payload, initial=agent_id == 'external-diagnosis-analysis')
        if self.change and agent_id != 'external-diagnosis-analysis':
            self.change(data)
        return data


def stage(store, text='外部诊断：未登记组件响应变慢。\n独立读数尚未提供。'):
    job_id = store.create('解释外部诊断结果。')['job_id']
    store.put_file(job_id, 'diagnosis.txt', text.encode('utf-8'))
    store.put_file(job_id, 'analysis_input.json', json.dumps({
        'schema_version': 'astrograph-result-log-review.v1',
        'analysis_mode': 'EXTERNAL_DIAGNOSIS_ANALYSIS', 'diagnosis_results': ['diagnosis.txt'],
    }).encode())
    store.run(job_id)
    return job_id


@pytest.fixture
def prepared(tmp_path):
    backend = Backend()
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=
        PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend))
    job_id = stage(store)
    backend.calls.clear()
    return store, job_id, backend


def ask(prepared, *, question='为什么需要独立读数？', request_id='request-0001', **extra):
    store, job_id, _backend = prepared
    input_hash = store.report(job_id)['analysis']['knowledge_enrichment']['model_explanation']['input_sha256']
    return conversation().ask_follow_up(store, job_id, question=question,
                                       input_sha256=input_hash, request_id=request_id, **extra)


def test_follow_up_uses_all_task_sources_and_preserves_original_artifacts(prepared):
    store, job_id, backend = prepared
    output = store._job_root(job_id) / 'output'
    paths = [output / 'report.json', output / 'report.md', output / 'external_diagnosis_input.json',
             output / 'knowledge/external_diagnosis_analysis.json']
    before = {path: path.read_bytes() for path in paths}
    state = ask(prepared)
    assert state['revision'] == 1
    assert state['analysis_verification'] == 'UNVERIFIED_MODEL_ANALYSIS'
    assert state['automatic_control_authorized'] is False
    assert state['turns'][0]['status'] == 'SUCCEEDED'
    assert state['turns'][0]['citation_reference_validation'] == 'PASS'
    assert len(backend.calls) == 1 and backend.calls[0][0] == 'external-diagnosis-follow-up'
    payload = backend.calls[0][1]
    assert payload['sources'] == state['sources']
    assert payload['question'] == '为什么需要独立读数？'
    assert payload['prior_analysis']['summary'] == '响应延迟需要复核。'
    assert payload['history'] == []
    assert 'untrusted' in payload['system_policy'].lower()
    assert all(path.read_bytes() == value for path, value in before.items())
    reopened = UploadReportJobStore(store.root, knowledge_service=store.knowledge_service)
    assert conversation().get_conversation(reopened, job_id) == state


def test_idempotence_and_history_come_from_saved_task(prepared):
    first = ask(prepared)
    assert ask(prepared) == first
    assert len(prepared[2].calls) == 1
    second = ask(prepared, request_id='request-0002', question='应如何比对？')
    assert second['revision'] == 2
    history = prepared[2].calls[-1][1]['history']
    assert len(history) == 1 and history[0]['question'] == first['turns'][0]['question']
    assert history[0]['answer'] == first['turns'][0]['answer']
    with pytest.raises(conversation().ConversationConflict, match='FOLLOW_UP_REQUEST_CONFLICT'):
        ask(prepared, question='复用同一编号但换问题')
    assert len(prepared[2].calls) == 2


@pytest.mark.parametrize('change,reason', [
    (lambda d: d['answer'][0].update(citation_ids=['foreign-source']), 'CITATION_REFERENCE_INVALID'),
    (lambda d: d['answer'][0].update(citation_ids=[]), 'MODEL_OUTPUT_INVALID'),
    (lambda d: d['answer'][0].update(citation_ids=['same', 'same']), 'MODEL_OUTPUT_INVALID'),
    (lambda d: d.update(answer=[]), 'MODEL_OUTPUT_INVALID'),
    (lambda d: d.update(control_command='execute'), 'MODEL_OUTPUT_INVALID'),
])
def test_invalid_answer_is_never_published_and_prior_history_remains(prepared, change, reason):
    first = ask(prepared)
    prepared[2].change = change
    state = ask(prepared, request_id='request-0002')
    assert state['turns'][0] == first['turns'][0]
    failed = state['turns'][-1]
    assert failed['status'] == 'FAILED' and failed['reason_code'] == reason
    assert 'answer' not in failed
    assert state['sources'] == first['sources']


def test_constructed_model_is_revalidated(prepared):
    class InvalidBackend:
        def generate_structured(self, **kwargs):
            return kwargs['output_model'].model_construct(answer=[], information_gaps=[])
    prepared[0].knowledge_service.llm_backend = InvalidBackend()
    assert ask(prepared)['turns'][0]['reason_code'] == 'MODEL_OUTPUT_INVALID'


def test_provider_error_is_fixed_and_not_used_as_history(prepared):
    class BrokenBackend:
        last_call_metadata = {'api_key': 'PRIVATE_RECEIPT'}
        def generate_structured(self, **kwargs):
            raise RuntimeError('SECRET_BEARER_PROVIDER_BODY')
    store, job_id, backend = prepared
    store.knowledge_service.llm_backend = BrokenBackend()
    state = ask(prepared)
    assert state['turns'][0]['reason_code'] == 'MODEL_GENERATION_FAILED'
    serialized = (store._job_root(job_id) / 'output/external_diagnosis_conversation.json').read_text()
    assert 'PRIVATE_RECEIPT' not in serialized and 'SECRET_BEARER' not in serialized
    store.knowledge_service.llm_backend = backend
    ask(prepared, request_id='request-0002')
    assert backend.calls[0][1]['history'] == []


def test_no_model_and_large_input_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("astrograph.integrations.external_diagnosis_conversation.MAX_SOURCE_CHARACTERS", 60_000)
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig()))
    job_id = stage(store)
    state = ask((store, job_id, None))
    assert state['turns'][0]['status'] == 'NOT_CONFIGURED'
    assert 'answer' not in state['turns'][0]
    large_id = stage(store, '字' * 60001)
    state = ask((store, large_id, None))
    assert state['turns'][0]['status'] == 'INPUT_TOO_LARGE'
    assert len(state['sources'][0]['text']) == 60001


def test_selected_import_only_engine_does_not_use_global_backend(prepared):
    store, job_id, backend = prepared
    job = store.get(job_id)
    job['analysis_engine_id'] = 'rules'
    store._save(job)
    assert ask(prepared)['turns'][0]['status'] == 'NOT_CONFIGURED'
    assert backend.calls == []


@pytest.mark.parametrize('target', ['bundle', 'report', 'client_hash'])
def test_changed_bindings_fail_before_model(prepared, target):
    store, job_id, backend = prepared
    state = conversation().get_conversation(store, job_id)
    ask(prepared)
    backend.calls.clear()
    output = store._job_root(job_id) / 'output'
    if target == 'bundle':
        path = output / 'external_diagnosis_input.json'
        bundle = json.loads(path.read_text())
        bundle['sources'][0]['text'] = '篡改原文'
        path.write_text(json.dumps(bundle))
    elif target == 'report':
        path = output / 'report.json'
        report = json.loads(path.read_text())
        report['task'] = '修改后的任务'
        path.write_text(json.dumps(report))
    with pytest.raises(conversation().ConversationConflict):
        conversation().ask_follow_up(store, job_id, question='继续分析', request_id='request-0002',
            input_sha256='0' * 64 if target == 'client_hash' else state['input_sha256'])
    assert backend.calls == []


def test_non_external_or_unfinished_job_cannot_enter_follow_up(prepared):
    store, job_id, backend = prepared
    other = store.create('未完成')['job_id']
    with pytest.raises(conversation().ConversationConflict):
        conversation().get_conversation(store, other)
    path = store._job_root(job_id) / 'output/report.json'
    report = json.loads(path.read_text())
    report['analysis']['analysis_mode'] = 'MATERIAL_REVIEW'
    path.write_text(json.dumps(report))
    with pytest.raises(conversation().ConversationConflict):
        conversation().get_conversation(store, job_id)
    assert backend.calls == []


def test_actual_history_character_limit_stops_without_dropping_sources(prepared):
    backend = prepared[2]
    backend.change = lambda d: d.update(answer=[{
        'text': '已生成的未验证分析' * 300, 'citation_ids': d['answer'][0]['citation_ids'],
    }])
    last = None
    for i in range(12):
        last = ask(prepared, question='请进一步分析：' + '问题' * 1800, request_id=f'long-history-{i:02d}')
        if last['turns'][-1]['status'] == 'CONTEXT_TOO_LARGE':
            break
    assert last['turns'][-1]['status'] == 'CONTEXT_TOO_LARGE'
    assert len(backend.calls) == len(last['turns']) - 1
    assert len(last['sources']) == 2


def test_saved_attempt_limit_and_idempotent_read_at_limit(prepared):
    for i in range(12):
        final = ask(prepared, request_id=f'request-{i:04d}')
    assert len(final['turns']) == 12
    assert ask(prepared, request_id='request-0011') == final
    with pytest.raises(conversation().ConversationConflict, match='FOLLOW_UP_LIMIT_REACHED'):
        ask(prepared, request_id='request-over-limit')
    assert len(prepared[2].calls) == 12


def test_concurrent_request_rejected_and_lock_released(prepared):
    store, job_id, backend = prepared
    entered, release = threading.Event(), threading.Event()
    original = backend.generate_structured
    def blocking(**kwargs):
        entered.set()
        assert release.wait(10)
        return original(**kwargs)
    backend.generate_structured = blocking
    result, errors = [], []
    def worker():
        try:
            result.append(ask(prepared))
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(conversation().ConversationConflict, match='FOLLOW_UP_BUSY'):
            ask(prepared, request_id='request-concurrent')
    finally:
        release.set()
        thread.join(10)
    assert not errors and result[0]['turns'][0]['status'] == 'SUCCEEDED'
    backend.generate_structured = original
    assert ask(prepared, request_id='request-next')['revision'] == 2


def test_separate_worker_crash_releases_conversation_lock(tmp_path):
    path = tmp_path / 'worker.lock'
    code = (
        'from pathlib import Path\n'
        'import sys\n'
        'from astrograph.integrations.external_diagnosis_conversation import _exclusive_call\n'
        'with _exclusive_call(Path(sys.argv[1])):\n'
        ' print("LOCKED", flush=True)\n'
        ' sys.stdin.read()\n'
    )
    process = subprocess.Popen([sys.executable, '-c', code, str(path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'LOCKED'
        with pytest.raises(conversation().ConversationConflict, match='FOLLOW_UP_BUSY'):
            with conversation()._exclusive_call(path):
                pytest.fail('second worker acquired an active lock')
    finally:
        process.kill()
        process.communicate(timeout=10)
    with conversation()._exclusive_call(path):
        assert path.exists()  # an existing lock file is not a permanently active lock


def test_report_change_while_model_is_running_prevents_publication(prepared):
    store, job_id, backend = prepared
    original = backend.generate_structured
    def modify_report(**kwargs):
        path = store._job_root(job_id) / 'output/report.json'
        data = json.loads(path.read_text())
        data['task'] = 'changed while calling provider'
        path.write_text(json.dumps(data))
        return original(**kwargs)
    backend.generate_structured = modify_report
    with pytest.raises(conversation().ConversationConflict, match='FOLLOW_UP_INPUT_CHANGED'):
        ask(prepared)
    assert not (store._job_root(job_id) / 'output/external_diagnosis_conversation.json').exists()


def test_corrupt_saved_history_is_not_sent_to_model(prepared):
    store, job_id, backend = prepared
    ask(prepared)
    path = store._job_root(job_id) / 'output/external_diagnosis_conversation.json'
    history = json.loads(path.read_text())
    history['turns'][0]['answer'][0]['citation_ids'] = ['foreign-task-source']
    path.write_text(json.dumps(history))
    backend.calls.clear()
    with pytest.raises(conversation().ConversationConflict, match='FOLLOW_UP_HISTORY_INVALID'):
        ask(prepared, request_id='request-after-corruption')
    assert backend.calls == []


@contextmanager
def model_wire():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            payload = json.loads(body['messages'][-1]['content'])['payload']
            calls.append(payload)
            initial = 'question' not in payload
            raw = json.dumps({'choices': [{'message': {'content': json.dumps(output_for(payload, initial=initial), ensure_ascii=False)},
                                         'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_api_real_http_follow_up_permissions_idempotence_and_immutable_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE', 'bearer')
    monkeypatch.setenv('ASTROGRAPH_API_TOKENS_JSON', json.dumps({'op-token': ['operator'], 'read-token': ['viewer']}))
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path / 'platform'))
    with model_wire() as (url, calls):
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            provider='scripted', base_url=url, model='test-model', tokenizer_mode='disabled',
            timeout_s=15, max_tokens=1200, response_format_strategy='json_object'))
        store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=
            PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend))
        app = create_app()
        app.state.upload_report_store = store
        op = {'Authorization': 'Bearer op-token'}
        viewer = {'Authorization': 'Bearer read-token'}
        with TestClient(app) as client:
            for asset in ('/platform/mission/assets/external_follow_up.js',
                          '/platform/assets/v0.21.4/external_follow_up.js'):
                script = client.get(asset)
                assert script.status_code == 200 and 'ExternalDiagnosisFollowUp' in script.text
                assert 'javascript' in script.headers['content-type']
            assert '/platform/assets/v0.21.4/external_follow_up.js' in client.get('/platform/legacy').text
            job_id = client.post('/v1/platform/jobs', json={'task': '分析外部文本'}, headers=op).json()['job_id']
            prefix = f'/v1/platform/jobs/{job_id}'
            manifest = {'schema_version': 'astrograph-result-log-review.v1', 'analysis_mode': 'EXTERNAL_DIAGNOSIS_ANALYSIS', 'diagnosis_results': ['diagnosis.txt']}
            for name, data in [('diagnosis.txt', '外部诊断响应延迟。'.encode()), ('analysis_input.json', json.dumps(manifest).encode())]:
                assert client.put(prefix + '/files/' + name, content=data, headers=op).status_code == 200
            assert client.post(prefix + '/preflight', headers=op).json()['can_start']
            assert client.post(prefix + '/run', headers=op).status_code == 200
            before = client.get(prefix + '/report', headers=viewer).content
            initial = client.get(prefix + '/follow-up', headers=viewer)
            assert initial.status_code == 200
            assert initial.headers['cache-control'] == 'no-store'
            request = {'question': '为何需要核查？', 'input_sha256': initial.json()['input_sha256'], 'request_id': 'request-api-01'}
            assert client.post(prefix + '/follow-up', json=request, headers=viewer).status_code == 403
            assert client.get(prefix + '/follow-up').status_code == 401
            response = client.post(prefix + '/follow-up', json=request, headers=op)
            assert response.status_code == 200 and response.json()['turns'][0]['status'] == 'SUCCEEDED'
            assert response.headers['cache-control'] == 'no-store'
            assert client.post(prefix + '/follow-up', json=request, headers=op).json() == response.json()
            assert len(calls) == 2
            assert client.get(prefix + '/report', headers=viewer).content == before
            assert client.post(prefix + '/follow-up', json={**request, 'question': '换问题'}, headers=op).status_code == 409
            for bad in ({**request, 'sources': []}, {**request, 'question': ' '}, {**request, 'question': 'x' * 4001}, {**request, 'request_id': '../bad'}):
                assert client.post(prefix + '/follow-up', json=bad, headers=op).status_code == 422
