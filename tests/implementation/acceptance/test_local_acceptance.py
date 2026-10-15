"""Acceptance wrappers are independently checked; these are not accuracy tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.acceptance import run
from scripts.acceptance.common import CATALOG_PATH, ROOT, Blocked, Context, digest, load, resolve_order, save, table
from scripts.acceptance.reporting import build
from scripts.acceptance.workspace import approve, initialize


def init(tmp_path, *extra):
    args = run.parser().parse_args(['init', '--workspace', str(tmp_path/'workspace'), *map(str, extra)])
    return initialize(args)


def stage(key):
    return next(x for x in load(CATALOG_PATH)['stages'] if x['id'] == key)


def test_catalog_unique_and_every_stage_has_implementation():
    c = load(CATALOG_PATH)
    keys = [s['id'] for s in c['stages']]
    assert len(set(keys)) == len(keys) == len(run.handlers())
    assert set(keys) == set(run.handlers())
    assert all(s['acceptance'] and s['artifacts'] and s['scope'] for s in c['stages'])
    assert len(resolve_order(keys, c)) == len(keys)


def test_dependency_order_and_deduplication():
    ordered = resolve_order(['11-detection', '12-localization'], load(CATALOG_PATH))
    assert ordered == ['01-input-contract', '10-data-inference', '11-detection', '12-localization']


def test_no_deps_means_no_implicit_execution():
    assert resolve_order(['13-classification'], load(CATALOG_PATH), dependencies=False) == ['13-classification']


@pytest.mark.parametrize('bad', ['unknown', '../10-data-inference', ''])
def test_unknown_stage_rejected(bad):
    with pytest.raises(ValueError):
        resolve_order([bad], load(CATALOG_PATH))


def test_cycle_rejected():
    c = {'stages': [{'id':'a', 'dependencies':['b']}, {'id':'b', 'dependencies':['a']}]}
    with pytest.raises(ValueError, match='cycle'):
        resolve_order(['a'], c)


def test_init_has_no_training_approval_and_no_production_memory(tmp_path):
    ws = init(tmp_path)
    assert load(ws/'training-approvals.json') == {}
    config = load(ws/'acceptance.json')
    assert 'ALREADY_USED' in config['telemetry_scope']
    s = Context(ws, stage('25-memory')).settings()
    assert s.memory_db is None and s.memory_policy is None
    assert Path(config['training_manifest']).is_file()


def test_reinitialization_never_overwrites(tmp_path):
    init(tmp_path)
    with pytest.raises(ValueError, match='already exists'):
        init(tmp_path)


def test_source_requires_own_paired_nominal(tmp_path):
    with pytest.raises(ValueError, match='together'):
        init(tmp_path, '--source', tmp_path/'a.csv')


def test_custom_training_does_not_silently_use_demo_inference(tmp_path):
    ws = init(tmp_path, '--training-manifest', tmp_path/'custom.json')
    assert load(ws/'acceptance.json')['candidate_features'] is None


def test_approval_bound_to_manifest_and_no_release(tmp_path):
    ws = init(tmp_path)
    manifest = Path(load(ws/'acceptance.json')['training_manifest'])
    r = approve(ws, manifest, 'synthetic tool check only')
    assert not r['release_authorized'] and 'SYNTHETIC' in r['scope']
    assert digest(manifest) in load(ws/'training-approvals.json')
    with pytest.raises(ValueError):
        approve(ws, tmp_path/'unselected.json', 'wrong manifest')


def test_model_hash_not_auto_approved(tmp_path):
    model = tmp_path/'other.pt'
    model.write_bytes(b'not approved')
    ws = init(tmp_path, '--model', model)
    ctx = Context(ws, stage('01-input-contract'))
    assert digest(model) not in ctx.settings().approved_model_sha256


def test_changed_input_detected_before_execution(tmp_path):
    ws = init(tmp_path)
    config = load(ws/'acceptance.json')
    Path(config['documents'][0]).write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Bound input changed'):
        run.stage_identity(ws, stage('00-environment'))


def test_missing_dependency_is_blocked_not_passed(tmp_path):
    ws = init(tmp_path)
    r = run.execute(ws, stage('11-detection'))
    assert r['status'] == 'BLOCKED' and r['error']['type'] == 'Blocked'
    assert (ws/'stages/11-detection/stderr.log').exists()


def test_missing_external_stays_blocked_even_if_protocol_tests_exist(tmp_path):
    ws = init(tmp_path)
    r = run.execute(ws, stage('34-paperseek'))
    assert r['status'] == 'BLOCKED' and 'PaperSeek' in r['error']['message']


def test_long_execution_requires_explicit_permission(tmp_path):
    ws = init(tmp_path)
    r = run.execute(ws, stage('50-three-task-replay'))
    assert r['status'] == 'BLOCKED' and '--allow-long' in r['error']['message']


def test_stage_training_needs_approval(tmp_path):
    ws = init(tmp_path)
    r = run.execute(ws, stage('40-model-train'))
    assert r['status'] == 'BLOCKED' and 'approve-training' in r['error']['message']


def test_success_cache_is_verified_and_never_rerun(tmp_path, monkeypatch):
    ws = init(tmp_path)
    calls = []
    def fn(c):
        calls.append(1)
        c.write('answer.json', {'status':'ok'})
        return {'value':1}
    monkeypatch.setattr(run, 'handlers', lambda: {'00-environment':fn})
    first = run.execute(ws, stage('00-environment'))
    assert first['status'] == 'PASSED'
    second = run.execute(ws, stage('00-environment'))
    assert first == second and len(calls) == 1
    (ws/'stages/00-environment/answer.json').write_text('{}')
    with pytest.raises(ValueError, match='Cached artifact changed'):
        run.execute(ws, stage('00-environment'))


def test_retry_archives_previous_attempt(tmp_path):
    ws = init(tmp_path)
    run.execute(ws, stage('34-paperseek'))
    run.execute(ws, stage('34-paperseek'), retry=True)
    old = list((ws/'attempts/34-paperseek').glob('*/receipt.json'))
    assert len(old) == 1 and load(old[0])['status'] == 'BLOCKED'


def test_exception_keeps_trace_and_failure_receipt(tmp_path, monkeypatch):
    ws = init(tmp_path)
    def fail(c):
        raise ValueError('intentional failure')
    monkeypatch.setattr(run, 'handlers', lambda: {'00-environment':fail})
    r = run.execute(ws, stage('00-environment'))
    assert r['status'] == 'FAILED'
    assert 'intentional failure' in (ws/'stages/00-environment/stderr.log').read_text(encoding="utf-8")


def test_process_logs_and_environment_work_from_source(tmp_path):
    ws = init(tmp_path)
    c = Context(ws, stage('00-environment'))
    c.out.mkdir(parents=True)
    c.worker(['-c', 'import astrograph; print(astrograph.__version__)'], timeout=20)
    assert (c.out/'process.stdout.log').read_text(encoding="utf-8").strip()
    assert load(c.out/'process.json')['returncode'] == 0


def test_process_timeout_keeps_logs(tmp_path):
    ws = init(tmp_path)
    c = Context(ws, stage('00-environment'))
    c.out.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='timed out'):
        c.worker(['-c', 'import time; time.sleep(30)'], timeout=.05)
    assert load(c.out/'process.json')['status'] == 'TIMEOUT'


def test_process_nonzero_not_success(tmp_path):
    ws = init(tmp_path)
    c = Context(ws, stage('00-environment'))
    c.out.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='returned 4'):
        c.worker(['-c', 'raise SystemExit(4)'])
    assert load(c.out/'process.json')['returncode'] == 4


def test_output_cannot_escape_stage(tmp_path):
    ws = init(tmp_path)
    with pytest.raises(ValueError, match='escaped'):
        Context(ws, stage('00-environment')).path('../outside.json')


def test_report_distinguishes_notrun_and_escapes_html(tmp_path):
    ws = init(tmp_path)
    path = ws/'stages/34-paperseek/receipt.json'
    save(path, {'status':'BLOCKED','summary':{'untrusted':'<script>alert(1)</script>'},'error':None,'artifacts':[]})
    r = build(ws)
    assert r['counts']['BLOCKED'] == 1 and r['counts']['NOT_RUN'] > 0
    text = (ws/'index.html').read_text(encoding="utf-8")
    assert '<script>alert' not in text and '&lt;script&gt;' in text


def test_csv_uses_bom_and_serializes_nested_fields(tmp_path):
    p = tmp_path/'x.csv'
    table(p, [{'label':'故障', 'nested':{'a':1}}])
    assert p.read_bytes().startswith(b'\xef\xbb\xbf')
    assert '故障' in p.read_text(encoding='utf-8-sig')


def test_cli_list_from_unrelated_working_directory(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT/'scripts/acceptance/run.py'), 'list', '--json'],
                         cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and len(json.loads(out.stdout)['stages']) >= 30


def test_local_default_disables_external_providers(tmp_path):
    ws = init(tmp_path)
    s = Context(ws, stage('22-agent-normal')).settings()
    assert s.planner == 'deterministic' and s.llm_config is None
    assert not s.allow_external_literature and not s.allow_mineru and not s.allow_a2o


def test_raw_inference_is_not_label_joined_scoring_input(tmp_path):
    ws = init(tmp_path)
    assert load(ws/'acceptance.json')['records'].endswith('joined_evaluation.json')


def test_live_llm_acceptance_cannot_be_faked_by_narrative_keywords():
    from scripts.acceptance.stages_tools import live_llm_accepted
    rows = [{'mode':mode,'mission_status':'SUCCEEDED','plan':{'llm_calls':0},
             'records':[],'narrative':'ACCEPTED STRUCTURE_AND_CITATIONS_VALIDATED'} for mode in ('thinking','creative')]
    assert not live_llm_accepted(rows)


def test_live_llm_requires_both_actual_modes_and_quarantine():
    from scripts.acceptance.stages_tools import live_llm_accepted
    rows = [{'mode':mode,'mission_status':'SUCCEEDED','plan':{'llm_calls':2},
             'records':[{'phase':'TASK_PLAN','outcome':'ACCEPTED'},{'phase':'EXPLANATION','outcome':'STRUCTURE_AND_CITATIONS_VALIDATED'}],
             'narrative':{'content':{'paragraphs':[{'text':'fixture','citation_ids':['s']}]}},
             'exploration':{'status':'UNVERIFIED_PROPOSAL','hypotheses':[{'hypothesis':'fixture'}],'diagnostic_vote_allowed':False,
                            'memory_write':'NONE','automatic_execution':False}} for mode in ('thinking','creative')]
    assert live_llm_accepted(rows)
    rows[1]['records'].append({'phase':'EXPLORATION','outcome':'DETERMINISTIC_FALLBACK'})
    assert not live_llm_accepted(rows)


def test_source_structure_links_and_no_product_core_import_cycles():
    assert (ROOT/'START_HERE.md').is_file()
    assert (ROOT/'docs/local_acceptance/STAGE_COMMANDS.md').is_file()
    for source in (ROOT/'src/astrograph').rglob('*.py'):
        if source.name != '__init__.py':
            assert 'scripts.acceptance' not in source.read_text(encoding='utf-8')
