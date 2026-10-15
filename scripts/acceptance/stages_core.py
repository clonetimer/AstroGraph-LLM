"""Readouts of the existing diagnostic core, never a second implementation."""
from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
import json
import platform
import shutil
import sys

from .common import ROOT, Blocked, Context, check, digest, load, object_sha, save, table


def environment(c: Context) -> dict:
    packages = {}
    for name in ('pydantic', 'numpy', 'pandas', 'jsonschema', 'torch', 'scikit-learn', 'pytest',
                 'langgraph', 'langchain', 'sentence-transformers', 'mineru'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    c.write('environment.json', {'python': sys.version, 'executable': sys.executable,
                                'platform': platform.platform(), 'packages': packages,
                                'node': shutil.which('node'), 'npm': shutil.which('npm'),
                                'real_external_services_verified': False})
    from astrograph.agents.mission.service import MissionAgent
    c.write('capabilities.json', MissionAgent(c.settings()).inventory())
    required = ('pydantic', 'numpy', 'pandas', 'jsonschema')
    missing = [x for x in required if packages[x] is None]
    if missing:
        raise Blocked('Missing core dependencies: ' + ', '.join(missing))
    return {'core_available': True, 'optional_missing': [k for k, v in packages.items() if v is None],
            'gpu_required_for_local_sample': False}


def input_contract(c: Context) -> dict:
    import pandas as pd
    sources = {}
    for role in ('source', 'nominal'):
        path = c.required(role)
        frame = pd.read_csv(path, low_memory=False)
        check(len(frame) >= 20, f'{role}: fewer than 20 samples')
        check('time_s' in frame and frame['time_s'].notna().all(), f'{role}: invalid time_s')
        check(frame['time_s'].is_monotonic_increasing and frame['time_s'].is_unique,
              f'{role}: timestamps not strictly increasing')
        sources[role] = {'path': str(path), 'sha256': digest(path), 'shape': list(frame.shape),
                         'columns': frame.columns.tolist(), 'start_s': float(frame.time_s.iloc[0]),
                         'end_s': float(frame.time_s.iloc[-1]),
                         'missing_values_by_column': {k: int(v) for k, v in frame.isna().sum().items() if v}}
    model = c.required('model')
    approved = digest(model) in c.settings().approved_model_sha256
    check(approved, 'Model hash is not host-approved; do not silently approve a different model')
    output = {'sources': sources, 'model': {'path': str(model), 'sha256': digest(model), 'host_approved': approved},
              'scope': c.config['telemetry_scope'],
              'label_access': 'This audits raw inputs only. Exact semantic projection is performed by prepare-native.',
              'claim': 'Existing used ADCS slice by default; not a new holdout or full-dataset evaluation.'}
    c.write('inputs.json', output)
    return {'rows': {k: v['shape'][0] for k, v in sources.items()}, 'model_approved': approved}


def inference(c: Context) -> dict:
    s = c.settings()
    c.worker(['-m', 'astrograph.oag', 'prepare-native', '--domain', s.domain,
              '--evidence-profile', s.evidence_profile, '--no-mechanism-confirmation',
              '--source', str(c.required('source')), '--nominal', str(c.required('nominal')),
              '--model', str(c.required('model')), '--project-root', str(ROOT),
              '--graph-output', str(c.path('graph')), '--output', str(c.path('prepared-input.json'))],
             name='native-inference')
    from astrograph.oag.contracts import OAGInput
    data = OAGInput.model_validate_json(c.path('prepared-input.json').read_text(encoding='utf-8'))
    check(data.three_task_result is not None, 'Native provider did not emit three task heads')
    c.write('three-task.json', data.three_task_result.model_dump(mode='json'))
    c.write('evidence-bundle.json', data.bundle.model_dump(mode='json'))
    return {'model_executed': True, 'model_id': data.three_task_result.model_id,
            'episode_id': data.bundle.episode.episode_id, 'evaluation_labels_used': False}


def heads(c: Context) -> dict:
    value = load(c.previous('10-data-inference', 'three-task.json'))
    kind = c.stage['id']
    trace = value.get('head_trace')
    check(trace is not None, 'Independent head trace unavailable; do not infer it from the final class')
    if kind == '11-detection':
        out = {k: value[k] for k in ('detection_probability', 'detection_threshold', 'detected',
                                     'window_start_s', 'window_end_s', 'input_data_sha256', 'model_id')}
        out['interpretation'] = 'SINGLE_WINDOW_MODEL_OUTPUT_NOT_F1_OR_CALIBRATED_CONFIDENCE'
        c.write('detection.json', out)
        table(c.path('detection.csv'), [out])
        return out
    if kind == '12-localization':
        out = {'independent_head': trace['direct_component_probabilities'],
               'classification_derived': trace['derived_component_probabilities'],
               'deployed_distribution': value['component_probabilities'],
               'deployed_component': value['predicted_component'],
               'scope': 'Single-window component groups, not wheel instance identity or population F1.'}
        c.write('localization.json', out)
        table(c.path('localization.csv'), [{'distribution': name, 'component': key, 'probability': p}
              for name in ('independent_head', 'classification_derived', 'deployed_distribution')
              for key, p in out[name].items()])
        return {'independent_top': max(out['independent_head'], key=out['independent_head'].get),
                'deployed_component': out['deployed_component']}
    out = {'raw': trace['raw_class_probabilities'], 'conditioned': trace['conditioned_class_probabilities'],
           'class_to_runtime_fault': trace['class_to_runtime_fault'],
           'policy': trace['classification_policy'], 'discarded_class_mass': trace['discarded_class_mass'],
           'fusion_input_ranking': value['classification_ranking'],
           'scope': 'Before fusion and final decision; not a confirmed fault or classification accuracy.'}
    c.write('classification.json', out)
    rows = [{'view': view, 'class_id': key, 'score': p} for view in ('raw', 'conditioned') for key, p in (out[view] or {}).items()]
    rows += [{'view': 'fusion_input', **candidate} for candidate in value['classification_ranking']['candidates']]
    table(c.path('classification.csv'), rows)
    return {'raw_top': trace['raw_top_class'], 'policy': trace['classification_policy']}


def load_input(c: Context):
    from astrograph.oag.contracts import OAGInput
    from astrograph.oag.kernel import EvidenceDiagnosisKernel
    data = OAGInput.model_validate_json(c.previous('10-data-inference', 'prepared-input.json').read_text(encoding='utf-8'))
    kernel = EvidenceDiagnosisKernel(c.previous('10-data-inference', 'graph'))
    return data, kernel


def graph(c: Context) -> dict:
    from astrograph.knowledge import build_forward_evidence_query_plan, build_backward_fault_query_plan
    data, kernel = load_input(c)
    definitions = sorted(x.evidence_definition_id for x in data.bundle.evidence_definitions)
    faults = sorted({x.target_fault_id for x in data.bundle.evidence_definitions})
    plans = [build_forward_evidence_query_plan(definitions), build_backward_fault_query_plan(faults)]
    rows = [kernel.executor.execute(p, knowledge_version=kernel.knowledge_version).model_dump(mode='json') for p in plans]
    check(not any(x.get('truncated', False) for x in rows), 'Graph result truncated')
    c.write('reasoning.json', {'knowledge_version': kernel.knowledge_version, 'query_results': rows,
                              'directions': ['forward_evidence_to_fault', 'backward_fault_to_evidence'],
                              'final_diagnosis_generated': False})
    for name in ('nodes.csv', 'edges.csv'):
        shutil.copy2(c.previous('10-data-inference', 'graph/' + name), c.path(name))
    return {'queries': len(rows), 'knowledge_version': kernel.knowledge_version}


def fusion(c: Context) -> dict:
    data, kernel = load_input(c)
    out = kernel.evaluate(data)
    c.write('kernel-result.json', out)
    c.write('fusion.json', out['fusion'])
    c.write('diagnosis.json', out['diagnosis'])
    c.write('explanation.json', out['explanation'])
    c.path('explanation.md').write_text('# 原内核结构化解释\n\n```json\n' + json.dumps(out['explanation'], ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    candidates = out['fusion']['birrf_result']['candidates']
    table(c.path('fusion-ranking.csv'), candidates)
    return {'initial_diagnosis': out['diagnosis'], 'checks_executed': 0,
            'scope': 'Existing BiRRF and evidence decision before autonomous checks; ranks are not posterior probabilities.'}


def mission(c: Context, mode: str, *, external: bool = False, task_id: str | None = None, documents: bool = False):
    from astrograph.agents.mission.models import AgentRequest
    from astrograph.agents.mission.service import MissionAgent
    settings = c.settings(external=external)
    req = AgentRequest(task_id=task_id or ('acceptance-' + mode), task='diagnose', work_mode=mode,
                       artifacts={'input': str(c.previous('10-data-inference', 'prepared-input.json')),
                                  'graph_dir': str(c.previous('10-data-inference', 'graph')),
                                  'source': str(c.required('source')), 'nominal': str(c.required('nominal'))},
                       documents=c.config['documents'] if documents else [])
    c.write((task_id or mode) + '.settings.json', settings.model_dump(mode='json'))
    c.write((task_id or mode) + '.request.json', req.model_dump(mode='json'))
    result = MissionAgent(settings).run_supervised(req)
    check(result.status in ('SUCCEEDED', 'PARTIAL'), f'Mission failed: {result.output}')
    return result


def agent(c: Context) -> dict:
    mode = {'22-agent-normal': 'normal', '23-agent-thinking': 'thinking', '24-agent-creative': 'creative'}[c.stage['id']]
    result = mission(c, mode)
    c.write('result.json', result.model_dump(mode='json'))
    inner = result.output['diagnosis_result']
    c.write('explanation.json', inner['explanation'])
    c.path('explanation.md').write_text('# 原内核结构化解释\n\n```json\n' + json.dumps(inner['explanation'], ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    c.write('checks.json', {k: v for k, v in inner.items() if k not in ('fusion', 'summary', 'diagnosis', 'explanation')})
    initial = load(c.previous('21-fusion', 'diagnosis.json'))
    c.write('decision-delta.json', {'before_checks': initial, 'after_checks': inner['diagnosis'],
                                   'changed': initial != inner['diagnosis'],
                                   'new_accuracy_claim': False})
    if mode != 'normal':
        normal = load(c.previous('22-agent-normal', 'result.json'))['output']['diagnosis_result']
        c.write('mode-comparison.json', {'normal_diagnosis_sha256': object_sha(normal['diagnosis']),
                                        'this_diagnosis_sha256': object_sha(inner['diagnosis']),
                                        'diagnosis_equal': normal['diagnosis'] == inner['diagnosis'],
                                        'mode_receipt': result.mode_receipt,
                                        'live_llm_executed': False})
    if mode == 'creative':
        proposal = result.output['exploration']
        check(proposal['diagnostic_vote_allowed'] is False and proposal['memory_write'] == 'NONE', 'Exploration escaped quarantine')
        c.write('exploration.json', proposal)
    return {'mission_status': result.status, 'work_mode': mode, 'planner': 'deterministic',
            'checks_executed': inner['checks_executed'], 'diagnosis': inner['diagnosis'],
            'live_llm_executed': False}


def structured(c: Context) -> dict:
    from astrograph.agents.mission.models import AgentResult
    from astrograph.agents.mission.storage import Workspace
    data = load(c.previous('22-agent-normal', 'result.json'))
    AgentResult.model_validate(data)
    taskroot = c.previous('22-agent-normal', 'tasks/acceptance-normal')
    w = Workspace(taskroot.parent, [c.workspace, ROOT], taskroot.name)
    event_hash = w.verify_events()
    saved = w.begin(load(taskroot / 'request_identity.json'))
    check(saved == data, 'Completed task readback differs')
    c.write('result.schema.json', AgentResult.model_json_schema())
    out = {'schema_valid': True, 'event_chain_valid': True, 'event_tip': event_hash,
           'persisted_artifacts_valid': True, 'completed_task_replayed': False,
           'round_trip_equal': saved == data}
    c.write('validation.json', out)
    return out


HANDLERS = {'00-environment': environment, '01-input-contract': input_contract, '10-data-inference': inference,
            '11-detection': heads, '12-localization': heads, '13-classification': heads,
            '20-graph-reasoning': graph, '21-fusion': fusion, '22-agent-normal': agent,
            '23-agent-thinking': agent, '24-agent-creative': agent, '61-structured-output': structured}
