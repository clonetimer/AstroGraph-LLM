"""Approved candidate training and wrappers for existing replay/benchmark scripts."""
from __future__ import annotations

from pathlib import Path
import shutil

from .common import ROOT, Blocked, Context, check, digest, load, table


def train(c: Context) -> dict:
    from astrograph.agents.mission.models import AgentRequest
    from astrograph.agents.mission.service import MissionAgent
    manifest = c.required('training_manifest')
    sha = digest(manifest)
    approvals = load(c.workspace / 'training-approvals.json')
    if sha not in approvals:
        raise Blocked('Training requires explicit approval. Run approve-training --workspace ... --manifest PATH --reason TEXT; the demo manifest is inputs/demo-training.json.')
    s = c.settings().model_copy(update={'approved_training_manifest_sha256': [sha]})
    req = AgentRequest(task_id='acceptance-train', task='train_model', artifacts={'training_manifest': str(manifest)})
    c.write('request.json', req.model_dump(mode='json'))
    c.write('settings.json', s.model_dump(mode='json'))
    result = MissionAgent(s).run_supervised(req)
    c.write('result.json', result.model_dump(mode='json'))
    check(result.status == 'SUCCEEDED', f'Training task did not complete: {result.output}')
    out = result.output['task_output']
    c.write('training-report.json', out)
    check(out['weights_changed'] is True and out['active_model_modified'] is False, 'Weights did not change or production modified')
    return {'weights_changed': True, 'scope': approvals[sha]['scope'],
            'validation_metrics': out['validation_metrics'], 'production_admitted': False,
            'candidate_manifest': out['candidate_manifest']}


def candidate(c: Context) -> dict:
    from astrograph.agents.mission.models import AgentRequest
    from astrograph.agents.mission.service import MissionAgent
    package = load(c.previous('40-model-train', 'training-report.json'))['candidate_manifest']
    features = c.required('candidate_features')
    req = AgentRequest(task_id='acceptance-candidate', task='infer_candidate',
                       artifacts={'candidate_manifest': package, 'features': str(features)})
    c.write('request.json', req.model_dump(mode='json'))
    s = c.settings()
    s = s.model_copy(update={'read_roots': [*s.read_roots, str(Path(package).parent)]})
    out = MissionAgent(s).run_supervised(req)
    check(out.status == 'SUCCEEDED', f'Candidate inference failed: {out.output}')
    result = out.output['task_output']
    c.write('result.json', out.model_dump(mode='json'))
    c.write('candidate-inference.json', result)
    rows = []
    for head, values in result['heads'].items():
        for sample, probabilities in enumerate(values['probabilities']):
            check(abs(sum(probabilities) - 1) < .001, 'Unnormalized head probabilities')
            for class_id, p in zip(values['classes'], probabilities, strict=True):
                rows.append({'head': head, 'sample': sample, 'class_id': class_id, 'probability': p})
    table(c.path('candidate-heads.csv'), rows)
    return {'heads': list(result['heads']), 'score_semantics': result['score_semantics'],
            'production_admitted': False, 'labels_read': False}


def replay(c: Context) -> dict:
    c.long_task()
    zip_path = c.required('dataset_zip')
    out = c.path('replay')
    c.worker(['scripts/evaluate_three_task_agent_zip.py', '--dataset-zip', str(zip_path),
              '--project-root', str(ROOT), '--output', str(out), '--cohort', c.config['cohort']],
             timeout=c.config['long_timeout_s'], name='full-replay')
    return {'cohort': c.config['cohort'], 'output': str(out), 'new_holdout': False}


def rescore(c: Context) -> dict:
    records = c.required('records')
    c.worker(['scripts/score_three_task_replay.py', '--records', str(records), '--output', str(c.path('metrics.json'))],
             timeout=300, name='rescore')
    return {'records_sha256': digest(records), 'retrained': False, 'replayed_model': False,
            'scope': 'Only recomputes metrics from supplied records; not an independent test.'}


def comparison_prepare(c: Context) -> dict:
    c.long_task()
    c.worker(['scripts/prepare_mode_benchmark.py', '--dataset-zip', str(c.required('dataset_zip')),
              '--output', str(c.path('benchmark-data'))], timeout=c.config['long_timeout_s'], name='prepare-benchmark')
    return {'path': str(c.path('benchmark-data')), 'evaluation_side_reads_labels': True,
            'online_diagnosis_reads_labels': False}


def comparison_train(c: Context) -> dict:
    c.long_task()
    c.worker(['scripts/run_mode_benchmark.py', '--data', str(c.previous('52-comparison-prepare', 'benchmark-data')),
              '--output', str(c.path('trained-runs')), '--workers', str(c.config['workers']),
              '--acceptance-junit', str(c.previous('60-regression', 'junit.xml'))],
             timeout=c.config['long_timeout_s'], name='benchmark-training')
    return {'scope': 'SIX_REFERENCE_ADAPTATIONS_THREE_INITIALIZATIONS_NOT_OFFICIAL_2026_SOTA',
            'trained_runs': str(c.path('trained-runs'))}


def comparison_normal(c: Context) -> dict:
    c.long_task()
    data = c.previous('52-comparison-prepare', 'benchmark-data')
    trained = c.previous('53-comparison-train', 'trained-runs')
    variants = [('production', c.required('model'))]
    variants += [(str(seed), trained / f'residual_tcn__{seed}' / 'normal-candidate.pt')
                 for seed in (20260913, 20260914, 20260915)]
    for name, model in variants:
        c.worker(['scripts/replay_normal_mode_benchmark.py', '--dataset-zip', str(c.required('dataset_zip')),
                  '--data', str(data), '--output', str(c.path('normal-runs/' + name)),
                  '--model', str(model), '--name', name], timeout=c.config['long_timeout_s'], name='normal-' + name)
    return {'normal_runs': str(c.path('normal-runs')), 'production_model_replaced': False,
            'information_condition': 'Normal mode adds paired reference and knowledge/check observations.'}


def comparison_score(c: Context) -> dict:
    if c.config.get('benchmark_dir'):
        b = c.required('benchmark_dir')
        data, runs, normal = b/'benchmark_data', b/'trained_runs', b/'normal_runs'
    else:
        data = c.previous('52-comparison-prepare', 'benchmark-data')
        runs = c.previous('53-comparison-train', 'trained-runs')
        normal = c.previous('54-comparison-normal', 'normal-runs')
    if not all(p.is_dir() for p in (data, runs, normal)):
        raise Blocked('Unpack the benchmark ZIP and set --benchmark-dir to its root containing benchmark_data, trained_runs, normal_runs')
    c.worker(['scripts/score_mode_benchmark.py', '--data', str(data), '--runs', str(runs),
              '--normal-runs', str(normal), '--output', str(c.path('comparison'))], timeout=300, name='score-comparison')
    return {'scored_from': str(data.parent), 'retrained': False, 'model_rerun': False,
            'scope': 'Previously used data; official/latest SOTA replication is not claimed.'}


HANDLERS = {'40-model-train': train, '41-candidate-infer': candidate, '50-three-task-replay': replay,
            '51-three-task-rescore': rescore, '52-comparison-prepare': comparison_prepare,
            '53-comparison-train': comparison_train, '54-comparison-normal': comparison_normal,
            '55-comparison-rescore': comparison_score}
