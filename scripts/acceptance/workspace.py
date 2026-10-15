"""Explicit host-side preparation for local acceptance, without touching production."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .common import ROOT, check, digest, load, now, object_sha, save


def code_sha() -> str:
    # Do not bind history/result inventories: adding an acceptance receipt is not a code change.
    files = [ROOT/'pyproject.toml', ROOT/'configs/acceptance/stages.json']
    files += sorted((ROOT/'scripts/acceptance').glob('*.py'))
    files += sorted((ROOT/'src/astrograph').rglob('*.py'))
    return object_sha({str(p.relative_to(ROOT)): digest(p) for p in files})


def absolute(value: str | None) -> str | None:
    return str(Path(value).expanduser().resolve()) if value else None


def make_demo(path: Path) -> tuple[Path, Path]:
    import numpy as np
    rng = np.random.default_rng(4816)
    data = {}
    for split, count in (('train', 24), ('validation', 12)):
        y = np.tile(np.arange(3), count // 3)
        x = rng.normal(0, .08, (count, 20, 3)).astype('float32')
        x[y == 1, :, 0] += .7
        x[y == 2, :, 1] -= .7
        data[split + '_x'] = x
        data[split + '_anomaly'] = (y > 0).astype('int64')
        data[split + '_component'] = np.where(y > 0, y - 1, -1).astype('int64')
        data[split + '_fault'] = np.where(y > 0, y - 1, -1).astype('int64')
        data[split + '_group'] = np.array([f'{split}-synthetic-group-{i}' for i in range(count)])
    dataset = path/'demo-training.npz'
    np.savez_compressed(dataset, **data)
    features = path/'demo-inference.npz'
    np.savez_compressed(features, x=rng.normal(0, .1, (6, 20, 3)).astype('float32'),
                        channel_names=np.array(['signal_a', 'signal_b', 'signal_c']),
                        channel_units=np.array(['1', '1', '1']))
    manifest = path/'demo-training.json'
    save(manifest, {'schema_version': 'astrograph-agent-training.v1', 'data_path': str(dataset),
                    'data_sha256': digest(dataset), 'channel_names': ['signal_a', 'signal_b', 'signal_c'],
                    'channel_units': ['1', '1', '1'], 'fault_ids': ['DEMO_FAULT_A', 'DEMO_FAULT_B'],
                    'component_ids': ['DEMO_COMPONENT_A', 'DEMO_COMPONENT_B'],
                    'fault_to_component': {'DEMO_FAULT_A': 'DEMO_COMPONENT_A', 'DEMO_FAULT_B': 'DEMO_COMPONENT_B'},
                    'epochs': 2, 'width': 8, 'batch_size': 8, 'learning_rate': .001, 'seed': 4816})
    save(path/'demo-provenance.json', {'scope': 'SYNTHETIC_TOOL_SMOKE_NOT_ADCS_PERFORMANCE',
                                     'seed': 4816, 'generated_by': 'scripts/acceptance/workspace.py',
                                     'production_model_admission': False})
    return manifest, features


def initialize(args: argparse.Namespace) -> Path:
    from astrograph.agents.mission.models import AgentSettings
    path = Path(args.workspace).expanduser().resolve()
    if path.exists():
        raise ValueError('Workspace already exists; choose a new directory. Existing results are never overwritten by init.')
    if path == ROOT or any(path.is_relative_to(ROOT/x) for x in ('src', 'models', 'configs', 'data', 'governance', 'tests')):
        raise ValueError('Acceptance workspace cannot be placed inside immutable source/model/config/data directories')
    if bool(args.source) != bool(args.nominal):
        raise ValueError('Provide --source and --nominal together; do not pair user data with the built-in reference')
    sfile = Path(args.settings).resolve() if args.settings else ROOT/'configs/agent/mission.default.json'
    settings = load(sfile)
    # Existing example settings use project-relative paths. Normalize once; invocation cwd is irrelevant afterward.
    project = Path(settings.get('project_root', '.'))
    project = project.resolve() if project.is_absolute() else (ROOT/project).resolve()
    for key in ('llm_config', 'memory_policy', 'memory_db', 'a2o_source_root', 'paperseek_source_root',
                'embedding_model_path', 'reranker_model_path'):
        if settings.get(key):
            p = Path(settings[key]).expanduser()
            settings[key] = str(p.resolve() if p.is_absolute() else (project/p).resolve())
    settings['read_roots'] = [str(Path(x).resolve() if Path(x).is_absolute() else (project/x).resolve())
                              for x in settings.get('read_roots', ['.'])]
    settings['project_root'] = str(ROOT)
    AgentSettings.model_validate(settings)
    path.mkdir(parents=True, exist_ok=False)
    inputs = path/'inputs'
    inputs.mkdir()
    manifest, features = make_demo(inputs)
    # Exercise real cleaning while preserving negative signs, units and negation.
    note = inputs/'acceptance-note.txt'
    note.write_bytes(b'# Gyro bias\r\nGyro bias requires residual verification; not confirmed by text alone.  \r\n'
                     b'The limit is -0.05 rad/s; do not change signs or units.\x00\r\n\r\n'
                     b'# Wheel torque\r\nReaction wheel torque and speed require measured comparison.\r\n')
    (inputs/'retrieval-gold.md').write_text(
        '# Gyro\nGYRO_SENTINEL gyro bias residual requires measured checks.\n\n'
        '# Wheel\nWHEEL_SENTINEL reaction wheel torque speed requires measured checks.\n', encoding='utf-8')
    legacy = ROOT/'governance/implementation/post48_dev3/cli_real'
    config = {'schema_version': 'astrograph-local-acceptance.v1', 'created_at': now(),
              'project_root': str(ROOT), 'workspace': str(path),
              'telemetry_scope': 'USER_SUPPLIED' if args.source else 'ALREADY_USED_ADCS_SLICE_NOT_NEW_TEST',
              'source': absolute(args.source) or str(legacy/'source.csv'),
              'nominal': absolute(args.nominal) or str(legacy/'nominal.csv'),
              'model': absolute(args.model) or str(ROOT/'models/adcs_v5_2_2/p2_innovation_v2.pt'),
              'documents': [absolute(x) for x in args.document] if args.document else [str(note)],
              'nontext_document': absolute(args.nontext_document),
              'training_manifest': absolute(args.training_manifest) or str(manifest),
              'candidate_features': absolute(args.candidate_features) or str(features),
              'dataset_zip': absolute(args.dataset_zip), 'benchmark_dir': absolute(args.benchmark_dir),
              'phm_dataset_root': absolute(args.phm_dataset_root),
              'records': absolute(args.records) or str(ROOT/'governance/implementation/post48_dev3/used_test_final/joined_evaluation.json'),
              'query': args.query, 'cohort': args.cohort, 'workers': args.workers,
              'long_timeout_s': args.long_timeout, 'settings_origin': str(sfile),
              'external_services_verified': False}
    if args.training_manifest and not args.candidate_features:
        config['candidate_features'] = None  # never infer a custom model on mismatched demo channels
    save(path/'host-settings.json', settings)
    bound = {}
    for name in ('source', 'nominal', 'model', 'training_manifest', 'candidate_features', 'nontext_document'):
        value = config.get(name)
        if value and Path(value).is_file():
            bound[value] = digest(Path(value))
    for value in config['documents']:
        if Path(value).is_file():
            bound[value] = digest(Path(value))
    if config.get('phm_dataset_root'):
        marker=Path(config['phm_dataset_root'])/'sample_index.csv'
        if not marker.is_file(): raise ValueError('--phm-dataset-root requires sample_index.csv')
        bound[str(marker.resolve())]=digest(marker)
    bound[str(path/'host-settings.json')] = digest(path/'host-settings.json')
    config['input_bindings'] = bound
    save(path/'acceptance.json', config)
    save(path/'training-approvals.json', {})
    (path/'README.md').write_text('# 本地验收工作区\n\n先读 `index.html` 或 `SUMMARY.md`。\n'
                                '本目录允许写入；源码、原模型与生产记忆保持不变。\n'
                                '演示训练和确认案例都是显式夹具，不能作为诊断准确率证据。\n', encoding='utf-8')
    return path


def approve(path: Path, manifest: Path, reason: str) -> dict:
    from astrograph.agents.mission.training import TrainingManifest, validate_dataset
    config = load(path/'acceptance.json')
    if manifest.resolve() != Path(config['training_manifest']).resolve():
        raise ValueError('Approve only the manifest selected by this workspace; create a new workspace to change inputs')
    m = TrainingManifest.model_validate_json(manifest.read_text(encoding='utf-8'))
    data = Path(m.data_path)
    if not data.is_absolute():
        data = manifest.parent/data
    validate_dataset(m, data)
    sha = digest(manifest)
    approvals = load(path/'training-approvals.json')
    scope = 'SYNTHETIC_TOOL_SMOKE_NOT_ADCS_PERFORMANCE' if manifest.resolve() == path/'inputs/demo-training.json' else 'USER_APPROVED_OFFLINE_CANDIDATE_ONLY'
    receipt = {'manifest': str(manifest.resolve()), 'manifest_sha256': sha,
               'dataset_sha256': digest(data), 'reason': reason, 'approved_at': now(),
               'scope': scope, 'release_authorized': False, 'approval_source': 'EXPLICIT_LOCAL_HOST_COMMAND'}
    approvals[sha] = receipt
    save(path/'training-approvals.json', approvals)
    return receipt
