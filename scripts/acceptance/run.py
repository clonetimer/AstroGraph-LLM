#!/usr/bin/env python3
"""Cross-platform stage-by-stage local acceptance; run from any working directory."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for base in (ROOT, ROOT/'src'):
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

# Nested Mission training workers must see this source checkout without shell-specific exports.
os.environ['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from scripts.acceptance.common import (CATALOG_PATH, Blocked, Context, digest, load, now,
                                       object_sha, resolve_order, save)
from scripts.acceptance.workspace import approve, code_sha, initialize
from scripts.acceptance.reporting import build


def handlers():
    from scripts.acceptance import stages_core, stages_tools, stages_experiments, stages_p1_p4, stages_extended
    return {**stages_core.HANDLERS, **stages_tools.HANDLERS, **stages_experiments.HANDLERS, **stages_p1_p4.HANDLERS, **stages_extended.HANDLERS}


def verify_cached(workspace: Path, receipt: dict) -> None:
    for item in receipt.get('artifacts', []):
        p = workspace/item['path']
        if not p.resolve().is_relative_to(workspace.resolve()):
            raise ValueError('Receipt contains a path outside the acceptance workspace')
        if not p.is_file() or digest(p) != item['sha256']:
            raise ValueError('Cached artifact changed: ' + item['path'])


def stage_identity(workspace: Path, stage: dict) -> dict:
    config = load(workspace/'acceptance.json')
    for value, expected in config['input_bindings'].items():
        p = Path(value)
        if not p.is_file() or digest(p) != expected:
            raise ValueError('Bound input changed or missing; create a new workspace: ' + value)
    deps = {}
    for name in stage['dependencies']:
        path = workspace/'stages'/name/'receipt.json'
        if not path.is_file():
            raise Blocked(f'Required stage {name} has not run')
        r = load(path)
        if r['status'] != 'PASSED':
            raise Blocked(f'Required stage {name} is {r["status"]}')
        verify_cached(workspace, r)
        deps[name] = digest(path)
    result = {'config_sha256': object_sha(config), 'code_sha256': code_sha(), 'stage_sha256': object_sha(stage),
              'dependency_receipts': deps}
    if stage['id'] == '40-model-train':
        result['approvals_sha256'] = digest(workspace/'training-approvals.json')
    return result


def execute(workspace: Path, stage: dict, *, retry: bool = False, allow_long: bool = False) -> dict:
    c = Context(workspace, stage, allow_long=allow_long)
    old = c.out/'receipt.json'
    # Do not conflate an attempted stage with a passed stage. A blocked stage can
    # be rerun explicitly after supplying approval; previous attempts are retained.
    if c.out.exists():
        if old.exists() and not retry:
            receipt = load(old)
            if receipt['status'] != 'PASSED':
                print(f'{stage["id"]}: {receipt["status"]} (saved attempt; use --retry)', flush=True)
                return receipt
            identity = stage_identity(workspace, stage)
            if receipt.get('identity') != identity:
                raise ValueError(f'{stage["id"]}: code/config/dependency changed; use a new workspace or --retry')
            verify_cached(workspace, receipt)
            print(f'{stage["id"]}: PASSED (verified cached artifacts; not re-executed)', flush=True)
            return receipt
        if not retry:
            raise RuntimeError('Interrupted stage found. Review logs and use --retry; never silently re-execute uncertain side effects.')
        archive = workspace/'attempts'/stage['id']
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(c.out), str(archive/(now().replace(':', '-'))))
    c.out.mkdir(parents=True, exist_ok=False)
    started = now()
    status, error, summary, identity = 'FAILED', None, {}, None
    print(f'{stage["id"]}: STARTED — {stage["title"]}', flush=True)
    with c.path('stdout.log').open('w', encoding='utf-8') as stdout, c.path('stderr.log').open('w', encoding='utf-8') as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                identity = stage_identity(workspace, stage)
                summary = handlers()[stage['id']](c)
                status = summary.pop('_status', 'PASSED')
                if status not in ('PASSED', 'PARTIAL'):
                    raise ValueError('Stage returned an unsupported success status')
            except (Blocked, ModuleNotFoundError) as exc:
                status = 'BLOCKED'
                error = {'type': type(exc).__name__, 'message': str(exc)}
                traceback.print_exc()
            except Exception as exc:
                status = 'FAILED'
                error = {'type': type(exc).__name__, 'message': str(exc)}
                traceback.print_exc()
    artifacts = []
    for p in sorted(c.out.rglob('*')):
        if p.is_file() and p.name != 'receipt.json' and not p.is_symlink():
            artifacts.append({'path': p.relative_to(workspace).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size})
    receipt = {'schema_version': 'astrograph-local-stage-receipt.v1', 'stage': stage['id'], 'title': stage['title'],
               'started_at': started, 'finished_at': now(), 'status': status, 'scope': stage['scope'],
               'acceptance': stage['acceptance'], 'identity': identity, 'summary': summary, 'error': error,
               'commands': c.commands, 'artifacts': artifacts, 'production_model_modified': False,
               'production_knowledge_modified': False, 'production_memory_modified': False}
    save(old, receipt)
    eventfile = workspace/'events.jsonl'
    previous = '0'*64
    if eventfile.exists():
        records = eventfile.read_text(encoding='utf-8').splitlines()
        if records:
            previous = json.loads(records[-1])['event_sha256']
    event = {'at': now(), 'stage': stage['id'], 'status': status, 'receipt_sha256': digest(old), 'previous_sha256': previous}
    event['event_sha256'] = object_sha(event)
    with eventfile.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(f'{stage["id"]}: {status}' + (f' — {error["message"]}' if error else ''), flush=True)
    return receipt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='List all stages and their coverage')
    listing.add_argument('--json', action='store_true')
    init = sub.add_parser('init', help='Create a new isolated workspace and synthetic training fixture; never approves training')
    init.add_argument('--workspace', required=True, type=Path)
    init.add_argument('--settings', type=Path)
    for key in ('source', 'nominal', 'model', 'nontext-document', 'dataset-zip', 'records', 'benchmark-dir', 'training-manifest', 'candidate-features', 'phm-dataset-root'):
        init.add_argument('--' + key)
    init.add_argument('--document', action='append', default=[])
    init.add_argument('--query', default='gyro bias residual verification')
    init.add_argument('--cohort', choices=['development', 'used-test-regression'], default='used-test-regression')
    init.add_argument('--workers', type=int, default=3)
    init.add_argument('--long-timeout', type=float, default=14400)
    run = sub.add_parser('run', help='Run individual stages, a named suite, or all; full replays require --allow-long')
    run.add_argument('stages', nargs='*')
    run.add_argument('--suite', choices=['core', 'local', 'external', 'regression', 'extended', 'all'])
    run.add_argument('--workspace', type=Path, required=True)
    run.add_argument('--retry', action='store_true', help='Archive and rerun the selected stages/dependencies')
    run.add_argument('--no-deps', action='store_true', help='Do not run dependencies; absent prerequisites are BLOCKED')
    run.add_argument('--allow-long', action='store_true')
    approve_p = sub.add_parser('approve-training', help='Explicit local host approval; never production model promotion')
    approve_p.add_argument('--workspace', type=Path, required=True)
    approve_p.add_argument('--manifest', type=Path, help='Defaults to the manifest selected when the workspace was initialized')
    approve_p.add_argument('--reason', required=True)
    for command in ('report', 'show', 'commands'):
        q = sub.add_parser(command)
        q.add_argument('--workspace', type=Path, required=True)
        if command == 'show':
            q.add_argument('stage')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    catalog = load(CATALOG_PATH)
    try:
        if args.command == 'list':
            if args.json:
                # Machine-readable output must survive callers using the native
                # Windows code page instead of UTF-8 text mode.
                print(json.dumps(catalog, ensure_ascii=True, indent=2))
            else:
                for s in catalog['stages']:
                    print(f'{s["id"]:26} {s["category"]:10} {s["title"]} | {s["scope"]}')
            return 0
        ws = args.workspace.expanduser().resolve()
        if args.command == 'init':
            if args.workers < 1 or args.long_timeout <= 0:
                raise ValueError('workers and timeout must be positive')
            initialize(args)
            build(ws)
            print(json.dumps({'workspace': str(ws), 'index': str(ws/'index.html'), 'training_approved': False}, ensure_ascii=False))
            return 0
        if not (ws/'acceptance.json').is_file():
            raise ValueError('Run init first: missing acceptance.json')
        if args.command == 'approve-training':
            manifest = args.manifest.resolve() if args.manifest else Path(load(ws/'acceptance.json')['training_manifest'])
            print(json.dumps(approve(ws, manifest, args.reason), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'report':
            summary = build(ws)
            print(json.dumps({'counts': summary['counts'], 'html': str(ws/'index.html'), 'markdown': str(ws/'SUMMARY.md')}, ensure_ascii=False))
            return 0
        if args.command == 'show':
            if args.stage not in {s['id'] for s in catalog['stages']}:
                raise ValueError('Unknown stage')
            path = ws/'stages'/args.stage/'receipt.json'
            print(json.dumps(load(path), ensure_ascii=False, indent=2) if path.exists() else 'NOT_RUN')
            return 0 if path.exists() else 2
        if args.command == 'commands':
            for s in catalog['stages']:
                print(f'python scripts/acceptance/run.py run {s["id"]} --workspace "{ws}"' + (' --allow-long' if s['category'] == 'long' else ''))
            return 0
        wanted = list(args.stages)
        if args.suite:
            wanted += catalog['suites'][args.suite]
        if not wanted:
            raise ValueError('Specify a stage or --suite core/local/external/regression/all')
        if 'all' in wanted:
            wanted = catalog['suites']['all']
        ordered = resolve_order(wanted, catalog, dependencies=not args.no_deps)
        stage_map = {s['id']: s for s in catalog['stages']}
        lock = ws/'.runner.lock'
        try:
            with lock.open('x', encoding='utf-8') as f:
                f.write(json.dumps({'pid': os.getpid(), 'started_at': now()}))
        except FileExistsError:
            raise RuntimeError('Another acceptance run may be active. Inspect .runner.lock; do not delete it while that process runs.')
        statuses = []
        try:
            for stage_id in ordered:
                result = execute(ws, stage_map[stage_id], retry=args.retry, allow_long=args.allow_long)
                statuses.append(result['status'])
                build(ws)
        finally:
            lock.unlink(missing_ok=True)
            build(ws)
        print('Results:', ws/'index.html', flush=True)
        return 1 if 'FAILED' in statuses else 2 if any(x in ('BLOCKED', 'PARTIAL') for x in statuses) else 0
    except Exception as exc:
        print(json.dumps({'status': 'ERROR', 'error_type': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
