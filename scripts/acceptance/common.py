"""Acceptance-only workspace, hashing, process logs and tabular exports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / 'configs/acceptance/stages.json'


class Blocked(RuntimeError):
    """A required input, approval, dependency or service has not been supplied."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: a partial write must not be mistaken for a complete receipt.
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def table(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def resolve_order(stage_ids: list[str], catalog: dict, *, dependencies: bool = True) -> list[str]:
    """Order *local acceptance* checks. This is not a production Agent DAG."""
    stages = {s['id']: s for s in catalog['stages']}
    order: list[str] = []
    visiting: set[str] = set()

    def visit(key: str) -> None:
        if key not in stages:
            raise ValueError(f'Unknown acceptance stage: {key}')
        if key in visiting:
            raise ValueError(f'Acceptance dependency cycle at {key}')
        if key in order:
            return
        visiting.add(key)
        if dependencies:
            for dep in stages[key]['dependencies']:
                visit(dep)
        visiting.remove(key)
        order.append(key)

    for key in stage_ids:
        visit(key)
    return order


class Context:
    def __init__(self, workspace: Path, stage: dict, *, allow_long: bool = False):
        self.workspace = workspace.resolve()
        self.config = load(self.workspace / 'acceptance.json')
        self.stage = stage
        self.out = self.workspace / 'stages' / stage['id']
        self.allow_long = allow_long
        self.commands: list[list[str]] = []
        self.notices: list[str] = []

    def path(self, name: str) -> Path:
        p = self.out / name
        if not p.resolve().is_relative_to(self.out.resolve()):
            raise ValueError('stage output escaped its directory')
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, name: str, value: Any) -> None:
        save(self.path(name), value)

    def previous(self, stage: str, filename: str) -> Path:
        path = self.workspace / 'stages' / stage / filename
        if not path.exists():
            raise Blocked(f'Run {stage} first; missing artifact {path}')
        return path

    def required(self, key: str) -> Path:
        value = self.config.get(key)
        if not value:
            raise Blocked(f'Missing {key}; initialize a NEW workspace with --{key.replace("_", "-")} PATH')
        path = Path(value)
        if not path.exists():
            raise Blocked(f'Missing configured {key}: {path}')
        return path

    def settings(self, *, external: bool = False, planner: str | None = None):
        from astrograph.agents.mission.models import AgentSettings
        payload = load(self.workspace / 'host-settings.json')
        # Always isolate output and memory from a production host configuration.
        payload.update(project_root=str(ROOT), output_root=str(self.out / 'tasks'),
                       memory_db=None, memory_policy=None)
        roots = [*payload.get('read_roots', []), str(ROOT), str(self.workspace)]
        for key in ('source', 'nominal', 'model', 'nontext_document', 'training_manifest', 'candidate_features'):
            if self.config.get(key):
                roots.append(str(Path(self.config[key]).parent))
        roots += [str(Path(p).parent) for p in self.config.get('documents', [])]
        if self.config.get('phm_dataset_root'): roots.append(str(Path(self.config['phm_dataset_root']).parent))
        payload['read_roots'] = sorted(set(roots))
        if not external:
            payload.update(planner='deterministic', llm_config=None,
                           allow_external_literature=False, allow_a2o=False, allow_mineru=False,
                           embedding_model_path=None, reranker_model_path=None)
        if planner is not None:
            payload['planner'] = planner
        return AgentSettings.model_validate(payload)

    def worker(self, args: list[str], *, timeout: float = 600, name: str = 'process') -> subprocess.CompletedProcess:
        cmd = [sys.executable, *map(str, args)]
        self.commands.append(cmd)
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        started = now()
        print('EXEC:', ' '.join(cmd), flush=True)
        from astrograph.integrations.process_control import _terminate_process_tree
        with self.path(name + '.stdout.log').open('w', encoding='utf-8') as stdout, \
             self.path(name + '.stderr.log').open('w', encoding='utf-8') as stderr:
            try:
                process = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                           start_new_session=(os.name == 'posix'),
                                           creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) if os.name == 'nt' else 0)
                try:
                    returncode = process.wait(timeout=timeout)
                except BaseException:
                    _terminate_process_tree(process)
                    raise
                proc = subprocess.CompletedProcess(cmd, returncode)
            except subprocess.TimeoutExpired:
                self.write(name + '.json', {'command': cmd, 'started': started, 'finished': now(), 'status': 'TIMEOUT'})
                raise RuntimeError(f'{name} timed out; inspect the retained process logs')
        self.write(name + '.json', {'command': cmd, 'started': started, 'finished': now(), 'returncode': proc.returncode})
        if proc.returncode:
            detail = self.path(name + '.stderr.log').read_text(encoding='utf-8')[-1600:]
            raise RuntimeError(f'{name} returned {proc.returncode}: {detail}')
        return proc

    def long_task(self) -> None:
        if not self.allow_long:
            raise Blocked('This is an explicit full replay/training stage. Re-run with --allow-long, or use 51/55 to rescore saved predictions.')

    def documents(self, *, external: bool = False):
        from astrograph.agents.mission.documents import DocumentTools
        from astrograph.agents.mission.storage import Workspace
        settings = self.settings(external=external)
        ws = Workspace(self.out, settings.read_roots, 'document-work')
        ws.root.mkdir(parents=True, exist_ok=False)
        return DocumentTools(ws, settings)


def parse_junit(path: Path) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    cases = root.findall('.//testcase')
    return {'tests': len(cases), 'passed': sum(not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases),
            'failures': sum(c.find('failure') is not None for c in cases),
            'errors': sum(c.find('error') is not None for c in cases),
            'skipped': sum(c.find('skipped') is not None for c in cases),
            'scope': 'SELECTED_TESTS_NOT_FULL_REPOSITORY'}
