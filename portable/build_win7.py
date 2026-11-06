"""Build the Win7 SP1 x64/Python3.8 trial from hash-locked local materials.

No network, pip, system modification, model download, or user-data import.
The old Win10 builders are historical references and are not used here.
"""
from __future__ import annotations
import argparse
from email.parser import BytesParser
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import zipfile

VERSION = '2.7.1-win7-py38-preview.1'
PACKAGE_NAME = 'AstroGraph-LLM-Win7-x64-Python38'
ARCHIVE_NAME = PACKAGE_NAME + '-V2.7.1-preview.zip'
RUNTIME_SHA256 = 'abbe314e9b41603dde0a823b76f5bbbe17b3de3e5ac4ef06b759da5466711271'
RUNTIME_URL = 'https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-amd64.zip'
BASELINE_SHA256 = 'ff12ff5c565ec5086e5a73c43b9a155f6c8988b829b90804aad997fbc9f364a1'
FIXED_TIME = (2026, 11, 4, 0, 0, 0)

_spec = importlib.util.spec_from_file_location('portable_stage_builder', Path(__file__).with_name('build.py'))
_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stage)
sha256 = _stage.sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def verified_wheels(directory, lockfile):
    lock = json.loads(Path(lockfile).read_text(encoding='utf-8'))
    items = lock['wheels']
    if len(items) != len({item['filename'] for item in items}):
        raise ValueError('duplicate pinned wheel')
    wheels = []
    for item in items:
        filename = item['filename']
        if Path(filename).name != filename or '/' in filename or '\\' in filename:
            raise ValueError('unsafe wheel filename')
        wheel = Path(directory)/filename
        if not wheel.is_file():
            raise ValueError('missing pinned wheel: ' + filename)
        if sha256(wheel) != item['sha256']:
            raise ValueError('wheel SHA-256 mismatch: ' + filename)
        with zipfile.ZipFile(wheel) as z:
            metadata_names = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError('ambiguous wheel metadata: ' + filename)
            metadata = BytesParser().parsebytes(z.read(metadata_names[0]))
            normalized = lambda value: re.sub(r'[-_.]+', '-', value).lower()
            if (normalized(metadata['Name']) != normalized(item['name'])
                    or metadata['Version'] != item['version']):
                raise ValueError('wheel name/version mismatch: ' + filename)
            native = normalized(item['name']) == 'pydantic-core'
            tag = '-cp38-cp38-win_amd64.whl' if native else '-py3-none-any.whl'
            if not filename.endswith(tag):
                raise ValueError('unsupported runtime wheel tag: ' + filename)
            if any(n.lower().endswith(('.ttf', '.otf', '.woff', '.woff2')) for n in z.namelist()):
                raise ValueError('unexpected font asset in runtime wheel')
        wheels.append(wheel)
    actual = {p.name for p in Path(directory).glob('*.whl')}
    if actual != {item['filename'] for item in items}:
        raise ValueError('unexpected wheel outside locked runtime closure')
    return wheels, lock


def deterministic_zip(root, archive):
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(root.name + '/' + path.relative_to(root).as_posix(), FIXED_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError('final archive CRC failure')


def copy_source(repo_root, destination):
    _stage.stage_source(repo_root, destination)


def build_distribution(repo_root, runtime, wheels_dir, output_dir):
    repo_root, runtime, wheels_dir, output_dir = map(Path, (repo_root, runtime, wheels_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('refusing existing output directory: ' + str(output_dir))
    if sha256(runtime) != RUNTIME_SHA256:
        raise ValueError('official Python3.8 runtime SHA-256 mismatch')
    wheels, lock = verified_wheels(wheels_dir, repo_root/'portable/win7_dependencies.lock.json')
    output_dir.mkdir(parents=True)
    package = output_dir/PACKAGE_NAME
    package.mkdir()
    _stage._extract_archive(runtime, package/'runtime')
    (package/'runtime/python38._pth').write_text('python38.zip\n.\n../packages\n../app\n', encoding='ascii')
    for wheel in wheels:
        _stage._extract_archive(wheel, package/'packages')
    _stage.stage_app(repo_root, package/'app')
    (package/'data').mkdir()
    _stage.stage_bootstrap(repo_root, package, 'win7')
    copy_source(repo_root, package/'source')
    provenance = package/'provenance'; provenance.mkdir()
    write_json(provenance/'materials.json', {
        'runtime': {'version': '3.8.10', 'url': RUNTIME_URL, 'sha256': RUNTIME_SHA256},
        'wheels': lock['wheels'], 'runtime_source': 'Official Python.org Windows embeddable ZIP',
        'source_baseline_win10_zip_sha256': BASELINE_SHA256,
        'source_baseline': 'AstroGraph Portable V2.7.1',
        'materials_workflow_run': 41820347322,
        'runtime_windows_host_workflow_run': 41821203263,
        'native_windows7_verified': False,
        'system_updates_included': False,
        'model_software_or_weights_included': False,
        'browser_included': False,
    })
    pe_files = [p for p in package.rglob('*') if p.suffix.lower() in ('.exe', '.dll', '.pyd')]
    for path in pe_files:
        _stage._verify_amd64(path)
    for path in package.rglob('*.pyd'):
        if path.parent.name == 'pydantic_core' and not path.name.endswith('.cp38-win_amd64.pyd'):
            raise ValueError('wrong Python extension ABI')
    if any('cp314' in p.name or p.name.startswith('python314') for p in package.rglob('*')):
        raise ValueError('Python3.14 material leaked into Win7 distribution')
    if any(p.suffix.lower() in ('.ttf', '.otf', '.woff', '.woff2') for p in package.rglob('*')):
        raise ValueError('font file found in output')
    version = _stage.application_version(repo_root) + '-win7-py38'
    write_json(package/'MANIFEST.json', {
        'schema_version': 'astrograph-portable-package.v1', 'version': version,
        'shared_source_closure': list(_stage.CORE_FILES), 'profile': 'win7',
        'target': 'Windows 7 SP1 x64 (compatibility trial; native target validation pending)',
        'python_version': '3.8.10', 'runtime_archive_sha256': RUNTIME_SHA256,
        'dependencies': {item['name']: item['version'] for item in lock['wheels']},
        'native_windows7_execution_verified': False,
        'native_windows_execution_verified': False,
        'windows_server_runtime_dependency_smoke_verified': True,
        'built_at': '2026-11-04T00:00:00Z',
        'amd64_pe_files': len(pe_files),
        'readable_application_source_included': True,
        'files': [{'path': p.relative_to(package).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in sorted(package.rglob('*')) if p.is_file()],
    })
    archive = output_dir/(PACKAGE_NAME + '-V' + _stage.application_version(repo_root) + '.zip')
    deterministic_zip(package, archive)
    archive.with_suffix('.zip.sha256').write_text(sha256(archive)+'  '+archive.name+'\n', encoding='ascii')
    return package, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--wheels-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    package, archive = build_distribution(Path(__file__).resolve().parents[1], args.runtime, args.wheels_dir, args.output_dir)
    print(json.dumps({'package': str(package), 'zip': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha256(archive)}, indent=2))

if __name__=='__main__':
    main()
