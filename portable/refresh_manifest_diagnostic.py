"""Create V2.6.1 diagnostics-only ZIP from the SHA-pinned V2.6 preview.

Uses only stdlib and bundled source. No Internet, pip, or Windows runtime execution.
Never overwrites a destination folder. Does not import or bundle user data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

BASE_SHA256 = '96f169274c741dcfdcbbc02087ccf9ada2352845f8b21151799826f6308d364f'
VERSION = '2.6.1-preview.1'
STAMP = (2026, 11, 3, 0, 0, 0)
ZIP_NAME = 'AstroGraph-LLM-Win10-x64-V2.6.1-diagnostics-preview.zip'
PACKAGE_NAME = 'AstroGraph-LLM-Win10-x64'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_files(root: Path):
    for directory in ('src', 'portable', 'docs', '.github'):
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                yield path
    for name in ('README.md', '.gitignore', 'SOURCE_ORIGIN.json'):
        file = root / name
        if file.is_file():
            yield file


def rebuild(source: Path, baseline: Path, destination: Path) -> Path:
    source = Path(source).resolve()
    baseline = Path(baseline).resolve()
    destination = Path(destination).resolve()
    if sha256(baseline.read_bytes()) != BASE_SHA256:
        raise ValueError('Refusing unverified V2.6 baseline ZIP (SHA-256 mismatch)')
    if destination.exists():
        raise FileExistsError('Output directory already exists; never overwrite')
    with tempfile.TemporaryDirectory(prefix='astro-checkdiag-') as temp:
        temp = Path(temp)
        with zipfile.ZipFile(baseline) as original:
            damaged = original.testzip()
            if damaged is not None:
                raise ValueError('Input ZIP has CRC damage: ' + damaged)
            for member in original.infolist():
                rel = PurePosixPath(member.filename)
                if (rel.is_absolute() or '..' in rel.parts or '\\' in member.filename
                        or ':' in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError('Unsafe archive member: ' + member.filename)
            original.extractall(temp)
        package = temp / PACKAGE_NAME
        original_manifest = json.loads((package / 'MANIFEST.json').read_text(encoding='utf-8'))
        for item in original_manifest['files']:
            file = package.joinpath(*PurePosixPath(item['path']).parts)
            if not file.is_file() or file.stat().st_size != item['bytes'] or sha256(file.read_bytes()) != item['sha256']:
                raise ValueError('Input ZIP manifest mismatch: ' + item['path'])
        for rel in ('verify_manifest.py', 'CHECK.bat'):
            shutil.copy2(source / 'portable' / rel, package / rel)
            shutil.copy2(source / 'portable' / rel, package / 'source' / 'portable' / rel)
        shutil.copy2(source / 'portable' / 'README_zh-CN.md', package / 'README_zh-CN.md')
        (package / 'README_zh-CN.txt').write_text(
            (source / 'portable' / 'README_zh-CN.md').read_text(encoding='utf-8'), encoding='utf-8-sig')
        for rel in ('test_manifest_diagnostics.py', 'refresh_manifest_diagnostic.py'):
            shutil.copy2(source / 'portable' / rel, package / 'source' / 'portable' / rel)
        shutil.copy2(source / 'docs' / 'RELEASE_NOTES_V2.6.1.md', package / 'RELEASE_NOTES_V2.6.1.md')
        shutil.copy2(source / 'docs' / 'RELEASE_NOTES_V2.6.1.md', package / 'source' / 'docs' / 'RELEASE_NOTES_V2.6.1.md')
        p_path = package / 'provenance' / 'refresh.json'
        provenance = json.loads(p_path.read_text(encoding='utf-8'))
        entries = []
        for path in source_files(source):
            content = path.read_bytes()
            entries.append({'path': path.relative_to(source).as_posix(), 'bytes': len(content), 'sha256': sha256(content)})
        provenance.update(version=VERSION, diagnostics_base_sha256=BASE_SHA256, source_files=entries,
            source_inventory_sha256=sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()),
            native_windows_execution_verified=False)
        p_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest = original_manifest.copy()
        manifest.update(version=VERSION, diagnostic_update=True,
            native_windows_execution_verified=False,
            files=[{'path':p.relative_to(package).as_posix(), 'bytes':p.stat().st_size, 'sha256':sha256(p.read_bytes())}
                   for p in sorted(package.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'])
        (package / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Verify the newly written manifest independently of the bundled checker.
        for item in manifest['files']:
            file = package / item['path']
            if sha256(file.read_bytes()) != item['sha256']:
                raise ValueError('New manifest check failed: ' + item['path'])
        destination.mkdir(parents=True)
        shutil.copytree(package, destination / PACKAGE_NAME)
        zip_path = destination / ZIP_NAME
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as final:
            for path in sorted((destination / PACKAGE_NAME).rglob('*')):
                if not path.is_file():
                    continue
                member = zipfile.ZipInfo(path.relative_to(destination).as_posix(), STAMP)
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                final.writestr(member, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(zip_path) as final:
            if final.testzip() is not None:
                raise ValueError('Generated ZIP CRC failure')
        (destination / (ZIP_NAME + '.sha256')).write_text(
            sha256(zip_path.read_bytes()) + '  ' + ZIP_NAME + '\n', encoding='ascii')
        return zip_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    result = rebuild(Path(__file__).resolve().parents[1], args.baseline, args.output_dir)
    print(json.dumps({'zip': str(result), 'sha256': sha256(result.read_bytes())}, indent=2))
