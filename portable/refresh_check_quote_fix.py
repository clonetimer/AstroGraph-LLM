"""Deterministic, offline repair release from the SHA-pinned V2.6.1 ZIP.

Restages only the CHECK launcher plus the Windows path regression test and
release notes; runtime, dependencies and application logic stay identical.
No user data is imported. The output directory must not already exist.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

BASE_SHA256 = '853ad3aca9142c614609356a8c71ce67a87c54599bfa7754a812198402148d99'
VERSION = '2.6.2-preview.1'
ZIP_NAME = 'AstroGraph-LLM-Win10-x64-V2.6.2-pathfix-preview.zip'
PACKAGE_NAME = 'AstroGraph-LLM-Win10-x64'
STAMP = (2026, 11, 3, 0, 0, 0)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def source_files(root: Path):
    for name in ('src', 'portable', 'docs', '.github'):
        directory = root / name
        if directory.exists():
            for path in sorted(directory.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    yield path
    for name in ('README.md', '.gitignore', 'SOURCE_ORIGIN.json'):
        path = root / name
        if path.is_file():
            yield path


def rebuild(source: Path, baseline: Path, output: Path) -> Path:
    source, baseline, output = Path(source).resolve(), Path(baseline).resolve(), Path(output).resolve()
    if digest_file(baseline) != BASE_SHA256:
        raise ValueError('Incorrect V2.6.1 baseline SHA-256; refusing to build')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing output directory')

    with tempfile.TemporaryDirectory(prefix='astro-quote-fix-') as temp:
        tempdir = Path(temp)
        with zipfile.ZipFile(baseline) as archive:
            if archive.testzip() is not None:
                raise ValueError('Baseline ZIP CRC check failed')
            seen = set()
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                mode = (member.external_attr >> 16) & 0o170000
                if (member.filename in seen or path.is_absolute() or '..' in path.parts
                        or '\\' in member.filename or ':' in member.filename or mode == 0o120000):
                    raise ValueError('Unsafe baseline ZIP member: ' + member.filename)
                seen.add(member.filename)
            archive.extractall(tempdir)
        package = tempdir / PACKAGE_NAME
        old = json.loads((package / 'MANIFEST.json').read_text(encoding='utf-8'))
        for item in old['files']:
            path = package.joinpath(*PurePosixPath(item['path']).parts)
            if not path.is_file() or path.stat().st_size != item['bytes'] or digest_file(path) != item['sha256']:
                raise ValueError('Baseline manifest mismatch: ' + item['path'])

        # Root and source copy are both part of the verified distribution.
        for rel in ('CHECK.bat', 'test_check_startup.py'):
            src = source / 'portable' / rel
            if rel == 'CHECK.bat':
                (package / rel).write_bytes(src.read_bytes())
            (package / 'source' / 'portable' / rel).write_bytes(src.read_bytes())
        (package / 'source' / 'portable' / 'refresh_check_quote_fix.py').write_bytes(
            (source / 'portable' / 'refresh_check_quote_fix.py').read_bytes())
        notes = (source / 'docs' / 'RELEASE_NOTES_V2.6.2.md').read_bytes()
        (package / 'RELEASE_NOTES_V2.6.2.md').write_bytes(notes)
        (package / 'source' / 'docs' / 'RELEASE_NOTES_V2.6.2.md').write_bytes(notes)

        provenance_file = package / 'provenance' / 'refresh.json'
        provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
        entries = [{'path': p.relative_to(source).as_posix(), 'bytes': p.stat().st_size,
                    'sha256': digest_file(p)} for p in source_files(source)]
        provenance.update(version=VERSION,
                          quote_fix_base_sha256=BASE_SHA256,
                          native_windows_execution_verified=False,
                          source_files=entries,
                          source_inventory_sha256=digest_bytes(
                              json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()))
        provenance_file.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        manifest = dict(old)
        manifest.update(version=VERSION, check_path_quoting_fix=True,
                        native_windows_execution_verified=False)
        manifest['files'] = [{'path': p.relative_to(package).as_posix(),
                              'bytes': p.stat().st_size, 'sha256': digest_file(p)}
                             for p in sorted(package.rglob('*'))
                             if p.is_file() and p.name != 'MANIFEST.json']
        (package / 'MANIFEST.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        output.mkdir(parents=True)
        shutil.copytree(package, output / PACKAGE_NAME)
        archive_file = output / ZIP_NAME
        with zipfile.ZipFile(archive_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted((output / PACKAGE_NAME).rglob('*')):
                if not path.is_file():
                    continue
                member = zipfile.ZipInfo(path.relative_to(output).as_posix(), STAMP)
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                archive.writestr(member, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(archive_file) as archive:
            if archive.testzip() is not None:
                raise ValueError('New ZIP CRC failure')
        (output / (ZIP_NAME + '.sha256')).write_text(
            digest_file(archive_file) + '  ' + ZIP_NAME + '\n', encoding='ascii')
        return archive_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    z = rebuild(Path(__file__).resolve().parents[1], args.baseline, args.output_dir)
    print(json.dumps({'archive': str(z), 'sha256': digest_file(z)}, indent=2))
