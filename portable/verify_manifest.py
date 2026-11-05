"""Offline file-integrity verifier; reports the exact failing file and OS error."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys


def _io_error(error: OSError) -> str:
    """Keep the actual OS description and codes needed for Windows support."""
    details = [type(error).__name__]
    winerror = getattr(error, 'winerror', None)
    if winerror is not None:
        details.append('WinError ' + str(winerror))
    if error.errno is not None:
        details.append('errno ' + str(error.errno))
    return '[' + ', '.join(details) + '] ' + str(error)


def verify(root: Path) -> list[str]:
    """Return named file failures, never confuse access errors with corrupt bytes."""
    root = Path(root).resolve()
    manifest_path = root / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = []
    for item in manifest['files']:
        name = item['path']
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            errors.append(f'unsafe manifest path: {name}')
            continue
        path = root.joinpath(*rel.parts)
        try:
            if not path.is_file() or path.is_symlink():
                errors.append(f'missing or symbolic link: {name}')
                continue
            if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                errors.append(f'changed file: {name}')
        except OSError as error:
            errors.append(f'cannot access file: {name}; {_io_error(error)}')
    return errors


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    try:
        failures = verify(root)
        for item in failures:
            print(item)
        print('Package integrity FAILED.' if failures else 'Package integrity passed.')
        raise SystemExit(bool(failures))
    except OSError as error:
        print('Could not read package MANIFEST.json or resolve package root:',
              str(root / 'MANIFEST.json') + ';', _io_error(error))
        raise SystemExit(1)
    except (ValueError, KeyError, TypeError) as error:
        print('Invalid package MANIFEST.json:', type(error).__name__, str(error))
        raise SystemExit(1)
