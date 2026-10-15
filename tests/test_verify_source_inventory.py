import hashlib
import importlib.util
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_source.py'
spec = importlib.util.spec_from_file_location('merged_verify_source', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def checksum(root, names):
    (root / 'SHA256SUMS.txt').write_text(''.join(
        hashlib.sha256((root / name).read_bytes()).hexdigest() + '  ' + name + '\n'
        for name in names
    ))


def test_checkout_keeps_local_state_outside_inventory_but_checks_managed_files(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('/.astrograph-state/\n/data/\n')
    (tmp_path / 'source.py').write_text('x = 1\n')
    (tmp_path / '.astrograph-state').mkdir()
    (tmp_path / '.astrograph-state/local.json').write_text('{}')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/frozen.json').write_text('{}')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '-f', 'data/frozen.json'], check=True)
    checksum(tmp_path, ['.gitignore', 'source.py', 'data/frozen.json'])
    assert module.verify(tmp_path)['status'] == 'PASS'
    (tmp_path / 'data/frozen.json').write_text('changed')
    assert module.verify(tmp_path)['errors'] == ['CHANGED: data/frozen.json']
    (tmp_path / 'data/frozen.json').write_text('{}')
    (tmp_path / 'new_source.py').write_text('x = 2\n')
    assert module.verify(tmp_path)['errors'] == ['UNLISTED: new_source.py']


def test_archive_without_git_checks_every_source_file(tmp_path):
    (tmp_path / 'source.py').write_text('x = 1\n')
    checksum(tmp_path, ['source.py'])
    assert module.verify(tmp_path)['status'] == 'PASS'
    (tmp_path / 'new_source.py').write_text('x = 2\n')
    assert module.verify(tmp_path)['errors'] == ['UNLISTED: new_source.py']
