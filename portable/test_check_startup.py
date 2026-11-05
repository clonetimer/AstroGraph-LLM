"""Protect Windows CHECK.bat argument quoting, including trailing slash roots.

The MS C runtime interprets a backslash preceding a double quote as an
escape. The directory supplied by Windows %%~dp0 ends in a backslash; passing
it verbatim as a quoted argument corrupts the Python process's argv.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
ROOT = PACKAGE_ROOT if (PACKAGE_ROOT / 'MANIFEST.json').is_file() else HERE


def windows_crt_argv(command: str) -> list[str]:
    """Model MS CRT argument quote/backslash rules for a simple command line."""
    args = []
    i = 0
    while i < len(command):
        while i < len(command) and command[i].isspace():
            i += 1
        if i == len(command):
            break
        quoted = False
        chars = []
        while i < len(command):
            c = command[i]
            if not quoted and c.isspace():
                break
            if c == '\\':
                start = i
                while i < len(command) and command[i] == '\\':
                    i += 1
                count = i - start
                if i < len(command) and command[i] == '"':
                    chars.extend('\\' * (count // 2))
                    if count % 2:
                        chars.append('"')
                    else:
                        quoted = not quoted
                    i += 1
                else:
                    chars.extend('\\' * count)
                continue
            if c == '"':
                quoted = not quoted
            else:
                chars.append(c)
            i += 1
        args.append(''.join(chars))
    return args


def command_argv(batch_text: str, windows_root: str) -> list[str]:
    """Get argv passed to embedded Python for the manifest check."""
    line = next(line for line in batch_text.splitlines()
                if 'verify_manifest.py' in line and 'runtime\\python.exe' in line)
    return windows_crt_argv(line.split(' > ', 1)[0].replace('%~dp0', windows_root))


class WindowsCheckStartupTests(unittest.TestCase):
    def test_manifest_verifier_receives_only_script_not_quoted_trailing_directory(self):
        for root in ('C:\\Users\\Administrator\\Desktop\\LLM\\llm\\',
                     'C:\\Users\\Administrator\\Desktop\\诊断 报告\\'):
            for check in (ROOT / 'CHECK.bat', HERE / 'CHECK.bat'):
                with self.subTest(root=root, check=str(check)):
                    argv = command_argv(check.read_text(encoding='ascii'), root)
                    self.assertEqual(argv, [
                        root + 'runtime\\python.exe', '-X', 'utf8', '-B',
                        root + 'verify_manifest.py',
                    ])

    def test_manifest_verifier_default_root_works_from_different_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='arbitrary-cwd-') as other_cwd:
            folder = Path(other_cwd) / '中文 package'; folder.mkdir()
            (folder / 'asset.txt').write_bytes(b'original')
            (folder / 'MANIFEST.json').write_text(json.dumps({'files': [
                {'path': 'asset.txt', 'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
            ]}), encoding='utf-8')
            (folder / 'verify_manifest.py').write_bytes((HERE / 'verify_manifest.py').read_bytes())
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(folder / 'verify_manifest.py')],
                                    cwd=other_cwd, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Package integrity passed.', result.stdout)

    def test_integrity_check_still_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'data.txt').write_bytes(b'tampered')
            (folder / 'MANIFEST.json').write_text(json.dumps({'files': [
                {'path': 'data.txt', 'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
            ]}), encoding='utf-8')
            (folder / 'verify_manifest.py').write_bytes((ROOT / 'verify_manifest.py').read_bytes())
            result = subprocess.run([sys.executable, str(folder / 'verify_manifest.py')],
                                    cwd=tmp, text=True, capture_output=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('changed file: data.txt', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
