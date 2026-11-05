"""Regressions for Windows CHECK.bat's opaque manifest OSError diagnostics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("verify_manifest", HERE / "verify_manifest.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def fixture(root: Path):
    """Create a minimal valid package that cannot be confused with actual runtime."""
    asset = root / "packages" / "pypdf" / "_reader.py"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"good")
    manifest = {"files": [{"path": "packages/pypdf/_reader.py", "bytes": 4,
                            "sha256": hashlib.sha256(b"good").hexdigest()}]}
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return asset


class ManifestDiagnosticsTests(unittest.TestCase):
    def test_read_denied_prints_path_and_real_exception_not_only_oserror(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            asset = fixture(root)
            real_read = Path.read_bytes
            def guarded_read(path):
                if path.name == asset.name:
                    raise PermissionError(13, "Access denied", str(asset))
                return real_read(path)
            with mock.patch.object(Path, "read_bytes", guarded_read):
                failures = checker.verify(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("packages/pypdf/_reader.py", failures[0])
            self.assertIn("PermissionError", failures[0])
            self.assertIn("13", failures[0])

    def test_stat_failure_returns_named_error_instead_of_aborting(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            asset = fixture(root)
            real_stat = Path.stat
            def guarded_stat(path, *args, **kwargs):
                if path.name == asset.name:
                    raise OSError(123, "Invalid path", str(asset))
                return real_stat(path, *args, **kwargs)
            with mock.patch.object(Path, "stat", guarded_stat):
                failures = checker.verify(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("packages/pypdf/_reader.py", failures[0])
            self.assertIn("Invalid path", failures[0])

    def test_missing_manifest_names_manifest_and_error(self):
        with tempfile.TemporaryDirectory() as name:
            process = subprocess.run([sys.executable, str(HERE / "verify_manifest.py"), name],
                                     capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("MANIFEST.json", process.stdout)
            self.assertIn("No such file", process.stdout)

    def test_actual_extracted_package_still_valid(self):
        package = HERE.parents[1]
        if not (package / "MANIFEST.json").exists():
            self.skipTest("standalone source checkout")
        self.assertEqual(checker.verify(package), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
