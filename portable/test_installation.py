import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class InstallationTests(unittest.TestCase):
    def load_checker(self):
        script=Path(__file__).with_name('check_installation.py')
        self.assertTrue(script.exists(), 'package verification entry point is missing')
        spec=importlib.util.spec_from_file_location('check_installation', script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def fixture(self, root):
        (root/'app').mkdir();(root/'app/workbench.py').write_bytes(b'print("ok")\n')
        data=(root/'app/workbench.py').read_bytes()
        manifest={'files':[{'path':'app/workbench.py','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
        (root/'MANIFEST.json').write_text(json.dumps(manifest))
        return manifest

    def test_complete_manifest_passes_and_modified_source_is_rejected(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root)
            self.assertEqual(checker.verify_files(root)['status'],'PASS')
            (root/'app/workbench.py').write_bytes(b'changed')
            self.assertEqual(checker.verify_files(root)['status'],'FAIL')

    def test_missing_runtime_file_returns_relative_file_and_not_success(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);(root/'app/workbench.py').unlink()
            report=checker.verify_files(root)
            self.assertEqual(report['status'],'FAIL')
            self.assertEqual(report['failures'][0]['path'],'app/workbench.py')

    def test_unsafe_manifest_never_reads_outside_package(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);m=self.fixture(root);m['files'][0]['path']='../outside.txt'
            (root/'MANIFEST.json').write_text(json.dumps(m))
            self.assertEqual(checker.verify_files(root)['status'],'FAIL')

if __name__=='__main__':unittest.main(verbosity=2)
