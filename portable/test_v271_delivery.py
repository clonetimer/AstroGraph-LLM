"""V2.7.1 offline refresh refuses untrusted bases and existing output paths."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

class PolishDeliveryTests(unittest.TestCase):
    def load_builder(self):
        script=Path(__file__).with_name('refresh_v271.py')
        self.assertTrue(script.is_file(), 'V2.7.1 refresh builder missing')
        spec=importlib.util.spec_from_file_location('v271_builder',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_refuses_untrusted_base_without_creating_destination(self):
        module=self.load_builder()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'base.zip';base.write_bytes(b'not a verified archive')
            with self.assertRaisesRegex(ValueError,'baseline SHA-256'):
                module.refresh_distribution(Path(__file__).resolve().parent.parent,base,root/'new')
            self.assertFalse((root/'new').exists())

    def test_refuses_existing_output_without_touching_user_file(self):
        module=self.load_builder()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'fixture.zip';base.write_bytes(b'fixture header')
            out=root/'existing';out.mkdir();sentinel=out/'user-report.txt';sentinel.write_bytes(b'keep me')
            # Boundary test: exact fixture digest is approved only for this call;
            # output refusal must occur before opening/extracting the fixture archive.
            with patch.object(module,'BASELINE_SHA256',hashlib.sha256(base.read_bytes()).hexdigest()):
                with self.assertRaises(FileExistsError):
                    module.refresh_distribution(Path(__file__).resolve().parent.parent,base,out)
            self.assertEqual(sentinel.read_bytes(),b'keep me')
            self.assertEqual([p.name for p in out.iterdir()],['user-report.txt'])

if __name__=='__main__':unittest.main(verbosity=2)
