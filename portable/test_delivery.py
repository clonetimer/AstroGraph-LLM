"""Delivery checks test real package bytes, not version labels."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from test_workbench import builder

class DeliveryTests(unittest.TestCase):
    def load_refresh(self):
        path=Path(__file__).with_name('refresh_distribution.py')
        self.assertTrue(path.is_file(), 'offline refresh builder is missing')
        spec=importlib.util.spec_from_file_location('refresh_distribution',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_refuses_wrong_baseline_before_creating_output(self):
        module=self.load_refresh()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fake=root/'base.zip'; fake.write_bytes(b'not the verified release')
            with self.assertRaisesRegex(ValueError,'baseline SHA-256'):
                module.refresh_distribution(Path(__file__).resolve().parent.parent,fake,root/'out')
            self.assertFalse((root/'out').exists())

    def test_manifest_verification_detects_modified_file(self):
        path=Path(__file__).with_name('verify_manifest.py')
        self.assertTrue(path.is_file(), 'manifest checker is missing')
        spec=importlib.util.spec_from_file_location('verify_manifest',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import hashlib,json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); file=root/'asset.txt';file.write_bytes(b'original')
            (root/'MANIFEST.json').write_text(json.dumps({'files':[{'path':'asset.txt','bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}]}))
            self.assertEqual(module.verify(root),[])
            file.write_bytes(b'changed!')
            self.assertIn('asset.txt',module.verify(root)[0])

if __name__=='__main__':unittest.main()
