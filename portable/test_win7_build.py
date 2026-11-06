"""The offline Win7 builder must fail closed before modifying user paths."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

class Win7BuildTests(unittest.TestCase):
    def module(self):
        script = Path(__file__).with_name('build_win7.py')
        self.assertTrue(script.is_file(), 'Win7 offline builder missing')
        spec = importlib.util.spec_from_file_location('build_win7', script)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    def test_source_payload_excludes_unrelated_full_platform_modules(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'source'
            mod.copy_source(Path(__file__).resolve().parents[1], target)
            self.assertFalse((target / 'src/astrograph/data_driven').exists())
            self.assertTrue((target / 'src/astrograph/integrations/external_diagnosis_analysis.py').is_file())
            self.assertFalse((target / 'src/astrograph/integrations/__init__.py').read_text().count('sat_sim'))

    def test_untrusted_runtime_does_not_create_destination(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runtime = root/'runtime.zip'; runtime.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'runtime SHA-256'):
                mod.build_distribution(Path(__file__).resolve().parents[1], runtime, root/'wheels', root/'out')
            self.assertFalse((root/'out').exists())

    def test_missing_pinned_wheels_are_not_silently_skipped(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'missing pinned wheel'):
                mod.verified_wheels(Path(directory), Path(__file__).with_name('win7_dependencies.lock.json'))

    def test_existing_destination_is_untouched(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'out'; out.mkdir()
            sentinel=out/'user-report.txt'; sentinel.write_bytes(b'keep me')
            with self.assertRaises(FileExistsError):
                mod.build_distribution(Path(__file__).resolve().parents[1], root/'absent.zip', root/'wheels', out)
            self.assertEqual(sentinel.read_bytes(), b'keep me')
            self.assertEqual([x.name for x in out.iterdir()], ['user-report.txt'])

if __name__=='__main__': unittest.main(verbosity=2)
