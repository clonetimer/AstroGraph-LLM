"""Regression: a built deployment must include every new shared P2/P3 module."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('closure_builder', ROOT / 'portable/build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

class RecoveryClosureTests(unittest.TestCase):
    def test_p2p3_app_and_export_keep_shared_modules_and_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, stage in [('app', builder.stage_app), ('source', builder.stage_source)]:
                destination = Path(directory) / name
                stage(ROOT, destination)
                core = destination / ('astrograph' if name == 'app' else 'src/astrograph')
                for relative in ('integrations/reviewed_context.py', 'integrations/stage_execution.py',
                                 'resources/workspace_view.js', 'resources/workspace_view.css'):
                    with self.subTest(deployment=name, path=relative):
                        self.assertTrue((core / relative).is_file(), relative + ' omitted from delivery')
                        self.assertEqual((core / relative).read_bytes(), (ROOT / 'src/astrograph' / relative).read_bytes())

if __name__ == '__main__':
    unittest.main()
