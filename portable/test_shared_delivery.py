"""Both runtime profiles must use the same app and a complete startup/source closure."""
from pathlib import Path
import tempfile
import unittest
from test_workbench import builder

ROOT=Path(__file__).resolve().parents[1]
class SharedDeliveryTests(unittest.TestCase):
    def test_bootstrap_contains_the_checker_used_by_check_bat_for_each_profile(self):
        self.assertTrue(hasattr(builder,'stage_bootstrap'),'shared complete startup builder missing')
        for profile in ['win10','win7']:
            with self.subTest(profile=profile),tempfile.TemporaryDirectory() as d:
                target=Path(d)/'package';target.mkdir()
                builder.stage_bootstrap(ROOT,target,profile)
                self.assertTrue((target/'verify_manifest.py').is_file())
                self.assertIn('verify_manifest.py',(target/'CHECK.bat').read_text())
                verify_line=next(line for line in (target/'CHECK.bat').read_text().splitlines() if 'python.exe' in line and 'verify_manifest.py' in line)
                self.assertNotIn('"%~dp0"',verify_line)
                if profile=='win7':self.assertTrue((target/'win7_environment.py').is_file())
                self.assertTrue((target/'START.bat').is_file())

    def test_source_export_is_shared_minimal_and_preserves_repository_init(self):
        self.assertTrue(hasattr(builder,'stage_source'),'shared source exporter missing')
        before=(ROOT/'src/astrograph/integrations/__init__.py').read_bytes()
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'source';builder.stage_source(ROOT,target)
            self.assertTrue((target/'portable/profiles/win7/CHECK.bat').is_file())
            self.assertTrue((target/'src/astrograph/integrations/execution_observability.py').is_file())
            self.assertTrue((target/'src/astrograph/resources/execution_view.js').is_file())
            self.assertFalse((target/'src/astrograph/data_driven').exists())
            self.assertFalse((target/'data').exists())
            self.assertEqual(before,(ROOT/'src/astrograph/integrations/__init__.py').read_bytes())
            with self.assertRaises(FileExistsError):builder.stage_source(ROOT,target)

    def test_version_metadata_is_read_from_canonical_application(self):
        self.assertTrue(hasattr(builder,'application_version'),'source-bound version metadata missing')
        import ast
        version=next(x.value.value for x in ast.parse((ROOT/'portable/workbench.py').read_text()).body
                     if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VERSION' for t in x.targets))
        self.assertEqual(builder.application_version(ROOT),version)

    def test_production_shell_loads_shared_renderer_before_application(self):
        import re
        scripts=re.findall(r'<script[^>]*src="/([^"]+)"', (ROOT/'portable/assets/index.html').read_text())
        self.assertIn('execution_view.js',scripts)
        self.assertLess(scripts.index('execution_view.js'),scripts.index('app.js'))
