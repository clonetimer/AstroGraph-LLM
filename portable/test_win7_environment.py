"""Validate legacy environment checks without requiring Windows to unit-test rules."""
import importlib.util
from pathlib import Path
import unittest

class Win7EnvironmentTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location("win7_environment", Path(__file__).with_name("win7_environment.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
    def test_windows7_requires_sp1_and_x64(self):
        check = self.module().platform_issues
        self.assertEqual(check((3,8,10),64,(6,1,1)), [])
        self.assertIn("WINDOWS7_SP1_REQUIRED", check((3,8,10),64,(6,1,0)))
        self.assertIn("X64_RUNTIME_REQUIRED", check((3,8,10),32,(6,1,1)))
    def test_wrong_interpreter_does_not_pass_as_python38(self):
        self.assertIn("PYTHON38_REQUIRED",self.module().platform_issues((3,14,1),64,(10,0,0)))
    def test_newer_windows_can_test_same_runtime_but_xp_is_rejected(self):
        check=self.module().platform_issues
        self.assertEqual(check((3,8,10),64,(10,0,0)),[])
        self.assertIn("WINDOWS7_OR_NEWER_REQUIRED",check((3,8,10),64,(5,1,3)))
if __name__ == '__main__': unittest.main()
