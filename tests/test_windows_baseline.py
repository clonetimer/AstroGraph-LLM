from __future__ import annotations

import importlib.util
import platform
from pathlib import Path


def test_windows_baseline_reports_real_interpreter() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/acceptance/run_windows_baseline.py"
    spec = importlib.util.spec_from_file_location("run_windows_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.collect()
    is_windows = platform.system() == "Windows"
    assert result["windows"] is is_windows
    assert result["status"] == ("PASS" if is_windows else "NOT_WINDOWS")
    assert result["platform"] == platform.system()
    assert result["executable"]
