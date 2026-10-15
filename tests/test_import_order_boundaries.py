from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_import(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )


def test_data_driven_models_import_in_a_fresh_interpreter() -> None:
    result = _run_import(
        "from astrograph.data_driven import ClassProbability, DataDrivenPrediction; "
        "print(ClassProbability.__name__, DataDrivenPrediction.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ClassProbability DataDrivenPrediction"


def test_lazy_assurance_model_helpers_remain_public() -> None:
    result = _run_import(
        "from astrograph.assurance import build_temporal_window_contract_registry; "
        "print(build_temporal_window_contract_registry.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "build_temporal_window_contract_registry"
