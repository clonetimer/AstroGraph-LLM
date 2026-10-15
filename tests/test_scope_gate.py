from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/acceptance/run_scope_gate.py"
    spec = importlib.util.spec_from_file_location("run_scope_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g1_gate_passes_after_independent_protocol_approval(tmp_path: Path) -> None:
    sat_root = tmp_path / "sat_agent_platform"
    acceptance = sat_root / "configs" / "acceptance"
    acceptance.mkdir(parents=True)
    (acceptance / "object_scope.json").write_text("{}\n", encoding="utf-8")
    (acceptance / "scenario_baseline.json").write_text("{}\n", encoding="utf-8")
    report = _module().evaluate(ROOT, sat_root)
    assert report["technical_status"] == "PASS"
    assert report["status"] == "PASS"
    assert report["checks"]["project_owner_approved"] is True
    assert report["checks"]["independent_evaluator_approved"] is True
    assert all(report["checks"].values())
