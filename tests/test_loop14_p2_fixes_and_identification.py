"""Loop-14 tests: the three P2 fixes + the full-mechanism identification (D)
and cross-object separability (E) report artifacts.

Work items ①③⑥ of loop 14 (P2-A spinning_rotor wiring / P2-B catalog id /
P2-C EPS wording) plus the experiment D/E outputs.  Verifies on committed
configs and generated reports only (no external Sat-Agent path required):

1. ``actuator.spinning_rotor`` is registered in the core semantic archetype
   catalog AND its physics contract angular_rate bounds are symmetric
   (lower = -upper, matching the abs() saturation guard) — P2-A.
2. The experiment D report covers all 9 objects and honestly reports
   fail-closed / non-identifiable mechanisms — work item ④.
3. The experiment E report covers the cross-object separability matrix and
   records the honest cross-object OVERLAP findings — work item ⑤.
4. The graph projection now carries spinning_rotor feature tokens (the P2-A
   feature-contract wiring is live).

S0: verification only, no fitting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import ComponentArchetypeCatalog

ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = ROOT / "configs" / "semantics"
TRACES = ROOT / "data" / "generated_traces"
PROJECTION = ROOT / "data" / "graph_projection" / "projection_summary.json"

EXPECTED_OBJECTS = {
    "battery", "imu", "reaction_wheel", "thermal", "mtb",
    "data_queue", "onboard_storage", "solar_panel", "eps",
}


class TestSpinningRotorWiring:
    def test_spinning_rotor_registered_in_semantic_catalog(self):
        catalog = ComponentArchetypeCatalog.model_validate_json(
            (SEMANTICS / "core_component_archetypes.json").read_text(encoding="utf-8")
        )
        archetype = catalog.archetype("actuator.spinning_rotor")
        assert "state.rate" in {p.port_role_id for p in archetype.port_roles}
        assert "command.effort" in {p.port_role_id for p in archetype.port_roles}
        assert "response.effort" in {p.port_role_id for p in archetype.port_roles}

    def test_spinning_rotor_angular_rate_bounds_symmetric(self):
        """P2-A: lower_bound=0.0 contradicted the abs() saturation guard; a wheel
        may spin in reverse to dump momentum, so bounds must be symmetric."""
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        state = contract.states[0]
        assert state.state_id == "angular_rate"
        assert state.lower_bound == -state.upper_bound
        assert state.upper_bound > 0.0
        guard = contract.event_modes[0].guard_expression
        assert "abs(angular_rate)" in guard


class TestExperimentDReport:
    def test_report_covers_all_nine_objects(self):
        if not (TRACES / "identification_D_report.md").is_file():
            pytest.skip("identification_D_report.md not present in this checkout")
        text = (TRACES / "identification_D_report.md").read_text(encoding="utf-8")
        sections = {
            "battery": "## 1. battery",
            "reaction_wheel": "## 2. reaction_wheel",
            "thermal": "## 3. thermal",
            "IMU": "## 4. IMU",
            "MTB": "## 5. MTB",
            "data_queue": "## 6. data_queue",
            "onboard_storage": "## 7. onboard_storage",
            "solar_panel": "## 8. solar_panel",
            "EPS": "## 9. EPS",
        }
        for obj, prefix in sections.items():
            assert prefix in text, obj
        # summary table covers all 9 objects (case as written in the report)
        for obj in ("battery", "reaction_wheel", "thermal", "IMU", "MTB",
                    "data_queue", "onboard_storage", "solar_panel", "EPS"):
            assert f"| {obj} |" in text, obj

    def test_report_is_honest_about_fail_closed(self):
        if not (TRACES / "identification_D_report.md").is_file():
            pytest.skip("identification_D_report.md not present in this checkout")
        text = (TRACES / "identification_D_report.md").read_text(encoding="utf-8")
        assert "FAIL-CLOSED" in text
        assert "检测" in text
        assert "不（遥测信号）" in text or "不（容量不变）" in text


class TestExperimentEReport:
    def test_report_has_cross_object_matrix_and_honest_overlaps(self):
        if not (TRACES / "identification_E_report.md").is_file():
            pytest.skip("identification_E_report.md not present in this checkout")
        text = (TRACES / "identification_E_report.md").read_text(encoding="utf-8")
        assert "跨对象可分性" in text
        assert "31" in text  # 31 mechanism entries
        # Honest confusions the task explicitly asked about are recorded.
        assert "battery-SCL035" in text and "EPS-SCL" in text
        assert "OVERLAP" in text
        assert "不可分" in text or "混淆" in text


class TestProjectionReflectsSpinningRotorWiring:
    def test_reaction_wheel_has_feature_tokens_in_projection(self):
        if not PROJECTION.is_file():
            pytest.skip("loop-13 graph projection output not present in this checkout")
        import json

        summary = json.loads(PROJECTION.read_text(encoding="utf-8"))
        by_object = {r["component_instance_id"]: r for r in summary["records"]}
        rw = by_object["reaction_wheel"]
        # P2-A feature-contract wiring: the spinning_rotor archetype now emits
        # feature tokens (was 0 before the loop-14 fix).
        assert rw["token_count"] > 0
        assert rw["estimate_status"] == "OK"
