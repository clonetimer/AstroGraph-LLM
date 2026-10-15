from __future__ import annotations

from scripts.experiments.run_ops_sat_fixed_probe_comparator import FIXED_ACTION


def test_fixed_probe_comparator_freezes_one_active_action() -> None:
    assert FIXED_ACTION == "inspect_early_late_shift"
