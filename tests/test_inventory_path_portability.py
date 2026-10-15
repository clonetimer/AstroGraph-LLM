from __future__ import annotations

from pathlib import Path

from astrograph.data_driven import IndependentRunRecord, SplitName, resolve_run_scenario_dir


def test_resolves_portable_scenario_dir_relative_to_inventory_runtime_outputs(tmp_path: Path) -> None:
    inventory = tmp_path / "independent_run_inventory.json"
    scenario = tmp_path / "runtime_outputs" / "runs" / "RUN-1" / "nominal"
    scenario.mkdir(parents=True)
    record = IndependentRunRecord(
        run_id="RUN-1",
        class_ids=["NORMAL"],
        split=SplitName.TEST,
        metadata={"scenario_dir": "runs/RUN-1/nominal", "scenario_dir_base": "CAMPAIGN_OUTPUT_ROOT"},
    )

    assert resolve_run_scenario_dir(record, inventory_path=inventory) == scenario.resolve()


def test_rebinds_legacy_windows_absolute_scenario_dir(tmp_path: Path) -> None:
    inventory = tmp_path / "independent_run_inventory.json"
    scenario = tmp_path / "runtime_outputs" / "runs" / "RUN-2" / "fault_a"
    scenario.mkdir(parents=True)
    record = IndependentRunRecord(
        run_id="RUN-2",
        class_ids=["F-A"],
        split=SplitName.TEST,
        metadata={
            "scenario_dir": r"E:\\timer\\AstroGraph\\outputs\\v0516_campaign\\runtime_outputs\\runs\\RUN-2\\fault_a"
        },
    )

    assert resolve_run_scenario_dir(record, inventory_path=inventory) == scenario.resolve()
