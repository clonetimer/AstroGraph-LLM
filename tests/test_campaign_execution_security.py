from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.data_driven import (
    CampaignScenario,
    IndependentRunCampaignSpec,
    build_independent_run_campaign_plan,
    campaign_plan_sha256,
    execute_independent_run_campaign,
)


def _plan():
    return build_independent_run_campaign_plan(
        IndependentRunCampaignSpec(
            campaign_id="SECURITY-TEST",
            runtime_script_name="registered_runtime.py",
            scenarios=[CampaignScenario(scenario_id="fault_a", class_ids=["F-A"])],
            seeds=[1, 2, 3],
            duration_s=1.0,
        )
    )


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "registered_runtime.py").write_text(
        """
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--scenario', required=True)
parser.add_argument('--duration-s', required=True)
parser.add_argument('--seed', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
(out / 'registered_script_ran.txt').write_text(args.scenario, encoding='utf-8')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return root


def test_execute_rejects_invalid_plan_hash(tmp_path: Path) -> None:
    plan = _plan().model_copy(update={"plan_sha256": "0" * 64})

    with pytest.raises(ValueError, match="SHA256 validation failed"):
        execute_independent_run_campaign(
            plan,
            runtime_root=_runtime_root(tmp_path),
            output_root=tmp_path / "outputs",
        )


def test_executor_ignores_tampered_command_template_even_with_recomputed_hash(tmp_path: Path) -> None:
    plan = _plan()
    malicious_marker = tmp_path / "unregistered_command_ran.txt"
    first = plan.runs[0].model_copy(
        update={
            "command_argv_template": [
                "python",
                "-c",
                f"from pathlib import Path; Path({str(malicious_marker)!r}).write_text('bad')",
            ]
        }
    )
    mutated = plan.model_copy(update={"runs": [first, *plan.runs[1:]], "plan_sha256": "0" * 64})
    mutated = mutated.model_copy(update={"plan_sha256": campaign_plan_sha256(mutated)})

    report = execute_independent_run_campaign(
        mutated,
        runtime_root=_runtime_root(tmp_path),
        output_root=tmp_path / "outputs",
    )

    assert report.status == "PASS"
    assert not malicious_marker.exists()
    assert (Path(report.rows[0].output_root) / "registered_script_ran.txt").is_file()
    assert report.rows[0].command[1].endswith("registered_runtime.py")


def test_execute_rejects_output_directory_escape_even_with_recomputed_hash(tmp_path: Path) -> None:
    plan = _plan()
    first = plan.runs[0].model_copy(
        update={
            "output_relative_root": "../outside",
            "expected_scenario_relative_dir": "../outside/nominal",
        }
    )
    mutated = plan.model_copy(update={"runs": [first, *plan.runs[1:]], "plan_sha256": "0" * 64})
    mutated = mutated.model_copy(update={"plan_sha256": campaign_plan_sha256(mutated)})

    with pytest.raises(ValueError, match="unsafe path segment"):
        execute_independent_run_campaign(
            mutated,
            runtime_root=_runtime_root(tmp_path),
            output_root=tmp_path / "outputs",
        )
    assert not (tmp_path / "outside").exists()
