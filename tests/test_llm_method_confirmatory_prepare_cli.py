from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_cli_uses_external_workdir_and_emits_fail_closed_launch_plan(tmp_path: Path):
    work = tmp_path / "confirmatory"
    command = [
        sys.executable,
        str(ROOT / "scripts/experiments/prepare_llm_method_confirmatory_campaign.py"),
        "--output-dir", str(work),
        "--vllm-url", "http://127.0.0.1:1/v1",
        "--vllm-model", "Qwen3.5-9B",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 2
    readiness = json.loads((work / "readiness.json").read_text(encoding="utf-8"))
    launch = json.loads((work / "launch_plan.json").read_text(encoding="utf-8"))
    mount = json.loads((work / "dataset_mount_receipt.json").read_text(encoding="utf-8"))
    snapshot = json.loads((work / "dataset_snapshot_receipt.json").read_text(encoding="utf-8"))
    assert readiness["status"] == "BLOCKED"
    assert mount["status"] == "INVALID"
    assert mount["schema_version"] == "astrograph-llm-method-confirmatory-dataset-mount-receipt.v1"
    assert snapshot["schema_version"] == "astrograph-llm-method-confirmatory-dataset-snapshot-receipt.v1"
    assert snapshot["status"] == "INVALID"
    assert readiness["preparation"]["source_tree_mutated"] is False
    assert Path(readiness["case_manifest_path"]).parent == work
    assert launch["readiness_status"] == "BLOCKED"
    assert launch["dataset_mount_receipt_path"] == str(work / "dataset_mount_receipt.json")
    assert launch["dataset_snapshot_receipt_path"] == str(work / "dataset_snapshot_receipt.json")
    assert launch["schema_version"] == "astrograph-llm-method-confirmatory-launch-plan.v6"
    assert launch["materialization_requested"] is False
    assert launch["snapshot_materialization_argv"][launch["snapshot_materialization_argv"].index("--output-root") + 1] == str(work / "materialized_snapshot")
    assert launch["snapshot_export_argv"][launch["snapshot_export_argv"].index("--snapshot-root") + 1] == str(work / "materialized_snapshot")
    assert launch["snapshot_verify_argv"][launch["snapshot_verify_argv"].index("--snapshot-root") + 1] == str(work / "materialized_snapshot")
    assert launch["runner_argv"] is None
    assert "CLIENT_ENVIRONMENT_LOCK_MISSING" in launch["runner_identity_blockers"]
    assert "MODEL_RUNTIME_ATTESTATION_SOURCE_MISSING" in launch["runner_identity_blockers"]
    assert launch["runtime_identity_status"] == "BLOCKED"
