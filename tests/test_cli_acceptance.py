from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    return env


def test_nested_cli_generates_and_probes_synthetic_dataset(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    generated = subprocess.run(
        [
            sys.executable,
            "-m",
            "astrograph.cli",
            "simulate",
            "generate",
            "--spec",
            str(PROJECT_ROOT / "examples" / "synthetic_diagnosis_spec.json"),
            "--output-root",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr
    manifest = json.loads(generated.stdout)
    assert manifest["dataset_id"] == "r4f3-regression-example"

    evidence = tmp_path / "probe"
    probed = subprocess.run(
        [
            sys.executable,
            "-m",
            "astrograph.cli",
            "dataset",
            "probe",
            "--source-type",
            "simulation",
            "--root",
            str(output),
            "--output",
            str(evidence),
            "--max-entities",
            "5",
            "--max-rows",
            "100",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert probed.returncode == 0, probed.stderr
    assert (evidence / "schema_profile.json").is_file()
    profile = json.loads((evidence / "schema_profile.json").read_text(encoding="utf-8"))
    assert profile["raw_data_included"] is False
    assert profile["absolute_paths_included"] is False


def test_local_esa_acceptance_emits_redacted_evidence_only(tmp_path: Path) -> None:
    dataset = tmp_path / "ESA-ADB-private"
    dataset.mkdir()
    pd.DataFrame(
        {
            "time_s": range(20),
            "channel_a": [float(index) for index in range(20)],
            "label": [0] * 18 + [1, 1],
        }
    ).to_csv(dataset / "data.csv", index=False)
    output = tmp_path / "acceptance"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "acceptance" / "run_esa_adb_acceptance.py"),
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--max-entities",
            "5",
            "--max-rows",
            "20",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "acceptance_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["raw_dataset_uploaded"] is False
    assert summary["raw_rows_persisted"] is False
    assert summary["diagnosis_status"] == "NOT_REQUESTED"

    absolute_root = str(dataset.resolve())
    for path in output.rglob("*"):
        if path.is_file():
            assert absolute_root not in path.read_text(encoding="utf-8", errors="ignore")


def _write_raw_esa_mission(root: Path) -> None:
    (root / "channels").mkdir(parents=True)
    (root / "telecommands").mkdir()
    pd.DataFrame(
        [
            {
                "Channel": "channel_1",
                "Subsystem": "EPS",
                "Physical Unit": "V",
                "Group": "power",
                "Target": "battery",
            }
        ]
    ).to_csv(root / "channels.csv", index=False)
    pd.DataFrame(columns=["Telecommand", "Priority"]).to_csv(
        root / "telecommands.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "ID": "A-1",
                "Channel": "channel_1",
                "StartTime": "2020-01-01T00:01:00Z",
                "EndTime": "2020-01-01T00:02:00Z",
            }
        ]
    ).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "ID": "A-1",
                "Class": "Point",
                "Subclass": "Spike",
                "Category": "Anomaly",
                "Dimensionality": "Univariate",
                "Locality": "Local",
                "Length": "Short",
            }
        ]
    ).to_csv(root / "anomaly_types.csv", index=False)
    index = pd.date_range("2020-01-01", periods=4, freq="min", tz="UTC")
    pd.DataFrame({"channel_1": [1.0, 1.1, 4.0, 1.0]}, index=index).to_pickle(
        root / "channels" / "channel_1.zip"
    )


def test_local_esa_acceptance_requires_trust_and_proves_pickle_zip_read(tmp_path: Path) -> None:
    dataset = tmp_path / "ESA-Mission1"
    _write_raw_esa_mission(dataset)
    output = tmp_path / "acceptance_raw"

    refused = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "acceptance" / "run_esa_adb_acceptance.py"),
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--max-entities",
            "1",
            "--max-rows",
            "4",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2
    refused_summary = json.loads((output / "acceptance_summary.json").read_text(encoding="utf-8"))
    assert refused_summary["status"] == "FAIL"

    accepted_output = tmp_path / "acceptance_raw_trusted"
    accepted = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "acceptance" / "run_esa_adb_acceptance.py"),
            "--root",
            str(dataset),
            "--output",
            str(accepted_output),
            "--max-entities",
            "1",
            "--max-rows",
            "4",
            "--trust-local-pickle",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    summary = json.loads(
        (accepted_output / "acceptance_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert summary["mission_layout_detected"] is True
    assert summary["telemetry_archive_gate"] == "PASS"
    assert summary["telemetry_archive_entity"] == "channel_1"
