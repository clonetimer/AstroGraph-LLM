from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SCRIPT = ROOT / "scripts" / "legacy" / "build_sat_sim_fault_dataset.py"


def _matrix(root: Path, *, proxy_value: str | None = None) -> Path:
    case = root / "basilisk" / "case-raw"
    case.mkdir(parents=True)
    result = {
        "status": "PASS",
        "effect_id": "gyro_bias_step",
        "kind": "fault",
        "capability_id": "subsystem.adcs",
    }
    (case / "result.json").write_text(json.dumps(result), encoding="utf-8")
    for name, values in (("nominal_trace.csv", [0.0, 0.1]), ("effect_trace.csv", [0.0, 0.2])):
        with (case / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["time_s", "sensor.gyro.rate"])
            writer.writeheader()
            writer.writerows([
                {"time_s": 0, "sensor.gyro.rate": values[0]},
                {"time_s": 1, "sensor.gyro.rate": proxy_value or values[1]},
            ])
    return root


def _run(matrix: Path, public: Path, truth: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [str(PYTHON), str(SCRIPT), "--matrix-root", str(matrix), "--output-root", str(public)]
    if truth is not None:
        command.extend(["--truth-output", str(truth)])
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def test_export_requires_physically_external_truth_and_emits_consumable_public_case(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix")
    public = tmp_path / "public"

    missing = _run(matrix, public)
    assert missing.returncode != 0
    nested = _run(matrix, public, public / "truth.json")
    assert nested.returncode != 0
    assert "physically disjoint" in nested.stderr

    truth = tmp_path / "restricted" / "truth.json"
    built = _run(matrix, public, truth)
    assert built.returncode == 0, built.stderr
    assert truth.is_file()
    assert not any("truth" in path.name.lower() for path in public.rglob("*"))

    case = public / "CASE-001"
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "simulation_handoff.v2"
    schema = json.loads(
        (ROOT / "protocols" / "integration" / "simulation_handoff.v2.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["telemetry"]["sha256"] == hashlib.sha256((case / "telemetry.csv").read_bytes()).hexdigest()
    serialized = json.dumps(manifest).lower()
    assert "gyro_bias_step" not in serialized
    assert "capability_id" not in serialized

    from astrograph.integrations.sat_sim_public_case import _load_public_case

    loaded, telemetry = _load_public_case(case)
    assert loaded["case_id"] == "CASE-001"
    assert telemetry.name == "telemetry.csv"


def test_export_fails_closed_on_answer_proxy_values(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix", proxy_value="event.gyro_bias_step")
    result = _run(matrix, tmp_path / "public", tmp_path / "restricted" / "truth.json")
    assert result.returncode != 0
    assert "answer-proxy values" in result.stderr


def test_export_consumes_actual_sat_agent_campaign_index_shape(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    contracts = []
    for role in ("fault", "nominal"):
        dataset_id = f"run-{role}"
        case = campaign / "cases" / dataset_id / "astrograph"
        case.mkdir(parents=True)
        (case / "telemetry.csv").write_text(
            "time_s,sensor.gyro.rate,label.class_id\n0,0.1,ANSWER\n1,0.2,ANSWER\n",
            encoding="utf-8",
        )
        contracts.append({
            "dataset_id": dataset_id,
            "pairing": {"case_role": role, "nominal_case_id": "run-nominal"},
            "ground_truth": {
                "case_role": role, "class_id": "GYRO_BIAS", "simulator_effect": "gyro_bias_step",
                "simulator_event_kind": "fault",
            },
            "simulation": {"backend": "python", "seed": 7, "spec_hash": "a" * 64},
            "telemetry": {"file": "astrograph/telemetry.csv"},
            "quality": {"physics_validation_passed": True, "formal_training_eligible": False},
        })
    (campaign / "astrograph_dataset_index.json").write_text(json.dumps({
        "campaign_status": "complete", "dry_run": False, "contracts": contracts,
    }), encoding="utf-8")

    public = tmp_path / "public"
    result = subprocess.run([
        str(PYTHON), str(SCRIPT), "--campaign-root", str(campaign), "--output-root", str(public),
        "--truth-output", str(tmp_path / "restricted" / "truth.json"),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    dataset = json.loads((public / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert dataset["source_runtimes"] == ["python"]
    assert dataset["training_qualified"] is False
    with (public / "CASE-001" / "telemetry.csv").open(encoding="utf-8") as handle:
        assert "label.class_id" not in next(handle)
