from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _bundle() -> dict:
    baseline = {"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"}
    predictions, truths, contexts = [], [], []
    index = 0
    for partition, count in [
        ("CALIBRATION_FIT", 8),
        ("DEVELOPMENT_SELECTION", 4),
        ("CONFIRMATORY_EVALUATION", 8),
        ("EXTERNAL_VALIDATION", 8),
    ]:
        prefix = partition.split("_")[0].lower()
        for item in range(count):
            group_id = f"{prefix}-g{item}"
            true_id = "A" if item % 2 == 0 else "B"
            wrong = item in ({6, 7} if count >= 8 else {3})
            p_a = (0.8 if true_id == "B" else 0.2) if wrong else (0.9 if true_id == "A" else 0.1)
            probs = {"A": p_a, "B": 1.0 - p_a}
            sample_id = f"s{index}"
            index += 1
            predictions.append({
                "sample_id": sample_id,
                "group_id": group_id,
                "probabilities": probs,
                "predicted_id": max(probs, key=probs.get),
                "prediction_sha256": f"{index:064x}"[-64:],
            })
            truths.append({"sample_id": sample_id, "group_id": group_id, "true_id": true_id})
            contexts.append({"sample_id": sample_id, "group_id": group_id, "partition": partition, "strata": baseline})
    return {
        "model_id": "model-r19",
        "model_version": "1",
        "model_artifact_sha256": "2" * 64,
        "calibration_id": "temp-r19",
        "calibration_dataset_id": "calib-r19",
        "calibration_dataset_sha256": "1" * 64,
        "external_validation_dataset_id": "external-r19",
        "external_validation_dataset_sha256": "3" * 64,
        "baseline_values": baseline,
        "predictions": predictions,
        "truths": truths,
        "contexts": contexts,
    }


def _run(root: Path, input_path: Path, design_path: Path, output_dir: Path, fit_receipt: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(root / "scripts/experiments/run_probability_calibration_qualification.py"),
        "--exposure-dataset-id", "SYNTHETIC_PROTOCOL",
        "--input", str(input_path),
        "--design", str(design_path),
        "--output-dir", str(output_dir),
    ]
    if fit_receipt is not None:
        command.extend(["--fit-receipt", str(fit_receipt)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)


def test_r19_cli_fit_then_external_admission_is_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "bundle.json"
    input_path.write_text(json.dumps(_bundle(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    design = root / "configs/evaluation/calibration_fitting_r19.v1.json"
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = _run(root, input_path, design, first)
    assert one.returncode == 0, one.stderr
    two = _run(root, input_path, design, second, first / "calibrator_fit_receipt.json")
    assert two.returncode == 0, two.stderr
    assert (first / "calibrator_fit_receipt.json").read_bytes() == (second / "calibrator_fit_receipt.json").read_bytes()
    assert (first / "probability_calibration_qualification_receipt.json").read_bytes() == (second / "probability_calibration_qualification_receipt.json").read_bytes()
    qualification = json.loads((first / "calibration_qualification_report.json").read_text(encoding="utf-8"))
    assert qualification["qualification_status"] == "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW"
    assert qualification["positive_group_count_by_class"]["EXTERNAL_VALIDATION"] == {"A": 4, "B": 4}
