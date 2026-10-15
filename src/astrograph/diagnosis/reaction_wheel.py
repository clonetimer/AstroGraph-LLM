from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class D3RWDiagnosis:
    run_id: str
    truth_case_id: str
    split: str
    diagnosis_status: str
    predicted_case_id: str | None
    inferred_wheel_index: int | None
    evidence: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_rw_residual_evidence(
    wheel_evidence: list[dict[str, float]],
    *,
    nominal_case: bool = False,
) -> tuple[str, str | None, int | None, dict[str, float], str]:
    """Classify two D3 reaction-wheel mechanisms with deterministic E0 rules.

    Direct driver/availability status is authoritative for torque-authority loss.
    Friction requires the wheel to remain available and a friction estimate plus
    current/speed response. Missing or contradictory status cannot be forced into
    a unique conclusion.
    """
    if nominal_case:
        return "DIAGNOSED", "nominal", None, {}, "Nominal paired residual baseline."
    if not wheel_evidence:
        return "UNKNOWN", None, None, {}, "No wheel evidence was supplied."

    def score(item: dict[str, float]) -> float:
        return max(
            item.get("availability_drop_fraction", 0.0) * 10.0,
            item.get("driver_code_peak_abs", 0.0),
            item.get("friction_estimate_peak_abs_nm", 0.0) / 0.0005,
            item.get("actual_torque_peak_abs_nm", 0.0) / 0.0005,
            item.get("motor_current_peak_abs_a", 0.0) / 0.01,
            item.get("wheel_speed_peak_abs_rad_s", 0.0) / 0.05,
        )

    selected = max(wheel_evidence, key=score)
    wheel_index = int(selected["wheel_index"])
    availability = selected.get("availability_drop_fraction")
    driver = selected.get("driver_code_peak_abs")
    friction = selected.get("friction_estimate_peak_abs_nm", 0.0)
    torque = selected.get("actual_torque_peak_abs_nm", 0.0)
    current = selected.get("motor_current_peak_abs_a", 0.0)
    speed = selected.get("wheel_speed_peak_abs_rad_s", 0.0)

    status_missing = availability is None or driver is None or np.isnan(float(availability)) or np.isnan(float(driver))
    shared_response = current >= 0.01 and speed >= 0.05
    friction_evidence = friction >= 0.0005 and shared_response
    availability_status = False if status_missing else float(availability) >= 0.05
    driver_status = False if status_missing else float(driver) >= 0.5
    torque_status = availability_status and driver_status
    status_conflict = (availability_status != driver_status) if not status_missing else False
    torque_response = torque >= 0.0005 and shared_response

    if status_missing and (friction_evidence or torque_response or shared_response):
        return (
            "AMBIGUOUS",
            None,
            wheel_index,
            selected,
            "Driver availability/status is missing; shared wheel response cannot uniquely separate friction from torque-authority loss.",
        )
    if status_conflict:
        return (
            "AMBIGUOUS",
            None,
            wheel_index,
            selected,
            "Wheel availability and driver-code status disagree; the mechanism cannot be uniquely resolved.",
        )
    if torque_status:
        return (
            "DIAGNOSED",
            "rw_torque_authority_loss",
            wheel_index,
            selected,
            "Wheel availability/driver status indicates torque-authority loss.",
        )
    if friction_evidence:
        return (
            "DIAGNOSED",
            "rw_friction_increase",
            wheel_index,
            selected,
            "Wheel remains available while friction estimate, current and speed residuals increase.",
        )
    if shared_response or torque_response:
        return (
            "AMBIGUOUS",
            None,
            wheel_index,
            selected,
            "Wheel response is abnormal but direct mechanism evidence is insufficient.",
        )
    return "UNKNOWN", None, wheel_index, selected, "No configured D3 mechanism threshold was met."


def run_d3_deterministic_diagnosis(
    package_root: str | Path,
    split_manifest_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    root = Path(package_root)
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8-sig"))
    split_lookup = {
        str(run_id): str(split_name)
        for split_name, run_ids in split["run_ids"].items()
        for run_id in run_ids
    }
    manifests: dict[str, dict] = {}
    for directory in (root / "03_run_audit").iterdir():
        path = directory / "run_manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            manifests[str(manifest["run_id"])] = manifest

    raw = pd.read_csv(root / "02_dataset" / "aggregate" / "telemetry_adcs_raw.csv", low_memory=False)
    feature_columns = [column for column in raw.columns if column not in {"record_id", "run_id", "time_s"}]
    frames = {
        str(run_id): frame.sort_values("time_s").reset_index(drop=True)
        for run_id, frame in raw.groupby("run_id")
    }

    output_rows: list[dict] = []
    for run_id, manifest in manifests.items():
        current = frames[run_id]
        pair_id = run_id if manifest["case_type"] == "nominal" else str(manifest["paired_nominal_run_id"])
        baseline = frames[pair_id]
        merged = current[["time_s", *feature_columns]].merge(
            baseline[["time_s", *feature_columns]],
            on="time_s",
            suffixes=("_cur", "_base"),
            how="inner",
        )
        wheel_evidence: list[dict[str, float]] = []
        for wheel in (0, 1, 2):
            residual = {
                column: (
                    merged[f"{column}_cur"].to_numpy(dtype=float)
                    - merged[f"{column}_base"].to_numpy(dtype=float)
                )
                for column in feature_columns
            }
            availability_current = merged[f"status.adcs.rw.available_{wheel}_cur"].to_numpy(dtype=float)
            availability_base = merged[f"status.adcs.rw.available_{wheel}_base"].to_numpy(dtype=float)
            wheel_evidence.append(
                {
                    "wheel_index": wheel,
                    "availability_drop_fraction": float(np.mean((availability_base - availability_current) > 0.5)),
                    "driver_code_peak_abs": float(np.max(np.abs(residual[f"status.adcs.rw.driver_code_{wheel}"]))),
                    "friction_estimate_peak_abs_nm": float(
                        np.max(np.abs(residual[f"estimate.adcs.rw.friction_torque_nm_{wheel}"]))
                    ),
                    "actual_torque_peak_abs_nm": float(
                        np.max(np.abs(residual[f"estimate.adcs.rw.actual_torque_nm_{wheel}"]))
                    ),
                    "motor_current_peak_abs_a": float(
                        np.max(np.abs(residual[f"sensor.adcs.rw.motor_current_a_{wheel}"]))
                    ),
                    "wheel_speed_peak_abs_rad_s": float(
                        np.max(np.abs(residual[f"sensor.adcs.rw.speed_rad_s_{wheel}"]))
                    ),
                    "winding_temp_peak_abs_c": float(
                        np.max(np.abs(residual[f"sensor.adcs.rw.winding_temp_c_{wheel}"]))
                    ),
                }
            )
        status, prediction, wheel_index, evidence, explanation = classify_rw_residual_evidence(
            wheel_evidence,
            nominal_case=manifest["case_type"] == "nominal",
        )
        row = D3RWDiagnosis(
            run_id=run_id,
            truth_case_id=str(manifest["case_id"]),
            split=split_lookup[run_id],
            diagnosis_status=status,
            predicted_case_id=prediction,
            inferred_wheel_index=wheel_index,
            evidence=evidence,
            explanation=explanation,
        ).to_dict()
        row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False, sort_keys=True)
        output_rows.append(row)

    frame = pd.DataFrame(output_rows)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame
