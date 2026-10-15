from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from astrograph.data.reaction_wheel_adapter import D3DatasetAdapter
from astrograph.schemas import AnomalyEvidence


@dataclass(frozen=True)
class D3RWRule:
    anomaly_id: str
    name: str
    signal_template: str
    threshold: float
    semantic_role: str
    minimum_consecutive_samples: int = 2


DEFAULT_D3_RW_RULES: tuple[D3RWRule, ...] = (
    D3RWRule(
        anomaly_id="ANOM_RW_FRICTION_ESTIMATE_HIGH",
        name="反作用轮摩擦力矩估计增大",
        signal_template="estimate.adcs.rw.friction_torque_nm_{wheel}",
        threshold=5.0e-4,
        semantic_role="direct",
    ),
    D3RWRule(
        anomaly_id="ANOM_RW_ACTUAL_TORQUE_LOSS",
        name="反作用轮实际力矩能力下降",
        signal_template="estimate.adcs.rw.actual_torque_nm_{wheel}",
        threshold=5.0e-4,
        semantic_role="component_response",
    ),
    D3RWRule(
        anomaly_id="ANOM_RW_CURRENT_DEVIATION",
        name="反作用轮电机电流偏移",
        signal_template="sensor.adcs.rw.motor_current_a_{wheel}",
        threshold=1.0e-2,
        semantic_role="component_response",
    ),
    D3RWRule(
        anomaly_id="ANOM_RW_SPEED_DEVIATION",
        name="反作用轮轮速偏移",
        signal_template="sensor.adcs.rw.speed_rad_s_{wheel}",
        threshold=5.0e-2,
        semantic_role="component_response",
    ),
)


class D3RWPairedResidualDetector:
    """Convert D3 paired nominal residuals into generic AstroGraph anomalies.

    The detector only consumes raw telemetry and the paired-normal relation. It does
    not use the fault label to decide which anomaly is emitted. The synthetic case
    identifier is copied into metadata strictly for offline evaluation.
    """

    detector_id = "d3_rw_paired_residual_v1"
    rule_set_version = "d3-rw-anomaly-rules-0.4.2-e0"

    def __init__(
        self,
        package_root: str | Path,
        *,
        split_override: str | Path | None = None,
        rules: Iterable[D3RWRule] = DEFAULT_D3_RW_RULES,
    ) -> None:
        self.adapter = D3DatasetAdapter(package_root, split_override=split_override)
        self.rules = tuple(rules)
        raw_path = self.adapter.aggregate / "telemetry_adcs_raw.csv"
        raw = pd.read_csv(raw_path, low_memory=False)
        numeric = [column for column in raw.columns if column not in {"record_id", "run_id"}]
        for column in numeric:
            raw[column] = pd.to_numeric(raw[column], errors="raise")
        self._frames = {
            str(run_id): frame.sort_values("time_s").reset_index(drop=True)
            for run_id, frame in raw.groupby("run_id")
        }

    @property
    def dataset_id(self) -> str:
        return self.adapter.dataset_id

    @staticmethod
    def _merge(current: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [column for column in current.columns if column not in {"record_id", "run_id", "time_s"}]
        return current[["time_s", *feature_columns]].merge(
            baseline[["time_s", *feature_columns]],
            on="time_s",
            suffixes=("_cur", "_base"),
            how="inner",
            validate="one_to_one",
        )

    @staticmethod
    def _residual(merged: pd.DataFrame, signal: str) -> np.ndarray:
        return (
            merged[f"{signal}_cur"].to_numpy(dtype=float)
            - merged[f"{signal}_base"].to_numpy(dtype=float)
        )

    @staticmethod
    def _intervals(mask: np.ndarray, times: np.ndarray, minimum: int) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        start: int | None = None
        for index, active in enumerate(mask.astype(bool)):
            if active and start is None:
                start = index
            if start is not None and (not active or index == len(mask) - 1):
                end = index if active and index == len(mask) - 1 else index - 1
                if end - start + 1 >= minimum:
                    intervals.append((start, end))
                start = None
        return intervals

    @staticmethod
    def _confidence(peak: float, threshold: float) -> float:
        if peak <= 0:
            return 0.0
        return float(min(1.0, peak / (peak + threshold)))

    def _wheel_score(self, merged: pd.DataFrame, wheel: int) -> float:
        availability = merged[f"status.adcs.rw.available_{wheel}_cur"].to_numpy(dtype=float)
        availability_base = merged[f"status.adcs.rw.available_{wheel}_base"].to_numpy(dtype=float)
        availability_drop = float(np.mean((availability_base - availability) > 0.5))
        driver = np.abs(self._residual(merged, f"status.adcs.rw.driver_code_{wheel}"))
        friction = np.abs(self._residual(merged, f"estimate.adcs.rw.friction_torque_nm_{wheel}"))
        torque = np.abs(self._residual(merged, f"estimate.adcs.rw.actual_torque_nm_{wheel}"))
        current = np.abs(self._residual(merged, f"sensor.adcs.rw.motor_current_a_{wheel}"))
        speed = np.abs(self._residual(merged, f"sensor.adcs.rw.speed_rad_s_{wheel}"))
        return max(
            availability_drop * 10.0,
            float(driver.max(initial=0.0)),
            float(friction.max(initial=0.0)) / 5.0e-4,
            float(torque.max(initial=0.0)) / 5.0e-4,
            float(current.max(initial=0.0)) / 1.0e-2,
            float(speed.max(initial=0.0)) / 5.0e-2,
        )

    def detect_run(self, run_id: str) -> list[AnomalyEvidence]:
        descriptor = self.adapter.get_run(run_id)
        pair_id = descriptor.paired_nominal_run_id or run_id
        current = self._frames[run_id]
        baseline = self._frames[pair_id]
        merged = self._merge(current, baseline)
        times = merged["time_s"].to_numpy(dtype=float)
        selected_wheel = max((0, 1, 2), key=lambda wheel: self._wheel_score(merged, wheel))
        output: list[AnomalyEvidence] = []

        def emit(
            *,
            anomaly_id: str,
            name: str,
            signal: str,
            values: np.ndarray,
            mask: np.ndarray,
            threshold: float,
            semantic_role: str,
            minimum: int = 2,
            metadata_extra: dict[str, Any] | None = None,
        ) -> None:
            for start_index, end_index in self._intervals(mask, times, minimum):
                segment = np.abs(values[start_index : end_index + 1])
                peak = float(segment.max(initial=0.0))
                output.append(
                    AnomalyEvidence(
                        anomaly_id=anomaly_id,
                        name=name,
                        signal=signal,
                        confidence=self._confidence(peak, threshold),
                        observed_at_s=float(times[start_index]),
                        metadata={
                            "dataset_id": self.dataset_id,
                            "run_id": run_id,
                            "paired_nominal_run_id": pair_id,
                            "case_id": descriptor.case_id,
                            "split": descriptor.split,
                            "wheel_index": selected_wheel,
                            "active_start_s": float(times[start_index]),
                            "active_end_s": float(times[end_index]),
                            "detector_id": self.detector_id,
                            "rule_set_version": self.rule_set_version,
                            "semantic_role": semantic_role,
                            "threshold_abs": threshold,
                            "peak_residual_abs": peak,
                            "source_is_truth_free": True,
                            "evaluation_case_id": descriptor.case_id,
                            **(metadata_extra or {}),
                        },
                    )
                )

        for rule in self.rules:
            signal = rule.signal_template.format(wheel=selected_wheel)
            values = self._residual(merged, signal)
            abs_values = np.abs(values)
            emit(
                anomaly_id=rule.anomaly_id,
                name=rule.name,
                signal=signal,
                values=values,
                mask=abs_values >= rule.threshold,
                threshold=rule.threshold,
                semantic_role=rule.semantic_role,
                minimum=rule.minimum_consecutive_samples,
            )

        availability_cur = merged[f"status.adcs.rw.available_{selected_wheel}_cur"].to_numpy(dtype=float)
        availability_base = merged[f"status.adcs.rw.available_{selected_wheel}_base"].to_numpy(dtype=float)
        driver_residual = self._residual(merged, f"status.adcs.rw.driver_code_{selected_wheel}")
        unavailable_mask = ((availability_base - availability_cur) > 0.5) | (np.abs(driver_residual) >= 0.5)
        status_strength = np.maximum(np.abs(availability_base - availability_cur), np.abs(driver_residual))
        emit(
            anomaly_id="ANOM_RW_DRIVER_UNAVAILABLE",
            name="反作用轮驱动器不可用",
            signal=f"status.adcs.rw.available_{selected_wheel}|status.adcs.rw.driver_code_{selected_wheel}",
            values=status_strength,
            mask=unavailable_mask,
            threshold=0.5,
            semantic_role="direct",
            minimum=2,
        )

        # A healthy availability state is contextual absence-of-fault evidence, not an
        # anomaly episode. It is intentionally not emitted into a run-level anomaly
        # bag because post-fault recovery would otherwise contradict a genuine
        # torque-authority-loss episode after temporal aggregation.

        gyro_signals = [f"sensor.adcs.gyro_rate_rad_s_{axis}" for axis in ("x", "y", "z")]
        gyro_values = np.vstack([np.abs(self._residual(merged, signal)) for signal in gyro_signals])
        gyro_peak = gyro_values.max(axis=0)
        emit(
            anomaly_id="ANOM_GYRO_RATE_DEVIATION",
            name="陀螺角速度偏移",
            signal="|".join(gyro_signals),
            values=gyro_peak,
            mask=gyro_peak >= 2.0e-5,
            threshold=2.0e-5,
            semantic_role="subsystem_propagation",
            minimum=2,
        )
        return sorted(
            output,
            key=lambda item: (
                float(item.metadata.get("active_start_s", item.observed_at_s or 0.0)),
                item.anomaly_id,
                item.signal,
            ),
        )

    def detect_all_fault_runs(self) -> list[AnomalyEvidence]:
        output: list[AnomalyEvidence] = []
        for descriptor in self.adapter.iter_runs():
            if descriptor.case_type == "fault":
                output.extend(self.detect_run(descriptor.run_id))
        return output


def write_anomaly_jsonl(path: str | Path, anomalies: Iterable[AnomalyEvidence]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.model_dump(mode="json") for item in anomalies]
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return len(rows)
