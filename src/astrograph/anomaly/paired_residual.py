from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astrograph.anomaly.rules import AnomalyRule, AnomalyRuleSet
from astrograph.conditions import ConditionCatalog, OperatingConditionResolver
from astrograph.data.raw_adapter import D0DatasetAdapter, STREAM_FILES
from astrograph.schemas import AnomalyEvidence


@dataclass(frozen=True)
class DetectionRecord:
    run_id: str
    paired_nominal_run_id: str
    case_id: str
    stream: str
    channel_id: str
    anomaly_id: str
    rule_id: str
    detected: bool
    threshold_abs: float
    peak_residual_abs: float
    signed_peak_residual: float
    detection_time_s: float | None
    detection_delay_s: float | None
    active_start_s: float
    active_end_s: float
    confidence: float
    expected_direction: str
    direction_matched: bool
    semantic_role: str
    rule_set_version: str
    baseline_condition_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class PairedResidualDetector:
    """Configuration-driven deterministic paired-residual anomaly baseline.

    Confidence is a normalized residual-strength indicator, never a calibrated
    probability of fault occurrence.
    """

    detector_id = "paired_residual_v2"

    def __init__(
        self,
        adapter: D0DatasetAdapter,
        *,
        rule_set: AnomalyRuleSet | None = None,
        rules_path: str | Path | None = None,
        condition_resolver: OperatingConditionResolver | None = None,
    ):
        self.adapter = adapter
        if rule_set is not None and rules_path is not None:
            raise ValueError("provide rule_set or rules_path, not both")
        if rule_set is None:
            path = Path(rules_path) if rules_path else Path(__file__).resolve().parents[3] / "configs" / "anomaly_rules.json"
            rule_set = AnomalyRuleSet.from_json(path)
        self.rule_set = rule_set
        self.rules = {rule.channel_id: rule for rule in rule_set.enabled_rules()}
        if condition_resolver is None:
            condition_path = Path(__file__).resolve().parents[3] / "configs" / "operating_conditions.json"
            condition_resolver = OperatingConditionResolver(ConditionCatalog.from_json(condition_path))
        self.condition_resolver = condition_resolver

    @staticmethod
    def _confidence(peak: float, threshold: float) -> float:
        return min(0.999, max(0.0, peak) / (max(0.0, peak) + max(1e-12, threshold)))

    @staticmethod
    def _align(fault: pd.DataFrame, nominal: pd.DataFrame, channel: str) -> pd.DataFrame:
        left = fault[["time_s", channel]].rename(columns={channel: "fault_value"})
        right = nominal[["time_s", channel]].rename(columns={channel: "nominal_value"})
        joined = left.merge(right, on="time_s", how="inner", validate="one_to_one")
        joined["residual"] = pd.to_numeric(joined["fault_value"], errors="coerce") - pd.to_numeric(
            joined["nominal_value"], errors="coerce"
        )
        return joined.dropna(subset=["residual"]).sort_values("time_s").reset_index(drop=True)

    @staticmethod
    def _direction_mask(residuals: np.ndarray, rule: AnomalyRule, threshold: float) -> np.ndarray:
        if rule.expected_direction == "increase":
            return residuals >= threshold
        if rule.expected_direction == "decrease":
            return residuals <= -threshold
        return np.abs(residuals) >= threshold

    @staticmethod
    def _direction_matched(signed_peak: float, rule: AnomalyRule) -> bool:
        if rule.expected_direction == "increase":
            return signed_peak > 0
        if rule.expected_direction == "decrease":
            return signed_peak < 0
        return True

    def detect_run_records(self, run_id: str) -> list[DetectionRecord]:
        run = self.adapter.get_run(run_id)
        if run.case_type != "fault" or not run.paired_nominal_run_id:
            return []
        event = self.adapter.fault_event(run_id)
        if event is None:
            raise ValueError(f"Fault run {run_id} contains no resolved fault event")
        start_s = float(event["start_s"])
        end_s = start_s + float(event["duration_s"])
        nominal_legacy = self.adapter.load_stream(run.paired_nominal_run_id, "legacy")
        nominal_adcs = self.adapter.load_stream(run.paired_nominal_run_id, "adcs")
        nominal_legacy_active = nominal_legacy[(nominal_legacy["time_s"] >= start_s) & (nominal_legacy["time_s"] <= end_s)]
        nominal_adcs_active = nominal_adcs[(nominal_adcs["time_s"] >= start_s) & (nominal_adcs["time_s"] <= end_s)]
        baseline_condition = self.condition_resolver.resolve(nominal_legacy_active, nominal_adcs_active)
        baseline_condition_ids = set(baseline_condition.condition_ids)

        records: list[DetectionRecord] = []
        for stream in STREAM_FILES:
            fault = self.adapter.load_stream(run_id, stream)
            nominal = self.adapter.load_stream(run.paired_nominal_run_id, stream)
            stream_rules = [
                rule
                for rule in self.rules.values()
                if rule.stream == stream and rule.channel_id in fault.columns and rule.channel_id in nominal.columns
            ]
            for rule in stream_rules:
                if rule.applicable_condition_ids and not baseline_condition_ids.intersection(rule.applicable_condition_ids):
                    continue
                aligned = self._align(fault, nominal, rule.channel_id)
                if aligned.empty:
                    continue
                pre = aligned[aligned["time_s"] < start_s]["residual"].abs()
                noise_floor = float(pre.median()) if not pre.empty else 0.0
                threshold = max(rule.minimum_change, noise_floor * rule.noise_multiplier)
                active = aligned[(aligned["time_s"] >= start_s) & (aligned["time_s"] <= end_s)].copy()
                if active.empty:
                    continue
                residuals = active["residual"].to_numpy(dtype=float)
                absolute = np.abs(residuals)
                peak_position = int(np.argmax(absolute))
                peak = float(absolute[peak_position])
                signed_peak = float(residuals[peak_position])
                above = self._direction_mask(residuals, rule, threshold)
                first_detection: float | None = None
                consecutive = 0
                times = active["time_s"].to_numpy(dtype=float)
                for position, flag in enumerate(above):
                    consecutive = consecutive + 1 if bool(flag) else 0
                    if consecutive >= rule.persistence_samples:
                        first_detection = float(times[position])
                        break
                records.append(
                    DetectionRecord(
                        run_id=run_id,
                        paired_nominal_run_id=run.paired_nominal_run_id,
                        case_id=run.case_id,
                        stream=stream,
                        channel_id=rule.channel_id,
                        anomaly_id=rule.anomaly_id,
                        rule_id=rule.rule_id,
                        detected=first_detection is not None,
                        threshold_abs=threshold,
                        peak_residual_abs=peak,
                        signed_peak_residual=signed_peak,
                        detection_time_s=first_detection,
                        detection_delay_s=None if first_detection is None else first_detection - start_s,
                        active_start_s=start_s,
                        active_end_s=end_s,
                        confidence=self._confidence(peak, threshold),
                        expected_direction=rule.expected_direction,
                        direction_matched=self._direction_matched(signed_peak, rule),
                        semantic_role=rule.semantic_role,
                        rule_set_version=self.rule_set.rule_set_version,
                        baseline_condition_ids=sorted(baseline_condition_ids),
                    )
                )
        return sorted(records, key=lambda item: (item.stream, item.channel_id))

    def detect_run(self, run_id: str) -> list[AnomalyEvidence]:
        evidences: dict[str, AnomalyEvidence] = {}
        rules_by_anomaly = {rule.anomaly_id: rule for rule in self.rules.values()}
        for record in self.detect_run_records(run_id):
            if not record.detected:
                continue
            rule = rules_by_anomaly[record.anomaly_id]
            evidence = AnomalyEvidence(
                anomaly_id=record.anomaly_id,
                name=rule.anomaly_name_zh,
                signal=record.channel_id,
                confidence=record.confidence,
                confidence_type=self.rule_set.confidence_semantics,
                observed_at_s=record.detection_time_s,
                metadata={
                    "detector_id": self.detector_id,
                    "rule_id": record.rule_id,
                    "rule_set_version": self.rule_set.rule_set_version,
                    "run_id": record.run_id,
                    "paired_nominal_run_id": record.paired_nominal_run_id,
                    "case_id": record.case_id,
                    "stream": record.stream,
                    "threshold_abs": record.threshold_abs,
                    "peak_residual_abs": record.peak_residual_abs,
                    "signed_peak_residual": record.signed_peak_residual,
                    "expected_direction": record.expected_direction,
                    "direction_matched": record.direction_matched,
                    "semantic_role": record.semantic_role,
                    "applicable_condition_ids": rule.applicable_condition_ids,
                    "baseline_condition_ids": record.baseline_condition_ids,
                    "condition_catalog_version": self.condition_resolver.catalog.catalog_version,
                    "detection_delay_s": record.detection_delay_s,
                    "active_start_s": record.active_start_s,
                    "active_end_s": record.active_end_s,
                    "calibration_status": "synthetic_uncalibrated",
                },
            )
            previous = evidences.get(record.anomaly_id)
            if previous is None or evidence.confidence > previous.confidence:
                evidences[record.anomaly_id] = evidence
        return sorted(evidences.values(), key=lambda item: item.anomaly_id)
