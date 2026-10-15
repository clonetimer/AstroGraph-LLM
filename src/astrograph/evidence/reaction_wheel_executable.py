from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astrograph.data.reaction_wheel_adapter import D3DatasetAdapter
from astrograph.evidence.executable import (
    EpisodeEvidenceBundle,
    EpisodeWindow,
    EvidenceDefinition,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
    evaluate_evidence,
    stable_id,
)
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, OperatorInput
from astrograph.semantics import TelemetrySemanticsRegistry


class D3RWExecutableEvidenceCatalog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.document = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self.catalog_version = str(self.document["catalog_version"])
        self.registered_faults = {
            str(item["fault_id"]): dict(item) for item in self.document.get("registered_faults", [])
        }
        self.feature_definitions = [
            FeatureDefinition.model_validate(item) for item in self.document.get("feature_definitions", [])
        ]
        self.evidence_definitions = [
            EvidenceDefinition.model_validate(item) for item in self.document.get("evidence_definitions", [])
        ]
        feature_ids = {item.feature_definition_id for item in self.feature_definitions}
        missing = sorted(
            {item.feature_definition_id for item in self.evidence_definitions} - feature_ids
        )
        if missing:
            raise ValueError(f"evidence catalog references unknown feature definitions: {missing}")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


class D3RWExecutableEvidenceBuilder:
    builder_id = "d3-rw-executable-evidence-builder-v1"
    builder_version = "0.5.0"

    def __init__(
        self,
        package_root: str | Path,
        *,
        semantics_registry: TelemetrySemanticsRegistry,
        catalog: D3RWExecutableEvidenceCatalog,
    ) -> None:
        self.adapter = D3DatasetAdapter(package_root)
        self.semantics = semantics_registry
        self.catalog = catalog
        raw = pd.read_csv(self.adapter.aggregate / "telemetry_adcs_raw.csv", low_memory=False)
        numeric = [column for column in raw.columns if column not in {"record_id", "run_id"}]
        for column in numeric:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        self._frames = {
            str(run_id): frame.sort_values("time_s").reset_index(drop=True)
            for run_id, frame in raw.groupby("run_id")
        }

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
    def _residual(merged: pd.DataFrame, field: str) -> np.ndarray:
        return (
            merged[f"{field}_cur"].to_numpy(dtype=float)
            - merged[f"{field}_base"].to_numpy(dtype=float)
        )

    def _wheel_score(self, merged: pd.DataFrame, wheel: int) -> float:
        availability_current = merged[f"status.adcs.rw.available_{wheel}_cur"].to_numpy(dtype=float)
        availability_base = merged[f"status.adcs.rw.available_{wheel}_base"].to_numpy(dtype=float)
        availability_drop = float(np.mean((availability_base - availability_current) > 0.5))
        driver = np.abs(self._residual(merged, f"status.adcs.rw.driver_code_{wheel}"))
        friction = np.abs(self._residual(merged, f"estimate.adcs.rw.friction_torque_nm_{wheel}"))
        torque = np.abs(self._residual(merged, f"estimate.adcs.rw.actual_torque_nm_{wheel}"))
        current = np.abs(self._residual(merged, f"sensor.adcs.rw.motor_current_a_{wheel}"))
        speed = np.abs(self._residual(merged, f"sensor.adcs.rw.speed_rad_s_{wheel}"))
        return max(
            availability_drop * 10.0,
            float(np.nanmax(driver, initial=0.0)),
            float(np.nanmax(friction, initial=0.0)) / 5.0e-4,
            float(np.nanmax(torque, initial=0.0)) / 5.0e-4,
            float(np.nanmax(current, initial=0.0)) / 1.0e-2,
            float(np.nanmax(speed, initial=0.0)) / 5.0e-2,
        )

    @staticmethod
    def _longest_interval(mask: np.ndarray, minimum_samples: int = 2) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        start: int | None = None
        for index, active in enumerate(mask.astype(bool)):
            if active and start is None:
                start = index
            if start is not None and (not active or index == len(mask) - 1):
                end = index if active and index == len(mask) - 1 else index - 1
                if end - start + 1 >= minimum_samples:
                    if best is None or end - start > best[1] - best[0]:
                        best = (start, end)
                start = None
        return best

    def _derive_episode(
        self,
        *,
        run_id: str,
        pair_id: str,
        merged: pd.DataFrame,
        wheel: int,
    ) -> tuple[EpisodeWindow, EpisodeWindow]:
        availability_current = merged[f"status.adcs.rw.available_{wheel}_cur"].to_numpy(dtype=float)
        availability_base = merged[f"status.adcs.rw.available_{wheel}_base"].to_numpy(dtype=float)
        availability_mask = (availability_base - availability_current) > 0.5
        driver_mask = np.abs(self._residual(merged, f"status.adcs.rw.driver_code_{wheel}")) >= 0.5
        friction_mask = (
            np.abs(self._residual(merged, f"estimate.adcs.rw.friction_torque_nm_{wheel}")) >= 5.0e-4
        )
        torque_status_mask = availability_mask | driver_mask
        actual_torque_mask = (
            np.abs(self._residual(merged, f"estimate.adcs.rw.actual_torque_nm_{wheel}")) >= 5.0e-4
        )
        current_mask = (
            np.abs(self._residual(merged, f"sensor.adcs.rw.motor_current_a_{wheel}")) >= 1.0e-2
        )
        speed_mask = np.abs(self._residual(merged, f"sensor.adcs.rw.speed_rad_s_{wheel}")) >= 5.0e-2
        response_mask = torque_status_mask | friction_mask | actual_torque_mask | current_mask | speed_mask
        times = merged["time_s"].to_numpy(dtype=float)

        # The mechanism window uses direct mechanism evidence only. Response signals can
        # persist after fault removal, so they are stored separately rather than being
        # treated as the physical injection interval.
        if torque_status_mask.any():
            mechanism_mask = torque_status_mask
            mechanism_source = "direct_torque_status_threshold"
        elif friction_mask.any():
            mechanism_mask = friction_mask
            mechanism_source = "direct_friction_residual_threshold"
        else:
            mechanism_mask = response_mask
            mechanism_source = "response_threshold_fallback"

        def make_window(mask: np.ndarray, source: str, suffix: str) -> EpisodeWindow:
            interval = self._longest_interval(mask)
            if interval is None:
                start_s, end_s = float(times[0]), float(times[-1])
                source = f"full_run_no_{suffix}_episode"
            else:
                start_s, end_s = float(times[interval[0]]), float(times[interval[1]])
            episode_id = stable_id(
                "EP" if suffix == "mechanism" else "RW",
                {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "wheel": wheel,
                    "source": source,
                    "window_kind": suffix,
                },
            )
            return EpisodeWindow(
                episode_id=episode_id,
                run_id=run_id,
                paired_nominal_run_id=pair_id,
                start_s=start_s,
                end_s=end_s,
                interval="closed",
                source=source,
                source_is_label_free=True,
                operating_condition={"campaign": "D3_RW_PRODUCTION", "window_kind": suffix},
            )

        return (
            make_window(mechanism_mask, mechanism_source, "mechanism"),
            make_window(response_mask, "paired_residual_response_threshold_union", "response"),
        )

    @staticmethod
    def _frame_hash(frame: pd.DataFrame) -> str:
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
        return hashlib.sha256(values.tobytes()).hexdigest()

    def _compute_feature(
        self,
        definition: FeatureDefinition,
        *,
        merged: pd.DataFrame,
        episode: EpisodeWindow,
        wheel: int,
        run_id: str,
        pair_id: str,
    ) -> FeatureResult:
        resolved_field = definition.source_field_template.format(wheel=wheel)
        semantic = self.semantics.get(resolved_field)
        if definition.aggregation == FeatureAggregation.AVAILABILITY_DROP_FRACTION:
            if semantic.unit != "bool" or definition.unit != "1":
                raise ValueError(
                    f"availability-drop feature requires bool source and dimensionless output: "
                    f"source={semantic.unit!r}, output={definition.unit!r}"
                )
        elif definition.unit != semantic.unit:
            raise ValueError(
                f"feature {definition.feature_definition_id} unit {definition.unit!r} does not match "
                f"telemetry semantic unit {semantic.unit!r} for {resolved_field}"
            )
        if definition.window_scope == "episode":
            selected = merged[(merged["time_s"] >= episode.start_s) & (merged["time_s"] <= episode.end_s)].copy()
        else:
            selected = merged.copy()
        quality_flags: list[str] = []
        if selected.empty:
            value: float | None = None
            quality_flags.append("MISSING_WINDOW_DATA")
        else:
            current_values = selected[f"{resolved_field}_cur"].to_numpy(dtype=float)
            baseline_values = selected[f"{resolved_field}_base"].to_numpy(dtype=float)
            time_values = selected["time_s"].to_numpy(dtype=float)
            if not np.isfinite(current_values).all() or not np.isfinite(baseline_values).all():
                quality_flags.append("NONFINITE_INPUT")
            value = BUILTIN_OPERATOR_REGISTRY.compute(
                definition.aggregation.value,
                OperatorInput(
                    current=current_values,
                    baseline=baseline_values,
                    time_s=time_values,
                ),
            )
        feature_result_id = stable_id(
            "FR",
            {
                "run_id": run_id,
                "pair_id": pair_id,
                "episode_id": episode.episode_id,
                "feature_definition_id": definition.feature_definition_id,
                "field": resolved_field,
                "value": value,
            },
        )
        return FeatureResult(
            feature_result_id=feature_result_id,
            feature_definition_id=definition.feature_definition_id,
            run_id=run_id,
            paired_nominal_run_id=pair_id,
            episode_id=episode.episode_id,
            resolved_field=resolved_field,
            component_instance=semantic.component_instance,
            value=value,
            unit=definition.unit,
            window_start_s=episode.start_s if definition.window_scope == "episode" else float(merged["time_s"].min()),
            window_end_s=episode.end_s if definition.window_scope == "episode" else float(merged["time_s"].max()),
            sample_count=int(len(selected)),
            algorithm_version=definition.algorithm_version,
            input_data_sha256=self._frame_hash(selected),
            quality_flags=quality_flags,
        )

    def build_run(self, run_id: str) -> EpisodeEvidenceBundle:
        descriptor = self.adapter.get_run(run_id)
        pair_id = descriptor.paired_nominal_run_id or run_id
        current = self._frames[run_id]
        baseline = self._frames[pair_id]
        merged = self._merge(current, baseline)
        wheel_scores = {wheel: self._wheel_score(merged, wheel) for wheel in (0, 1, 2)}
        analysis_wheel = max(wheel_scores, key=wheel_scores.get)
        selected_wheel = analysis_wheel if wheel_scores[analysis_wheel] > 1.0e-12 else None
        episode, response_window = self._derive_episode(
            run_id=run_id,
            pair_id=pair_id,
            merged=merged,
            wheel=analysis_wheel,
        )
        feature_results = [
            self._compute_feature(
                definition,
                merged=merged,
                episode=episode,
                wheel=analysis_wheel,
                run_id=run_id,
                pair_id=pair_id,
            )
            for definition in self.catalog.feature_definitions
        ]
        by_feature = {item.feature_definition_id: item for item in feature_results}
        facts = [
            evaluate_evidence(definition, by_feature.get(definition.feature_definition_id))
            for definition in self.catalog.evidence_definitions
        ]
        evaluation_metadata: dict[str, Any] = {
            "eligible_for_model_input": False,
            "truth_case_id": descriptor.case_id,
            "truth_fault_id": descriptor.fault_id,
            "split": descriptor.split,
            "paired_nominal_run_id": pair_id,
        }
        if descriptor.case_type == "fault":
            parameters = self.adapter.parameter_sample(run_id).get("sampled_values", {})
            onset = parameters.get("actual_fault_onset_s")
            duration = parameters.get("actual_fault_duration_s")
            if onset is not None and duration is not None:
                evaluation_metadata["label_window_closed"] = [float(onset), float(onset) + float(duration)]
                evaluation_metadata["label_window_is_model_input"] = False
            evaluation_metadata["injected_wheel_index"] = parameters.get("wheel_index")
            evaluation_metadata["injected_wheel_is_model_input"] = False
        return EpisodeEvidenceBundle(
            domain_id="D3_RW",
            episode=episode,
            response_window=response_window,
            analysis_wheel_index=analysis_wheel,
            analysis_component_instance=self.semantics.get(
                f"sensor.adcs.rw.speed_rad_s_{analysis_wheel}"
            ).component_instance,
            selected_wheel_index=selected_wheel,
            selected_component_instance=(
                None
                if selected_wheel is None
                else self.semantics.get(f"sensor.adcs.rw.speed_rad_s_{selected_wheel}").component_instance
            ),
            wheel_selection_status=(
                "MECHANISM_DETECTED" if selected_wheel is not None else "NO_MECHANISM_DETECTED"
            ),
            component_selection_status=(
                "MECHANISM_DETECTED" if selected_wheel is not None else "NO_MECHANISM_DETECTED"
            ),
            feature_definitions=self.catalog.feature_definitions,
            feature_results=feature_results,
            evidence_definitions=self.catalog.evidence_definitions,
            evidence_facts=facts,
            source_protocol_hashes={
                **self.semantics.validation.protocol_hashes,
                "d3_rw_executable_evidence_catalog": self.catalog.sha256,
            },
            evaluation_metadata=evaluation_metadata,
        )

    @staticmethod
    def core_evidence(bundle: EpisodeEvidenceBundle) -> list[dict[str, float]]:
        values = {item.feature_definition_id: item.value for item in bundle.feature_results}
        selected = bundle.analysis_wheel_index
        return [
            {
                "wheel_index": selected,
                "availability_drop_fraction": values.get("FD-RW-AVAILABLE-DROP-FRACTION-V1"),
                "driver_code_peak_abs": values.get("FD-RW-DRIVER-CODE-PEAK-ABS-V1"),
                "friction_estimate_peak_abs_nm": values.get("FD-RW-FRICTION-PEAK-ABS-V1", 0.0),
                "actual_torque_peak_abs_nm": values.get("FD-RW-ACTUAL-TORQUE-PEAK-ABS-V1", 0.0),
                "motor_current_peak_abs_a": values.get("FD-RW-CURRENT-PEAK-ABS-V1", 0.0),
                "wheel_speed_peak_abs_rad_s": values.get("FD-RW-SPEED-PEAK-ABS-V1", 0.0),
                "winding_temp_peak_abs_c": values.get("FD-RW-WINDING-TEMP-PEAK-ABS-V1", 0.0),
            }
        ]
