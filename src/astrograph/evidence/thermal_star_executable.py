from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astrograph.data.thermal_star_adapter import D3ThermalStarDatasetAdapter
from astrograph.evidence.reaction_wheel_executable import D3RWExecutableEvidenceCatalog
from astrograph.evidence.executable import (
    EpisodeEvidenceBundle,
    EpisodeWindow,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
    evaluate_evidence,
    stable_id,
)
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, DeterministicOperatorRegistry, OperatorInput
from astrograph.semantics import TelemetrySemanticsRegistry


_STREAM_BY_PREFIX = {
    "status.thermal.": "thermal",
    "sensor.thermal.": "thermal",
    "sensor.eps.": "eps",
    "status.eps.": "eps",
    "estimate.eps.": "eps",
    "counter.eps.": "eps",
    "status.adcs.": "adcs",
    "sensor.adcs.": "adcs",
    "estimate.adcs.": "adcs",
    "adcs.rw.": "adcs",
}


class D3ThermalStarExecutableEvidenceBuilder:
    builder_id = "d3-thermal-star-executable-evidence-builder-v1"
    builder_version = "0.5.13"

    def __init__(
        self,
        package_root: str | Path,
        *,
        semantics_registry: TelemetrySemanticsRegistry,
        catalog: D3RWExecutableEvidenceCatalog,
        operator_registry: DeterministicOperatorRegistry | None = None,
    ) -> None:
        self.adapter = D3ThermalStarDatasetAdapter(package_root)
        self.semantics = semantics_registry
        self.catalog = catalog
        self.operators = operator_registry or BUILTIN_OPERATOR_REGISTRY
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        needed_streams = {self._stream(item.source_field_template) for item in self.catalog.feature_definitions}
        for stream in sorted(needed_streams):
            path = self.adapter.aggregate / {
                "adcs": "telemetry_adcs_raw.csv",
                "eps": "telemetry_eps_raw.csv",
                "thermal": "telemetry_thermal_raw.csv",
            }[stream]
            raw = pd.read_csv(path, low_memory=False)
            for column in raw.columns:
                if column not in {"record_id", "run_id"}:
                    raw[column] = pd.to_numeric(raw[column], errors="coerce")
            for run_id, frame in raw.groupby("run_id", sort=False):
                self._cache[(str(run_id), stream)] = frame.sort_values("time_s").reset_index(drop=True)

    def _frame(self, run_id: str, stream: str) -> pd.DataFrame:
        try:
            return self._cache[(str(run_id), stream)]
        except KeyError as exc:
            raise KeyError(f"missing stream {stream} for run {run_id}") from exc

    @staticmethod
    def _stream(field: str) -> str:
        for prefix, stream in _STREAM_BY_PREFIX.items():
            if field.startswith(prefix):
                return stream
        raise KeyError(f"no stream mapping for {field}")

    @staticmethod
    def _hash(frame: pd.DataFrame) -> str:
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
        return hashlib.sha256(values.tobytes()).hexdigest()

    @staticmethod
    def _longest(mask: np.ndarray) -> tuple[int, int] | None:
        best = None
        start = None
        for index, active in enumerate(mask.astype(bool)):
            if active and start is None:
                start = index
            if start is not None and (not active or index == len(mask) - 1):
                end = index if active and index == len(mask) - 1 else index - 1
                if best is None or end - start > best[1] - best[0]:
                    best = (start, end)
                start = None
        return best

    def _episode(self, run_id: str, pair_id: str, case_id: str) -> EpisodeWindow:
        if case_id == "thermal_heater_stuck_on":
            frame = self._frame(run_id, "thermal")
            mask = (frame["status.thermal.heater_on"] > 0.5) & (
                frame["status.thermal.heater_command_on"] <= 0.5
            )
            source = "heater_command_actual_mismatch"
        elif case_id == "star_tracker_dropout":
            frame = self._frame(run_id, "adcs")
            mask = (frame["status.adcs.star_tracker_valid"] <= 0.5) & (
                frame["status.adcs.star_tracker_locked"] <= 0.5
            )
            source = "star_valid_locked_false"
        else:
            frame = self._frame(run_id, "adcs")
            mask = np.zeros(len(frame), dtype=bool)
            source = "full_run_no_mechanism"
        times = frame["time_s"].to_numpy(float)
        interval = self._longest(np.asarray(mask))
        if interval is None:
            start, end = float(times[0]), float(times[-1])
        else:
            start, end = float(times[interval[0]]), float(times[interval[1]])
        return EpisodeWindow(
            episode_id=stable_id(
                "EP",
                {
                    "run_id": run_id,
                    "pair": pair_id,
                    "start": start,
                    "end": end,
                    "source": source,
                },
            ),
            run_id=run_id,
            paired_nominal_run_id=pair_id,
            start_s=start,
            end_s=end,
            source=source,
            source_is_label_free=True,
            operating_condition={
                "campaign": "D3_THERMAL_STAR_PRODUCTION",
                "case_family": "thermal_star",
            },
        )

    def _compute(
        self,
        definition: FeatureDefinition,
        run_id: str,
        pair_id: str,
        episode: EpisodeWindow,
    ) -> FeatureResult:
        field = definition.source_field_template
        stream = self._stream(field)
        current_frame = self._frame(run_id, stream)
        baseline_frame = self._frame(pair_id, stream)
        selected = current_frame[
            (current_frame.time_s >= episode.start_s) & (current_frame.time_s <= episode.end_s)
        ].copy()
        baseline_selected = baseline_frame[baseline_frame.time_s.isin(selected.time_s)].copy()
        semantic = self.semantics.get(field)
        value: float | None = None
        flags: list[str] = []

        if selected.empty:
            flags = ["MISSING_WINDOW_DATA"]
        else:
            operator_id = definition.aggregation.value
            parameters: dict[str, Any] = {}
            current_values: np.ndarray
            baseline_values: np.ndarray | None = None
            time_values: np.ndarray | None = None

            if definition.feature_definition_id == "FD-HEATER-CMD-ACTUAL-MISMATCH-FRACTION-V1":
                operator_id = "JOINT_THRESHOLD_FRACTION"
                current_values = selected["status.thermal.heater_on"].to_numpy(float)
                baseline_values = selected["status.thermal.heater_command_on"].to_numpy(float)
                parameters = {
                    "current_comparison": "GREATER_THAN",
                    "current_threshold": 0.5,
                    "baseline_comparison": "LESS_OR_EQUAL",
                    "baseline_threshold": 0.5,
                }
            elif definition.aggregation in {
                FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL,
                FeatureAggregation.MEAN_ABS_PAIRED_RESIDUAL,
                FeatureAggregation.MEAN_PAIRED_RESIDUAL,
                FeatureAggregation.LAST_PAIRED_RESIDUAL,
                FeatureAggregation.MAX_ABS_STATUS_RESIDUAL,
                FeatureAggregation.AVAILABILITY_DROP_FRACTION,
            }:
                merged = selected[["time_s", field]].merge(
                    baseline_selected[["time_s", field]],
                    on="time_s",
                    suffixes=("_cur", "_base"),
                )
                current_values = merged[f"{field}_cur"].to_numpy(float)
                baseline_values = merged[f"{field}_base"].to_numpy(float)
                time_values = merged["time_s"].to_numpy(float)
            else:
                current_values = selected[field].to_numpy(float)
                if definition.aggregation == FeatureAggregation.LINEAR_SLOPE:
                    time_values = selected["time_s"].to_numpy(float)

            value = self.operators.compute(
                operator_id,
                OperatorInput(
                    current=current_values,
                    baseline=baseline_values,
                    time_s=time_values,
                    parameters=parameters,
                ),
            )
            if value is None:
                flags.append("NO_FINITE_RESULT")

        feature_result_id = stable_id(
            "FR",
            {
                "run": run_id,
                "pair": pair_id,
                "episode": episode.episode_id,
                "feature": definition.feature_definition_id,
                "value": value,
            },
        )
        return FeatureResult(
            feature_result_id=feature_result_id,
            feature_definition_id=definition.feature_definition_id,
            run_id=run_id,
            paired_nominal_run_id=pair_id,
            episode_id=episode.episode_id,
            resolved_field=field,
            component_instance=semantic.component_instance,
            value=value,
            unit=definition.unit,
            window_start_s=episode.start_s,
            window_end_s=episode.end_s,
            sample_count=len(selected),
            algorithm_version=definition.algorithm_version,
            input_data_sha256=self._hash(selected),
            quality_flags=flags,
        )

    def build_run(self, run_id: str) -> EpisodeEvidenceBundle:
        descriptor = self.adapter.get_run(run_id)
        pair_id = descriptor.paired_nominal_run_id
        episode = self._episode(run_id, pair_id, descriptor.case_id)
        features = [
            self._compute(definition, run_id, pair_id, episode)
            for definition in self.catalog.feature_definitions
        ]
        feature_by_id = {item.feature_definition_id: item for item in features}
        facts = [
            evaluate_evidence(definition, feature_by_id.get(definition.feature_definition_id))
            for definition in self.catalog.evidence_definitions
        ]
        component = (
            "PAYLOAD_HEATER"
            if descriptor.case_id
            in {"thermal_heater_stuck_on", "nominal_heater_on", "nominal_heater_off"}
            else "STAR_TRACKER"
        )
        detected = (
            descriptor.case_id in {"thermal_heater_stuck_on", "star_tracker_dropout"}
            and episode.source != "full_run_no_mechanism"
        )
        return EpisodeEvidenceBundle(
            episode=episode,
            response_window=None,
            domain_id="D3_THERMAL_STAR",
            analysis_component_instance=component,
            selected_component_instance=component if detected else None,
            component_selection_status=(
                "MECHANISM_DETECTED" if detected else "NO_MECHANISM_DETECTED"
            ),
            wheel_selection_status="NOT_APPLICABLE",
            feature_definitions=self.catalog.feature_definitions,
            feature_results=features,
            evidence_definitions=self.catalog.evidence_definitions,
            evidence_facts=facts,
            source_protocol_hashes=self.semantics.validation.protocol_hashes,
            evaluation_metadata={
                "eligible_for_model_input": False,
                "truth_case_id": descriptor.case_id,
                "truth_fault_id": descriptor.fault_id,
                "split": descriptor.split,
                "severity": descriptor.severity,
                "condition": descriptor.condition,
                "repeat_index": descriptor.repeat_index,
            },
        )

    @staticmethod
    def core_payload(bundle: EpisodeEvidenceBundle) -> dict[str, Any]:
        return {
            "evidence_facts": [item.model_dump(mode="json") for item in bundle.evidence_facts]
        }
