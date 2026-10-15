from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, DeterministicOperatorRegistry, OperatorInput
from astrograph.semantics import DEFAULT_UNIT_REGISTRY, is_prohibited_field
from astrograph.simulation.models import OnlineDataGateReport, SimulationRunCompleted


def column_schema_sha256(columns: list[str]) -> str:
    payload = json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ConfiguredChannelMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_field: str = Field(min_length=1)
    semantic_channel_id: str = Field(min_length=1)
    component_instance: str = Field(min_length=1)
    source_unit: str = Field(min_length=1)
    canonical_unit: str = Field(min_length=1)
    direction: Literal[-1, 1] = 1
    required: bool = True
    max_missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("source_field")
    @classmethod
    def reject_label_or_truth_fields(cls, value: str) -> str:
        if is_prohibited_field(value):
            raise ValueError("runtime channel mapping cannot read label, truth, event, or answer-proxy fields")
        return value


class ConfiguredTimeAxis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_field: str = Field(min_length=1)
    source_unit: str = "s"

    @field_validator("source_field")
    @classmethod
    def reject_label_or_truth_fields(cls, value: str) -> str:
        if is_prohibited_field(value):
            raise ValueError("runtime time mapping cannot use a prohibited field")
        return value


class ConfiguredRuntimeEvidencePackage(BaseModel):
    """Versioned, data-only contract for telemetry-to-evidence adaptation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-configured-runtime-evidence.v1"
    package_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain_id: str = Field(min_length=1)
    analysis_component_instance: str = Field(min_length=1)
    episode_mode: Literal["full_run"] = "full_run"
    time_axis: ConfiguredTimeAxis
    channels: list[ConfiguredChannelMapEntry] = Field(min_length=1)
    feature_definitions: list[FeatureDefinition] = Field(min_length=1)
    evidence_definitions: list[EvidenceDefinition] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "ConfiguredRuntimeEvidencePackage":
        source_fields = [self.time_axis.source_field, *(item.source_field for item in self.channels)]
        semantic_ids = [item.semantic_channel_id for item in self.channels]
        feature_ids = [item.feature_definition_id for item in self.feature_definitions]
        evidence_ids = [item.evidence_definition_id for item in self.evidence_definitions]
        for label, values in {
            "source fields": source_fields,
            "semantic channel IDs": semantic_ids,
            "feature IDs": feature_ids,
            "evidence IDs": evidence_ids,
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(f"configured runtime package contains duplicate {label}")
        channel_by_id = {item.semantic_channel_id: item for item in self.channels}
        feature_by_id = {item.feature_definition_id: item for item in self.feature_definitions}
        for definition in self.feature_definitions:
            try:
                channel = channel_by_id[definition.source_field_template]
            except KeyError as exc:
                raise ValueError(
                    f"feature {definition.feature_definition_id} references an unmapped semantic channel"
                ) from exc
            if definition.unit != channel.canonical_unit:
                raise ValueError(
                    f"feature {definition.feature_definition_id} unit does not match channel canonical unit"
                )
            if not BUILTIN_OPERATOR_REGISTRY.has(definition.aggregation.value):
                raise ValueError(f"feature references an unregistered operator: {definition.aggregation.value}")
        for definition in self.evidence_definitions:
            try:
                feature = feature_by_id[definition.feature_definition_id]
            except KeyError as exc:
                raise ValueError(
                    f"evidence {definition.evidence_definition_id} references an unknown feature"
                ) from exc
            if definition.threshold_unit != feature.unit:
                raise ValueError(
                    f"evidence {definition.evidence_definition_id} threshold unit does not match feature unit"
                )
            for conjunct in definition.and_feature_conditions:
                try:
                    conjunct_feature = feature_by_id[conjunct.feature_definition_id]
                except KeyError as exc:
                    raise ValueError(
                        f"evidence {definition.evidence_definition_id} conjunct references an unknown feature"
                    ) from exc
                if conjunct.threshold_unit != conjunct_feature.unit:
                    raise ValueError(
                        f"evidence {definition.evidence_definition_id} conjunct threshold unit "
                        "does not match feature unit"
                    )
        return self


class ConfiguredRuntimeEvidenceBuilder:
    builder_version = "0.7.1-configured-runtime-evidence-v1"

    def __init__(
        self,
        package_path: str | Path,
        *,
        operator_registry: DeterministicOperatorRegistry | None = None,
    ) -> None:
        self.package_path = Path(package_path).resolve()
        self.package = ConfiguredRuntimeEvidencePackage.model_validate_json(
            self.package_path.read_text(encoding="utf-8")
        )
        self.operators = operator_registry or BUILTIN_OPERATOR_REGISTRY
        self.package_sha256 = hashlib.sha256(self.package_path.read_bytes()).hexdigest()
        for channel in self.package.channels:
            source = DEFAULT_UNIT_REGISTRY.get(channel.source_unit)
            target = DEFAULT_UNIT_REGISTRY.get(channel.canonical_unit)
            if source.dimension != target.dimension:
                raise ValueError(
                    f"incompatible units for {channel.semantic_channel_id}: "
                    f"{channel.source_unit} -> {channel.canonical_unit}"
                )

    @staticmethod
    def _frame_hash(frame: pd.DataFrame) -> str:
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
        return hashlib.sha256(values.tobytes()).hexdigest()

    def _map_frame(self, source: pd.DataFrame) -> pd.DataFrame:
        mapped = pd.DataFrame(index=source.index)
        raw_time = pd.to_numeric(source[self.package.time_axis.source_field], errors="coerce")
        mapped["time_s"] = [
            DEFAULT_UNIT_REGISTRY.convert(value, source=self.package.time_axis.source_unit, target="s")
            if pd.notna(value)
            else np.nan
            for value in raw_time
        ]
        for channel in self.package.channels:
            if channel.source_field not in source.columns:
                continue
            raw = pd.to_numeric(source[channel.source_field], errors="coerce")
            converted = DEFAULT_UNIT_REGISTRY.convert_optional(
                [None if pd.isna(value) else float(value) for value in raw],
                source=channel.source_unit,
                target=channel.canonical_unit,
            )
            mapped[channel.semantic_channel_id] = [
                np.nan if value is None else float(channel.direction) * value for value in converted
            ]
        return mapped

    def _gate(
        self,
        *,
        event: SimulationRunCompleted,
        source: pd.DataFrame,
        nominal: pd.DataFrame,
    ) -> OnlineDataGateReport:
        required = [self.package.time_axis.source_field] + [
            item.source_field for item in self.package.channels if item.required
        ]
        missing = sorted(set(required) - set(source.columns))
        missing_nominal = sorted(set(required) - set(nominal.columns))
        warnings: list[str] = []
        if missing_nominal:
            warnings.append(f"paired nominal missing required fields: {missing_nominal}")
        source_hash = column_schema_sha256([str(item) for item in source.columns])
        nominal_hash = column_schema_sha256([str(item) for item in nominal.columns])
        if source_hash != self.package.source_schema_sha256:
            warnings.append("runtime source schema hash differs from configured source_schema_sha256")
        if nominal_hash != self.package.source_schema_sha256:
            warnings.append("paired nominal schema hash differs from configured source_schema_sha256")
        prohibited = sorted(
            item.source_field for item in self.package.channels if is_prohibited_field(item.source_field)
        )
        time = (
            pd.to_numeric(source[self.package.time_axis.source_field], errors="coerce")
            if self.package.time_axis.source_field in source.columns
            else pd.Series(dtype=float)
        )
        nominal_time = (
            pd.to_numeric(nominal[self.package.time_axis.source_field], errors="coerce")
            if self.package.time_axis.source_field in nominal.columns
            else pd.Series(dtype=float)
        )
        monotonic = bool(time.notna().all() and time.is_monotonic_increasing and time.is_unique)
        aligned = bool(
            len(time) == len(nominal_time)
            and time.notna().all()
            and nominal_time.notna().all()
            and np.array_equal(time.to_numpy(float), nominal_time.to_numpy(float))
        )
        checked = sorted(set(required) & set(source.columns))
        nominal_checked = sorted(set(required) & set(nominal.columns))
        missing_values = (
            int(source[checked].isna().sum().sum()) if checked else 0
        ) + (
            int(nominal[nominal_checked].isna().sum().sum()) if nominal_checked else 0
        )
        missing_violations = []
        for channel in self.package.channels:
            if channel.required and channel.source_field in source.columns:
                fraction = float(source[channel.source_field].isna().mean())
                if fraction > channel.max_missing_fraction:
                    missing_violations.append(
                        f"{channel.source_field} missing fraction {fraction:.6g} exceeds "
                        f"{channel.max_missing_fraction:.6g}"
                    )
            if channel.required and channel.source_field in nominal.columns:
                fraction = float(nominal[channel.source_field].isna().mean())
                if fraction > channel.max_missing_fraction:
                    missing_violations.append(
                        f"paired nominal {channel.source_field} missing fraction {fraction:.6g} exceeds "
                        f"{channel.max_missing_fraction:.6g}"
                    )
        if missing_violations:
            warnings.extend(missing_violations)
        passed = not any(
            (
                missing,
                missing_nominal,
                prohibited,
                source_hash != self.package.source_schema_sha256,
                nominal_hash != self.package.source_schema_sha256,
                not monotonic,
                not aligned,
                missing_violations,
            )
        )
        return OnlineDataGateReport(
            passed=passed,
            run_id=event.run_id,
            row_count=len(source),
            field_count=len(source.columns),
            required_field_count=len(required),
            time_monotonic=monotonic,
            time_aligned_with_nominal=aligned,
            missing_required_fields=sorted(set(missing + missing_nominal)),
            missing_values=missing_values,
            prohibited_model_input_fields=prohibited,
            protocol_version=f"{self.package.package_id}@{self.package.version}",
            warnings=warnings,
            claim_boundary=self.package.claim_boundary,
        )

    def _feature_result(
        self,
        *,
        definition: FeatureDefinition,
        current: pd.DataFrame,
        baseline: pd.DataFrame,
        episode: EpisodeWindow,
        event: SimulationRunCompleted,
    ) -> FeatureResult:
        field = definition.source_field_template
        channel = next(item for item in self.package.channels if item.semantic_channel_id == field)
        selected = current[
            (current["time_s"] >= episode.start_s) & (current["time_s"] <= episode.end_s)
        ][["time_s", field]].copy()
        baseline_selected = baseline[["time_s", field]].copy()
        merged = selected.merge(
            baseline_selected,
            on="time_s",
            suffixes=("_current", "_baseline"),
            how="inner",
            validate="one_to_one",
        )
        flags: list[str] = []
        value: float | None = None
        if merged.empty:
            flags.append("MISSING_WINDOW_DATA")
        else:
            operator = self.operators.get(definition.aggregation.value).spec
            current_values = merged[f"{field}_current"].to_numpy(float)
            baseline_values = (
                merged[f"{field}_baseline"].to_numpy(float) if operator.requires_baseline else None
            )
            time_values = merged["time_s"].to_numpy(float) if operator.requires_time else None
            value = self.operators.compute(
                definition.aggregation.value,
                OperatorInput(current=current_values, baseline=baseline_values, time_s=time_values),
            )
            if value is None:
                flags.append("NO_FINITE_RESULT")
        return FeatureResult(
            feature_result_id=stable_id(
                "FR",
                {
                    "run": event.run_id,
                    "episode": episode.episode_id,
                    "feature": definition.feature_definition_id,
                    "package_sha256": self.package_sha256,
                },
            ),
            feature_definition_id=definition.feature_definition_id,
            run_id=event.run_id,
            paired_nominal_run_id=Path(event.nominal_run_dir).name,
            episode_id=episode.episode_id,
            resolved_field=field,
            component_instance=channel.component_instance,
            value=value,
            unit=definition.unit,
            window_start_s=episode.start_s,
            window_end_s=episode.end_s,
            sample_count=len(merged),
            algorithm_version=definition.algorithm_version,
            input_data_sha256=self._frame_hash(merged),
            quality_flags=flags,
        )

    def build(
        self, event: SimulationRunCompleted
    ) -> tuple[EpisodeEvidenceBundle, OnlineDataGateReport]:
        source = pd.read_csv(Path(event.run_dir) / "telemetry.csv", low_memory=False)
        nominal = pd.read_csv(Path(event.nominal_run_dir) / "telemetry.csv", low_memory=False)
        return self.build_from_frames(event, source=source, nominal=nominal)

    def build_from_frames(
        self,
        event: SimulationRunCompleted,
        *,
        source: pd.DataFrame,
        nominal: pd.DataFrame,
    ) -> tuple[EpisodeEvidenceBundle, OnlineDataGateReport]:
        """Build evidence from caller-owned telemetry frames.

        This is semantically identical to :meth:`build`, but allows an offline
        evaluator to cache paired nominal telemetry and reuse the already-read
        source frame for a data-driven model.  The builder never mutates either
        caller-owned frame.
        """
        gate = self._gate(event=event, source=source, nominal=nominal)
        if not gate.passed:
            raise ValueError(f"configured runtime data gate failed: {gate.model_dump(mode='json')}")
        current = self._map_frame(source)
        baseline = self._map_frame(nominal)
        time = current["time_s"].to_numpy(float)
        episode = EpisodeWindow(
            episode_id=stable_id(
                "EP",
                {
                    "run": event.run_id,
                    "start": float(time[0]),
                    "end": float(time[-1]),
                    "package_sha256": self.package_sha256,
                },
            ),
            run_id=event.run_id,
            paired_nominal_run_id=Path(event.nominal_run_dir).name,
            start_s=float(time[0]),
            end_s=float(time[-1]),
            source="configured label-free full-run analysis",
            source_is_label_free=True,
            operating_condition={"runtime_id": event.runtime_id, "mapping_id": self.package.package_id},
        )
        results = [
            self._feature_result(
                definition=definition,
                current=current,
                baseline=baseline,
                episode=episode,
                event=event,
            )
            for definition in self.package.feature_definitions
        ]
        by_feature = {item.feature_definition_id: item for item in results}
        facts = [
            evaluate_evidence(
                definition, by_feature.get(definition.feature_definition_id),
                all_feature_results=results,
            )
            for definition in self.package.evidence_definitions
        ]
        satisfied_critical_definitions = [
            definition
            for fact, definition in zip(facts, self.package.evidence_definitions, strict=True)
            if fact.status.value == "SATISFIED" and definition.critical
        ]
        satisfied_components = sorted(
            {
                by_feature[definition.feature_definition_id].component_instance
                for definition in satisfied_critical_definitions
                if definition.feature_definition_id in by_feature
            }
        )
        # A multi-subsystem configured package must select the component that
        # produced the satisfied executable evidence.  Falling back to the
        # package-wide analysis scope here would silently report, for example,
        # an EPS fault as a thermal-heater fault.  Multiple simultaneous
        # components remain explicit and therefore do not manufacture a unique
        # component selection.
        selected_component = satisfied_components[0] if len(satisfied_components) == 1 else None
        bundle = EpisodeEvidenceBundle(
            episode=episode,
            domain_id=self.package.domain_id,
            analysis_component_instance=self.package.analysis_component_instance,
            selected_component_instance=selected_component,
            wheel_selection_status="NOT_APPLICABLE",
            component_selection_status=(
                "MECHANISM_DETECTED"
                if selected_component is not None
                else ("NOT_APPLICABLE" if satisfied_components else "NO_MECHANISM_DETECTED")
            ),
            analysis_metadata={
                "configured_runtime_package_id": self.package.package_id,
                "configured_runtime_package_version": self.package.version,
                "episode_mode": self.package.episode_mode,
                "satisfied_critical_fault_ids": sorted(
                    {item.target_fault_id for item in satisfied_critical_definitions}
                ),
                "satisfied_critical_components": satisfied_components,
            },
            feature_definitions=self.package.feature_definitions,
            feature_results=results,
            evidence_definitions=self.package.evidence_definitions,
            evidence_facts=facts,
            source_protocol_hashes={
                "configured_runtime_evidence_package": self.package_sha256,
                "source_column_schema": self.package.source_schema_sha256,
            },
            evaluation_metadata={"eligible_for_model_input": False},
            claim_boundary=self.package.claim_boundary,
        )
        return bundle, gate


__all__ = [
    "ConfiguredChannelMapEntry",
    "ConfiguredRuntimeEvidenceBuilder",
    "ConfiguredRuntimeEvidencePackage",
    "ConfiguredTimeAxis",
    "column_schema_sha256",
]
