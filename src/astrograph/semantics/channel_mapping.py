from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any

import pandas as pd

from .contracts import (
    ChannelMap,
    SamplingProfile,
    SemanticMappingReceipt,
    SemanticMappingResult,
    SemanticSchema,
    SemanticSignal,
    SemanticTelemetryFrame,
    UnmappedFieldPolicy,
)
from .leakage import is_prohibited_field
from .units import DEFAULT_UNIT_REGISTRY, TIME, UnitRegistry


class SemanticMappingError(ValueError):
    def __init__(self, code: str, message: str, *, fields: list[str] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.fields = sorted(fields or [])


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_frame_sha256(frame: pd.DataFrame) -> str:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _numeric_values(series: pd.Series, *, field: str) -> list[float | None]:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & converted.isna()
    if bool(invalid.any()):
        raise SemanticMappingError("NON_NUMERIC_VALUE", "mapped signal contains non-numeric values", fields=[field])
    values: list[float | None] = []
    for value in converted:
        if pd.isna(value):
            values.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise SemanticMappingError("NON_FINITE_VALUE", "mapped signal contains non-finite values", fields=[field])
        values.append(numeric)
    return values


class ChannelMapper:
    def __init__(
        self,
        *,
        semantic_schema: SemanticSchema,
        channel_map: ChannelMap,
        unit_registry: UnitRegistry = DEFAULT_UNIT_REGISTRY,
    ) -> None:
        if channel_map.semantic_schema_id != semantic_schema.semantic_schema_id:
            raise ValueError("channel map semantic_schema_id does not match semantic schema")
        self.semantic_schema = semantic_schema
        self.channel_map = channel_map
        self.unit_registry = unit_registry
        self._quantity_kinds = {
            item.quantity_kind_id: item for item in semantic_schema.quantity_kinds
        }
        self.unit_registry.require_dimension(channel_map.time_axis.source_unit, TIME)
        for entry in channel_map.entries:
            try:
                quantity_kind = self._quantity_kinds[entry.quantity_kind_id]
            except KeyError as exc:
                raise ValueError(f"channel map references unknown quantity kind: {entry.quantity_kind_id}") from exc
            self.unit_registry.require_dimension(entry.source_unit, quantity_kind.dimension)
            self.unit_registry.require_dimension(quantity_kind.canonical_unit, quantity_kind.dimension)

    def map_frame(self, source_frame: pd.DataFrame) -> SemanticMappingResult:
        source = source_frame.copy()
        source.columns = [str(column) for column in source.columns]
        if len(source.columns) != len(set(source.columns)):
            raise SemanticMappingError("DUPLICATE_SOURCE_FIELD", "source fields must be unique")
        if source.empty:
            raise SemanticMappingError("EMPTY_SOURCE_FRAME", "source telemetry contains no rows")

        prohibited = sorted(field for field in source.columns if is_prohibited_field(field))
        if prohibited:
            raise SemanticMappingError(
                "PROHIBITED_SOURCE_FIELD",
                "source telemetry contains label, truth, event, object, or answer-proxy fields",
                fields=prohibited,
            )

        mapped_fields = {
            self.channel_map.time_axis.source_field,
            *(entry.source_field for entry in self.channel_map.entries),
        }
        missing_required = sorted(
            entry.source_field
            for entry in self.channel_map.entries
            if entry.required and entry.source_field not in source.columns
        )
        if self.channel_map.time_axis.source_field not in source.columns:
            missing_required.append(self.channel_map.time_axis.source_field)
        if missing_required:
            raise SemanticMappingError(
                "MISSING_REQUIRED_SOURCE_FIELD",
                "required source fields are absent",
                fields=sorted(set(missing_required)),
            )
        ignored_clean = sorted(set(source.columns) - mapped_fields)
        if ignored_clean and self.channel_map.unmapped_field_policy == UnmappedFieldPolicy.REJECT:
            raise SemanticMappingError(
                "UNMAPPED_SOURCE_FIELD",
                "source contains clean fields without a declared mapping",
                fields=ignored_clean,
            )

        raw_time = _numeric_values(
            source[self.channel_map.time_axis.source_field],
            field=self.channel_map.time_axis.source_field,
        )
        if any(value is None for value in raw_time):
            raise SemanticMappingError(
                "MISSING_TIME_VALUE",
                "time axis cannot contain missing values",
                fields=[self.channel_map.time_axis.source_field],
            )
        timestamps_s = [
            self.unit_registry.convert(value, source=self.channel_map.time_axis.source_unit, target="s")
            for value in raw_time
            if value is not None
        ]
        if any(right <= left for left, right in zip(timestamps_s, timestamps_s[1:])):
            raise SemanticMappingError(
                "NON_MONOTONIC_TIME",
                "normalized timestamps must be strictly increasing",
                fields=[self.channel_map.time_axis.source_field],
            )

        signals: list[SemanticSignal] = []
        for entry in sorted(self.channel_map.entries, key=lambda item: item.semantic_channel_id):
            if entry.source_field not in source.columns:
                continue
            quantity_kind = self._quantity_kinds[entry.quantity_kind_id]
            raw_values = _numeric_values(source[entry.source_field], field=entry.source_field)
            converted = self.unit_registry.convert_optional(
                raw_values,
                source=entry.source_unit,
                target=quantity_kind.canonical_unit,
            )
            directed = [None if value is None else float(entry.direction) * value for value in converted]
            signals.append(
                SemanticSignal(
                    semantic_channel_id=entry.semantic_channel_id,
                    asset_instance_id=entry.asset_instance_id,
                    component_archetype_id=entry.component_archetype_id,
                    port_role_id=entry.port_role_id,
                    quantity_kind_id=entry.quantity_kind_id,
                    canonical_unit=quantity_kind.canonical_unit,
                    direction=entry.direction,
                    values=directed,
                    missing_mask=[value is None for value in directed],
                )
            )
        if not signals:
            raise SemanticMappingError("NO_MAPPED_SIGNALS", "source contains no available mapped signals")

        intervals = [right - left for left, right in zip(timestamps_s, timestamps_s[1:])]
        nominal = statistics.median(intervals) if intervals else None
        minimum = min(intervals) if intervals else None
        maximum = max(intervals) if intervals else None
        tolerance = max((nominal or 0.0) * self.channel_map.sampling_tolerance_ratio, 1e-12)
        irregular = bool(nominal is not None and any(abs(value - nominal) > tolerance for value in intervals))
        missing_count = sum(sum(signal.missing_mask) for signal in signals)

        semantic_payload = {
            "semantic_schema_id": self.semantic_schema.semantic_schema_id,
            "timestamps_s": timestamps_s,
            "signals": [signal.model_dump(mode="json") for signal in signals],
        }
        semantic_hash = _json_sha256(semantic_payload)
        frame = SemanticTelemetryFrame(
            frame_id=f"STF-{semantic_hash[:20].upper()}",
            semantic_schema_id=self.semantic_schema.semantic_schema_id,
            timestamps_s=timestamps_s,
            signals=signals,
            sampling=SamplingProfile(
                row_count=len(source),
                nominal_period_s=nominal,
                minimum_period_s=minimum,
                maximum_period_s=maximum,
                irregular=irregular,
                missing_value_count=missing_count,
            ),
            semantic_data_sha256=semantic_hash,
        )
        receipt = SemanticMappingReceipt(
            mapping_id=self.channel_map.mapping_id,
            mapping_version=self.channel_map.version,
            source_adapter_id=self.channel_map.source_adapter_id,
            source_schema_sha256=self.channel_map.source_schema_sha256,
            source_data_sha256=_source_frame_sha256(source),
            semantic_data_sha256=semantic_hash,
            source_to_semantic={
                entry.source_field: entry.semantic_channel_id for entry in self.channel_map.entries
                if entry.source_field in source.columns
            },
            ignored_clean_fields=ignored_clean,
            prohibited_fields=[],
        )
        return SemanticMappingResult(frame=frame, receipt=receipt)


__all__ = ["ChannelMapper", "SemanticMappingError"]
