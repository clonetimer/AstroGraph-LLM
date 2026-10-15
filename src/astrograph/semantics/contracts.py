from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .leakage import is_prohibited_field
from .units import DimensionVector


class ValueSemantics(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"
    BINARY = "BINARY"


class UnmappedFieldPolicy(str, Enum):
    REJECT = "REJECT"
    IGNORE_CLEAN = "IGNORE_CLEAN"


class QuantityKind(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quantity_kind_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
    dimension: DimensionVector
    canonical_unit: str = Field(min_length=1)
    value_semantics: ValueSemantics = ValueSemantics.CONTINUOUS
    description: str = ""


class SemanticSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-semantic-schema.v1"
    semantic_schema_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    quantity_kinds: list[QuantityKind] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_quantity_kinds(self) -> "SemanticSchema":
        ids = [item.quantity_kind_id for item in self.quantity_kinds]
        if len(ids) != len(set(ids)):
            raise ValueError("quantity_kind_id values must be unique")
        return self

    def quantity_kind(self, quantity_kind_id: str) -> QuantityKind:
        for item in self.quantity_kinds:
            if item.quantity_kind_id == quantity_kind_id:
                return item
        raise KeyError(f"unknown quantity kind: {quantity_kind_id}")


class TimeAxisMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_field: str = Field(min_length=1)
    source_unit: str = "s"

    @field_validator("source_field")
    @classmethod
    def public_time_field(cls, value: str) -> str:
        if is_prohibited_field(value):
            raise ValueError("time source field is prohibited")
        return value


class ChannelMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_field: str = Field(min_length=1)
    semantic_channel_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
    asset_instance_id: str = Field(min_length=1)
    component_archetype_id: str = Field(min_length=1)
    port_role_id: str | None = None
    quantity_kind_id: str = Field(min_length=1)
    source_unit: str = Field(min_length=1)
    direction: Literal[-1, 1] = 1
    required: bool = True

    @field_validator("source_field")
    @classmethod
    def public_source_field(cls, value: str) -> str:
        if is_prohibited_field(value):
            raise ValueError("channel source field is prohibited")
        return value


class ChannelMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-channel-map.v1"
    mapping_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_adapter_id: str = Field(min_length=1)
    source_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_schema_id: str = Field(min_length=1)
    time_axis: TimeAxisMap
    entries: list[ChannelMapEntry] = Field(min_length=1)
    unmapped_field_policy: UnmappedFieldPolicy = UnmappedFieldPolicy.REJECT
    sampling_tolerance_ratio: float = Field(default=1e-6, ge=0.0, le=0.1)

    @model_validator(mode="after")
    def unique_mapping_keys(self) -> "ChannelMap":
        source_fields = [self.time_axis.source_field, *(item.source_field for item in self.entries)]
        semantic_ids = [item.semantic_channel_id for item in self.entries]
        if len(source_fields) != len(set(source_fields)):
            raise ValueError("source fields must be unique across time and signal mappings")
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("semantic_channel_id values must be unique")
        return self


class ChannelMapCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-channel-map-catalog.v1"
    catalog_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    maps: list[ChannelMap] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_maps(self) -> "ChannelMapCatalog":
        ids = [item.mapping_id for item in self.maps]
        if len(ids) != len(set(ids)):
            raise ValueError("channel map IDs must be unique")
        return self

    def channel_map(self, mapping_id: str) -> ChannelMap:
        for item in self.maps:
            if item.mapping_id == mapping_id:
                return item
        raise KeyError(f"unknown channel map: {mapping_id}")


class SamplingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_count: int = Field(ge=1)
    nominal_period_s: float | None = Field(default=None, gt=0.0)
    minimum_period_s: float | None = Field(default=None, gt=0.0)
    maximum_period_s: float | None = Field(default=None, gt=0.0)
    irregular: bool
    missing_value_count: int = Field(ge=0)


class SemanticSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_channel_id: str
    asset_instance_id: str
    component_archetype_id: str
    port_role_id: str | None = None
    quantity_kind_id: str
    canonical_unit: str
    direction: Literal[-1, 1]
    values: list[float | None]
    missing_mask: list[bool]

    @model_validator(mode="after")
    def valid_values(self) -> "SemanticSignal":
        if len(self.values) != len(self.missing_mask):
            raise ValueError("values and missing_mask lengths differ")
        for value, missing in zip(self.values, self.missing_mask, strict=True):
            if missing != (value is None):
                raise ValueError("missing_mask must exactly identify None values")
            if value is not None and not math.isfinite(value):
                raise ValueError("semantic signal values must be finite or None")
        return self


class SemanticTelemetryFrame(BaseModel):
    """Dataset-neutral model input. Raw source field names live only in the receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-semantic-telemetry-frame.v1"
    frame_id: str = Field(min_length=1)
    semantic_schema_id: str = Field(min_length=1)
    timestamps_s: list[float] = Field(min_length=1)
    signals: list[SemanticSignal] = Field(min_length=1)
    sampling: SamplingProfile
    semantic_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False
    claim_boundary: str = (
        "Contains normalized public observations only; it does not contain source field names, labels, truth, "
        "dataset identity, or diagnostic conclusions."
    )

    @model_validator(mode="after")
    def valid_frame(self) -> "SemanticTelemetryFrame":
        if len(self.timestamps_s) != self.sampling.row_count:
            raise ValueError("timestamp count must equal sampling row_count")
        if any(not math.isfinite(value) for value in self.timestamps_s):
            raise ValueError("timestamps must be finite")
        if any(right <= left for left, right in zip(self.timestamps_s, self.timestamps_s[1:])):
            raise ValueError("timestamps must be strictly increasing")
        ids = [signal.semantic_channel_id for signal in self.signals]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic signal IDs must be unique")
        if any(len(signal.values) != self.sampling.row_count for signal in self.signals):
            raise ValueError("all signals must match sampling row_count")
        return self


class SemanticMappingReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-semantic-mapping-receipt.v1"
    mapping_id: str
    mapping_version: str
    source_adapter_id: str
    source_schema_sha256: str
    source_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_to_semantic: dict[str, str]
    ignored_clean_fields: list[str] = Field(default_factory=list)
    prohibited_fields: list[str] = Field(default_factory=list)
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False


class SemanticMappingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: SemanticTelemetryFrame
    receipt: SemanticMappingReceipt


__all__ = [
    "ChannelMap",
    "ChannelMapCatalog",
    "ChannelMapEntry",
    "QuantityKind",
    "SamplingProfile",
    "SemanticMappingReceipt",
    "SemanticMappingResult",
    "SemanticSchema",
    "SemanticSignal",
    "SemanticTelemetryFrame",
    "TimeAxisMap",
    "UnmappedFieldPolicy",
    "ValueSemantics",
]
