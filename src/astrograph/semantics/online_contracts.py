"""Truth-free online observations referencing offline DOR semantic releases.

These are AstroGraph application contracts. They are deliberately NOT fabricated
DOR DatasetManifests: inference has no train/calibration labels or forced reference.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ReleaseRef(FrozenContract):
    release_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChannelBinding(FrozenContract):
    channel_id: str = Field(min_length=1)
    semantic_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)

    @field_validator("channel_id", "semantic_id")
    @classmethod
    def no_supervision_channels(cls, value: str) -> str:
        tokens = set(re.split(r"[^a-z0-9]+", value.lower()))
        if tokens & {"truth", "label", "labels", "injection", "injected", "diagnosticclass"}:
            raise ValueError("supervision/injection channels are not online observations")
        if value.lower() in {"fault_id", "fault_class", "fault_type", "is_fault", "diagnostic_class"}:
            raise ValueError("supervision field is forbidden")
        return value


class ObservationFrame(FrozenContract):
    schema_version: Literal["astrograph-observation-frame.v1"] = "astrograph-observation-frame.v1"
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    channels: tuple[ChannelBinding, ...] = Field(min_length=1)
    observed_at: tuple[datetime, ...] = Field(min_length=1)
    available_at: tuple[datetime, ...] = Field(min_length=1)
    values: tuple[tuple[float | None, ...], ...] = Field(min_length=1)
    observed_mask: tuple[tuple[bool, ...], ...] = Field(min_length=1)
    # This hash is verified against the canonical content; it is not a signature.
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        channels = len(self.channels)
        if len({item.channel_id for item in self.channels}) != channels:
            raise ValueError("duplicate channel IDs")
        if len({item.semantic_id for item in self.channels}) != channels:
            raise ValueError("duplicate semantic IDs")
        count = len(self.observed_at)
        if any(len(value) != count for value in (self.available_at, self.values, self.observed_mask)):
            raise ValueError("time, values and mask lengths differ")
        for timestamp in (*self.observed_at, *self.available_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timezone-aware observation times are required")
        if any(a >= b for a, b in zip(self.observed_at, self.observed_at[1:])):
            raise ValueError("observation times must be strictly increasing")
        if any(a > b for a, b in zip(self.observed_at, self.available_at)):
            raise ValueError("an observation cannot be available before it was measured")
        for row, mask in zip(self.values, self.observed_mask):
            if len(row) != channels or len(mask) != channels:
                raise ValueError("channel/matrix shape mismatch")
            for value, present in zip(row, mask):
                if present and (value is None or not math.isfinite(value)):
                    raise ValueError("observed values must be finite")
                if not present and value is not None:
                    raise ValueError("missing values must be null, not imputed silently")
        if self.compute_hash() != self.payload_sha256:
            raise ValueError("observation payload checksum mismatch")
        return self

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"payload_sha256"})
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @classmethod
    def create(cls, **payload) -> "ObservationFrame":
        """Normalize JSON representation then validate both content and checksum."""
        # A private constructor is only used to compute the checksum; full validation follows.
        data = dict(payload)
        data.setdefault("schema_version", "astrograph-observation-frame.v1")
        data["channels"] = tuple(ChannelBinding.model_validate(x) for x in data["channels"])
        for key in ("observed_at", "available_at"):
            data[key] = tuple(datetime.fromisoformat(x.replace("Z", "+00:00")) if isinstance(x, str) else x
                              for x in data[key])
        data["values"] = tuple(tuple(None if x is None else float(x) for x in row) for row in data["values"])
        data["observed_mask"] = tuple(tuple(row) for row in data["observed_mask"])
        provisional = cls.model_construct(**data, payload_sha256="0" * 64)
        data["payload_sha256"] = provisional.compute_hash()
        return cls.model_validate(data)


class DiagnosticRequest(FrozenContract):
    schema_version: Literal["astrograph-online-request.v1"] = "astrograph-online-request.v1"
    request_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    operating_mode: str = Field(min_length=1)
    cutoff_at: datetime
    observation: ObservationFrame
    reference: ObservationFrame | None = None
    reference_applicability: Literal["ABSENT", "UNVERIFIED", "QUALIFIED"] = "ABSENT"
    reference_qualification: ReleaseRef | None = None
    semantic_release: ReleaseRef
    model_release: ReleaseRef
    knowledge_release: ReleaseRef
    plugin_release: ReleaseRef
    memory_policy_release: ReleaseRef | None = None

    @model_validator(mode="after")
    def check_online_visibility(self) -> Self:
        if self.cutoff_at.tzinfo is None or self.cutoff_at.utcoffset() is None:
            raise ValueError("cutoff needs timezone")
        for frame in (self.observation, self.reference):
            if frame and any(value > self.cutoff_at for value in (*frame.observed_at, *frame.available_at)):
                raise ValueError("future or not-yet-available observations are forbidden")
        if self.reference is None and self.reference_applicability != "ABSENT":
            raise ValueError("reference absent but applicability claimed")
        if self.reference is not None and self.reference_applicability == "ABSENT":
            raise ValueError("reference presence/applicability mismatch")
        if self.reference_applicability == "QUALIFIED" and self.reference_qualification is None:
            raise ValueError("qualified reference needs an independently issued qualification reference")
        if self.reference is None and self.reference_qualification is not None:
            raise ValueError("qualification without reference")
        return self

    def require_channels(self, required: tuple[ChannelBinding, ...], *, reference=False) -> tuple[tuple[float, ...], ...]:
        """Exact semantic/unit/order match; no reorder, scaling or fill is implicit."""
        frame = self.reference if reference else self.observation
        if frame is None:
            raise ValueError("reference unavailable")
        if frame.channels != required:
            raise ValueError("channel order, unit or semantic binding mismatch")
        if not all(all(row) for row in frame.observed_mask):
            raise ValueError("this tool requires complete observations")
        return tuple(tuple(float(value) for value in row) for row in frame.values)

    def dor_context(self) -> dict:
        """Online sidecar; external DOR qualification remains an offline step."""
        return {"contract": "astrograph-dor-online-binding.v1",
                "request_id": self.request_id,
                "semantic_release": self.semantic_release.model_dump(mode="json"),
                "source_sha256": self.observation.payload_sha256,
                "reference_applicability": self.reference_applicability,
                "supervision_available_to_runtime": False}
