"""Content-addressed physical semantics for tensor channels.

This contract complements the ordered-feature contract: position alone is not
sufficient.  Each model-active channel binds unit, frame, axis/sign convention,
time semantics, missingness/imputation, normalization and preprocessing source.
Unknown semantics remain explicit and prevent a contract from being frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED = "UNRESOLVED"


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


class ResamplingMethod(str, Enum):
    NONE = "NONE"
    EXACT = "EXACT"
    HOLD_LAST = "HOLD_LAST"
    LINEAR = "LINEAR"
    AGGREGATE = "AGGREGATE"
    UNRESOLVED = "UNRESOLVED"


class MissingnessPolicy(str, Enum):
    REJECT = "REJECT"
    MASK = "MASK"
    IMPUTE = "IMPUTE"
    UNRESOLVED = "UNRESOLVED"


class NormalizationKind(str, Enum):
    NONE = "NONE"
    ZSCORE = "ZSCORE"
    MINMAX = "MINMAX"
    AFFINE = "AFFINE"
    UNRESOLVED = "UNRESOLVED"


class NormalizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: NormalizationKind
    parameters: dict[str, float] = Field(default_factory=dict)
    fitted_on_groups: list[str] = Field(default_factory=list)
    receipt_sha256: str | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "NormalizationSpec":
        if self.receipt_sha256 is not None and not _SHA.fullmatch(self.receipt_sha256):
            raise ValueError("normalization receipt_sha256 must be lowercase SHA-256")
        if any(not math.isfinite(float(v)) for v in self.parameters.values()):
            raise ValueError("normalization parameters must be finite")
        if self.kind not in {NormalizationKind.NONE, NormalizationKind.UNRESOLVED} and self.receipt_sha256 is None:
            raise ValueError("fitted normalization requires a receipt_sha256")
        return self


class ChannelSemanticSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tensor_index: int = Field(ge=0)
    channel_name: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    physical_unit: str = Field(min_length=1)
    coordinate_frame: str = Field(min_length=1)
    axis_definition: str = Field(min_length=1)
    sign_convention: str = Field(min_length=1)
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    timestamp_semantics: str = Field(min_length=1)
    resampling_method: ResamplingMethod
    missingness_policy: MissingnessPolicy
    imputation_policy: str = Field(min_length=1)
    normalization: NormalizationSpec
    preprocessing_sha256: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("preprocessing_sha256")
    @classmethod
    def valid_sha(cls, value: str) -> str:
        if not _SHA.fullmatch(value):
            raise ValueError("preprocessing_sha256 must be lowercase SHA-256")
        return value

    def unresolved_fields(self) -> list[str]:
        fields = []
        for name in ("physical_unit", "coordinate_frame", "axis_definition", "sign_convention", "timestamp_semantics", "imputation_policy"):
            if getattr(self, name).strip().upper() == UNRESOLVED:
                fields.append(name)
        if self.resampling_method is ResamplingMethod.UNRESOLVED:
            fields.append("resampling_method")
        if self.missingness_policy is MissingnessPolicy.UNRESOLVED:
            fields.append("missingness_policy")
        if self.normalization.kind is NormalizationKind.UNRESOLVED:
            fields.append("normalization")
        if self.sampling_rate_hz is None:
            fields.append("sampling_rate_hz")
        return fields


class FeatureSemanticContractV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-feature-semantic-contract.v3"
    contract_id: str = Field(min_length=1)
    model_package_sha256: str | None = None
    channels: list[ChannelSemanticSpec] = Field(min_length=1)
    status: str = Field(pattern=r"^(DRAFT|AUDITED|FROZEN)$")
    source_manifest_sha256s: list[str] = Field(default_factory=list)
    claim_boundary: str = Field(min_length=1)
    contract_sha256: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "FeatureSemanticContractV3":
        if self.model_package_sha256 is not None and not _SHA.fullmatch(self.model_package_sha256):
            raise ValueError("model_package_sha256 must be lowercase SHA-256")
        for digest in self.source_manifest_sha256s:
            if not _SHA.fullmatch(digest):
                raise ValueError("source manifest hashes must be lowercase SHA-256")
        indices = [c.tensor_index for c in self.channels]
        names = [c.channel_name for c in self.channels]
        if indices != list(range(len(indices))):
            raise ValueError("tensor indices must be contiguous and ordered from zero")
        if len(names) != len(set(names)):
            raise ValueError("channel_name values must be unique")
        if self.status == "FROZEN":
            if self.model_package_sha256 is None:
                raise ValueError("frozen contract requires authoritative model package hash")
            unresolved = self.unresolved_semantics()
            if unresolved:
                raise ValueError(f"frozen contract contains unresolved semantics: {unresolved}")
        expected = self.compute_sha256()
        if self.contract_sha256 is not None and self.contract_sha256 != expected:
            raise ValueError("contract_sha256 mismatch")
        return self

    def payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude={"contract_sha256"})
        return data

    def compute_sha256(self) -> str:
        return _sha(self.payload())

    def unresolved_semantics(self) -> dict[str, list[str]]:
        return {c.channel_name: c.unresolved_fields() for c in self.channels if c.unresolved_fields()}


__all__ = [
    "ChannelSemanticSpec", "FeatureSemanticContractV3", "MissingnessPolicy",
    "NormalizationKind", "NormalizationSpec", "ResamplingMethod", "UNRESOLVED",
]
