"""Runtime contracts for the generic feature-operator runtime.

These are truth-blind: no labels, no truth, no case/scenario/seed/file identity
may enter an operator input or an emitted token. The token fields mirror the
frozen `astrograph.truth-blind-evidence-token.v1` contract exactly.
"""

from __future__ import annotations

import math
import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Identity/truth tokens that must never appear in operator input or output.
_FORBIDDEN_TOKENS = (
    "case", "class", "dataset", "effect", "event", "fault", "file",
    "health", "identity", "label", "mode", "modifier", "nominal", "object",
    "onset", "path", "provenance", "run_id", "scenario", "seed", "severity",
    "split", "status", "target", "truth", "anomaly",
)


def _assert_truth_blind(*fields: str) -> None:
    """Reject identity/truth tokens on whole-word boundaries.

    Word-boundary matching (not raw substring) avoids false positives like
    "profile" tripping on the token "file" while still catching a real
    identity field such as "file_identity" or "label.status".
    """
    pattern = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(t) for t in _FORBIDDEN_TOKENS), re.IGNORECASE)
    for field in fields:
        if pattern.search(field):
            raise ValueError(f"forbidden truth/identity token in field: {field}")


class FeatureInput(BaseModel):
    """One semantic signal fed to an operator."""

    model_config = ConfigDict(extra="forbid")

    semantic_channel_id: str
    quantity_kind_id: str
    port_role_id: str
    direction: Literal[-1, 1] = 1
    timestamps_s: list[float] = Field(min_length=1)
    values: list[float | None] = Field(min_length=1)
    missing_mask: list[bool] = Field(min_length=1)

    @model_validator(mode="after")
    def aligned(self) -> "FeatureInput":
        n = len(self.timestamps_s)
        if len(self.values) != n or len(self.missing_mask) != n:
            raise ValueError("timestamps, values and missing_mask must be aligned")
        _assert_truth_blind(self.semantic_channel_id, self.quantity_kind_id, self.port_role_id)
        return self


class OperatorResult(BaseModel):
    """Raw output of one feature operator (pre-token)."""

    model_config = ConfigDict(extra="forbid")

    operator_id: str
    value: float
    uncertainty: float = 0.0
    availability: float = Field(ge=0.0, le=1.0)
    window_descriptor: str = ""
    quality_flags: list[str] = Field(default_factory=list)


class FeatureOperator(Protocol):
    """Uniform interface every generic operator implements."""

    operator_id: str

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        """Produce one or more OperatorResults from the given inputs."""
        ...


class EvidenceToken(BaseModel):
    """Unified token emitted by all operators — mirrors the frozen contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-token.v1"
    component_archetype_id: str
    functional_role_id: str = ""
    quantity_kind_id: str
    port_role_id: str
    operator_id: str
    value: float
    uncertainty: float = 0.0
    availability: float = Field(ge=0.0, le=1.0)
    window_descriptor: str = ""
    quality_flags: list[str] = Field(default_factory=list)
    provenance_hash: str = ""

    # Guards — never permitted in a token.
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False
    identity_fields_allowed: Literal[False] = False

    @model_validator(mode="after")
    def truth_blind(self) -> "EvidenceToken":
        _assert_truth_blind(
            self.component_archetype_id, self.functional_role_id,
            self.quantity_kind_id, self.port_role_id, self.operator_id,
            self.window_descriptor,
        )
        return self


class TokenResult(BaseModel):
    """What token_builder returns: tokens plus a truth-blind receipt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-token-result.v1"
    tokens: list[EvidenceToken]
    token_count: int
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False


def nan_safe(value: float, *, default: float = 0.0) -> float:
    """Return default when value is NaN or infinite (zero-fit, numeric safety)."""
    return default if not math.isfinite(value) else value
