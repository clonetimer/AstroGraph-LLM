from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ExtensionCapability(str, Enum):
    SEMANTIC_AUDIT = "SEMANTIC_AUDIT"
    CONTROL_PLANE_AUDIT = "CONTROL_PLANE_AUDIT"
    EVIDENCE_CALIBRATION = "EVIDENCE_CALIBRATION"
    MECHANISM_EVIDENCE = "MECHANISM_EVIDENCE"
    ACTIVE_DISCOVERY = "ACTIVE_DISCOVERY"
    MODEL_CHALLENGER = "MODEL_CHALLENGER"


class ExtensionDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-extension-descriptor.v1"
    extension_id: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    capabilities: tuple[ExtensionCapability, ...] = ()
    online_allowed: bool = False
    fail_closed: bool = True
    claim_boundary: str = Field(min_length=1)


@runtime_checkable
class ExtensionProvider(Protocol):
    descriptor: ExtensionDescriptor

    def invoke(self, capability: ExtensionCapability, payload: dict[str, Any]) -> dict[str, Any]:
        ...
