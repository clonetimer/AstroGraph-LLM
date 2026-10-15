from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .contracts import UpstreamProviderCapability


class UpstreamRecomputeProvider(ABC):
    """Narrow provider abstraction for optional upstream recompute backends.

    Core diagnosis never imports provider-specific packages. Automatic providers may
    live in optional connectors; the built-in manual provider exchanges immutable
    request/response bundles instead of executing a simulator.
    """

    provider_id: str

    @abstractmethod
    def capability(self) -> UpstreamProviderCapability:
        raise NotImplementedError

    def automatic_execution_available(self) -> bool:
        return self.capability().status == "AVAILABLE"

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"upstream provider {self.provider_id} does not implement automatic execution")


@dataclass(frozen=True)
class OptionalProviderDescriptor(UpstreamRecomputeProvider):
    provider_id: str
    display_name: str
    execution_mode: str
    configured: bool
    reason: str
    claim_boundary: str

    def capability(self) -> UpstreamProviderCapability:
        return UpstreamProviderCapability(
            provider_id=self.provider_id,
            display_name=self.display_name,
            status="UNAVAILABLE",
            execution_mode=self.execution_mode,  # type: ignore[arg-type]
            configured=self.configured,
            reason=self.reason,
            claim_boundary=self.claim_boundary,
        )
