from __future__ import annotations

import os
from collections import OrderedDict

from .contracts import (
    MANUAL_PROVIDER_ID,
    MATLAB_PROVIDER_ID,
    SAT_AGENT_PROVIDER_ID,
    UPSTREAM_PROVIDER_CAPABILITIES_SCHEMA_VERSION,
    UpstreamProviderCapability,
)
from .provider import OptionalProviderDescriptor, UpstreamRecomputeProvider


class ManualProviderDescriptor(UpstreamRecomputeProvider):
    provider_id = MANUAL_PROVIDER_ID

    def capability(self) -> UpstreamProviderCapability:
        return UpstreamProviderCapability(
            provider_id=MANUAL_PROVIDER_ID,
            display_name="Manual request/response exchange",
            status="MANUAL",
            execution_mode="MANUAL_EXCHANGE",
            configured=True,
            reason="Built into AstroGraph core; exports a request bundle and strictly admits a returned runtime/mapping bundle.",
            claim_boundary="Manual mode does not execute an upstream simulator. It only freezes request identity and verifies returned governed artifacts before child creation.",
        )


class UpstreamProviderRegistry:
    def __init__(self) -> None:
        self._providers: OrderedDict[str, UpstreamRecomputeProvider] = OrderedDict()

    def register(self, provider: UpstreamRecomputeProvider) -> None:
        if provider.provider_id in self._providers:
            raise ValueError(f"duplicate upstream provider: {provider.provider_id}")
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> UpstreamRecomputeProvider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"unknown upstream provider: {provider_id}") from exc

    def capabilities(self) -> dict:
        return {
            "schema_version": UPSTREAM_PROVIDER_CAPABILITIES_SCHEMA_VERSION,
            "default_provider_id": MANUAL_PROVIDER_ID,
            "automatic_provider_required": False,
            "providers": [provider.capability().model_dump(mode="json") for provider in self._providers.values()],
            "claim_boundary": "AstroGraph diagnosis and existing maintenance workflows remain available when every optional automatic upstream provider is unavailable.",
        }


def build_default_registry() -> UpstreamProviderRegistry:
    registry = UpstreamProviderRegistry()
    registry.register(ManualProviderDescriptor())

    sat_url = (os.environ.get("ASTROGRAPH_SAT_AGENT_RECOMPUTE_URL") or "").strip()
    registry.register(OptionalProviderDescriptor(
        provider_id=SAT_AGENT_PROVIDER_ID,
        display_name="Satellite Agent Platform",
        execution_mode="REMOTE_OPTIONAL" if sat_url else "LOCAL_OPTIONAL",
        configured=bool(sat_url),
        reason=(
            "Optional provider endpoint is configured, but the automatic connector is not part of AstroGraph core R13. Use manual exchange until that connector is installed."
            if sat_url else
            "Optional provider not configured; AstroGraph core remains fully usable and manual exchange is available."
        ),
        claim_boundary="sat_agent_platform is optional and must never become an import-time or runtime dependency of AstroGraph core.",
    ))

    matlab_url = (os.environ.get("ASTROGRAPH_MATLAB_RECOMPUTE_URL") or "").strip()
    registry.register(OptionalProviderDescriptor(
        provider_id=MATLAB_PROVIDER_ID,
        display_name="MATLAB / Simulink",
        execution_mode="REMOTE_OPTIONAL" if matlab_url else "LOCAL_OPTIONAL",
        configured=bool(matlab_url),
        reason=(
            "Optional MATLAB/Simulink endpoint is configured, but the automatic connector is not part of AstroGraph core R13. Use manual exchange until that connector is installed."
            if matlab_url else
            "MATLAB/Simulink is optional and not configured; no MATLAB installation is required for AstroGraph core."
        ),
        claim_boundary="MATLAB/Simulink remains an external recompute backend, not a Python dependency of AstroGraph core.",
    ))
    return registry
