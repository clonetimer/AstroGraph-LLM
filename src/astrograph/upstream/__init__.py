"""Optional upstream recompute providers and zero-dependency manual exchange."""

from .contracts import (
    MANUAL_PROVIDER_ID,
    UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION,
    UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION,
    UpstreamProviderCapability,
    UpstreamRecomputeRequest,
    UpstreamRecomputeResponse,
    build_pipeline_runtime_recompute_request,
    provider_capabilities,
)
from .manual_provider import (
    ManualRecomputeAdmission,
    build_manual_request_bundle,
    build_manual_response_bundle,
    verify_manual_response_bundle,
)

__all__ = [
    "MANUAL_PROVIDER_ID",
    "UPSTREAM_RECOMPUTE_REQUEST_SCHEMA_VERSION",
    "UPSTREAM_RECOMPUTE_RESPONSE_SCHEMA_VERSION",
    "UpstreamProviderCapability",
    "UpstreamRecomputeRequest",
    "UpstreamRecomputeResponse",
    "build_pipeline_runtime_recompute_request",
    "provider_capabilities",
    "ManualRecomputeAdmission",
    "build_manual_request_bundle",
    "build_manual_response_bundle",
    "verify_manual_response_bundle",
    "UpstreamRecomputeProvider",
    "UpstreamProviderRegistry",
    "build_default_registry",
]

from .provider import UpstreamRecomputeProvider
from .registry import UpstreamProviderRegistry, build_default_registry
