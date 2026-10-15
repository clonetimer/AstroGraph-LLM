"""Policy-governed agentization layer for the AstroGraph scientific core.

This package exposes content-addressed capabilities without changing the
scientific algorithms or promoting development evidence beyond its claim scope.
"""
from .protocol import PROTOCOL_VERSION, CapabilityRequest, CapabilityResponse
from .profiles import AgentProfile

__all__ = ["PROTOCOL_VERSION", "CapabilityRequest", "CapabilityResponse", "AgentProfile"]
