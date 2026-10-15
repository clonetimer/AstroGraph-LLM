"""Callable stability and Step-6 capabilities for Cordis/Harness adapters."""
from __future__ import annotations

from typing import Any, Mapping

from .lineage_resolution import Step6AssessmentInput, assess_step6_route
from .stable_release import StabilityInput, assess_stability

CAPABILITY_SCHEMAS = {
    "lineage.step6.assess": Step6AssessmentInput.model_json_schema(),
    "release.stability.assess": StabilityInput.model_json_schema(),
}


def invoke_stability_capability(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if name == "lineage.step6.assess":
        return assess_step6_route(Step6AssessmentInput.model_validate(payload)).model_dump(mode="json")
    if name == "release.stability.assess":
        return assess_stability(StabilityInput.model_validate(payload)).model_dump(mode="json")
    raise KeyError(f"unknown stability capability: {name}")
