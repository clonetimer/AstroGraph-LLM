from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from astrograph.agents.base import ControlledAgent
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY
from astrograph.plugin_factory import ObjectPluginBuildSpec

T = TypeVar("T", bound=BaseModel)


class ObjectPluginDesignBackend(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self,
        *,
        agent_id: str,
        output_model: type[T],
        payload: dict[str, Any],
    ) -> T: ...


class ObjectPluginDesignAgent(ControlledAgent):
    """Design-time Agent that may produce only a structured plugin build spec.

    The Agent is deliberately separated from runtime code execution.  It cannot
    register a plugin, execute generated Python, or bypass deterministic review.
    """

    agent_id = "object-plugin-design-agent"
    agent_version = "0.5.13"

    def __init__(
        self,
        *args: Any,
        backend: ObjectPluginDesignBackend | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend

    def design(
        self,
        *,
        object_context: dict[str, Any],
        draft_build_spec: dict[str, Any] | None = None,
    ) -> ObjectPluginBuildSpec:
        if self.backend is None:
            if draft_build_spec is None:
                raise ValueError("deterministic design mode requires draft_build_spec")
            return ObjectPluginBuildSpec.model_validate(draft_build_spec)

        payload = {
            "object_context": object_context,
            "approved_operator_registry": [
                item.model_dump(mode="json") for item in BUILTIN_OPERATOR_REGISTRY.list_specs()
            ],
            "output_protocol": ObjectPluginBuildSpec.model_json_schema(),
            "design_rules": [
                "Prefer an existing approved operator over generated code.",
                "Use operator_pipeline only when no direct operator represents the feature.",
                "Use custom_formula only for an object-specific physical mechanism that cannot be expressed by approved operators.",
                "Every formula, threshold, unit, and telemetry binding must include a traceable source reference.",
                "Do not emit executable Python, shell, Cypher, or arbitrary tool instructions.",
            ],
        }
        result = self.backend.generate_structured(
            agent_id=self.agent_id,
            output_model=ObjectPluginBuildSpec,
            payload=payload,
        )
        return ObjectPluginBuildSpec.model_validate(result)
