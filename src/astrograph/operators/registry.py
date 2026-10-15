from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from astrograph.operators.models import OperatorInput, OperatorSpec


OperatorHandler = Callable[[OperatorInput], float | None]


@dataclass(frozen=True)
class OperatorDefinition:
    spec: OperatorSpec
    handler: OperatorHandler


class DeterministicOperatorRegistry:
    """White-list registry for reusable numerical feature operators."""

    def __init__(self) -> None:
        self._definitions: dict[str, OperatorDefinition] = {}

    def register(self, definition: OperatorDefinition) -> None:
        operator_id = definition.spec.operator_id
        if operator_id in self._definitions:
            raise ValueError(f"operator already registered: {operator_id}")
        if not definition.spec.deterministic:
            raise ValueError("object feature operators must be deterministic")
        self._definitions[operator_id] = definition

    def get(self, operator_id: str) -> OperatorDefinition:
        try:
            return self._definitions[operator_id]
        except KeyError as exc:
            raise KeyError(f"unknown deterministic operator: {operator_id}") from exc

    def has(self, operator_id: str) -> bool:
        return operator_id in self._definitions

    def list_specs(self) -> list[OperatorSpec]:
        return [self._definitions[key].spec for key in sorted(self._definitions)]

    def compute(self, operator_id: str, inputs: OperatorInput) -> float | None:
        definition = self.get(operator_id)
        if definition.spec.requires_baseline and inputs.baseline is None:
            raise ValueError(f"operator {operator_id} requires baseline input")
        if definition.spec.requires_time and inputs.time_s is None:
            raise ValueError(f"operator {operator_id} requires time_s input")
        return definition.handler(inputs)
