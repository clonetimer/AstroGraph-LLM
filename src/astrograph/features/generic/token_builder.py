"""Token builder — turn semantic signals into unified EvidenceTokens.

Given a SemanticTelemetryFrame and an archetype feature contract (primitives +
their operator families and input port roles), run the registered operators and
assemble the frozen EvidenceToken structure. This is the boundary that
guarantees: variable token count, truth-blind inputs, no identity fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import (
    EvidenceToken,
    FeatureInput,
    OperatorResult,
    TokenResult,
)
from .registry import FeatureOperatorRegistry, default_registry


def _frame_to_inputs(frame, port_role_id: str) -> list[FeatureInput]:
    """Build FeatureInput list for a port role from a SemanticTelemetryFrame."""
    inputs: list[FeatureInput] = []
    for signal in frame.signals:
        if signal.port_role_id != port_role_id:
            continue
        inputs.append(
            FeatureInput(
                semantic_channel_id=signal.semantic_channel_id,
                quantity_kind_id=signal.quantity_kind_id,
                port_role_id=signal.port_role_id,
                direction=signal.direction,
                timestamps_s=list(frame.timestamps_s),
                values=signal.values,
                missing_mask=signal.missing_mask,
            )
        )
    return inputs


def _provenance_hash(component: str, operator: str, window: str) -> str:
    payload = f"{component}|{operator}|{window}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


class TokenBuilder:
    """Runs operator primitives against a frame and emits EvidenceTokens."""

    def __init__(
        self,
        *,
        registry: FeatureOperatorRegistry | None = None,
        archetype: str | None = None,
        functional_role_id: str = "",
        contract_path: str | Path | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.archetype = archetype
        self.functional_role_id = functional_role_id
        self._primitives: list[tuple[str, str, list[str]]] | None = None
        if contract_path is not None:
            self.load_contract(contract_path)

    def load_contract(self, path: str | Path) -> None:
        """Load the archetype contract so primitives are known."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        contracts = data.get("archetype_contracts", [])
        if self.archetype is not None:
            contracts = [c for c in contracts if c["component_archetype_id"] == self.archetype]
        if not contracts:
            raise ValueError(f"no contract for archetype: {self.archetype}")
        contract = contracts[0]
        self._primitives = [
            (
                prim["primitive_id"],
                prim["operator_family"],
                prim.get("input_port_role_ids", []),
            )
            for prim in contract.get("primitives", [])
        ]

    def build(self, frame, *, component_archetype_id: str | None = None) -> TokenResult:
        """Run all primitives against a SemanticTelemetryFrame → TokenResult."""
        archetype = component_archetype_id or self.archetype
        if archetype is None:
            raise ValueError("archetype must be provided")
        if self._primitives is None:
            raise ValueError("contract not loaded; call load_contract or pass contract_path")

        tokens: list[EvidenceToken] = []
        for primitive_id, family, port_roles in self._primitives:
            operator = self.registry.get(family)

            # Every declared port role must contribute at least one input. A
            # missing role must never be silently filled by a signal from a
            # different port: positional substitution would feed the operator
            # fake inputs (e.g. two response signals standing in for an absent
            # command signal, which produced a fabricated -1.0 gain for MTB).
            # Fail closed with a masked token instead.
            inputs: list[FeatureInput] = []
            missing_roles: list[str] = []
            for role in port_roles:
                role_inputs = _frame_to_inputs(frame, role)
                if not role_inputs:
                    missing_roles.append(role)
                inputs.extend(role_inputs)

            if missing_roles:
                tokens.append(
                    EvidenceToken(
                        component_archetype_id=archetype,
                        functional_role_id=self.functional_role_id,
                        quantity_kind_id="",
                        port_role_id=",".join(port_roles),
                        operator_id=primitive_id,
                        value=0.0,
                        uncertainty=0.0,
                        availability=0.0,
                        window_descriptor="no-input",
                        quality_flags=["missing_required_port"],
                    )
                )
                continue

            results: list[OperatorResult]
            try:
                results = operator.compute(inputs)
            except ValueError:
                results = [
                    OperatorResult(
                        operator_id=primitive_id, value=0.0, uncertainty=0.0,
                        availability=0.0, window_descriptor="error",
                        quality_flags=["operator_error"],
                    )
                ]

            for res in results:
                qk = inputs[0].quantity_kind_id if inputs else ""
                tokens.append(
                    EvidenceToken(
                        component_archetype_id=archetype,
                        functional_role_id=self.functional_role_id,
                        quantity_kind_id=qk,
                        port_role_id=",".join(port_roles),
                        operator_id=res.operator_id or primitive_id,
                        value=res.value,
                        uncertainty=res.uncertainty,
                        availability=res.availability,
                        window_descriptor=res.window_descriptor,
                        quality_flags=res.quality_flags,
                        provenance_hash=_provenance_hash(
                            archetype, primitive_id, res.window_descriptor
                        ),
                    )
                )

        return TokenResult(tokens=tokens, token_count=len(tokens))


__all__ = ["TokenBuilder"]
