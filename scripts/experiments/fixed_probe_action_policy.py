"""Exploratory fixed-action comparator for the LLM active-observation study.

Only diagnostic action selection is deterministic. The same backend still handles
final hypothesis generation, and the host continues to enforce probe admission.
"""

from __future__ import annotations

from typing import Any

from astrograph.orchestration.llm_diagnostic_loop import DiagnosticActionProposal


class FixedProbeActionBackend:
    policy_id = "first-admitted-fusion-ordered-probe.v1"

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.fixed_action_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def generate_structured(
        self, *, agent_id: str, output_model: type, payload: dict[str, Any],
    ) -> Any:
        if not issubclass(output_model, DiagnosticActionProposal):
            return self.delegate.generate_structured(
                agent_id=agent_id, output_model=output_model, payload=payload,
            )
        available = {item["capability_id"] for item in payload["allowed_capabilities"]}
        executed = set(payload.get("already_executed_capability_ids", []))
        raw_probes = [
            item for item in payload.get("probe_index", [])
            if item.get("creates_new_raw_observation") is True
        ]
        probe_id: str | None = None
        if not executed and "inspect_candidate_fusion" in available:
            capability = "inspect_candidate_fusion"
        elif (
            "execute_diagnostic_probe" in available
            and "execute_diagnostic_probe" not in executed and raw_probes
        ):
            capability = "execute_diagnostic_probe"
            probe_id = str(raw_probes[0]["probe_id"])
        elif "inspect_deterministic_core" in available:
            capability = "inspect_deterministic_core"
        elif "synthesize_hypotheses" in available:
            capability = "synthesize_hypotheses"
        else:
            eligible = sorted(item for item in available if item != "execute_diagnostic_probe")
            if not eligible:
                raise ValueError("fixed action policy has no admissible next action")
            capability = eligible[0]
        observation_ids = [
            item["observation_id"] for item in payload.get("observation_index", [])
        ]
        self.fixed_action_count += 1
        return output_model.model_validate({
            "capability_id": capability,
            "objective": (
                "Compare the first host-admitted, fusion-ordered telemetry probe with the current candidates."
                if probe_id else "Apply the preregistered fixed investigation sequence."
            ),
            "cited_observation_ids": observation_ids[-1:],
            "probe_id": probe_id,
        })
