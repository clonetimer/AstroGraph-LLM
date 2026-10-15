from __future__ import annotations

import hashlib
import json


def _stable_id(prefix: str, payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20].upper()}"
from astrograph.investigation.models import (
    CritiqueResult,
    DecisionDisposition,
    DecisionRecord,
    HypothesisSet,
    HypothesisStatus,
    HypothesisType,
)


_CASE_TO_FAULT = {
    "rw_friction_increase": "ADCS_RW_FRICTION_INCREASE",
    "rw_torque_authority_loss": "ADCS_RW_TORQUE_AUTHORITY_LOSS",
    "thermal_heater_stuck_on": "THERMAL_HEATER_STUCK_ON",
    "star_tracker_dropout": "ADCS_STAR_TRACKER_DROPOUT",
    "reaction_wheel_jam": "ADCS_RW_JAM",
    "battery_internal_resistance_increase": "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE",
    "solar_array_string_failure": "EPS_SOLAR_ARRAY_STRING_FAILURE",
    "gyro_bias_drift": "ADCS_GYRO_BIAS_DRIFT",
    "transmitter_output_power_loss": "COMM_TRANSMITTER_OUTPUT_POWER_LOSS",
    "storage_write_failure": "DATA_STORAGE_WRITE_FAILURE",
    "battery_capacity_loss": "EPS_BATTERY_CAPACITY_LOSS",
    "solar_efficiency_loss": "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS",
    "radiator_rejection_loss": "THERMAL_RADIATOR_REJECTION_LOSS",
    "downlink_service_loss": "COMM_DOWNLINK_SERVICE_LOSS",
    "temperature_sensor_bias": "THERMAL_TEMPERATURE_SENSOR_BIAS",
    "temperature_sensor_drift": "THERMAL_TEMPERATURE_SENSOR_DRIFT",
}


class DiagnosticDecisionGovernor:
    governor_id = "decision-governor-v1"
    role = "DIAGNOSTIC_DECISION"
    governor_version = "0.5.9"

    def decide(
        self,
        *,
        hypothesis_set: HypothesisSet,
        critique: CritiqueResult,
        deterministic_core_result: dict,
        protocol_versions: dict[str, str],
    ) -> DecisionRecord:
        registered = [
            item
            for item in hypothesis_set.hypotheses
            if item.hypothesis_type == HypothesisType.REGISTERED_FAULT
        ]
        supported = [item for item in registered if item.status == HypothesisStatus.SUPPORTED]
        supported_no_fault = [
            item
            for item in hypothesis_set.hypotheses
            if item.hypothesis_type == HypothesisType.NO_REGISTERED_FAULT
            and item.status == HypothesisStatus.SUPPORTED
        ]
        reasons: list[str] = []
        unresolved: list[str] = []
        accepted_hypothesis_id: str | None = None
        accepted_fault_id: str | None = None
        requires_human_review = False

        if critique.blocking:
            disposition = DecisionDisposition.HUMAN_REVIEW
            requires_human_review = True
            reasons.append("Safety Critic reported one or more blocking findings.")
            unresolved.extend(item.code for item in critique.findings if item.severity.value == "BLOCKING")
        elif len(supported) == 1:
            disposition = DecisionDisposition.ACCEPT
            accepted_hypothesis_id = supported[0].hypothesis_id
            accepted_fault_id = supported[0].target_fault_id
            reasons.append("Exactly one registered hypothesis satisfies all critical executable evidence.")
        elif len(supported) > 1:
            disposition = DecisionDisposition.AMBIGUOUS
            requires_human_review = True
            reasons.append("Multiple registered hypotheses satisfy their critical evidence definitions.")
            unresolved.extend(item.target_fault_id or item.hypothesis_id for item in supported)
        elif len(supported_no_fault) == 1:
            disposition = DecisionDisposition.NO_REGISTERED_FAULT
            accepted_hypothesis_id = supported_no_fault[0].hypothesis_id
            reasons.append(
                "All currently registered fault hypotheses are contradicted by executable evidence."
            )
        elif len(supported_no_fault) > 1:
            disposition = DecisionDisposition.HUMAN_REVIEW
            requires_human_review = True
            reasons.append("Multiple no-registered-fault hypotheses were generated unexpectedly.")
        else:
            out_of_scope = [item for item in hypothesis_set.hypotheses if item.status == HypothesisStatus.OUT_OF_SCOPE]
            if out_of_scope:
                disposition = DecisionDisposition.OUT_OF_SCOPE
                reasons.append("Only out-of-scope or research hypotheses remain.")
            else:
                disposition = DecisionDisposition.INSUFFICIENT_EVIDENCE
                reasons.append("No registered hypothesis satisfies all critical executable evidence.")
                for item in registered:
                    unresolved.extend(item.missing_evidence_definition_ids)
                    unresolved.extend(item.contradicting_evidence_fact_ids)

        core_status = str(deterministic_core_result.get("diagnosis_status", ""))
        core_case = deterministic_core_result.get("predicted_case_id")
        core_fault = _CASE_TO_FAULT.get(str(core_case)) if core_case else None
        comparison_status = "NOT_COMPARABLE"
        composite_blocked = any(
            item.code == "COMPOSITE_SIMULATION_CONTEXT_NOT_FORMALLY_SUPPORTED"
            and item.severity.value == "BLOCKING"
            for item in critique.findings
        )
        if core_status == "DIAGNOSED" and core_fault:
            if disposition == DecisionDisposition.ACCEPT and accepted_fault_id == core_fault:
                comparison_status = "MATCH"
            elif composite_blocked and disposition == DecisionDisposition.HUMAN_REVIEW:
                comparison_status = "COMPOSITE_MEMBER_EVIDENCE_ONLY"
                reasons.append(
                    f"Deterministic Core identified single-fault member {core_fault}, but the declared "
                    "composite context remains blocked for formal diagnosis."
                )
            else:
                comparison_status = "MISMATCH"
                disposition = DecisionDisposition.HUMAN_REVIEW
                requires_human_review = True
                reasons.append(
                    f"Agentic shadow decision does not match deterministic Core: core={core_fault}, "
                    f"agentic={accepted_fault_id or 'none'}."
                )
        elif core_status == "DIAGNOSED" and str(core_case) == "nominal":
            if disposition == DecisionDisposition.NO_REGISTERED_FAULT:
                comparison_status = "MATCH"
            else:
                comparison_status = "MISMATCH"
                disposition = DecisionDisposition.HUMAN_REVIEW
                requires_human_review = True
                reasons.append(
                    "Deterministic Core identified the paired nominal case while Agentic investigation did not "
                    "return NO_REGISTERED_FAULT."
                )
        elif core_status in {"UNKNOWN", "AMBIGUOUS"}:
            if core_status == "AMBIGUOUS" and composite_blocked and disposition == DecisionDisposition.HUMAN_REVIEW:
                comparison_status = "COMPOSITE_MEMBER_EVIDENCE_ONLY"
                reasons.append(
                    "Deterministic Core found multiple supported single-fault members, but no formal "
                    "composite FaultMode is registered; members remain evidence for a research hypothesis only."
                )
            elif disposition in {
                DecisionDisposition.INSUFFICIENT_EVIDENCE,
                DecisionDisposition.AMBIGUOUS,
                DecisionDisposition.HUMAN_REVIEW,
            }:
                comparison_status = "CONSISTENT_NON_DIAGNOSIS"
            else:
                comparison_status = "MISMATCH"
                disposition = DecisionDisposition.HUMAN_REVIEW
                requires_human_review = True
                reasons.append("Agentic shadow accepted a fault while deterministic Core did not diagnose uniquely.")

        rejected = [
            item.hypothesis_id
            for item in hypothesis_set.hypotheses
            if item.hypothesis_id != accepted_hypothesis_id
        ]
        decision_id = _stable_id(
            "DR",
            {
                "episode_id": hypothesis_set.episode_id,
                "disposition": disposition.value,
                "accepted_fault_id": accepted_fault_id,
                "comparison_status": comparison_status,
            },
        )
        return DecisionRecord(
            decision_id=decision_id,
            episode_id=hypothesis_set.episode_id,
            disposition=disposition,
            accepted_hypothesis_id=accepted_hypothesis_id,
            accepted_fault_id=accepted_fault_id,
            rejected_hypothesis_ids=rejected,
            reasons=reasons,
            unresolved_gaps=sorted(set(unresolved)),
            formal_release_allowed=False,
            requires_human_review=requires_human_review,
            shadow_mode=True,
            deterministic_core_result=deterministic_core_result,
            comparison_status=comparison_status,
            protocol_versions=protocol_versions,
        )


# Backward-compatible public name. New code should use DiagnosticDecisionGovernor.
DecisionGovernor = DiagnosticDecisionGovernor

__all__ = ["DiagnosticDecisionGovernor", "DecisionGovernor"]
