from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence import EpisodeEvidenceBundle, FeatureResult
from astrograph.evidence.executable import evaluate_evidence, stable_id
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, BUILTIN_OPERATOR_VERSION, OperatorInput
from astrograph.services.reasoning import DeterministicReferenceReasoner


CAPSULE_SCHEMA_VERSION = "astrograph-maintenance-supplemental-feature-input.v1"
ADMISSION_SCHEMA_VERSION = "astrograph-maintenance-supplemental-feature-admission.v1"
OVERLAY_SCHEMA_VERSION = "astrograph-maintenance-supplemental-evidence-overlay.v1"

# R10 deliberately supports only operators whose input/output unit semantics are
# direct and whose parameters are fully fixed by the builtin implementation.
# Operators with hidden/extra parameters, time-derivative semantics or input-unit
# ambiguity must continue through the normal telemetry/profile admission path.
_SAFE_DIRECT_OPERATORS = {
    "MAX_VALUE",
    "MIN_VALUE",
    "MEAN_VALUE",
    "MAX_ABS_PAIRED_RESIDUAL",
    "MEAN_ABS_PAIRED_RESIDUAL",
    "MEAN_PAIRED_RESIDUAL",
    "LAST_PAIRED_RESIDUAL",
    "MAX_ABS_STATUS_RESIDUAL",
    "TRUE_FRACTION",
    "FALSE_FRACTION",
}


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _expected_builtin_algorithm_version(operator_id: str) -> str:
    return f"builtin-{operator_id.lower().replace('_', '-')}@{BUILTIN_OPERATOR_VERSION}"


class SupplementalFeatureInputCapsule(BaseModel):
    """Operator-ready, contract-bound maintenance input.

    This is intentionally *not* arbitrary telemetry access.  The caller supplies
    one already-resolved canonical channel window; AstroGraph re-runs the exact
    allow-listed deterministic operator and derives FeatureResult/EvidenceFact
    server-side.  The original frozen report is never modified.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CAPSULE_SCHEMA_VERSION] = CAPSULE_SCHEMA_VERSION
    contract_id: str = Field(min_length=1, max_length=240, pattern=r"^[A-Za-z0-9_.:-]+$")
    resolved_source_field: str = Field(min_length=1, max_length=300)
    unit: str = Field(min_length=1, max_length=80)
    window_start_s: float = Field(ge=0.0)
    window_end_s: float = Field(ge=0.0)
    sample_times_s: list[float] = Field(min_length=1, max_length=200_000)
    current_values: list[float | None] = Field(min_length=1, max_length=200_000)
    baseline_values: list[float | None] | None = Field(default=None, max_length=200_000)
    source_note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_shapes(self) -> "SupplementalFeatureInputCapsule":
        if self.window_end_s < self.window_start_s:
            raise ValueError("window_end_s must be >= window_start_s")
        if len(self.sample_times_s) != len(self.current_values):
            raise ValueError("sample_times_s length must match current_values")
        if self.baseline_values is not None and len(self.baseline_values) != len(self.current_values):
            raise ValueError("baseline_values length must match current_values")
        if any(not math.isfinite(float(value)) for value in self.sample_times_s):
            raise ValueError("sample_times_s must contain only finite numbers")
        if any(
            value is not None and not math.isfinite(float(value))
            for value in self.current_values
        ):
            raise ValueError("current_values may contain finite numbers or null only")
        if self.baseline_values is not None and any(
            value is not None and not math.isfinite(float(value))
            for value in self.baseline_values
        ):
            raise ValueError("baseline_values may contain finite numbers or null only")
        if any(self.sample_times_s[index] > self.sample_times_s[index + 1] for index in range(len(self.sample_times_s) - 1)):
            raise ValueError("sample_times_s must be non-decreasing")
        return self


def assess_supplemental_admission_support(
    *,
    feature_definition: Any | None,
    resolved_source_field: str | None,
    window_scope: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    operator_id = None
    frozen_algorithm_version = None
    if feature_definition is None:
        reasons.append("FEATURE_DEFINITION_UNAVAILABLE")
    else:
        operator_id = feature_definition.aggregation.value
        frozen_algorithm_version = feature_definition.algorithm_version
        if operator_id not in _SAFE_DIRECT_OPERATORS:
            reasons.append("OPERATOR_NOT_ELIGIBLE_FOR_DIRECT_SUPPLEMENTAL_ADMISSION")
        if not BUILTIN_OPERATOR_REGISTRY.has(operator_id):
            reasons.append("OPERATOR_NOT_REGISTERED")
        expected = _expected_builtin_algorithm_version(operator_id)
        if frozen_algorithm_version != expected:
            reasons.append("FROZEN_ALGORITHM_VERSION_NOT_EXACT_BUILTIN_MATCH")
    if not resolved_source_field:
        reasons.append("SOURCE_FIELD_UNRESOLVED")
    if window_scope != "episode":
        reasons.append("ONLY_EPISODE_SCOPED_SUPPLEMENTAL_ADMISSION_SUPPORTED")
    return {
        "status": "SUPPORTED" if not reasons else "UNSUPPORTED",
        "reason_codes": sorted(set(reasons)),
        "operator_id": operator_id,
        "operator_runtime_version": BUILTIN_OPERATOR_VERSION,
        "frozen_algorithm_version": frozen_algorithm_version,
        "capsule_schema_version": CAPSULE_SCHEMA_VERSION,
        "claim_boundary": (
            "R10 direct supplemental admission is limited to exact-version, allow-listed deterministic operators over one resolved "
            "episode-scoped canonical channel. Unsupported contracts must use the normal telemetry/profile admission path."
        ),
    }


def build_supplemental_input_template(contract: dict[str, Any]) -> dict[str, Any]:
    support = {
        "status": contract.get("supplemental_admission_status"),
        "reason_codes": list(contract.get("supplemental_admission_reason_codes") or []),
    }
    return {
        "schema_version": "astrograph-maintenance-supplemental-input-template.v1",
        "contract_id": contract.get("contract_id"),
        "support": support,
        "capsule": {
            "schema_version": CAPSULE_SCHEMA_VERSION,
            "contract_id": contract.get("contract_id"),
            "resolved_source_field": contract.get("resolved_source_field") or "",
            "unit": contract.get("required_unit") or "",
            "window_start_s": contract.get("required_window_start_s"),
            "window_end_s": contract.get("required_window_end_s"),
            "sample_times_s": [],
            "current_values": [],
            **({"baseline_values": []} if contract.get("paired_nominal_required") else {}),
            "source_note": "",
        },
        "claim_boundary": (
            "Template only. The server revalidates the current contract and recomputes the allow-listed operator; "
            "editing this JSON cannot override unit, window, field, algorithm or evidence thresholds."
        ),
    }


def _status_for_fault(hypothesis_payload: dict[str, Any], fault_id: str) -> str | None:
    for item in hypothesis_payload.get("hypotheses") or []:
        if item.get("target_fault_id") == fault_id:
            return str(item.get("status")) if item.get("status") is not None else None
    return None


def admit_supplemental_feature_input(
    *,
    bundle: EpisodeEvidenceBundle,
    contract: dict[str, Any],
    capsule: SupplementalFeatureInputCapsule,
    actor: str,
) -> tuple[dict[str, Any], EpisodeEvidenceBundle]:
    if capsule.contract_id != contract.get("contract_id"):
        raise ValueError("capsule contract_id does not match requested maintenance input contract")
    evidence_definition_id = str(contract.get("evidence_definition_id") or "")
    feature_definition_id = str(contract.get("feature_definition_id") or "")
    definitions = {item.evidence_definition_id: item for item in bundle.evidence_definitions}
    features = {item.feature_definition_id: item for item in bundle.feature_definitions}
    definition = definitions.get(evidence_definition_id)
    feature_definition = features.get(feature_definition_id)
    if definition is None or feature_definition is None:
        raise ValueError("current evidence bundle no longer contains the contract definition")
    if definition.feature_definition_id != feature_definition.feature_definition_id:
        raise ValueError("evidence definition no longer references the contracted feature definition")

    support = assess_supplemental_admission_support(
        feature_definition=feature_definition,
        resolved_source_field=contract.get("resolved_source_field"),
        window_scope=contract.get("window_scope"),
    )
    if support["status"] != "SUPPORTED":
        raise ValueError("supplemental admission is not supported for this contract: " + ",".join(support["reason_codes"]))

    required_field = str(contract.get("resolved_source_field") or "")
    if capsule.resolved_source_field != required_field:
        raise ValueError("resolved_source_field does not match the current input contract")
    required_unit = str(contract.get("required_unit") or "")
    if capsule.unit != required_unit or capsule.unit != feature_definition.unit:
        raise ValueError("supplemental input unit does not match the frozen FeatureDefinition")
    required_start = contract.get("required_window_start_s")
    required_end = contract.get("required_window_end_s")
    if required_start is None or required_end is None:
        raise ValueError("supplemental admission requires a resolved episode window")
    if abs(float(capsule.window_start_s) - float(required_start)) > 1e-9 or abs(float(capsule.window_end_s) - float(required_end)) > 1e-9:
        raise ValueError("supplemental input window does not match the frozen episode contract")
    if not capsule.sample_times_s:
        raise ValueError("supplemental input must contain at least one timestamped sample")
    if min(capsule.sample_times_s) < float(required_start) - 1e-9 or max(capsule.sample_times_s) > float(required_end) + 1e-9:
        raise ValueError("supplemental sample_times_s escape the frozen episode window")

    operator_id = feature_definition.aggregation.value
    operator = BUILTIN_OPERATOR_REGISTRY.get(operator_id)
    if operator.spec.requires_time:
        raise ValueError("time-dependent operators are not eligible for R10 direct supplemental admission")
    if operator.spec.requires_baseline and capsule.baseline_values is None:
        raise ValueError("contracted operator requires baseline_values")
    if not operator.spec.requires_baseline and capsule.baseline_values is not None:
        raise ValueError("baseline_values are not allowed for this non-paired operator")

    current = np.asarray([np.nan if value is None else float(value) for value in capsule.current_values], dtype=float)
    baseline = None if capsule.baseline_values is None else np.asarray(
        [np.nan if value is None else float(value) for value in capsule.baseline_values], dtype=float
    )
    if operator.spec.requires_baseline:
        assert baseline is not None
        finite_mask = np.isfinite(current) & np.isfinite(baseline)
    else:
        finite_mask = np.isfinite(current)
    sample_count = int(finite_mask.sum())
    if sample_count <= 0:
        raise ValueError("supplemental input contains no finite samples admitted by the operator")

    capsule_payload = capsule.model_dump(mode="json")
    input_sha256 = _canonical_sha256({key: value for key, value in capsule_payload.items() if key != "source_note"})
    prior_admissions = list(bundle.analysis_metadata.get("maintenance_supplemental_admissions") or [])
    if any(
        str(item.get("contract_id")) == capsule.contract_id
        and str(item.get("input_data_sha256")) == input_sha256
        for item in prior_admissions
        if isinstance(item, dict)
    ):
        raise ValueError("identical supplemental input capsule was already admitted for this contract")
    value = BUILTIN_OPERATOR_REGISTRY.compute(
        operator_id,
        OperatorInput(current=current, baseline=baseline),
    )
    if value is None or not math.isfinite(float(value)):
        raise ValueError("deterministic operator did not produce a finite FeatureResult value")

    episode = bundle.episode
    feature_result = FeatureResult(
        feature_result_id=stable_id(
            "FR",
            {
                "feature_definition_id": feature_definition.feature_definition_id,
                "episode_id": episode.episode_id,
                "input_data_sha256": input_sha256,
                "operator_id": operator_id,
                "operator_runtime_version": BUILTIN_OPERATOR_VERSION,
            },
        ),
        feature_definition_id=feature_definition.feature_definition_id,
        run_id=episode.run_id,
        paired_nominal_run_id=episode.paired_nominal_run_id,
        episode_id=episode.episode_id,
        resolved_field=required_field,
        component_instance=bundle.analysis_component_instance,
        value=float(value),
        unit=feature_definition.unit,
        window_start_s=float(required_start),
        window_end_s=float(required_end),
        sample_count=sample_count,
        algorithm_version=feature_definition.algorithm_version,
        input_data_sha256=input_sha256,
        # quality_flags describe observation/data validity only.  Supplemental
        # provenance is frozen in analysis_metadata plus admission/provenance
        # receipts; encoding provenance here would make a valid observation
        # unusable by OAG's causal-evidence validator.
        quality_flags=[],
    )

    updated_feature_results = [
        item for item in bundle.feature_results
        if not (
            item.feature_definition_id == feature_definition.feature_definition_id
            and item.episode_id == episode.episode_id
        )
    ]
    updated_feature_results.append(feature_result)
    result_by_definition = {item.feature_definition_id: item for item in updated_feature_results}
    updated_facts = [
        evaluate_evidence(
            item,
            result_by_definition.get(item.feature_definition_id),
            all_feature_results=updated_feature_results,
        )
        for item in bundle.evidence_definitions
    ]

    original_hypotheses = DeterministicReferenceReasoner.build(bundle).model_dump(mode="json")
    overlay_metadata = dict(bundle.analysis_metadata)
    history = list(overlay_metadata.get("maintenance_supplemental_admissions") or [])
    admission_identity = {
        "contract_id": capsule.contract_id,
        "evidence_definition_id": evidence_definition_id,
        "feature_definition_id": feature_definition.feature_definition_id,
        "input_data_sha256": input_sha256,
        "operator_id": operator_id,
        "operator_runtime_version": BUILTIN_OPERATOR_VERSION,
        "frozen_algorithm_version": feature_definition.algorithm_version,
        "actor": actor,
    }
    history.append(admission_identity)
    overlay_metadata["maintenance_supplemental_admissions"] = history
    overlay_metadata["maintenance_evidence_source_mode"] = "SUPPLEMENTAL_REANALYSIS_OVERLAY"
    overlay_bundle = bundle.model_copy(update={
        "feature_results": updated_feature_results,
        "evidence_facts": updated_facts,
        "analysis_metadata": overlay_metadata,
    })
    updated_hypotheses = DeterministicReferenceReasoner.build(overlay_bundle).model_dump(mode="json")

    old_fact = next((item for item in bundle.evidence_facts if item.evidence_definition_id == evidence_definition_id), None)
    new_fact = next((item for item in overlay_bundle.evidence_facts if item.evidence_definition_id == evidence_definition_id), None)
    target_fault_id = definition.target_fault_id
    receipt_core = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": "ADMITTED",
        "contract_id": capsule.contract_id,
        "actor": actor,
        "capsule_sha256": input_sha256,
        "operator_id": operator_id,
        "operator_runtime_version": BUILTIN_OPERATOR_VERSION,
        "frozen_algorithm_version": feature_definition.algorithm_version,
        "feature_result": feature_result.model_dump(mode="json"),
        "evidence_definition_id": evidence_definition_id,
        "evidence_fact_before": (old_fact.model_dump(mode="json") if old_fact is not None else None),
        "evidence_fact_after": (new_fact.model_dump(mode="json") if new_fact is not None else None),
        "target_fault_id": target_fault_id,
        "hypothesis_status_before": _status_for_fault(original_hypotheses, target_fault_id),
        "hypothesis_status_after": _status_for_fault(updated_hypotheses, target_fault_id),
        "deterministic_hypothesis_set_after": updated_hypotheses,
        "maintenance_overlay_generated": True,
        "original_report_mutated": False,
        "formal_release_eligible": False,
        "claim_boundary": (
            "The server recomputed an allow-listed deterministic feature and EvidenceFact from a contract-bound maintenance input capsule. "
            "The result is a maintenance reanalysis overlay only: it does not mutate the original frozen report, does not change release status, "
            "and must pass the full production pipeline again before becoming release evidence."
        ),
    }
    receipt = dict(receipt_core)
    receipt["receipt_sha256"] = _canonical_sha256(receipt_core)
    return receipt, overlay_bundle


__all__ = [
    "CAPSULE_SCHEMA_VERSION",
    "ADMISSION_SCHEMA_VERSION",
    "OVERLAY_SCHEMA_VERSION",
    "SupplementalFeatureInputCapsule",
    "assess_supplemental_admission_support",
    "build_supplemental_input_template",
    "admit_supplemental_feature_input",
]
