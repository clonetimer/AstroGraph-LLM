from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence import (
    EpisodeEvidenceBundle,
    EvidenceFactStatus,
    evaluate_evidence,
)


PerturbationType = Literal[
    "DROP_EVIDENCE",
    "UNIT_MISMATCH",
    "FORCE_STATUS",
    "NEAR_THRESHOLD",
    "SHIFT_FEATURE_WINDOW",
]


class EvidencePerturbationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_id: str = Field(min_length=1)
    perturbation_type: PerturbationType
    evidence_definition_id: str | None = None
    feature_definition_id: str | None = None
    forced_status: EvidenceFactStatus | None = None
    replacement_unit: str | None = None
    threshold_offset_ratio: float | None = None
    window_shift_s: float | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def validate_arguments(self) -> "EvidencePerturbationSpec":
        if self.perturbation_type in {"DROP_EVIDENCE", "FORCE_STATUS", "NEAR_THRESHOLD"} and not self.evidence_definition_id:
            raise ValueError(f"{self.perturbation_type} requires evidence_definition_id")
        if self.perturbation_type in {"UNIT_MISMATCH", "SHIFT_FEATURE_WINDOW"} and not self.feature_definition_id:
            raise ValueError(f"{self.perturbation_type} requires feature_definition_id")
        if self.perturbation_type == "FORCE_STATUS" and self.forced_status is None:
            raise ValueError("FORCE_STATUS requires forced_status")
        if self.perturbation_type == "UNIT_MISMATCH" and not self.replacement_unit:
            raise ValueError("UNIT_MISMATCH requires replacement_unit")
        if self.perturbation_type == "NEAR_THRESHOLD" and self.threshold_offset_ratio is None:
            raise ValueError("NEAR_THRESHOLD requires threshold_offset_ratio")
        if self.perturbation_type == "SHIFT_FEATURE_WINDOW" and self.window_shift_s is None:
            raise ValueError("SHIFT_FEATURE_WINDOW requires window_shift_s")
        return self


class PerturbationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_id: str
    perturbation_type: PerturbationType
    target_id: str
    before: dict = Field(default_factory=dict)
    after: dict = Field(default_factory=dict)


def apply_evidence_perturbations(
    bundle: EpisodeEvidenceBundle,
    specifications: list[EvidencePerturbationSpec],
) -> tuple[EpisodeEvidenceBundle, list[PerturbationAudit]]:
    if not specifications:
        return bundle, []
    updated = bundle.model_copy(deep=True)
    feature_by_definition = {item.feature_definition_id: item for item in updated.feature_results}
    definition_by_id = {item.evidence_definition_id: item for item in updated.evidence_definitions}
    fact_index = {item.evidence_definition_id: idx for idx, item in enumerate(updated.evidence_facts)}
    audits: list[PerturbationAudit] = []

    for spec in specifications:
        if spec.perturbation_type in {"DROP_EVIDENCE", "FORCE_STATUS", "NEAR_THRESHOLD"}:
            definition = definition_by_id.get(str(spec.evidence_definition_id))
            if definition is None:
                raise ValueError(f"unknown evidence definition for perturbation: {spec.evidence_definition_id}")
            idx = fact_index[definition.evidence_definition_id]
            fact = updated.evidence_facts[idx]
            feature = feature_by_definition.get(definition.feature_definition_id)
            before = fact.model_dump(mode="json")
            if spec.perturbation_type == "DROP_EVIDENCE":
                if feature is not None:
                    feature.value = None
                    feature.quality_flags = sorted(set([*feature.quality_flags, "PERTURBED_MISSING_DATA"]))
                new_fact = evaluate_evidence(definition, feature)
            elif spec.perturbation_type == "FORCE_STATUS":
                new_fact = fact.model_copy(
                    update={
                        "status": spec.forced_status,
                        "explanation": f"Perturbation {spec.perturbation_id} forced status to {spec.forced_status.value}.",
                    }
                )
            else:
                if feature is None:
                    raise ValueError(
                        "NEAR_THRESHOLD perturbation requires the referenced feature result"
                    )
                ratio = float(spec.threshold_offset_ratio)
                value = float(definition.threshold) * (1.0 + ratio)
                if definition.threshold == 0:
                    value = ratio
                feature.value = value
                feature.quality_flags = sorted(set([*feature.quality_flags, "PERTURBED_NEAR_THRESHOLD"]))
                new_fact = evaluate_evidence(definition, feature)
            updated.evidence_facts[idx] = new_fact
            audits.append(
                PerturbationAudit(
                    perturbation_id=spec.perturbation_id,
                    perturbation_type=spec.perturbation_type,
                    target_id=definition.evidence_definition_id,
                    before=before,
                    after=new_fact.model_dump(mode="json"),
                )
            )
        elif spec.perturbation_type in {"UNIT_MISMATCH", "SHIFT_FEATURE_WINDOW"}:
            feature = feature_by_definition.get(str(spec.feature_definition_id))
            if feature is None:
                raise ValueError(f"unknown feature definition for perturbation: {spec.feature_definition_id}")
            before = feature.model_dump(mode="json")
            if spec.perturbation_type == "UNIT_MISMATCH":
                feature.unit = str(spec.replacement_unit)
                for idx, fact in enumerate(updated.evidence_facts):
                    definition = definition_by_id[fact.evidence_definition_id]
                    if definition.feature_definition_id == feature.feature_definition_id:
                        updated.evidence_facts[idx] = fact.model_copy(update={"unit": str(spec.replacement_unit)})
            else:
                shift = float(spec.window_shift_s)
                feature.window_start_s = max(0.0, feature.window_start_s + shift)
                feature.window_end_s = max(feature.window_start_s, feature.window_end_s + shift)
                feature.quality_flags = sorted(set([*feature.quality_flags, "PERTURBED_WINDOW_SHIFT"]))
            audits.append(
                PerturbationAudit(
                    perturbation_id=spec.perturbation_id,
                    perturbation_type=spec.perturbation_type,
                    target_id=feature.feature_definition_id,
                    before=before,
                    after=feature.model_dump(mode="json"),
                )
            )
    return updated, audits
