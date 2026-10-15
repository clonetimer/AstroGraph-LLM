from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from .unified_models import DiagnosticEvidence

_TIME_COLUMNS = {"time_s", "sim_time_s", "timestamp_s"}
_DEFAULT_CATALOG = "expanded_single_fault_catalog.json"


@dataclass(frozen=True)
class ExpandedSingleFaultResult:
    status: str
    observation_class: str | None
    candidate_modes: list[str]
    primary_mode: str | None
    subsystem: str | None
    component: str | None
    screening_score: float | None
    nearest_mode: str | None
    cosine_distance: float | None
    evidence: list[DiagnosticEvidence]
    reason_codes: list[str]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observation_class": self.observation_class,
            "candidate_modes": self.candidate_modes,
            "primary_mode": self.primary_mode,
            "subsystem": self.subsystem,
            "component": self.component,
            "screening_score": self.screening_score,
            "nearest_mode": self.nearest_mode,
            "cosine_distance": self.cosine_distance,
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "reason_codes": self.reason_codes,
            "claim_boundary": self.claim_boundary,
        }


def _load_catalog() -> dict[str, Any]:
    path = files("astrograph.resources").joinpath(_DEFAULT_CATALOG)
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(frame: pd.DataFrame, field: str) -> np.ndarray:
    if field not in frame.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)


def extract_paired_residual_features(nominal: pd.DataFrame, effect: pd.DataFrame) -> dict[str, float]:
    """Extract case-id-independent paired residual features.

    The extractor only reads numeric telemetry. Labels, truth columns, target IDs,
    file paths, and case identifiers are not accepted as features.
    """
    output: dict[str, float] = {}
    for field in nominal.columns:
        if field not in effect.columns or field in _TIME_COLUMNS:
            continue
        a = _numeric(nominal, field)
        b = _numeric(effect, field)
        if len(a) != len(b):
            continue
        valid = np.isfinite(a) & np.isfinite(b)
        if int(valid.sum()) < 2:
            continue
        a = a[valid]
        b = b[valid]
        delta = b - a
        scale = max(float(np.nanpercentile(np.abs(a), 90)), float(np.nanstd(a)), 1e-9)
        relative = np.clip(delta / scale, -20.0, 20.0)
        denominator = float(np.mean(np.abs(delta))) + 1e-12
        output[f"{field}__mean_rel"] = float(np.mean(relative))
        output[f"{field}__mean_abs_rel"] = float(np.mean(np.abs(relative)))
        output[f"{field}__max_abs_rel"] = float(np.max(np.abs(relative)))
        output[f"{field}__final_rel"] = float(relative[-1])
        output[f"{field}__std_rel"] = float(np.std(relative))
        output[f"{field}__changed_frac"] = float(np.mean(np.abs(relative) > 1e-5))
        output[f"{field}__sign_mean"] = float(np.sign(np.mean(delta)))
        output[f"{field}__final_vs_mean"] = float(np.clip(delta[-1] / denominator, -20.0, 20.0))
    return output


def _feature_evidence(
    nominal: pd.DataFrame,
    effect: pd.DataFrame,
    *,
    maximum: int = 6,
) -> list[DiagnosticEvidence]:
    ranked: list[tuple[float, DiagnosticEvidence]] = []
    for field in nominal.columns:
        if field not in effect.columns or field in _TIME_COLUMNS:
            continue
        a = _numeric(nominal, field)
        b = _numeric(effect, field)
        if len(a) != len(b):
            continue
        valid = np.isfinite(a) & np.isfinite(b)
        if int(valid.sum()) < 2:
            continue
        a = a[valid]
        b = b[valid]
        delta = b - a
        nominal_scale = max(float(np.nanpercentile(np.abs(a), 90)), float(np.nanstd(a)), 1e-9)
        strength = float(np.mean(np.abs(delta)) / nominal_scale)
        if strength <= 1e-8:
            continue
        ranked.append((strength, DiagnosticEvidence(
            evidence_id="PAIRED_RESIDUAL_FEATURE",
            field=field,
            statistic="mean",
            nominal_value=float(np.mean(a)),
            effect_value=float(np.mean(b)),
            observed_delta=float(np.mean(delta)),
            interpretation="Paired Basilisk telemetry exhibits a repeatable residual in this field.",
        )))
    ranked.sort(key=lambda item: (item[0], item[1].field), reverse=True)
    return [item[1] for item in ranked[:maximum]]


class ExpandedSingleFaultEngine:
    """Development-prototype paired classifier with fail-closed equivalence groups."""

    def __init__(self, catalog: dict[str, Any] | None = None) -> None:
        self.catalog = catalog or _load_catalog()
        self.feature_names = list(self.catalog["feature_names"])
        self.center = np.asarray(self.catalog["center"], dtype=float)
        self.scale = np.asarray(self.catalog["scale"], dtype=float)
        self.prototypes = list(self.catalog["prototypes"])
        self.prototype_matrix = np.asarray([item["vector"] for item in self.prototypes], dtype=float)
        self.groups = dict(self.catalog["equivalence_groups"])
        self.mode_to_group = {
            mode: group_id
            for group_id, spec in self.groups.items()
            for mode in spec["members"]
        }

    @property
    def known_modes(self) -> list[str]:
        return sorted(item["mode"] for item in self.prototypes)

    def diagnose_frames(self, *, nominal: pd.DataFrame, effect: pd.DataFrame) -> ExpandedSingleFaultResult:
        columns = set(nominal.columns) & set(effect.columns)
        expanded_whole_spacecraft = {
            "imu_gyro_x_rad_s",
            "star_tracker_q0",
            "sun_sensor_output",
            "magnetometer_x_t",
            "thermal_electronics_temp_c",
            "radiator_electronics_on",
        }.issubset(columns)
        component_bench = any(
            field in columns for field in ("cmg_wheel_speed_rad_s", "mtb_torque_norm_nm")
        )
        if not (expanded_whole_spacecraft or component_bench):
            return ExpandedSingleFaultResult(
                status="OUT_OF_SCOPE_SCHEMA",
                observation_class=None,
                candidate_modes=[],
                primary_mode=None,
                subsystem=None,
                component=None,
                screening_score=None,
                nearest_mode=None,
                cosine_distance=None,
                evidence=[],
                reason_codes=["EXPANDED_TELEMETRY_PROFILE_NOT_PRESENT"],
                claim_boundary=self.catalog["claim_boundary"],
            )
        features = extract_paired_residual_features(nominal, effect)
        vector = np.asarray([features.get(name, 0.0) for name in self.feature_names], dtype=float)
        standardized = (vector - self.center) / self.scale
        norm = float(np.linalg.norm(standardized))
        if norm < 1e-12:
            return ExpandedSingleFaultResult(
                status="INSUFFICIENT_EVIDENCE",
                observation_class=None,
                candidate_modes=[],
                primary_mode=None,
                subsystem=None,
                component=None,
                screening_score=0.0,
                nearest_mode=None,
                cosine_distance=None,
                evidence=[],
                reason_codes=["NO_MEASURABLE_PAIRED_RESIDUAL"],
                claim_boundary=self.catalog["claim_boundary"],
            )
        standardized /= norm
        distances = 1.0 - self.prototype_matrix @ standardized
        candidate_indices = np.arange(len(self.prototypes))
        # Star-tracker faults directly perturb quaternion channels by orders of
        # magnitude, while reaction-wheel faults only move those channels through
        # downstream attitude dynamics. Restricting on this high-specificity direct
        # measurement prevents downstream control residuals from stealing the root.
        quaternion_fields = [f"star_tracker_q{i}" for i in range(4)]
        if all(field in nominal.columns and field in effect.columns for field in quaternion_fields):
            q_delta = np.sqrt(sum(
                np.square(_numeric(effect, field) - _numeric(nominal, field))
                for field in quaternion_fields
            ))
            q_strength = float(np.nanmean(np.abs(q_delta))) if len(q_delta) else 0.0
        else:
            q_strength = 0.0
        # Direct star-tracker accuracy/bias injections stay below 0.02 in mean
        # quaternion residual in the frozen development campaign. Larger residuals
        # are produced by downstream spacecraft motion (RW/thruster faults) and
        # must not be mistaken for a sensor root cause.
        if 1e-6 < q_strength < 0.02:
            direct = np.asarray([
                index for index, item in enumerate(self.prototypes)
                if item["component"] == "star_tracker"
            ], dtype=int)
            if len(direct):
                candidate_indices = direct
        nearest_index = int(candidate_indices[np.argmin(distances[candidate_indices])])
        nearest = self.prototypes[nearest_index]
        distance = float(distances[nearest_index])
        score = max(0.0, min(1.0, 1.0 - distance / 0.8))
        if distance > 0.8:
            return ExpandedSingleFaultResult(
                status="UNKNOWN",
                observation_class=None,
                candidate_modes=[],
                primary_mode=None,
                subsystem=None,
                component=None,
                screening_score=score,
                nearest_mode=str(nearest["mode"]),
                cosine_distance=distance,
                evidence=_feature_evidence(nominal, effect),
                reason_codes=["OUTSIDE_FROZEN_DEVELOPMENT_RESIDUAL_ENVELOPE"],
                claim_boundary=self.catalog["claim_boundary"],
            )
        nearest_mode = str(nearest["mode"])
        group_id = self.mode_to_group.get(nearest_mode)
        evidence = _feature_evidence(nominal, effect)
        if group_id is not None:
            group = self.groups[group_id]
            evidence.insert(0, DiagnosticEvidence(
                evidence_id="OBSERVATIONAL_EQUIVALENCE_CLASS",
                field="paired_residual_vector",
                statistic="cosine_nearest_prototype",
                nominal_value=None,
                effect_value=None,
                observed_delta=distance,
                threshold=0.8,
                interpretation=str(group["reason"]),
            ))
            return ExpandedSingleFaultResult(
                status="OBSERVATIONALLY_EQUIVALENT",
                observation_class=group_id,
                candidate_modes=list(group["members"]),
                primary_mode=None,
                subsystem=group.get("subsystem"),
                component=group.get("component"),
                screening_score=score,
                nearest_mode=nearest_mode,
                cosine_distance=distance,
                evidence=evidence,
                reason_codes=["SOURCE_NOT_IDENTIFIABLE_FROM_AVAILABLE_TELEMETRY"],
                claim_boundary=self.catalog["claim_boundary"],
            )
        return ExpandedSingleFaultResult(
            status="DIAGNOSED",
            observation_class=nearest_mode,
            candidate_modes=[nearest_mode],
            primary_mode=nearest_mode,
            subsystem=str(nearest["subsystem"]),
            component=str(nearest["component_display"]),
            screening_score=score,
            nearest_mode=nearest_mode,
            cosine_distance=distance,
            evidence=evidence,
            reason_codes=["FROZEN_DEVELOPMENT_RESIDUAL_PROTOTYPE_MATCH"],
            claim_boundary=self.catalog["claim_boundary"],
        )


__all__ = [
    "ExpandedSingleFaultEngine",
    "ExpandedSingleFaultResult",
    "extract_paired_residual_features",
]
