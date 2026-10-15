from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .model_release import model_source_fingerprint, resolve_release_artifact
from .sat_sim_catalog import (
    CatalogCandidate,
    CatalogEvidence,
    NORMAL_CLASS_ID,
    SatSimCatalogDiagnosisEngine,
    _is_prohibited,
    extract_catalog_features,
    load_catalog_artifact,
)


class OpenSetDiagnosisStatus(str):
    NORMAL = "NORMAL"
    KNOWN_FAULT = "KNOWN_FAULT"
    UNKNOWN_FAULT = "UNKNOWN_FAULT"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    DATA_ERROR = "DATA_ERROR"


class OpenSetDiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-sat-sim-open-set-diagnosis.v1"
    case_id: str
    backend: str
    status: str
    conclusion_effect_id: str | None = None
    catalog_membership_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    catalog_conformity_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_vote_share: float = Field(default=0.0, ge=0.0, le=1.0)
    vote_margin: float = 0.0
    profile_distance: float | None = Field(default=None, ge=0.0)
    resolution_basis: str = "NO_RESOLUTION"
    uncertainty_reason_codes: list[str] = Field(default_factory=list)
    candidates: list[CatalogCandidate] = Field(default_factory=list)
    evidence: list[CatalogEvidence] = Field(default_factory=list)
    observed_feature_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    prohibited_columns_used: list[str] = Field(default_factory=list)
    labels_used: bool = False
    truth_used: bool = False
    score_semantics: str = (
        "CATALOG_MEMBERSHIP_CONFIDENCE_CALIBRATED_ON_BASILISK_LAB_DATA; "
        "MODEL_VOTE_SHARE_NOT_CALIBRATED_CLASS_PROBABILITY"
    )
    claim_boundary: str = (
        "Laboratory open-set screening for the declared Sat-Sim catalog. Membership confidence estimates "
        "catalog fit, not flight reliability or calibrated fault-class probability. Unknown results require "
        "additional telemetry or expert review."
    )


def augment_derived_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only physically defined compatibility fields; never derive from labels/truth."""
    result = frame.copy()
    axes = ("x", "y", "z")
    for stem, target in (
        ("adcs.sensor.gyro_bias_rad_s_", "adcs.sensor.gyro_bias_norm_rad_s"),
        ("adcs.sensor.gyro_noise_rad_s_", "adcs.sensor.gyro_noise_norm_rad_s"),
    ):
        columns = [stem + axis for axis in axes]
        if target not in result.columns and all(column in result.columns for column in columns):
            values = [pd.to_numeric(result[column], errors="coerce").fillna(0.0) for column in columns]
            result[target] = np.sqrt(sum(value * value for value in values))
    target = "adcs.rw.effective_friction_torque_nm_2"
    if target not in result.columns:
        legacy = "estimate.adcs.rw.friction_torque_nm_2"
        if legacy in result.columns:
            result[target] = pd.to_numeric(result[legacy], errors="coerce")
        elif "adcs.rw.effective_drag_nms_2" in result.columns and "adcs.rw.speed_rad_s_2" in result.columns:
            result[target] = -pd.to_numeric(result["adcs.rw.effective_drag_nms_2"], errors="coerce") * pd.to_numeric(
                result["adcs.rw.speed_rad_s_2"], errors="coerce"
            )
    return result


def sanitize_telemetry(frame: pd.DataFrame) -> pd.DataFrame:
    prohibited = [column for column in frame.columns if _is_prohibited(column)]
    if prohibited:
        raise ValueError("prohibited columns: " + ",".join(sorted(prohibited)))
    return augment_derived_fields(frame)


def _distance(
    vector: np.ndarray,
    *,
    predicted: str,
    centers: Mapping[str, np.ndarray],
    scales: np.ndarray,
    mask: np.ndarray,
    importances: np.ndarray,
) -> float:
    center = np.asarray(centers[predicted], dtype=float)
    scale = np.maximum(np.asarray(scales, dtype=float), 1.0e-9)
    z = np.abs((vector - center) / scale)
    active = np.asarray(mask, dtype=bool) & np.isfinite(z)
    if not np.any(active):
        return math.inf
    raw_weights = np.asarray(importances, dtype=float)
    if raw_weights.shape == active.shape:
        weights = np.where(active, raw_weights, 0.0)
    else:
        weights = np.zeros_like(z, dtype=float)
        model_positions = np.flatnonzero(np.asarray(mask, dtype=bool))
        if len(model_positions) != len(raw_weights):
            raise ValueError("feature importance shape does not match active feature mask")
        weights[model_positions] = raw_weights
        weights = np.where(active, weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = active.astype(float)
    weights = weights / weights.sum()
    return float(np.sqrt(np.sum(weights * z * z)))


def _descriptor(
    *,
    base_artifact: Mapping[str, Any],
    calibration: Mapping[str, Any],
    backend: str,
    frame: pd.DataFrame,
) -> tuple[dict[str, float], np.ndarray, list[str], Any]:
    clean = sanitize_telemetry(frame)
    vector, names, ratio = extract_catalog_features(clean, backend=backend)
    backend_model = base_artifact["backend_models"][backend]
    medians = np.asarray(backend_model["medians"], dtype=float)
    vector = np.where(np.isfinite(vector), vector, medians)
    active_mask = np.asarray(backend_model["active_mask"], dtype=bool)
    model = backend_model["model"]
    probabilities = np.asarray(model.predict_proba([vector[active_mask]])[0], dtype=float)
    classes = np.asarray(model.classes_, dtype=str)
    order = np.argsort(probabilities)[::-1]
    predicted = str(classes[order[0]])
    top = float(probabilities[order[0]])
    second = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    distance = _distance(
        vector,
        predicted=predicted,
        centers=calibration["class_centers"],
        scales=np.asarray(calibration["global_scales"], dtype=float),
        mask=active_mask,
        importances=np.asarray(model.feature_importances_, dtype=float),
    )
    return {
        "top_vote": top,
        "vote_margin": top - second,
        "profile_distance": distance,
        "observed_ratio": ratio,
        "predicted": predicted,
    }, vector, names, clean



def _select_knownness_threshold(
    probabilities: np.ndarray,
    known_flags: np.ndarray,
) -> tuple[float, dict[str, float | str]]:
    """Choose a catalog-membership threshold from calibration evidence only.

    When known and unknown calibration probabilities have a clean separation
    gap, placing the threshold on an observed endpoint is brittle: a new known
    case can fall slightly below the lowest calibration probability while
    remaining orders of magnitude above every unknown example.  Use the
    midpoint in log-odds space so the gate is centred inside the evidence gap,
    not fitted to either endpoint.  If the calibration distributions overlap,
    fall back to the constrained balanced-recall search.
    """
    probs = np.asarray(probabilities, dtype=float)
    flags = np.asarray(known_flags, dtype=int)
    known = probs[flags == 1]
    unknown = probs[flags == 0]
    if known.size == 0 or unknown.size == 0:
        return 0.0, {
            "threshold_strategy": "NO_TWO_CLASS_CALIBRATION",
            "calibration_known_probability_min": float(np.min(known)) if known.size else 0.0,
            "calibration_unknown_probability_max": float(np.max(unknown)) if unknown.size else 0.0,
        }
    known_min = float(np.min(known))
    unknown_max = float(np.max(unknown))
    if unknown_max < known_min:
        eps = 1.0e-9
        k = float(np.clip(known_min, eps, 1.0 - eps))
        u = float(np.clip(unknown_max, eps, 1.0 - eps))
        midpoint_logit = (np.log(k / (1.0 - k)) + np.log(u / (1.0 - u))) / 2.0
        threshold = float(1.0 / (1.0 + np.exp(-midpoint_logit)))
        return threshold, {
            "threshold_strategy": "LOGIT_MIDPOINT_OF_CALIBRATION_SEPARATION_GAP",
            "calibration_known_probability_min": known_min,
            "calibration_unknown_probability_max": unknown_max,
        }

    candidates = sorted(set(float(value) for value in probs))
    best = (-1.0, -1.0, -1.0)
    threshold = 0.0
    for value in candidates:
        predicted = probs >= value
        known_recall = float(np.mean(predicted[flags == 1]))
        unknown_recall = float(np.mean(~predicted[flags == 0]))
        balanced = (known_recall + unknown_recall) / 2.0
        key = (1.0 if known_recall >= 0.95 else 0.0, balanced, unknown_recall)
        if key > best:
            best = key
            threshold = float(value)
    return threshold, {
        "threshold_strategy": "CONSTRAINED_BALANCED_RECALL_OVERLAP_FALLBACK",
        "calibration_known_probability_min": known_min,
        "calibration_unknown_probability_max": unknown_max,
    }

def train_open_set_artifact(
    *,
    base_artifact: Mapping[str, Any],
    training_frames: Mapping[str, Sequence[tuple[str, pd.DataFrame]]],
    known_calibration_frames: Mapping[str, Sequence[pd.DataFrame]],
    unknown_calibration_frames: Mapping[str, Sequence[pd.DataFrame]],
    artifact_version: str,
    random_state: int = 20260728,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for open-set calibration") from exc

    backend_payloads: dict[str, Any] = {}
    for backend, examples in training_frames.items():
        backend_model = base_artifact["backend_models"][backend]
        medians = np.asarray(backend_model["medians"], dtype=float)
        rows: list[np.ndarray] = []
        labels: list[str] = []
        for class_id, frame in examples:
            clean = sanitize_telemetry(frame)
            vector, _, _ = extract_catalog_features(clean, backend=backend)
            rows.append(np.where(np.isfinite(vector), vector, medians))
            labels.append(class_id)
        matrix = np.vstack(rows)
        global_scales = np.std(matrix, axis=0)
        positive = global_scales[global_scales > 1.0e-9]
        floor = float(np.quantile(positive, 0.1)) if positive.size else 1.0
        global_scales = np.where(global_scales > 1.0e-9, global_scales, floor)
        class_centers = {
            class_id: np.mean(matrix[np.asarray(labels, dtype=str) == class_id], axis=0)
            for class_id in sorted(set(labels))
        }
        skeleton = {
            "class_centers": class_centers,
            "global_scales": global_scales,
        }
        features: list[list[float]] = []
        known_flags: list[int] = []
        nonconformity_known: list[float] = []
        for frame in known_calibration_frames.get(backend, []):
            descriptor, _, _, _ = _descriptor(
                base_artifact=base_artifact, calibration=skeleton, backend=backend, frame=frame
            )
            features.append([
                descriptor["top_vote"], descriptor["vote_margin"], -descriptor["profile_distance"], descriptor["observed_ratio"]
            ])
            known_flags.append(1)
            nonconformity_known.append(
                descriptor["profile_distance"] + (1.0 - descriptor["top_vote"]) + max(0.0, 0.15 - descriptor["vote_margin"])
            )
        for frame in unknown_calibration_frames.get(backend, []):
            descriptor, _, _, _ = _descriptor(
                base_artifact=base_artifact, calibration=skeleton, backend=backend, frame=frame
            )
            features.append([
                descriptor["top_vote"], descriptor["vote_margin"], -descriptor["profile_distance"], descriptor["observed_ratio"]
            ])
            known_flags.append(0)
        calibrator = None
        threshold = 0.0
        if len(set(known_flags)) == 2:
            calibrator = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=2000)
            calibrator.fit(np.asarray(features, dtype=float), np.asarray(known_flags, dtype=int))
            probabilities = calibrator.predict_proba(np.asarray(features, dtype=float))[:, 1]
            threshold, threshold_evidence = _select_knownness_threshold(
                probabilities, np.asarray(known_flags, dtype=int)
            )
        else:
            threshold_evidence = {
                "threshold_strategy": "NO_TWO_CLASS_CALIBRATION",
                "calibration_known_probability_min": 0.0,
                "calibration_unknown_probability_max": 0.0,
            }
        backend_payloads[backend] = {
            **skeleton,
            "knownness_calibrator": calibrator,
            "knownness_threshold": threshold,
            **threshold_evidence,
            "known_nonconformity_scores": np.sort(np.asarray(nonconformity_known, dtype=float)),
            "minimum_observed_field_ratio": float(backend_model["minimum_observed_field_ratio"]),
            "calibration_count_known": int(sum(known_flags)),
            "calibration_count_unknown": int(len(known_flags) - sum(known_flags)),
        }
    return {
        "schema_version": "astrograph-sat-sim-open-set-model.v1",
        "artifact_version": artifact_version,
        "base_artifact": dict(base_artifact),
        "backend_calibration": backend_payloads,
        "score_semantics": (
            "CATALOG_MEMBERSHIP_CONFIDENCE_CALIBRATED_ON_BASILISK_LAB_DATA; "
            "MODEL_VOTE_SHARE_NOT_CALIBRATED_CLASS_PROBABILITY"
        ),
        "claim_boundary": (
            "Open-set screening is calibrated on limited Basilisk laboratory known/unknown examples. "
            "It is not flight confidence, causal proof, or a guarantee for arbitrary unknown faults."
        ),
    }


def save_open_set_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(artifact), target)


@lru_cache(maxsize=8)
def _load_open_set_artifact_cached(fingerprint: tuple[str, int, int, str, int, int]) -> dict[str, Any]:
    import joblib

    artifact_path, _ = resolve_release_artifact(
        fingerprint[0],
        manifest_schema="astrograph-sat-sim-open-set-model-manifest.v1",
        artifact_schema="astrograph-sat-sim-open-set-model.v1",
    )
    payload = joblib.load(artifact_path)
    if payload.get("schema_version") != "astrograph-sat-sim-open-set-model.v1":
        raise ValueError("unsupported open-set model schema")
    return payload


def load_open_set_artifact(path: str | Path) -> dict[str, Any]:
    return _load_open_set_artifact_cached(model_source_fingerprint(path))


class SatSimOpenSetDiagnosisEngine:
    engine_id = "astrograph-sat-sim-open-set-v1"

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("schema_version") != "astrograph-sat-sim-open-set-model.v1":
            raise ValueError("unsupported open-set artifact")
        self.artifact = artifact
        self.base_engine = SatSimCatalogDiagnosisEngine(artifact["base_artifact"])

    @classmethod
    def from_path(cls, path: str | Path) -> "SatSimOpenSetDiagnosisEngine":
        return cls(load_open_set_artifact(path))

    def diagnose_frame(self, *, case_id: str, backend: str, telemetry: pd.DataFrame) -> OpenSetDiagnosisResult:
        if backend not in self.artifact.get("backend_calibration", {}):
            return OpenSetDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=OpenSetDiagnosisStatus.OUT_OF_SCOPE,
                uncertainty_reason_codes=["UNSUPPORTED_BACKEND"],
                resolution_basis="BACKEND_SCOPE_GATE",
            )
        prohibited = [column for column in telemetry.columns if _is_prohibited(column)]
        if prohibited:
            return OpenSetDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=OpenSetDiagnosisStatus.DATA_ERROR,
                prohibited_columns_used=sorted(prohibited),
                uncertainty_reason_codes=["PROHIBITED_INPUT_COLUMN"],
                resolution_basis="INPUT_INTEGRITY_GATE",
            )
        calibration = self.artifact["backend_calibration"][backend]
        try:
            descriptor, _, _, clean = _descriptor(
                base_artifact=self.artifact["base_artifact"], calibration=calibration, backend=backend, frame=telemetry
            )
        except ValueError:
            return OpenSetDiagnosisResult(case_id=case_id, backend=backend, status=OpenSetDiagnosisStatus.DATA_ERROR)
        ratio = float(descriptor["observed_ratio"])
        if ratio < float(calibration["minimum_observed_field_ratio"]):
            return OpenSetDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=OpenSetDiagnosisStatus.INSUFFICIENT_EVIDENCE,
                observed_feature_ratio=ratio,
                uncertainty_reason_codes=["REQUIRED_TELEMETRY_COVERAGE_BELOW_MODEL_CONTRACT"],
                resolution_basis="EVIDENCE_COVERAGE_GATE",
            )
        feature = np.asarray([[
            descriptor["top_vote"], descriptor["vote_margin"], -descriptor["profile_distance"], descriptor["observed_ratio"]
        ]], dtype=float)
        calibrator = calibration.get("knownness_calibrator")
        membership = None if calibrator is None else float(calibrator.predict_proba(feature)[0, 1])
        nonconformity = (
            descriptor["profile_distance"] + (1.0 - descriptor["top_vote"]) + max(0.0, 0.15 - descriptor["vote_margin"])
        )
        calibration_scores = np.asarray(calibration["known_nonconformity_scores"], dtype=float)
        conformity_p = float((1 + np.sum(calibration_scores >= nonconformity)) / (len(calibration_scores) + 1))
        base_result = self.base_engine.diagnose_frame(case_id=case_id, backend=backend, telemetry=clean)
        rejected_by_membership = membership is not None and membership < float(calibration["knownness_threshold"])
        rejected_by_conformity = calibrator is None and conformity_p < 0.05
        if rejected_by_membership or rejected_by_conformity:
            return OpenSetDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=OpenSetDiagnosisStatus.UNKNOWN_FAULT,
                catalog_membership_confidence=membership,
                catalog_conformity_p_value=conformity_p,
                model_vote_share=float(descriptor["top_vote"]),
                vote_margin=float(descriptor["vote_margin"]),
                profile_distance=float(descriptor["profile_distance"]),
                resolution_basis="CALIBRATED_CATALOG_MEMBERSHIP_REJECTION",
                uncertainty_reason_codes=["OUTSIDE_CALIBRATED_KNOWN_CATALOG_ENVELOPE"],
                candidates=base_result.candidates,
                evidence=base_result.evidence,
                observed_feature_ratio=ratio,
            )
        if descriptor["vote_margin"] < 0.02 and (membership is None or membership < 0.75):
            return OpenSetDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=OpenSetDiagnosisStatus.AMBIGUOUS,
                catalog_membership_confidence=membership,
                catalog_conformity_p_value=conformity_p,
                model_vote_share=float(descriptor["top_vote"]),
                vote_margin=float(descriptor["vote_margin"]),
                profile_distance=float(descriptor["profile_distance"]),
                resolution_basis="CALIBRATED_AMBIGUITY_GATE",
                uncertainty_reason_codes=["TOP_CANDIDATES_NOT_SEPARATED"],
                candidates=base_result.candidates,
                evidence=base_result.evidence,
                observed_feature_ratio=ratio,
            )
        status = OpenSetDiagnosisStatus.NORMAL if base_result.status == "NORMAL" else OpenSetDiagnosisStatus.KNOWN_FAULT
        return OpenSetDiagnosisResult(
            case_id=case_id,
            backend=backend,
            status=status,
            conclusion_effect_id=base_result.conclusion_effect_id,
            catalog_membership_confidence=membership,
            catalog_conformity_p_value=conformity_p,
            diagnostic_confidence=None,
            model_vote_share=base_result.model_vote_share,
            vote_margin=base_result.vote_margin,
            profile_distance=float(descriptor["profile_distance"]),
            resolution_basis=base_result.resolution_basis,
            candidates=base_result.candidates,
            evidence=base_result.evidence,
            observed_feature_ratio=ratio,
        )

    def diagnose_path(self, *, case_id: str, backend: str, telemetry_path: str | Path) -> OpenSetDiagnosisResult:
        return self.diagnose_frame(
            case_id=case_id,
            backend=backend,
            telemetry=pd.read_csv(telemetry_path, low_memory=False),
        )


__all__ = [
    "OpenSetDiagnosisStatus",
    "OpenSetDiagnosisResult",
    "SatSimOpenSetDiagnosisEngine",
    "augment_derived_fields",
    "sanitize_telemetry",
    "train_open_set_artifact",
    "save_open_set_artifact",
    "load_open_set_artifact",
]
