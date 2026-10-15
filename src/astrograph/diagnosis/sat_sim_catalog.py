from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from astrograph.semantics import is_prohibited_field

from .model_release import model_source_fingerprint, resolve_release_artifact
from .sat_sim_paired import PROHIBITED_CONTAINS, PROHIBITED_PREFIXES, SIGNATURES


NORMAL_CLASS_ID = "NORMAL"
_FEATURE_STATS: tuple[str, ...] = (
    "mean", "std", "min", "max", "first", "last", "delta", "rms",
    "early_mean", "late_mean", "late_minus_early", "slope",
)
_EXTRA_FIELDS: Mapping[str, tuple[str, ...]] = {
    "composite": (
        "adcs_control_power_w",
        "battery_storage_j",
        "net_power_w",
        "attitude_error_deg",
        "attitude_error_norm",
        "rate_error_norm_rad_s",
    ),
    "unified": (),
}


class CatalogDiagnosisStatus(str):
    NORMAL = "NORMAL"
    DIAGNOSED = "DIAGNOSED"
    AMBIGUOUS = "AMBIGUOUS"
    DATA_ERROR = "DATA_ERROR"


class CatalogEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float
    predicted_class_median: float
    normal_class_median: float | None = None
    evidence_strength: float
    interpretation: str


class CatalogCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str
    vote_share: float = Field(ge=0.0, le=1.0)


class CatalogDiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-sat-sim-catalog-diagnosis.v1"
    case_id: str
    backend: str
    status: str
    conclusion_effect_id: str | None = None
    model_vote_share: float = Field(default=0.0, ge=0.0, le=1.0)
    vote_margin: float = 0.0
    resolution_basis: str = "NO_RESOLUTION"
    candidates: list[CatalogCandidate] = Field(default_factory=list)
    evidence: list[CatalogEvidence] = Field(default_factory=list)
    observed_feature_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    prohibited_columns_used: list[str] = Field(default_factory=list)
    labels_used: bool = False
    truth_used: bool = False
    claim_boundary: str = (
        "Known-catalog single-trace Basilisk laboratory diagnosis. Model vote shares are not "
        "calibrated probabilities and do not establish open-set, flight, or cross-platform validity."
    )


def _allowed_fields() -> dict[str, tuple[str, ...]]:
    fields: dict[str, set[str]] = {"composite": set(), "unified": set()}
    for signature in SIGNATURES:
        fields.setdefault(signature.backend, set())
        for rule in signature.rules:
            fields[signature.backend].update(rule.aliases)
    for backend, extras in _EXTRA_FIELDS.items():
        fields.setdefault(backend, set()).update(extras)
    return {backend: tuple(sorted(values)) for backend, values in fields.items()}


ALLOWED_FIELDS = _allowed_fields()


def _is_prohibited(column: str) -> bool:
    return is_prohibited_field(column)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)


def _series_features(series: pd.Series) -> tuple[float, ...]:
    if series.empty:
        return (math.nan,) * len(_FEATURE_STATS)
    values = series.to_numpy(dtype=float)
    size = len(values)
    quarter = max(1, size // 4)
    x = np.linspace(0.0, 1.0, size)
    slope = float(np.polyfit(x, values, 1)[0]) if size > 1 else 0.0
    early = float(np.mean(values[:quarter]))
    late = float(np.mean(values[-quarter:]))
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=0)),
        float(np.min(values)),
        float(np.max(values)),
        float(values[0]),
        float(values[-1]),
        float(values[-1] - values[0]),
        float(math.sqrt(float(np.mean(values * values)))),
        early,
        late,
        late - early,
        slope,
    )


def extract_catalog_features(frame: pd.DataFrame, *, backend: str) -> tuple[np.ndarray, list[str], float]:
    if backend not in ALLOWED_FIELDS:
        raise ValueError(f"unsupported backend: {backend}")
    prohibited = [column for column in frame.columns if _is_prohibited(column)]
    if prohibited:
        raise ValueError("prohibited columns: " + ",".join(sorted(prohibited)))
    values: list[float] = []
    names: list[str] = []
    observed_fields = 0
    fields = ALLOWED_FIELDS[backend]
    for column in fields:
        series = _numeric_series(frame, column)
        if not series.empty:
            observed_fields += 1
        values.extend(_series_features(series))
        names.extend(f"{column}__{stat}" for stat in _FEATURE_STATS)
    ratio = observed_fields / len(fields) if fields else 0.0
    return np.asarray(values, dtype=float), names, ratio


def _profile(values: np.ndarray, feature_names: Sequence[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    if values.size == 0:
        return result
    for index, name in enumerate(feature_names):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        result[name] = {
            "median": float(median),
            "iqr": float(max(q75 - q25, 1.0e-12)),
        }
    return result


def _class_profiles(matrix: np.ndarray, labels: Sequence[str], feature_names: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
    label_array = np.asarray(labels, dtype=str)
    return {
        class_id: _profile(matrix[label_array == class_id], feature_names)
        for class_id in sorted(set(label_array.tolist()))
    }


def _resolver_metrics(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    required = ("attitude_error_deg", "attitude_error_norm", "rate_error_norm_rad_s")
    if any(column not in frame.columns for column in required):
        return None, None
    attitude = _numeric_series(frame, "attitude_error_deg")
    attitude_norm = _numeric_series(frame, "attitude_error_norm")
    rate = _numeric_series(frame, "rate_error_norm_rad_s")
    if min(len(attitude), len(attitude_norm), len(rate)) < 2:
        return None, None
    rate_mean = float(rate.mean())
    attitude_delta = float(attitude.iloc[-1] - attitude.iloc[0])
    attitude_norm_delta = float(attitude_norm.iloc[-1] - attitude_norm.iloc[0])
    if abs(rate_mean) < 1.0e-12 or abs(attitude_norm_delta) < 1.0e-12:
        return None, None
    return attitude_delta / rate_mean, rate_mean / attitude_norm_delta


def _train_resolver_thresholds(examples: Sequence[tuple[str, pd.DataFrame]]) -> dict[str, float]:
    bearing: list[tuple[float, float]] = []
    compound: list[tuple[float, float]] = []
    for class_id, frame in examples:
        metrics = _resolver_metrics(frame)
        if metrics[0] is None or metrics[1] is None:
            continue
        if class_id == "adcs_rw_bearing_seizure":
            bearing.append((float(metrics[0]), float(metrics[1])))
        elif class_id == "adcs_rw_friction_and_sensor_noise":
            compound.append((float(metrics[0]), float(metrics[1])))
    if len(bearing) < 2 or len(compound) < 2:
        return {}
    a_threshold = (min(value[0] for value in bearing) + max(value[0] for value in compound)) / 2.0
    b_threshold = (max(value[1] for value in bearing) + min(value[1] for value in compound)) / 2.0
    if not (max(value[0] for value in compound) < min(value[0] for value in bearing)):
        return {}
    if not (max(value[1] for value in bearing) < min(value[1] for value in compound)):
        return {}
    return {
        "attitude_delta_over_rate_mean_max": float(a_threshold),
        "rate_mean_over_attitude_norm_delta_min": float(b_threshold),
    }


def train_catalog_artifact(
    *,
    training_frames: Mapping[str, Sequence[tuple[str, pd.DataFrame]]],
    model_version: str,
    random_state: int = 20260728,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as exc:  # pragma: no cover - actionable runtime boundary
        raise RuntimeError("scikit-learn is required for catalog model training") from exc

    backend_payloads: dict[str, Any] = {}
    for backend, examples in training_frames.items():
        if backend not in ALLOWED_FIELDS:
            raise ValueError(f"unsupported backend: {backend}")
        rows: list[np.ndarray] = []
        labels: list[str] = []
        feature_names: list[str] | None = None
        effect_rows: list[np.ndarray] = []
        effect_labels: list[str] = []
        resolver_examples: list[tuple[str, pd.DataFrame]] = []
        observed_ratios: list[float] = []
        for class_id, frame in examples:
            vector, names, observed_ratio = extract_catalog_features(frame, backend=backend)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("feature schema changed within backend")
            rows.append(vector)
            labels.append(class_id)
            observed_ratios.append(observed_ratio)
            if class_id != NORMAL_CLASS_ID:
                effect_rows.append(vector)
                effect_labels.append(class_id)
                resolver_examples.append((class_id, frame))
        if not rows or feature_names is None:
            raise ValueError(f"no training rows for backend: {backend}")
        matrix = np.vstack(rows)
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        active_mask = np.std(matrix, axis=0) > 1.0e-12
        if not np.any(active_mask):
            raise ValueError(f"no active features for backend: {backend}")
        model = ExtraTreesClassifier(
            n_estimators=1000,
            max_features=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(matrix[:, active_mask], labels)

        effect_matrix = np.vstack(effect_rows)
        effect_matrix = np.where(np.isfinite(effect_matrix), effect_matrix, medians)
        effect_active_mask = np.std(effect_matrix, axis=0) > 1.0e-12
        effect_model = ExtraTreesClassifier(
            n_estimators=1000,
            max_features=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        effect_model.fit(effect_matrix[:, effect_active_mask], effect_labels)

        backend_payloads[backend] = {
            "model": model,
            "effect_model": effect_model,
            "feature_names": feature_names,
            "medians": medians,
            "active_mask": active_mask,
            "effect_active_mask": effect_active_mask,
            "class_profiles": _class_profiles(matrix, labels, feature_names),
            "resolver_thresholds": _train_resolver_thresholds(resolver_examples) if backend == "composite" else {},
            "minimum_observed_field_ratio": float(max(0.5, min(observed_ratios) - 0.05)),
        }
    return {
        "schema_version": "astrograph-sat-sim-catalog-model.v1",
        "model_version": model_version,
        "engine_id": "astrograph-sat-sim-catalog-extra-trees-v1",
        "backend_models": backend_payloads,
        "normal_class_id": NORMAL_CLASS_ID,
        "score_semantics": "ENSEMBLE_VOTE_SHARE_NOT_CALIBRATED_PROBABILITY",
        "claim_boundary": (
            "Known-catalog single-trace Basilisk laboratory model trained on external labels. "
            "No open-set, flight, cross-platform, or calibrated-confidence claim."
        ),
    }


def save_catalog_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("joblib is required to save catalog models") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(artifact), target)


@lru_cache(maxsize=8)
def _load_catalog_artifact_cached(fingerprint: tuple[str, int, int, str, int, int]) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("joblib is required to load catalog models") from exc
    artifact_path, _ = resolve_release_artifact(
        fingerprint[0],
        manifest_schema="astrograph-sat-sim-catalog-release.v1",
        artifact_schema="astrograph-sat-sim-catalog-model.v1",
    )
    payload = joblib.load(artifact_path)
    if payload.get("schema_version") != "astrograph-sat-sim-catalog-model.v1":
        raise ValueError("unsupported catalog model schema")
    return payload


def load_catalog_artifact(path: str | Path) -> dict[str, Any]:
    return _load_catalog_artifact_cached(model_source_fingerprint(path))



def _battery_capacity_loss_observed(telemetry: pd.DataFrame) -> bool:
    """Return true when telemetry contains a real, sustained capacity reduction.

    The runtime capacity ratio is physical state telemetry, not a label.  A
    reduction below 90% with a non-trivial within-trace transition separates
    an actual capacity-loss episode from a normal battery trace.
    """
    if "eps.battery_capacity_ratio" not in telemetry.columns:
        return False
    values = pd.to_numeric(telemetry["eps.battery_capacity_ratio"], errors="coerce").dropna()
    if values.empty:
        return False
    return float(values.min()) <= 0.90 and float(values.max() - values.min()) >= 0.05

class SatSimCatalogDiagnosisEngine:
    """Single-trace known-catalog diagnosis for validated Sat-Agent Basilisk telemetry.

    Inference receives one telemetry frame and backend identity.  It never receives the
    paired nominal trace, case truth, effect identifier, directory name, or Sat-Agent
    result.json.  Training labels are external and are recorded in the model manifest.
    """

    engine_id = "astrograph-sat-sim-catalog-extra-trees-v1"

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("schema_version") != "astrograph-sat-sim-catalog-model.v1":
            raise ValueError("unsupported catalog artifact")
        self.artifact = artifact

    @classmethod
    def from_path(cls, path: str | Path) -> "SatSimCatalogDiagnosisEngine":
        return cls(load_catalog_artifact(path))

    def diagnose_frame(self, *, case_id: str, backend: str, telemetry: pd.DataFrame) -> CatalogDiagnosisResult:
        prohibited = [column for column in telemetry.columns if _is_prohibited(column)]
        if prohibited:
            return CatalogDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=CatalogDiagnosisStatus.DATA_ERROR,
                prohibited_columns_used=sorted(prohibited),
            )
        backend_model = self.artifact.get("backend_models", {}).get(backend)
        if backend_model is None:
            return CatalogDiagnosisResult(case_id=case_id, backend=backend, status=CatalogDiagnosisStatus.DATA_ERROR)
        try:
            vector, feature_names, observed_ratio = extract_catalog_features(telemetry, backend=backend)
        except ValueError as exc:
            return CatalogDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=CatalogDiagnosisStatus.DATA_ERROR,
                prohibited_columns_used=[str(exc)] if "prohibited" in str(exc) else [],
            )
        if feature_names != list(backend_model["feature_names"]):
            return CatalogDiagnosisResult(case_id=case_id, backend=backend, status=CatalogDiagnosisStatus.DATA_ERROR)
        if observed_ratio < float(backend_model["minimum_observed_field_ratio"]):
            return CatalogDiagnosisResult(
                case_id=case_id,
                backend=backend,
                status=CatalogDiagnosisStatus.DATA_ERROR,
                observed_feature_ratio=observed_ratio,
            )
        medians = np.asarray(backend_model["medians"], dtype=float)
        vector = np.where(np.isfinite(vector), vector, medians)
        active_mask = np.asarray(backend_model["active_mask"], dtype=bool)
        model = backend_model["model"]
        probabilities = np.asarray(model.predict_proba([vector[active_mask]])[0], dtype=float)
        classes = np.asarray(model.classes_, dtype=str)
        order = np.argsort(probabilities)[::-1]
        predicted = str(classes[order[0]])
        top_score = float(probabilities[order[0]])
        second_score = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        resolution_basis = "CATALOG_ENSEMBLE_VOTE"

        effect_model = backend_model["effect_model"]
        effect_mask = np.asarray(backend_model["effect_active_mask"], dtype=bool)
        effect_probabilities = np.asarray(effect_model.predict_proba([vector[effect_mask]])[0], dtype=float)
        effect_classes = np.asarray(effect_model.classes_, dtype=str)
        effect_order = np.argsort(effect_probabilities)[::-1]
        effect_prediction = str(effect_classes[effect_order[0]])
        effect_score = float(effect_probabilities[effect_order[0]])

        resolver = dict(backend_model.get("resolver_thresholds") or {})
        compound_id = "adcs_rw_friction_and_sensor_noise"
        bearing_id = "adcs_rw_bearing_seizure"
        catalog_top_pair = {str(classes[index]) for index in order[:2]}
        if (
            backend == "composite"
            and resolver
            and predicted == bearing_id
            and effect_prediction == bearing_id
            and catalog_top_pair == {bearing_id, compound_id}
        ):
            metric_a, metric_b = _resolver_metrics(telemetry)
            if (
                metric_a is not None
                and metric_b is not None
                and metric_a < float(resolver["attitude_delta_over_rate_mean_max"])
                and metric_b > float(resolver["rate_mean_over_attitude_norm_delta_min"])
            ):
                predicted = compound_id
                selected = int(np.where(classes == compound_id)[0][0])
                top_score = float(probabilities[selected])
                second_score = float(max(probabilities[index] for index in range(len(classes)) if index != selected))
                resolution_basis = "COMPOUND_DYNAMIC_RESOLVER"

        battery_capacity_id = "eps_battery_capacity_loss"
        if (
            backend == "unified"
            and predicted == NORMAL_CLASS_ID
            and effect_prediction == battery_capacity_id
            and catalog_top_pair == {NORMAL_CLASS_ID, battery_capacity_id}
            and _battery_capacity_loss_observed(telemetry)
        ):
            predicted = battery_capacity_id
            selected = int(np.where(classes == battery_capacity_id)[0][0])
            top_score = float(probabilities[selected])
            second_score = float(max(probabilities[index] for index in range(len(classes)) if index != selected))
            resolution_basis = "BATTERY_CAPACITY_PHYSICAL_RESOLVER"

        status = CatalogDiagnosisStatus.NORMAL if predicted == NORMAL_CLASS_ID else CatalogDiagnosisStatus.DIAGNOSED
        evidence = self._evidence(
            backend_model=backend_model,
            feature_names=feature_names,
            vector=vector,
            predicted=predicted,
        )
        candidates = [
            CatalogCandidate(class_id=str(classes[index]), vote_share=float(probabilities[index]))
            for index in order[:5]
        ]
        if resolution_basis in {"COMPOUND_DYNAMIC_RESOLVER", "BATTERY_CAPACITY_PHYSICAL_RESOLVER"}:
            candidates.insert(0, CatalogCandidate(class_id=predicted, vote_share=top_score))
            unique: list[CatalogCandidate] = []
            seen: set[str] = set()
            for candidate in candidates:
                if candidate.class_id not in seen:
                    unique.append(candidate)
                    seen.add(candidate.class_id)
            candidates = unique[:5]
        return CatalogDiagnosisResult(
            case_id=case_id,
            backend=backend,
            status=status,
            conclusion_effect_id=None if predicted == NORMAL_CLASS_ID else predicted,
            model_vote_share=top_score,
            vote_margin=top_score - second_score,
            resolution_basis=resolution_basis,
            candidates=candidates,
            evidence=evidence,
            observed_feature_ratio=observed_ratio,
            labels_used=False,
            truth_used=False,
        )

    def diagnose_path(self, *, case_id: str, backend: str, telemetry_path: str | Path) -> CatalogDiagnosisResult:
        telemetry = pd.read_csv(telemetry_path, low_memory=False)
        return self.diagnose_frame(case_id=case_id, backend=backend, telemetry=telemetry)

    @staticmethod
    def _evidence(
        *,
        backend_model: Mapping[str, Any],
        feature_names: Sequence[str],
        vector: np.ndarray,
        predicted: str,
    ) -> list[CatalogEvidence]:
        profiles = backend_model.get("class_profiles", {})
        predicted_profile = profiles.get(predicted, {})
        normal_profile = profiles.get(NORMAL_CLASS_ID, {})
        ranked: list[tuple[float, CatalogEvidence]] = []
        for index, name in enumerate(feature_names):
            if name not in predicted_profile or not np.isfinite(vector[index]):
                continue
            class_median = float(predicted_profile[name]["median"])
            class_iqr = float(predicted_profile[name]["iqr"])
            normal = normal_profile.get(name)
            class_distance = abs(float(vector[index]) - class_median) / max(class_iqr, 1.0e-12)
            if normal is None:
                normal_distance = 0.0
                normal_median = None
            else:
                normal_median = float(normal["median"])
                normal_iqr = float(normal["iqr"])
                normal_distance = abs(float(vector[index]) - normal_median) / max(normal_iqr, 1.0e-12)
            strength = float(normal_distance - class_distance)
            if predicted == NORMAL_CLASS_ID:
                strength = float(-class_distance)
            ranked.append((
                strength,
                CatalogEvidence(
                    feature=name,
                    value=float(vector[index]),
                    predicted_class_median=class_median,
                    normal_class_median=normal_median,
                    evidence_strength=strength,
                    interpretation=(
                        "Observed aggregate telemetry is closer to the predicted catalog profile than "
                        "to the learned normal profile; this is model evidence, not causal proof."
                    ),
                ),
            ))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:6]]


__all__ = [
    "NORMAL_CLASS_ID",
    "CatalogDiagnosisStatus",
    "CatalogEvidence",
    "CatalogCandidate",
    "CatalogDiagnosisResult",
    "SatSimCatalogDiagnosisEngine",
    "extract_catalog_features",
    "train_catalog_artifact",
    "save_catalog_artifact",
    "load_catalog_artifact",
    "_battery_capacity_loss_observed",
]
