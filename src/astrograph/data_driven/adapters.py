from __future__ import annotations

import math
from typing import Iterable

from astrograph.evidence.executable import stable_id

from .contracts import (
    DataDrivenModelManifest,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    RawModelOutput,
)
from .models import ClassProbability, DataDrivenPrediction


def rebind_prediction_episode(
    prediction: DataDrivenPrediction, episode_id: str
) -> DataDrivenPrediction:
    """Bind an externally prepared prediction to an episode and recompute its stable ID."""

    prediction_id = stable_id(
        "PRED",
        {
            "episode_id": episode_id,
            "manifest_id": prediction.model_manifest_id,
            "classes": [
                (item.class_id, round(item.probability, 12)) for item in prediction.classes
            ],
            "input_data_sha256": prediction.input_data_sha256,
        },
    )
    return prediction.model_copy(
        update={"episode_id": episode_id, "prediction_id": prediction_id}
    )


def _softmax(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    exponentials = {key: math.exp(float(value) - maximum) for key, value in values.items()}
    total = sum(exponentials.values())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("softmax normalization failed")
    return {key: value / total for key, value in exponentials.items()}


def _normalize_nonnegative(values: dict[str, float]) -> dict[str, float]:
    if any(float(value) < 0.0 for value in values.values()):
        raise ValueError("nonnegative score adapter received a negative value")
    total = sum(float(value) for value in values.values())
    if total <= 0.0:
        raise ValueError("nonnegative score mass must be positive")
    return {key: float(value) / total for key, value in values.items()}


def _validate_probability_mass(values: dict[str, float]) -> dict[str, float]:
    if any(float(value) < 0.0 or float(value) > 1.0 for value in values.values()):
        raise ValueError("probabilities must be within [0, 1]")
    total = sum(float(value) for value in values.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"probabilities must sum to 1 within tolerance, got {total}")
    return {key: float(value) / total for key, value in values.items()}


def _anomaly_attribution_probabilities(
    *,
    anomaly_score: float,
    attribution: dict[str, float],
    normal_class_id: str,
) -> dict[str, float]:
    anomaly_score = float(anomaly_score)
    if not 0.0 <= anomaly_score <= 1.0:
        raise ValueError("anomaly score must be within [0, 1]")
    if anomaly_score == 0.0:
        return {normal_class_id: 1.0, **{key: 0.0 for key in attribution}}
    normalized = _normalize_nonnegative(attribution)
    result = {key: anomaly_score * value for key, value in normalized.items()}
    result[normal_class_id] = 1.0 - anomaly_score
    return result


def adapt_raw_model_output(
    manifest: DataDrivenModelManifest,
    request: ModelInferenceRequest,
    raw: RawModelOutput,
) -> DataDrivenPrediction:
    """Convert a model-family-specific output into AstroGraph's common ranking protocol."""

    if not manifest.feeds_birrf_directly:
        raise ValueError(f"model {manifest.manifest_id} does not declare FAULT_RANKING capability")
    if raw.model_id != manifest.model_id or raw.model_version != manifest.model_version:
        raise ValueError("raw output model identity does not match manifest")
    if raw.episode_id != request.episode_id:
        raise ValueError("raw output episode ID does not match inference request")
    if raw.output_semantics != manifest.output_semantics:
        raise ValueError("raw output semantics does not match model manifest")
    if request.model_id != manifest.model_id:
        raise ValueError("inference request model ID does not match manifest")
    if request.model_version and request.model_version != manifest.model_version:
        raise ValueError("inference request model version does not match manifest")
    if manifest.feature_schema_ids and request.feature_schema_id not in set(manifest.feature_schema_ids):
        raise ValueError("inference request feature schema is not supported by model manifest")

    if raw.output_semantics == ModelOutputSemantics.PROBABILITIES:
        probabilities = _validate_probability_mass(raw.class_values)
        normalization = "IDENTITY_PROBABILITY_VALIDATION"
    elif raw.output_semantics == ModelOutputSemantics.LOGITS:
        probabilities = _softmax(raw.class_values)
        normalization = "SOFTMAX_FROM_LOGITS"
    elif raw.output_semantics == ModelOutputSemantics.NONNEGATIVE_SCORES:
        probabilities = _normalize_nonnegative(raw.class_values)
        normalization = "MASS_NORMALIZATION_FROM_NONNEGATIVE_SCORES"
    elif raw.output_semantics == ModelOutputSemantics.ANOMALY_ATTRIBUTION:
        probabilities = _anomaly_attribution_probabilities(
            anomaly_score=float(raw.anomaly_score or 0.0),
            attribution=raw.fault_attribution,
            normal_class_id=manifest.normal_class_id,
        )
        normalization = "ANOMALY_MASS_TIMES_FAULT_ATTRIBUTION"
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(f"unsupported output semantics: {raw.output_semantics}")

    allowed_faults = set(manifest.supported_fault_ids)
    registered = set(request.registered_fault_ids)
    output_classes = set(probabilities)
    unexpected = output_classes - allowed_faults - {manifest.normal_class_id}
    if unexpected:
        raise ValueError(f"model output contains classes outside manifest: {sorted(unexpected)}")
    if registered:
        unregistered = output_classes - registered - {manifest.normal_class_id}
        if unregistered:
            raise ValueError(f"model output contains unregistered fault IDs: {sorted(unregistered)}")

    # Include absent manifest faults with zero mass so downstream rank fusion sees a complete,
    # deterministic class universe. The normal class is also guaranteed to exist.
    for fault_id in manifest.supported_fault_ids:
        if not registered or fault_id in registered:
            probabilities.setdefault(fault_id, 0.0)
    probabilities.setdefault(manifest.normal_class_id, 0.0)

    ordered = sorted(probabilities.items(), key=lambda item: (-float(item[1]), item[0]))
    classes = [
        ClassProbability(class_id=class_id, probability=float(probability), rank=index)
        for index, (class_id, probability) in enumerate(ordered, start=1)
    ]
    calibrated = (
        raw.output_semantics == ModelOutputSemantics.PROBABILITIES
        and ModelCapability.CALIBRATED_PROBABILITY in manifest.capabilities
    )
    return DataDrivenPrediction(
        prediction_id=stable_id(
            "PRED",
            {
                "episode_id": request.episode_id,
                "manifest_id": manifest.manifest_id,
                "classes": [(item.class_id, round(item.probability, 12)) for item in classes],
                "input_data_sha256": request.input_data_sha256,
            },
        ),
        episode_id=request.episode_id,
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        model_manifest_id=manifest.manifest_id,
        model_family=manifest.runtime.value,
        task_type=manifest.task_type.value,
        output_semantics=manifest.output_semantics.value,
        class_value_semantics=(
            "CALIBRATED_CLASS_PROBABILITY"
            if calibrated
            else (
                "UNCALIBRATED_CLASS_PROBABILITY"
                if raw.output_semantics == ModelOutputSemantics.PROBABILITIES
                else "NORMALIZED_RANKING_MASS"
            )
        ),
        classes=classes,
        normal_class_id=manifest.normal_class_id,
        feature_schema_id=request.feature_schema_id,
        input_data_sha256=request.input_data_sha256,
        calibrated_probabilities=calibrated,
        calibration_id=manifest.calibration_id,
        out_of_distribution_score=raw.out_of_distribution_score,
        metadata={
            "normalization": normalization,
            "provider_runtime": manifest.runtime.value,
            "model_capabilities": [item.value for item in manifest.capabilities],
            "raw_output_metadata": raw.metadata,
            # Arbitrary request metadata is intentionally not copied into the
            # prediction: it may contain case identity, truth labels, or injection
            # timing.  A typed observation scope is retained as trusted provenance.
            "observation_scope": (
                request.observation_scope.model_dump(mode="json")
                if request.observation_scope is not None
                else None
            ),
            "uncertainty": raw.uncertainty,
            "class_value_semantics": (
                "CALIBRATED_CLASS_PROBABILITY"
                if calibrated
                else (
                    "UNCALIBRATED_CLASS_PROBABILITY"
                    if raw.output_semantics == ModelOutputSemantics.PROBABILITIES
                    else "NORMALIZED_RANKING_MASS"
                )
            ),
        },
    )


def ranked_class_ids(prediction: DataDrivenPrediction, *, include_normal: bool = False) -> list[str]:
    return [
        item.class_id
        for item in sorted(prediction.classes, key=lambda value: value.rank)
        if include_normal or item.class_id != prediction.normal_class_id
    ]
