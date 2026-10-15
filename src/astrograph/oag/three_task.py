"""Project native three-task outputs into a compact, content-bound model context."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from astrograph.data_driven.models import DataDrivenPrediction
from astrograph.fusion import FaultRanking


class EvidenceProjectionReceipt(BaseModel):
    """Hash bindings emitted by the native paired-frame bridge, not a signature."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["astrograph-native-projection.v1"] = "astrograph-native-projection.v1"
    source_window_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nominal_window_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_input_hashes: dict[str, str]
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_start_s: float
    window_end_s: float
    construction: Literal["NATIVE_PAIRED_FRAME_PROJECTIONS"] = "NATIVE_PAIRED_FRAME_PROJECTIONS"


class ThreeTaskHeadTrace(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    classification_policy: Literal['hard_component', 'unmasked', 'soft_component']
    raw_class_probabilities: dict[str, float]
    conditioned_class_probabilities: dict[str, float] | None = None
    direct_component_probabilities: dict[str, float]
    derived_component_probabilities: dict[str, float]
    raw_top_class: str
    discarded_class_mass: float = Field(ge=0, le=1)
    class_to_runtime_fault: dict[str, str]
    score_semantics: Literal['UNCALIBRATED_MODEL_OUTPUT_SHARED_EVIDENCE']

    @model_validator(mode='after')
    def check_distributions(self):
        for values in (self.raw_class_probabilities, self.direct_component_probabilities,
                       self.derived_component_probabilities):
            if not values or any(not k.strip() or not 0 <= v <= 1 for k, v in values.items()) or abs(sum(values.values()) - 1) > .001:
                raise ValueError('head trace requires normalized finite distributions')
        if self.conditioned_class_probabilities is not None:
            values = self.conditioned_class_probabilities
            if set(values) != set(self.raw_class_probabilities) or any(not 0 <= v <= 1 for v in values.values()) or abs(sum(values.values()) - 1) > .001:
                raise ValueError("conditioned class distribution must match raw class universe")
        if self.raw_top_class not in self.raw_class_probabilities or self.raw_class_probabilities[self.raw_top_class] != max(self.raw_class_probabilities.values()):
            raise ValueError('raw top class differs from distribution')
        if not set(self.class_to_runtime_fault) <= set(self.raw_class_probabilities):
            raise ValueError('class map references unknown raw class')
        return self


class ThreeTaskResult(BaseModel):
    """Predictions are model output, never substituted for observed evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["astrograph-three-task-context.v1"] = "astrograph-three-task-context.v1"
    episode_id: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_start_s: float
    window_end_s: float
    detection_probability: float = Field(ge=0, le=1)
    detection_threshold: float = Field(ge=0, le=1)
    detected: bool
    component_probabilities: dict[str, float]
    predicted_component: str = Field(min_length=1)
    classification_ranking: FaultRanking
    evidence_projection: EvidenceProjectionReceipt | None = None
    head_trace: ThreeTaskHeadTrace | None = None
    interpretation: Literal["MODEL_OUTPUT_NOT_OBSERVATION"] = "MODEL_OUTPUT_NOT_OBSERVATION"

    @model_validator(mode="after")
    def validate_heads(self):
        if self.window_start_s >= self.window_end_s:
            raise ValueError("three-task window must have positive duration")
        if self.detected != (self.detection_probability >= self.detection_threshold):
            raise ValueError("detection decision differs from the registered threshold")
        p = self.component_probabilities
        if not p or any(not k.strip() or not 0 <= v <= 1 for k, v in p.items()):
            raise ValueError("invalid component probabilities")
        if abs(sum(p.values()) - 1) > .001:
            raise ValueError("component probabilities must sum to one")
        if self.predicted_component not in p or p[self.predicted_component] != max(p.values()):
            raise ValueError("localization must match a maximum-probability component")
        if self.evidence_projection is not None:
            p = self.evidence_projection
            if p.model_input_sha256 != self.input_data_sha256 or (p.window_start_s, p.window_end_s) != (self.window_start_s, self.window_end_s):
                raise ValueError("native projection does not match model input/window")
        r = self.classification_ranking
        if r.episode_id != self.episode_id or r.producer_id != self.model_id or r.producer_version != self.model_version:
            raise ValueError("three-task heads must share episode and model provenance")
        if r.source.value != "DATA_DRIVEN":
            raise ValueError("classification must come from the data-driven branch")
        return self

    @classmethod
    def from_prediction(cls, prediction: DataDrivenPrediction, *, registered_fault_ids: list[str],
                        start_s: float, end_s: float, evidence_projection=None) -> "ThreeTaskResult":
        """Read only declared head outputs from the existing provider adapter."""
        raw = prediction.metadata.get("raw_output_metadata", prediction.metadata)
        fields = ("detection_probability", "detection_threshold", "detected",
                  "component_probabilities", "predicted_component")
        missing = [key for key in fields if key not in raw]
        if missing:
            raise ValueError("provider has no complete three-task outputs: " + ", ".join(missing))
        return cls(episode_id=prediction.episode_id, prediction_id=prediction.prediction_id,
                   model_id=prediction.model_id, model_version=prediction.model_version,
                   input_data_sha256=prediction.input_data_sha256, evidence_projection=evidence_projection,
                   window_start_s=start_s, window_end_s=end_s,
                   classification_ranking=prediction.to_fault_ranking(registered_fault_ids),
                   head_trace=({**{key: raw['head_trace'][key] for key in (
                       'classification_policy', 'raw_class_probabilities',
                       'direct_component_probabilities', 'derived_component_probabilities',
                       'raw_top_class', 'discarded_class_mass', 'class_to_runtime_fault',
                       'score_semantics')}, 'conditioned_class_probabilities': raw.get('dataset_class_probabilities')} if raw.get('head_trace') is not None else None),
                   **{key: raw[key] for key in fields})

    def prompt_projection(self) -> dict:
        """Exclude opaque IDs, arbitrary provider metadata and model file paths."""
        return {
            "status": "AVAILABLE", "interpretation": self.interpretation,
            "input_data_sha256": self.input_data_sha256,
            "window": {"start_s": self.window_start_s, "end_s": self.window_end_s},
            "model": {"id": self.model_id, "version": self.model_version},
            "detection": {"score": self.detection_probability, "threshold": self.detection_threshold,
                          "detected": self.detected, "semantics": "MODEL_PROBABILITY_NOT_CALIBRATED_CONFIDENCE"},
            "localization": {"component": self.predicted_component,
                             "component_probabilities": self.component_probabilities},
            "classification": [{"fault_id": c.fault_id, "rank": c.rank, "score": c.score}
                               for c in self.classification_ranking.candidates],
            "classification_score_semantics": self.classification_ranking.score_semantics.value,
            "head_trace": self.head_trace.model_dump(mode='json') if self.head_trace is not None else None,
        }
