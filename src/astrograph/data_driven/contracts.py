from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer


class DataDrivenTaskType(str, Enum):
    """Task types supported by the data-driven plugin boundary.

    Only tasks with FAULT_RANKING capability can feed BiRRF directly. Other tasks
    may still emit EvidenceFact-compatible signals through a project adapter.
    """

    MULTICLASS_FAULT_CLASSIFICATION = "MULTICLASS_FAULT_CLASSIFICATION"
    MULTILABEL_FAULT_RANKING = "MULTILABEL_FAULT_RANKING"
    ANOMALY_ATTRIBUTION = "ANOMALY_ATTRIBUTION"
    PROGNOSTICS = "PROGNOSTICS"
    REPRESENTATION = "REPRESENTATION"


class ModelRuntime(str, Enum):
    SCIKIT_LEARN = "SCIKIT_LEARN"
    XGBOOST_COMPATIBLE = "XGBOOST_COMPATIBLE"
    LIGHTGBM_COMPATIBLE = "LIGHTGBM_COMPATIBLE"
    PYTORCH = "PYTORCH"
    TENSORFLOW = "TENSORFLOW"
    ONNX_RUNTIME = "ONNX_RUNTIME"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    CUSTOM_CALLABLE = "CUSTOM_CALLABLE"


class ModelOutputSemantics(str, Enum):
    PROBABILITIES = "PROBABILITIES"
    LOGITS = "LOGITS"
    NONNEGATIVE_SCORES = "NONNEGATIVE_SCORES"
    ANOMALY_ATTRIBUTION = "ANOMALY_ATTRIBUTION"


class ModelCapability(str, Enum):
    FAULT_RANKING = "FAULT_RANKING"
    NORMAL_CLASS = "NORMAL_CLASS"
    OOD_SCORE = "OOD_SCORE"
    CALIBRATED_PROBABILITY = "CALIBRATED_PROBABILITY"
    UNCERTAINTY = "UNCERTAINTY"
    TEMPORAL_CONTEXT = "TEMPORAL_CONTEXT"
    MULTI_FAULT = "MULTI_FAULT"


class ProbabilityCalibrationReceipt(BaseModel):
    """Content-bound evidence that one model's probability calibration was evaluated/frozen.

    The receipt does not assert operational or flight qualification. It binds the
    calibration method and held-out calibration material so a bare calibration ID
    cannot be treated as sufficient evidence of calibrated probability semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-probability-calibration-receipt.v1"
    calibration_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    calibration_dataset_id: str = Field(min_length=1)
    calibration_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_method: str = Field(min_length=1)
    calibration_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    calibration_sample_count: int = Field(ge=1)
    calibration_group_count: int = Field(ge=1)
    claim_boundary: str = (
        "Content-bound calibration evidence only. It does not establish transfer under distribution shift, "
        "selective-risk guarantees, HIL/flight qualification, or fault-model correctness."
    )

    @property
    def receipt_sha256(self) -> str:
        raw = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class ProbabilityCalibrationQualificationReceipt(BaseModel):
    """R19 content-bound calibration qualification after external validation.

    This is stricter than the legacy v1 ProbabilityCalibrationReceipt. It binds
    the fit receipt, confirmatory/external validation report and the development
    threshold selected without using confirmatory or external groups.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-probability-calibration-qualification-receipt.v1"
    evaluation_task_id: str | None = Field(default=None, min_length=1)
    label_semantics_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_universe_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    calibration_dataset_id: str = Field(min_length=1)
    calibration_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_method: str = Field(min_length=1)
    fit_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    external_validation_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    calibration_sample_count: int = Field(ge=1)
    calibration_group_count: int = Field(ge=1)
    development_group_count: int = Field(ge=1)
    confirmatory_group_count: int = Field(ge=1)
    external_validation_dataset_id: str = Field(min_length=1)
    external_validation_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_validation_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_validation_group_count: int = Field(ge=1)
    selective_threshold_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_status: str
    claim_boundary: str = (
        "Engineering calibration qualification after group-disjoint fitting, development selection, confirmatory and "
        "external validation. It does not establish arbitrary-shift calibration, HIL/flight qualification, or release safety."
    )

    @model_validator(mode="after")
    def valid_status(self) -> "ProbabilityCalibrationQualificationReceipt":
        identity = (self.evaluation_task_id, self.label_semantics_sha256, self.target_universe_sha256)
        if any(v is not None for v in identity) and not all(v is not None for v in identity):
            raise ValueError("TASK_IDENTITY_REQUIRED")
        if self.qualification_status not in {
            "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW",
            "BLOCKED_BY_CALIBRATION_GUARDRAIL",
        }:
            raise ValueError("unsupported calibration qualification status")
        return self

    @model_serializer(mode="wrap")
    def serialize_task_identity(self, handler):
        payload = handler(self)
        # Keep archived R19 receipt bytes/hashes compatible when task is absent.
        for key in ("evaluation_task_id", "label_semantics_sha256", "target_universe_sha256"):
            if payload.get(key) is None: payload.pop(key, None)
        return payload

    def require_external_task(self, external: dict) -> None:
        from astrograph.evaluation.external_dataset_contract import require_same_task
        require_same_task(self.model_dump(mode="json"), external)

    @property
    def receipt_sha256(self) -> str:
        raw = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class DataDrivenModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-data-driven-model-manifest.v1"
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    task_type: DataDrivenTaskType
    runtime: ModelRuntime
    output_semantics: ModelOutputSemantics
    capabilities: list[ModelCapability] = Field(default_factory=list)
    supported_fault_ids: list[str] = Field(default_factory=list)
    normal_class_id: str = "NORMAL"
    feature_schema_ids: list[str] = Field(default_factory=list)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    calibration_id: str | None = None
    calibration_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_dataset_ids: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "Model plugin output is an engineering candidate signal. It cannot become a formal diagnosis "
        "without graph evidence, deterministic validation, and Decision Governor."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capabilities", "supported_fault_ids", "feature_schema_ids")
    @classmethod
    def unique_values(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("manifest list values must be unique")
        return value

    @model_validator(mode="after")
    def validate_direct_ranking_contract(self) -> "DataDrivenModelManifest":
        direct_tasks = {
            DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
            DataDrivenTaskType.MULTILABEL_FAULT_RANKING,
            DataDrivenTaskType.ANOMALY_ATTRIBUTION,
        }
        if self.task_type in direct_tasks and ModelCapability.FAULT_RANKING not in self.capabilities:
            raise ValueError("ranking-capable task must declare FAULT_RANKING capability")
        if ModelCapability.CALIBRATED_PROBABILITY in self.capabilities and not self.calibration_id:
            raise ValueError("calibrated probability capability requires calibration_id")
        if self.calibration_receipt_sha256 is not None:
            if ModelCapability.CALIBRATED_PROBABILITY not in self.capabilities or not self.calibration_id:
                raise ValueError(
                    "calibration_receipt_sha256 requires calibrated probability capability and calibration_id"
                )
        if self.normal_class_id in set(self.supported_fault_ids):
            raise ValueError("normal_class_id must not be listed as a registered fault")
        return self

    @property
    def manifest_id(self) -> str:
        return f"{self.model_id}@{self.model_version}"

    @property
    def feeds_birrf_directly(self) -> bool:
        return ModelCapability.FAULT_RANKING in self.capabilities




class ObservationScope(BaseModel):
    """Trusted observation identity kept outside the provider-visible request.

    The scope is used to derive evidence/correlation lineage.  It is intentionally
    not included in the request passed to a strict model provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-observation-scope.v1"
    scope_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    window_start_s: float | None = None
    window_end_s: float | None = None
    source_data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    trajectory_group_id: str | None = None

    @model_validator(mode="after")
    def valid_window(self) -> "ObservationScope":
        if (self.window_start_s is None) != (self.window_end_s is None):
            raise ValueError("observation scope window requires both start and end")
        if (
            self.window_start_s is not None
            and self.window_end_s is not None
            and self.window_end_s < self.window_start_s
        ):
            raise ValueError("observation scope window_end_s must be >= window_start_s")
        return self


class ModelInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = "astrograph-model-inference-request.v1"
    episode_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    feature_schema_id: str = Field(min_length=1)
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registered_fault_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    observation_scope: ObservationScope | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("registered_fault_ids")
    @classmethod
    def unique_registered_faults(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("registered fault IDs must be unique")
        return value


class RawModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-raw-model-output.v1"
    episode_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    output_semantics: ModelOutputSemantics
    class_values: dict[str, float] = Field(default_factory=dict)
    anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)
    fault_attribution: dict[str, float] = Field(default_factory=dict)
    out_of_distribution_score: float | None = Field(default=None, ge=0.0)
    uncertainty: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_values", "fault_attribution")
    @classmethod
    def finite_values(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(float(item)) for item in value.values()):
            raise ValueError("model outputs must be finite")
        return value

    @model_validator(mode="after")
    def validate_semantics_payload(self) -> "RawModelOutput":
        if self.output_semantics == ModelOutputSemantics.ANOMALY_ATTRIBUTION:
            if self.anomaly_score is None:
                raise ValueError("anomaly attribution output requires anomaly_score")
            if self.anomaly_score > 0.0 and not self.fault_attribution:
                raise ValueError("positive anomaly score requires fault_attribution")
        elif not self.class_values:
            raise ValueError("classification/ranking output requires class_values")
        return self
