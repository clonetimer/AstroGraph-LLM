from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.evidence.executable import stable_id
from astrograph.fusion import FaultRanking, RankedFault, RankingSource, ScoreSemantics


class ClassProbability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


ClassValueSemantics = Literal[
    "CALIBRATED_CLASS_PROBABILITY",
    "UNCALIBRATED_CLASS_PROBABILITY",
    "NORMALIZED_RANKING_MASS",
]


class DataDrivenPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-data-driven-prediction.v1"
    prediction_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_manifest_id: str | None = None
    model_family: str = "UNSPECIFIED"
    task_type: str = "MULTICLASS_FAULT_CLASSIFICATION"
    output_semantics: str = "PROBABILITIES"
    class_value_semantics: ClassValueSemantics | None = None
    classes: list[ClassProbability]
    normal_class_id: str = "NORMAL"
    feature_schema_id: str = Field(min_length=1)
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibrated_probabilities: bool = False
    calibration_id: str | None = None
    out_of_distribution_score: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Data-driven class values are candidate-model outputs. A normalized class value is not a "
        "probability unless class_value_semantics says probability, and it is not a calibrated "
        "probability unless calibrated_probabilities=true with a calibration_id. No class value or "
        "rank-fusion score can bypass graph evidence, deterministic validation, or Decision Governor."
    )

    @field_validator("classes")
    @classmethod
    def unique_class_ids_and_ranks(cls, value: list[ClassProbability]) -> list[ClassProbability]:
        ids = [item.class_id for item in value]
        ranks = [item.rank for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("class IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("class ranks must be unique")
        return value

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "DataDrivenPrediction":
        total = sum(item.probability for item in self.classes)
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"class probabilities must sum to 1 within tolerance, got {total}")
        if self.calibrated_probabilities and not self.calibration_id:
            raise ValueError("calibrated_probabilities=True requires calibration_id")
        if not self.calibrated_probabilities and self.calibration_id is not None:
            raise ValueError("calibration_id requires calibrated_probabilities=True")
        if self.calibrated_probabilities and self.output_semantics != "PROBABILITIES":
            raise ValueError("calibrated probabilities require PROBABILITIES output semantics")
        inferred: ClassValueSemantics
        if self.calibrated_probabilities:
            inferred = "CALIBRATED_CLASS_PROBABILITY"
        elif self.output_semantics == "PROBABILITIES":
            inferred = "UNCALIBRATED_CLASS_PROBABILITY"
        else:
            inferred = "NORMALIZED_RANKING_MASS"
        if self.class_value_semantics is None:
            self.class_value_semantics = inferred
        elif self.class_value_semantics != inferred:
            raise ValueError(
                "class_value_semantics is inconsistent with output_semantics/calibration status"
            )
        return self

    def to_fault_ranking(self, registered_fault_ids: list[str]) -> FaultRanking:
        registered = set(registered_fault_ids)
        unknown = {item.class_id for item in self.classes} - registered - {self.normal_class_id}
        if unknown:
            raise ValueError(f"prediction contains unregistered classes: {sorted(unknown)}")
        fault_classes = [item for item in sorted(self.classes, key=lambda item: item.rank) if item.class_id in registered]
        raw_lineage = self.metadata.get("evidence_lineage_by_fault", {})
        if raw_lineage is None:
            raw_lineage = {}
        if not isinstance(raw_lineage, dict):
            raise ValueError("metadata.evidence_lineage_by_fault must be an object")

        lineage_by_fault: dict[str, dict[str, Any]] = {}
        for fault_id, raw_entry in raw_lineage.items():
            if fault_id not in registered:
                raise ValueError(
                    "metadata.evidence_lineage_by_fault contains an unregistered fault: "
                    f"{fault_id}"
                )
            if raw_entry is None:
                raw_entry = {}
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    "metadata.evidence_lineage_by_fault entries must be objects: "
                    f"{fault_id}"
                )
            allowed = {"source_evidence_ids", "correlation_group_ids", "lineage_complete"}
            unknown_keys = set(raw_entry) - allowed
            if unknown_keys:
                raise ValueError(
                    "metadata.evidence_lineage_by_fault contains unsupported keys for "
                    f"{fault_id}: {sorted(unknown_keys)}"
                )
            source_ids = raw_entry.get("source_evidence_ids", [])
            group_ids = raw_entry.get("correlation_group_ids", [])
            complete = raw_entry.get("lineage_complete", False)
            if not isinstance(source_ids, list) or not all(
                isinstance(value, str) and value.strip() for value in source_ids
            ):
                raise ValueError(f"source_evidence_ids must be a list of nonempty strings: {fault_id}")
            if not isinstance(group_ids, list) or not all(
                isinstance(value, str) and value.strip() for value in group_ids
            ):
                raise ValueError(f"correlation_group_ids must be a list of nonempty strings: {fault_id}")
            if not isinstance(complete, bool):
                raise ValueError(f"lineage_complete must be boolean: {fault_id}")
            if complete and not source_ids:
                raise ValueError(
                    f"lineage_complete=True requires source_evidence_ids: {fault_id}"
                )
            lineage_by_fault[str(fault_id)] = {
                "source_evidence_ids": source_ids,
                "correlation_group_ids": group_ids,
                "lineage_complete": complete,
            }

        candidates = []
        for index, item in enumerate(fault_classes, start=1):
            lineage = lineage_by_fault.get(item.class_id, {})
            candidates.append(
                RankedFault(
                    fault_id=item.class_id,
                    rank=index,
                    score=item.probability,
                    source=RankingSource.DATA_DRIVEN,
                    source_evidence_ids=list(lineage.get("source_evidence_ids", [])),
                    correlation_group_ids=list(lineage.get("correlation_group_ids", [])),
                    lineage_complete=bool(lineage.get("lineage_complete", False)),
                    contributes_to_fusion=item.probability > 0.0,
                    metadata={"original_class_rank": item.rank},
                )
            )
        return FaultRanking(
            ranking_id=stable_id(
                "RANK",
                {
                    "prediction_id": self.prediction_id,
                    "registered_fault_ids": sorted(registered),
                    "ordered_faults": [item.fault_id for item in candidates],
                },
            ),
            episode_id=self.episode_id,
            source=RankingSource.DATA_DRIVEN,
            score_semantics={
                "CALIBRATED_CLASS_PROBABILITY": ScoreSemantics.CALIBRATED_CLASS_PROBABILITY,
                "UNCALIBRATED_CLASS_PROBABILITY": ScoreSemantics.UNCALIBRATED_CLASS_PROBABILITY,
                "NORMALIZED_RANKING_MASS": ScoreSemantics.NORMALIZED_RANKING_MASS,
            }[self.class_value_semantics or "UNCALIBRATED_CLASS_PROBABILITY"],
            calibration_id=self.calibration_id if self.calibrated_probabilities else None,
            candidates=candidates,
            producer_id=self.model_id,
            producer_version=self.model_version,
            metadata={
                "prediction_id": self.prediction_id,
                "feature_schema_id": self.feature_schema_id,
                "calibrated_probabilities": self.calibrated_probabilities,
                "calibration_id": self.calibration_id,
                "model_manifest_id": self.model_manifest_id,
                "model_family": self.model_family,
                "task_type": self.task_type,
                "output_semantics": self.output_semantics,
                "class_value_semantics": self.class_value_semantics,
            },
        )

    @property
    def normal_score(self) -> float | None:
        item = next((value for value in self.classes if value.class_id == self.normal_class_id), None)
        return item.probability if item else None

    @property
    def normal_probability(self) -> float | None:
        """Legacy numeric accessor. Consult class_value_semantics before treating it as probability."""
        return self.normal_score

    @property
    def normal_rank(self) -> int | None:
        item = next((value for value in self.classes if value.class_id == self.normal_class_id), None)
        return item.rank if item else None


class DataDrivenRankingProvider(Protocol):
    model_id: str
    model_version: str

    def predict(self, payload: dict[str, Any]) -> DataDrivenPrediction: ...


class StaticDataDrivenRankingProvider:
    """Deterministic provider for replay, fixtures, and externally computed predictions."""

    def __init__(self, prediction: DataDrivenPrediction):
        self.prediction = prediction
        self.model_id = prediction.model_id
        self.model_version = prediction.model_version

    def predict(self, payload: dict[str, Any]) -> DataDrivenPrediction:
        expected_episode = payload.get("episode_id")
        if expected_episode and str(expected_episode) != self.prediction.episode_id:
            raise ValueError("static prediction episode ID does not match request")
        return self.prediction
