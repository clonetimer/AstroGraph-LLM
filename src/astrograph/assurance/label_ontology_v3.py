"""Versioned fault-label semantics for AstroGraph scientific evaluation.

A class name alone is not sufficient evidence for temporal localisation or root
cause attribution.  This module records the mechanism, component, manifestation,
time support, causal role, provenance, and uncertainty separately.  Unknown
fields remain explicitly ``UNRESOLVED``; they are never inferred from a label
name by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

UNRESOLVED = "UNRESOLVED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")


def _semantic_text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be blank; use {UNRESOLVED} when unknown")
    return text


def is_resolved(value: Optional[str]) -> bool:
    return value is not None and value.strip().upper() != UNRESOLVED


class Multiplicity(str, Enum):
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"
    UNKNOWN = "UNKNOWN"


class CausalRole(str, Enum):
    INITIATING_EVENT = "INITIATING_EVENT"
    ROOT_CAUSE = "ROOT_CAUSE"
    INTERMEDIATE_CAUSE = "INTERMEDIATE_CAUSE"
    SYMPTOM = "SYMPTOM"
    DOWNSTREAM_EFFECT = "DOWNSTREAM_EFFECT"
    UNKNOWN = "UNKNOWN"


class LabelSourceKind(str, Enum):
    SIMULATOR_DEFINITION = "SIMULATOR_DEFINITION"
    EXPERIMENT_MANIFEST = "EXPERIMENT_MANIFEST"
    PUBLIC_DATASET_LABEL = "PUBLIC_DATASET_LABEL"
    EXPERT_ANNOTATION = "EXPERT_ANNOTATION"
    MAINTENANCE_RECORD = "MAINTENANCE_RECORD"
    OTHER = "OTHER"


class ReviewStatus(str, Enum):
    SOURCE_PINNED = "SOURCE_PINNED"
    DOMAIN_REVIEWED = "DOMAIN_REVIEWED"
    UNREVIEWED = "UNREVIEWED"


@dataclass(frozen=True)
class TimeInterval:
    start: float
    end: float
    time_unit: str
    reference: str
    schema_version: str = "astrograph-label-time-interval.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("time interval bounds must be finite")
        if self.end < self.start:
            raise ValueError("time interval end must be greater than or equal to start")
        if not self.time_unit.strip() or not self.reference.strip():
            raise ValueError("time_unit and reference must not be blank")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimeInterval":
        allowed = {"schema_version", "start", "end", "time_unit", "reference"}
        _strict_keys(data, allowed, "TimeInterval")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-label-time-interval.v1")),
            start=float(data["start"]),
            end=float(data["end"]),
            time_unit=str(data["time_unit"]),
            reference=str(data["reference"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "start": self.start,
            "end": self.end,
            "time_unit": self.time_unit,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class LabelSource:
    source_id: str
    source_kind: LabelSourceKind
    source_sha256: str
    source_location: str
    review_status: ReviewStatus
    schema_version: str = "astrograph-label-source.v1"

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.source_location.strip():
            raise ValueError("source_id and source_location must not be blank")
        if not _SHA256_RE.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelSource":
        allowed = {
            "schema_version", "source_id", "source_kind", "source_sha256",
            "source_location", "review_status",
        }
        _strict_keys(data, allowed, "LabelSource")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-label-source.v1")),
            source_id=str(data["source_id"]),
            source_kind=LabelSourceKind(data["source_kind"]),
            source_sha256=str(data["source_sha256"]),
            source_location=str(data["source_location"]),
            review_status=ReviewStatus(data["review_status"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "source_sha256": self.source_sha256,
            "source_location": self.source_location,
            "review_status": self.review_status.value,
        }


@dataclass(frozen=True)
class TaskSupport:
    supported: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "errors": list(self.errors), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class LabelOntologyRecord:
    label_id: str
    fault_universe_id: str
    mechanism: str
    component: str
    manifestation: str
    injection_location: str
    onset_interval: Optional[TimeInterval]
    onset_uncertainty: Optional[float]
    active_interval: Optional[TimeInterval]
    severity: str
    causal_role: CausalRole
    upstream_causes: tuple[str, ...]
    downstream_effects: tuple[str, ...]
    multiplicity: Multiplicity
    label_source: LabelSource
    label_confidence: Optional[float]
    supporting_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = "astrograph-label-ontology.v3"

    def __post_init__(self) -> None:
        for field_name in (
            "label_id", "fault_universe_id", "mechanism", "component", "manifestation",
            "injection_location", "severity",
        ):
            value = _semantic_text(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        if self.onset_uncertainty is not None:
            if not math.isfinite(self.onset_uncertainty) or self.onset_uncertainty < 0:
                raise ValueError("onset_uncertainty must be finite and non-negative")
        if self.label_confidence is not None:
            if not math.isfinite(self.label_confidence) or not 0.0 <= self.label_confidence <= 1.0:
                raise ValueError("label_confidence must be in [0, 1]")
        if self.onset_interval and self.active_interval:
            if self.active_interval.end < self.onset_interval.start:
                raise ValueError("active_interval cannot end before onset_interval starts")
        for collection_name in ("upstream_causes", "downstream_effects", "supporting_evidence_ids"):
            values = tuple(_semantic_text(x, collection_name) for x in getattr(self, collection_name))
            if len(set(values)) != len(values):
                raise ValueError(f"{collection_name} contains duplicates")
            object.__setattr__(self, collection_name, values)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelOntologyRecord":
        allowed = {
            "schema_version", "label_id", "fault_universe_id", "mechanism", "component",
            "manifestation", "injection_location", "onset_interval", "onset_uncertainty",
            "active_interval", "severity", "causal_role", "upstream_causes",
            "downstream_effects", "multiplicity", "label_source", "label_confidence",
            "supporting_evidence_ids", "notes",
        }
        _strict_keys(data, allowed, "LabelOntologyRecord")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-label-ontology.v3")),
            label_id=str(data["label_id"]),
            fault_universe_id=str(data["fault_universe_id"]),
            mechanism=str(data.get("mechanism", UNRESOLVED)),
            component=str(data.get("component", UNRESOLVED)),
            manifestation=str(data.get("manifestation", UNRESOLVED)),
            injection_location=str(data.get("injection_location", UNRESOLVED)),
            onset_interval=None if data.get("onset_interval") is None else TimeInterval.from_dict(data["onset_interval"]),
            onset_uncertainty=None if data.get("onset_uncertainty") is None else float(data["onset_uncertainty"]),
            active_interval=None if data.get("active_interval") is None else TimeInterval.from_dict(data["active_interval"]),
            severity=str(data.get("severity", UNRESOLVED)),
            causal_role=CausalRole(data.get("causal_role", CausalRole.UNKNOWN.value)),
            upstream_causes=tuple(str(x) for x in data.get("upstream_causes", ())),
            downstream_effects=tuple(str(x) for x in data.get("downstream_effects", ())),
            multiplicity=Multiplicity(data.get("multiplicity", Multiplicity.UNKNOWN.value)),
            label_source=LabelSource.from_dict(data["label_source"]),
            label_confidence=None if data.get("label_confidence") is None else float(data["label_confidence"]),
            supporting_evidence_ids=tuple(str(x) for x in data.get("supporting_evidence_ids", ())),
            notes=tuple(str(x) for x in data.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "fault_universe_id": self.fault_universe_id,
            "mechanism": self.mechanism,
            "component": self.component,
            "manifestation": self.manifestation,
            "injection_location": self.injection_location,
            "onset_interval": None if self.onset_interval is None else self.onset_interval.to_dict(),
            "onset_uncertainty": self.onset_uncertainty,
            "active_interval": None if self.active_interval is None else self.active_interval.to_dict(),
            "severity": self.severity,
            "causal_role": self.causal_role.value,
            "upstream_causes": list(self.upstream_causes),
            "downstream_effects": list(self.downstream_effects),
            "multiplicity": self.multiplicity.value,
            "label_source": self.label_source.to_dict(),
            "label_confidence": self.label_confidence,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "notes": list(self.notes),
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def support_for_task(self, task_type: str) -> TaskSupport:
        errors: list[str] = []
        warnings: list[str] = []

        if task_type == "ANOMALY_DETECTION":
            if not is_resolved(self.manifestation) and not is_resolved(self.mechanism):
                errors.append("anomaly detection requires a resolved manifestation or mechanism")
        elif task_type == "TEMPORAL_LOCALIZATION":
            if self.onset_interval is None and self.active_interval is None:
                errors.append("temporal localization requires an onset_interval or active_interval")
            if self.onset_uncertainty is None:
                warnings.append("onset uncertainty is unresolved and must be reported as such")
        elif task_type == "FAULT_IDENTIFICATION":
            if not (is_resolved(self.mechanism) or is_resolved(self.component)):
                errors.append("fault identification requires a resolved mechanism or component")
        elif task_type == "ROOT_CAUSE_ATTRIBUTION":
            if self.causal_role not in {CausalRole.INITIATING_EVENT, CausalRole.ROOT_CAUSE}:
                errors.append("root-cause attribution requires causal_role ROOT_CAUSE or INITIATING_EVENT")
            resolved_core = sum(
                int(is_resolved(value))
                for value in (self.mechanism, self.component, self.injection_location)
            )
            if resolved_core < 2:
                errors.append("root-cause attribution requires at least two resolved fields among mechanism, component, and injection_location")
            if self.label_source.review_status is ReviewStatus.UNREVIEWED:
                errors.append("root-cause attribution requires a source-pinned or domain-reviewed label source")
            if self.label_confidence is None:
                errors.append("root-cause attribution requires an explicit label_confidence")
        elif task_type in {"CALIBRATION", "SELECTIVE_PREDICTION"}:
            if not (is_resolved(self.manifestation) or is_resolved(self.mechanism) or is_resolved(self.component)):
                errors.append(f"{task_type} requires a resolved target semantics")
        elif task_type == "EXPLANATION_FIDELITY":
            if not self.supporting_evidence_ids:
                errors.append("explanation fidelity requires supporting_evidence_ids")
        else:
            errors.append(f"unsupported task type: {task_type}")

        if self.multiplicity is Multiplicity.UNKNOWN:
            warnings.append("single-versus-multiple-fault status is unresolved")
        return TaskSupport(not errors, tuple(errors), tuple(warnings))


@dataclass(frozen=True)
class LabelOntologyBundle:
    fault_universe_id: str
    records: tuple[LabelOntologyRecord, ...]
    schema_version: str = "astrograph-label-ontology-bundle.v3"

    def __post_init__(self) -> None:
        if not self.fault_universe_id.strip():
            raise ValueError("fault_universe_id must not be blank")
        ids = [record.label_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("label_id values must be unique")
        wrong = sorted(record.label_id for record in self.records if record.fault_universe_id != self.fault_universe_id)
        if wrong:
            raise ValueError(f"records bind a different fault universe: {wrong}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelOntologyBundle":
        allowed = {"schema_version", "fault_universe_id", "records"}
        _strict_keys(data, allowed, "LabelOntologyBundle")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-label-ontology-bundle.v3")),
            fault_universe_id=str(data["fault_universe_id"]),
            records=tuple(LabelOntologyRecord.from_dict(x) for x in data["records"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fault_universe_id": self.fault_universe_id,
            "records": [record.to_dict() for record in sorted(self.records, key=lambda x: x.label_id)],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def by_sha256(self) -> dict[str, LabelOntologyRecord]:
        return {record.sha256: record for record in self.records}
