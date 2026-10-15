"""Dataset-label to runtime-ontology alignment audit.

Semantic mechanism coverage, strict registered fault mapping and executable
runtime evidence are deliberately distinct layers.  Only strict, executable
mappings may vote in fusion.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AlignmentStatus(str, Enum):
    EXACT_REGISTERED = "EXACT_REGISTERED"
    SEMANTIC_ONLY = "SEMANTIC_ONLY"
    UNMAPPED = "UNMAPPED"


class AlignmentRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_class_id: str
    mechanism_id: str | None = None
    registered_fault_id: str | None = None
    status: AlignmentStatus
    executable_runtime_evidence: bool = False
    fusion_eligible: bool = False

    @model_validator(mode="after")
    def guard_fusion(self) -> "AlignmentRow":
        should = self.status is AlignmentStatus.EXACT_REGISTERED and self.registered_fault_id is not None and self.executable_runtime_evidence
        if self.fusion_eligible != should:
            raise ValueError("fusion_eligible must equal exact registered mapping AND executable runtime evidence")
        return self


class OntologyAlignmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-ontology-alignment-report.v1"
    dataset_id: str
    rows: list[AlignmentRow]
    report_sha256: str | None = None

    @model_validator(mode="after")
    def unique_rows(self) -> "OntologyAlignmentReport":
        ids = [r.dataset_class_id for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate dataset class IDs")
        expected = self.compute_sha256()
        if self.report_sha256 is not None and self.report_sha256 != expected:
            raise ValueError("report_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def counts(self) -> dict[str, int]:
        return {
            "dataset": len(self.rows),
            "semantic": sum(r.mechanism_id is not None for r in self.rows),
            "strict_registered": sum(r.status is AlignmentStatus.EXACT_REGISTERED for r in self.rows),
            "executable": sum(r.executable_runtime_evidence for r in self.rows),
            "fusion_eligible": sum(r.fusion_eligible for r in self.rows),
        }


def build_alignment_report(
    *, dataset_id: str, dataset_classes: list[str], mechanism_bindings: Mapping[str, str],
    strict_mappings: Mapping[str, str], executable_fault_ids: set[str],
) -> OntologyAlignmentReport:
    rows = []
    for class_id in dataset_classes:
        mechanism = mechanism_bindings.get(class_id)
        registered = strict_mappings.get(class_id)
        if registered is not None:
            status = AlignmentStatus.EXACT_REGISTERED
        elif mechanism is not None:
            status = AlignmentStatus.SEMANTIC_ONLY
        else:
            status = AlignmentStatus.UNMAPPED
        executable = registered in executable_fault_ids if registered is not None else False
        rows.append(AlignmentRow(
            dataset_class_id=class_id, mechanism_id=mechanism, registered_fault_id=registered,
            status=status, executable_runtime_evidence=executable,
            fusion_eligible=(status is AlignmentStatus.EXACT_REGISTERED and executable),
        ))
    report = OntologyAlignmentReport(dataset_id=dataset_id, rows=rows)
    return report.model_copy(update={"report_sha256": report.compute_sha256()})


__all__ = ["AlignmentRow", "AlignmentStatus", "OntologyAlignmentReport", "build_alignment_report"]
