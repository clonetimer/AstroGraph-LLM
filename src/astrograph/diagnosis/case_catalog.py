from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FaultCaseCatalog(BaseModel):
    """Versioned presentation mapping for deterministic fault conclusions.

    Fault identifiers remain the diagnostic identity.  Case identifiers are
    output labels for regression/report compatibility and therefore belong in
    a reviewed data asset, not in the execution function.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-fault-case-catalog.v1"
    catalog_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    normal_case_id: str = Field(min_length=1)
    mappings: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_mappings(self) -> "FaultCaseCatalog":
        if any(not fault_id or not case_id for fault_id, case_id in self.mappings.items()):
            raise ValueError("fault and case identifiers must be nonempty")
        if len(self.mappings.values()) != len(set(self.mappings.values())):
            raise ValueError("case identifiers must be unique")
        if self.normal_case_id in self.mappings.values():
            raise ValueError("normal_case_id must not alias a registered fault case")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "FaultCaseCatalog":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8-sig"))

    @classmethod
    def packaged_default(cls) -> "FaultCaseCatalog":
        resource = files("astrograph.resources").joinpath("registered_fault_case_catalog.v1.json")
        return cls.model_validate_json(resource.read_text(encoding="utf-8"))

    def case_id_for_fault(self, fault_id: str) -> str:
        try:
            return self.mappings[fault_id]
        except KeyError as exc:
            raise ValueError(f"registered fault is missing from case catalog: {fault_id}") from exc


__all__ = ["FaultCaseCatalog"]
