from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from astrograph.semantics.models import TelemetryFieldSemantics, TelemetrySemanticsValidation


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


class TelemetrySemanticsRegistry:
    """Validated registry joining the production telemetry dictionary and physical semantics.

    This registry is intentionally read-only. It keeps direct telemetry, synthetic
    onboard estimates, validation truth, and labels separated according to the
    supplied production protocols.
    """

    registry_id = "phm-project-a-telemetry-semantics-v1.1"
    registry_version = "1.1"

    def __init__(self, protocol_root: str | Path) -> None:
        self.protocol_root = Path(protocol_root).resolve()
        self.dataset_contract_path = self.protocol_root / "dataset_contract.json"
        self.dictionary_path = self.protocol_root / "telemetry_dictionary.csv"
        self.physical_csv_path = self.protocol_root / "telemetry_physical_semantics_v1.csv"
        self.physical_contract_path = self.protocol_root / "physical_semantics_contract.json"

        self.dataset_contract = json.loads(self.dataset_contract_path.read_text(encoding="utf-8-sig"))
        self.physical_contract = json.loads(self.physical_contract_path.read_text(encoding="utf-8-sig"))
        self.registry_version = str(self.physical_contract.get("protocol_version", self.registry_version))
        self.registry_id = f"phm-project-a-telemetry-semantics-v{self.registry_version}"
        self._dictionary_rows = _read_csv(self.dictionary_path)
        self._physical_rows = _read_csv(self.physical_csv_path)
        self.validation = self._validate()
        if not self.validation.passed:
            raise ValueError(f"telemetry semantics protocol validation failed: {self.validation.model_dump()}")
        dictionary = {row["field"]: row for row in self._dictionary_rows}
        physical = {row["field"]: row for row in self._physical_rows}
        self._fields = {
            field: TelemetryFieldSemantics(
                field=field,
                chinese_name=dictionary[field].get("chinese_name", ""),
                component_instance=physical[field]["component_instance"],
                observable_property=physical[field]["observable_property"],
                unit=physical[field]["unit"],
                reference_frame=physical[field]["reference_frame"],
                axis=physical[field]["axis"],
                sign_convention=physical[field]["sign_convention"],
                sample_period_s=float(physical[field]["sample_period_s"]),
                valid_range=physical[field].get("valid_range", ""),
                status_code_definition=physical[field].get("status_code_definition", ""),
                source=physical[field]["source"],
                source_class=dictionary[field].get("source_class", ""),
                dataset_role=dictionary[field].get("dataset_role", ""),
                measurement_status=dictionary[field].get("measurement_status", ""),
                missing_value_rule=dictionary[field].get("missing_value_rule", ""),
                definition_status=physical[field].get("definition_status", ""),
                source_evidence=physical[field].get("source_evidence", ""),
                limitations=physical[field].get("limitations", ""),
            )
            for field in sorted(dictionary)
        }

    def _validate(self) -> TelemetrySemanticsValidation:
        expected = [str(item) for item in self.dataset_contract.get("raw_telemetry_fields", [])]
        expected_set = set(expected)
        dictionary_fields = [row.get("field", "") for row in self._dictionary_rows]
        physical_fields = [row.get("field", "") for row in self._physical_rows]
        dictionary_set = set(dictionary_fields)
        physical_set = set(physical_fields)
        duplicates = sorted(
            {field for field in dictionary_fields if dictionary_fields.count(field) > 1}
            | {field for field in physical_fields if physical_fields.count(field) > 1}
        )
        dictionary_periods = {row["field"]: float(row["sample_period_s"]) for row in self._dictionary_rows}
        physical_periods = {row["field"]: float(row["sample_period_s"]) for row in self._physical_rows}
        mismatches = [
            {
                "field": field,
                "dictionary_sample_period_s": dictionary_periods[field],
                "physical_sample_period_s": physical_periods[field],
            }
            for field in sorted(dictionary_set & physical_set)
            if dictionary_periods[field] != physical_periods[field]
        ]
        field_sets_equal = expected_set == dictionary_set == physical_set
        sample_periods_equal = not mismatches
        hashes = {
            "dataset_contract.json": sha256_path(self.dataset_contract_path),
            "telemetry_dictionary.csv": sha256_path(self.dictionary_path),
            "telemetry_physical_semantics_v1.csv": sha256_path(self.physical_csv_path),
            "physical_semantics_contract.json": sha256_path(self.physical_contract_path),
        }
        return TelemetrySemanticsValidation(
            field_count=len(dictionary_set),
            expected_field_count=int(self.physical_contract.get("field_count", len(expected_set))),
            field_sets_equal=field_sets_equal,
            sample_periods_equal=sample_periods_equal,
            duplicate_fields=duplicates,
            missing_from_dictionary=sorted(expected_set - dictionary_set),
            missing_from_physical_semantics=sorted(expected_set - physical_set),
            extra_in_dictionary=sorted(dictionary_set - expected_set),
            extra_in_physical_semantics=sorted(physical_set - expected_set),
            mismatched_sample_periods=mismatches,
            passed=(
                field_sets_equal
                and sample_periods_equal
                and not duplicates
                and len(dictionary_set) == int(self.physical_contract.get("field_count", -1))
            ),
            protocol_hashes=hashes,
            claim_boundary=str(self.physical_contract.get("claim_boundary", "")),
        )

    def get(self, field: str) -> TelemetryFieldSemantics:
        try:
            return self._fields[field]
        except KeyError as exc:
            raise KeyError(f"unknown telemetry field in semantics registry: {field}") from exc

    def list_fields(self) -> list[TelemetryFieldSemantics]:
        return [self._fields[field] for field in sorted(self._fields)]

    def require_fields(self, fields: Iterable[str]) -> list[TelemetryFieldSemantics]:
        return [self.get(field) for field in fields]

    def model_input_fields(self) -> list[str]:
        return [
            item.field
            for item in self.list_fields()
            if item.dataset_role == "raw_model_input"
        ]

    @property
    def event_window_semantics(self) -> dict:
        return dict(self.physical_contract.get("event_window", {}))

    @property
    def rw_installation(self) -> dict:
        return dict(self.physical_contract.get("rw_installation", {}))
