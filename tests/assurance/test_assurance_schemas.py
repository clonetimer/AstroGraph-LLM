from __future__ import annotations

import json
from pathlib import Path

from astrograph.assurance import AdmissionReceipt, FeatureContract, ModelAdmissionManifest


ROOT = Path(__file__).resolve().parents[2]


def _expected(model, schema_id: str) -> dict:
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = schema_id
    schema["title"] = model.__name__
    ordered = {key: schema[key] for key in ("$schema", "$id", "title")}
    ordered.update({key: value for key, value in schema.items() if key not in ordered})
    return ordered


def test_feature_admission_schema_is_strict_and_synchronized() -> None:
    path = ROOT / "configs/assurance/feature_admission_contract.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == _expected(
        FeatureContract,
        "https://astrograph.local/schemas/feature-admission-contract-v1.json",
    )
    assert document["additionalProperties"] is False
    assert document["$defs"]["InputBinding"]["additionalProperties"] is False


def test_model_admission_schema_is_strict_and_synchronized() -> None:
    path = ROOT / "configs/assurance/model_admission_manifest.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == _expected(
        ModelAdmissionManifest,
        "https://astrograph.local/schemas/model-admission-manifest-v1.json",
    )
    assert document["additionalProperties"] is False


def test_model_admission_receipt_schema_is_strict_and_synchronized() -> None:
    path = ROOT / "configs/assurance/model_admission_receipt.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == _expected(
        AdmissionReceipt,
        "https://astrograph.local/schemas/model-admission-receipt-v1.json",
    )
    assert document["additionalProperties"] is False
