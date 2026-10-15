"""Validated, configuration-driven mechanism feature families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def load_feature_bank(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "astrograph-mechanism-feature-banks.v1":
        raise ValueError("unsupported mechanism feature bank schema")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("mechanism feature bank requires datasets")
    for dataset_id, dataset in datasets.items():
        families = dataset.get("families")
        profiles = dataset.get("profiles")
        if not isinstance(families, dict) or not isinstance(profiles, dict):
            raise ValueError(f"dataset {dataset_id} requires families and profiles")
        for family, columns in families.items():
            if not isinstance(columns, list) or not columns or any(
                not isinstance(column, str) or not column for column in columns
            ):
                raise ValueError(f"invalid feature family {dataset_id}/{family}")
            if len(columns) != len(set(columns)):
                raise ValueError(f"duplicate column in feature family {dataset_id}/{family}")
        for profile, family_names in profiles.items():
            if not isinstance(family_names, list) or any(name not in families for name in family_names):
                raise ValueError(f"invalid feature profile {dataset_id}/{profile}")
    return payload


def profile_columns(payload: dict[str, Any], dataset_id: str, profile: str) -> list[str]:
    try:
        dataset = payload["datasets"][dataset_id]
        family_names = dataset["profiles"][profile]
    except KeyError as exc:
        raise ValueError(f"unknown mechanism feature profile: {dataset_id}/{profile}") from exc
    result: list[str] = []
    for family in family_names:
        for column in dataset["families"][family]:
            if column not in result:
                result.append(column)
    return result


def compose_columns(base: Sequence[str], additions: Sequence[str]) -> list[str]:
    result = list(base)
    if len(result) != len(set(result)):
        raise ValueError("base feature columns contain duplicates")
    for column in additions:
        if column not in result:
            result.append(column)
    return result


def require_available(columns: Sequence[str], available: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(f"configured mechanism feature columns are missing: {missing}")
