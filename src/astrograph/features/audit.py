from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


_IDENTIFIER_COLUMNS = {"run_id", "case_id", "split", "paired_nominal_run_id"}
_BANNED_FEATURE_TOKENS = ("label", "truth", "case_id", "run_id", "record_id", "paired_nominal")
_ALLOWED_STATS = {"mean", "std", "max_abs", "mean_abs", "last", "delta"}
_ALLOWED_CHANNEL_PREFIXES = ("sensor.", "estimate.", "command.", "state.", "status.", "counter.", "data.", "adcs.", "payload.", "comm.")


class FeatureFileAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    path: str
    row_count: int
    feature_count: int
    identifier_columns: list[str]
    feature_columns: list[str]
    duplicate_columns: list[str] = Field(default_factory=list)
    non_numeric_features: list[str] = Field(default_factory=list)
    non_finite_features: list[str] = Field(default_factory=list)
    constant_features: list[str] = Field(default_factory=list)
    leakage_suspects: list[str] = Field(default_factory=list)
    invalid_feature_names: list[str] = Field(default_factory=list)
    feature_family_counts: dict[str, int] = Field(default_factory=dict)
    status: str


class UnifiedFeatureAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-unified-feature-audit.v1"
    audit_version: str = "feature-audit-0.4.3"
    status: str
    d1: FeatureFileAudit
    d3: FeatureFileAudit
    shared_feature_count: int
    d3_subset_of_d1: bool
    d3_features_missing_from_d1: list[str] = Field(default_factory=list)
    d1_only_feature_count: int
    claim_boundary: str = (
        "Engineering feature-protocol and leakage audit only; it does not establish model validity or flight performance."
    )


def _family(column: str) -> str:
    channel = column.rsplit("__", 1)[0]
    return channel.split(".", 1)[0] if "." in channel else "unknown"


def audit_feature_file(path: str | Path, *, dataset_id: str) -> FeatureFileAudit:
    source = Path(path)
    frame = pd.read_csv(source)
    duplicate_columns = sorted([name for name, count in Counter(frame.columns).items() if count > 1])
    identifiers = [name for name in frame.columns if name in _IDENTIFIER_COLUMNS]
    features = [name for name in frame.columns if name not in _IDENTIFIER_COLUMNS]
    non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(frame[name])]
    non_finite: list[str] = []
    constant: list[str] = []
    leakage: list[str] = []
    invalid_names: list[str] = []
    for name in features:
        lowered = name.lower()
        if any(token in lowered for token in _BANNED_FEATURE_TOKENS):
            leakage.append(name)
        match = re.match(r"^(.+)__([a-z0-9_]+)$", name)
        if not match or match.group(2) not in _ALLOWED_STATS or not match.group(1).startswith(_ALLOWED_CHANNEL_PREFIXES):
            invalid_names.append(name)
        if name in non_numeric:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            non_finite.append(name)
        if values.nunique(dropna=False) <= 1:
            constant.append(name)
    family_counts = dict(sorted(Counter(_family(name) for name in features).items()))
    hard_failures = duplicate_columns + non_numeric + non_finite + leakage + invalid_names
    return FeatureFileAudit(
        dataset_id=dataset_id,
        path=str(source),
        row_count=len(frame),
        feature_count=len(features),
        identifier_columns=identifiers,
        feature_columns=features,
        duplicate_columns=duplicate_columns,
        non_numeric_features=non_numeric,
        non_finite_features=non_finite,
        constant_features=constant,
        leakage_suspects=leakage,
        invalid_feature_names=invalid_names,
        feature_family_counts=family_counts,
        status="PASS" if not hard_failures else "FAIL",
    )


def audit_d1_d3_feature_contracts(d1_path: str | Path, d3_path: str | Path) -> UnifiedFeatureAuditReport:
    d1 = audit_feature_file(d1_path, dataset_id="D1")
    d3 = audit_feature_file(d3_path, dataset_id="D3_RW")
    d1_set = set(d1.feature_columns)
    d3_set = set(d3.feature_columns)
    missing = sorted(d3_set - d1_set)
    status = "PASS" if d1.status == "PASS" and d3.status == "PASS" and not missing else "FAIL"
    return UnifiedFeatureAuditReport(
        status=status,
        d1=d1,
        d3=d3,
        shared_feature_count=len(d1_set & d3_set),
        d3_subset_of_d1=not missing,
        d3_features_missing_from_d1=missing,
        d1_only_feature_count=len(d1_set - d3_set),
    )
