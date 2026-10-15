from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnomalyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    anomaly_name_zh: str = Field(min_length=1)
    stream: Literal["legacy", "adcs", "eps", "comm", "thermal"]
    minimum_change: float = Field(gt=0.0)
    persistence_samples: int = Field(default=2, ge=1)
    noise_multiplier: float = Field(default=6.0, ge=0.0)
    expected_direction: Literal["either", "increase", "decrease"] = "either"
    semantic_role: Literal["direct_observation", "component_response", "cascade_effect"] = "component_response"
    applicable_condition_ids: list[str] = Field(default_factory=list)
    evidence_level: str = "E0"
    review_status: str = "provisional"
    enabled: bool = True


class AnomalyRuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-anomaly-rule-set.v1"
    rule_set_version: str = Field(min_length=1)
    confidence_semantics: str = "synthetic_paired_residual_strength_not_calibrated"
    rules: list[AnomalyRule]

    @model_validator(mode="after")
    def validate_unique(self) -> "AnomalyRuleSet":
        rule_ids = [item.rule_id for item in self.rules]
        channels = [item.channel_id for item in self.rules if item.enabled]
        anomaly_ids = [item.anomaly_id for item in self.rules if item.enabled]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule_id must be unique")
        if len(channels) != len(set(channels)):
            raise ValueError("one enabled rule per channel is required")
        if len(anomaly_ids) != len(set(anomaly_ids)):
            raise ValueError("one enabled rule per anomaly_id is required")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "AnomalyRuleSet":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def enabled_rules(self) -> list[AnomalyRule]:
        return [item for item in self.rules if item.enabled]
