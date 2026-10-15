"""Read-only numerical checks over host-admitted telemetry, not model-written facts."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from astrograph.evidence.executable import FeatureResult, stable_id
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, OperatorInput
from astrograph.semantics import is_prohibited_field
from .contracts import content_hash


@dataclass(frozen=True)
class EvidenceCheckTool:
    """An explicitly host-registered, trusted numerical function.

    This is deliberately not a new plugin loader or a sandbox. Third-party code
    belongs behind the existing ToolExecutor; only trusted built-ins run here.
    """
    tool_id: str
    tool_version: str
    feature_definition_ids: frozenset[str]
    input_hashes: frozenset[str]
    handler: Callable[[tuple[str, ...]], list[FeatureResult]]
    cost: float = 1.0

    def __post_init__(self):
        if not self.tool_id or not self.tool_version or not self.feature_definition_ids:
            raise ValueError("tool identity, version and feature scope required")
        if not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("tool cost must be finite and positive")
        if not self.input_hashes or any(len(h) != 64 for h in self.input_hashes):
            raise ValueError("tool requires admitted input hashes")


class FeatureSeries(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    feature_definition_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    time_s: tuple[float, ...] = Field(min_length=1)
    current: tuple[float, ...] = Field(min_length=1)
    baseline: tuple[float, ...] | None = None
    available_at_s: float

    @model_validator(mode="after")
    def check_series(self):
        if is_prohibited_field(self.field):
            raise ValueError("truth or answer-proxy field cannot enter a telemetry check")
        if len(self.current) != len(self.time_s):
            raise ValueError("unaligned current values and time")
        if self.baseline is not None and len(self.baseline) != len(self.time_s):
            raise ValueError("unaligned reference values")
        if any(a >= b for a, b in zip(self.time_s, self.time_s[1:])):
            raise ValueError("time must be strictly increasing")
        if self.available_at_s < self.time_s[-1]:
            raise ValueError("availability cannot precede measurement")
        return self


class MeasurementBank(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-admitted-measurements.v1"] = "astrograph-admitted-measurements.v1"
    episode_id: str = Field(min_length=1)
    series: tuple[FeatureSeries, ...] = ()

    @model_validator(mode="after")
    def unique_features(self):
        ids = [s.feature_definition_id for s in self.series]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate feature series")
        return self


def tools_from_measurements(bank: MeasurementBank, bundle) -> tuple[EvidenceCheckTool, ...]:
    """Compile approved feature definitions to existing deterministic operators.

    Each resulting tool computes a value from actual numeric samples. It cannot
    return a fault label, change thresholds, or mark its own evidence SATISFIED.
    """
    bank = MeasurementBank.model_validate_json(bank.model_dump_json())
    if bank.episode_id != bundle.episode.episode_id:
        raise ValueError("measurement bank belongs to another episode")
    defs = {d.feature_definition_id: d for d in bundle.feature_definitions}
    old = {f.feature_definition_id: f for f in bundle.feature_results}
    tools = []
    for series in bank.series:
        if series.feature_definition_id not in defs:
            raise ValueError("measurement has no registered feature definition")
        definition = defs[series.feature_definition_id]
        prior = old.get(series.feature_definition_id)
        fields = {definition.source_field_template}
        if prior is not None:
            fields.add(prior.resolved_field)
        if series.field not in fields or series.unit != definition.unit:
            raise ValueError("measurement field/unit differs from registered semantics")
        window = bundle.episode
        if (series.time_s[0] != window.start_s or series.time_s[-1] != window.end_s
                or series.available_at_s > window.end_s):
            raise ValueError("check must use the exact admitted window without future data")
        operator = BUILTIN_OPERATOR_REGISTRY.get(definition.aggregation.value).spec
        if operator.requires_baseline and series.baseline is None:
            # Inapplicable tools are omitted; no fabricated healthy reference.
            continue
        digest = content_hash(series.model_dump(mode="json"))
        def compute(requested, *, series=series, definition=definition, digest=digest, prior=prior):
            if requested != (definition.feature_definition_id,):
                raise ValueError("check requested a feature outside the admitted scope")
            value = BUILTIN_OPERATOR_REGISTRY.compute(
                definition.aggregation.value,
                OperatorInput(current=series.current, baseline=series.baseline, time_s=series.time_s),
            )
            if value is None or not math.isfinite(value):
                return []
            return [FeatureResult(
                feature_result_id=stable_id("FR-CHECK", {"feature": definition.feature_definition_id,
                                                        "input": digest, "value": value}),
                feature_definition_id=definition.feature_definition_id,
                run_id=window.run_id, paired_nominal_run_id=window.paired_nominal_run_id,
                episode_id=window.episode_id, resolved_field=series.field,
                component_instance=prior.component_instance if prior else bundle.analysis_component_instance,
                value=value, unit=definition.unit, window_start_s=window.start_s, window_end_s=window.end_s,
                sample_count=len(series.time_s), algorithm_version=definition.algorithm_version,
                input_data_sha256=digest, quality_flags=[],
            )]
        tools.append(EvidenceCheckTool(
            tool_id="telemetry.compute:" + definition.feature_definition_id,
            tool_version="registered-operator.v1", feature_definition_ids=frozenset({definition.feature_definition_id}),
            input_hashes=frozenset({digest}), handler=compute,
        ))
    return tuple(tools)
