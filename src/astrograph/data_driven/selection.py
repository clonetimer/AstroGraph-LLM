from __future__ import annotations

import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.evidence.executable import stable_id

from .contracts import ModelCapability, ModelRuntime
from .registry import DataDrivenModelRegistry


class ModelSelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"


class FaultModelBinding(BaseModel):
    """Versioned binding between a registered fault mode and a model provider.

    Bindings do not load model code. They constrain selection among providers that have
    already been registered through ``DataDrivenModelRegistry``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fault-model-binding.v1"
    binding_id: str = ""
    fault_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    feature_schema_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("feature_schema_ids")
    @classmethod
    def unique_feature_schemas(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("feature_schema_ids must be unique")
        return value

    @model_validator(mode="after")
    def assign_binding_id(self) -> "FaultModelBinding":
        if not self.binding_id:
            self.binding_id = stable_id(
                "FMB",
                {
                    "fault_id": self.fault_id,
                    "model_id": self.model_id,
                    "model_version": self.model_version,
                    "feature_schema_ids": self.feature_schema_ids,
                },
            )
        return self


class FaultModelBindingSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fault-model-binding-set.v1"
    binding_set_id: str = Field(min_length=1)
    bindings: list[FaultModelBinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bindings")
    @classmethod
    def unique_binding_ids(cls, value: list[FaultModelBinding]) -> list[FaultModelBinding]:
        ids = [item.binding_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("binding IDs must be unique")
        return value

    @classmethod
    def from_json(cls, path: str | Path) -> "FaultModelBindingSet":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class ModelSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-model-selection-request.v1"
    candidate_fault_ids: list[str] = Field(min_length=1)
    feature_schema_id: str = Field(min_length=1)
    required_capabilities: list[ModelCapability] = Field(
        default_factory=lambda: [ModelCapability.FAULT_RANKING]
    )
    preferred_runtimes: list[ModelRuntime] = Field(default_factory=list)
    allow_partial_fault_coverage: bool = True
    allow_unbound_fallback: bool = True

    @field_validator("candidate_fault_ids", "required_capabilities", "preferred_runtimes")
    @classmethod
    def unique_values(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("model selection request lists must contain unique values")
        return value


class ModelSelectionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_id: str
    model_id: str
    model_version: str
    runtime: ModelRuntime
    matched_fault_ids: list[str]
    missing_fault_ids: list[str]
    binding_ids: list[str]
    coverage: float = Field(ge=0.0, le=1.0)
    priority_score: int = Field(ge=0)
    runtime_preference_rank: int = Field(ge=0)
    selection_score: float
    used_unbound_fallback: bool = False


class ModelSelectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-model-selection-decision.v1"
    selection_id: str
    status: ModelSelectionStatus
    selected_model_id: str | None = None
    selected_model_version: str | None = None
    candidates: list[ModelSelectionCandidate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    claim_boundary: str = (
        "Model selection chooses a registered inference provider; it does not establish diagnostic validity."
    )


class FaultModelBindingRegistry:
    """Deterministic model router for fault-specific and generic model adapters."""

    def __init__(self, bindings: list[FaultModelBinding] | None = None) -> None:
        self._bindings: dict[str, FaultModelBinding] = {}
        for binding in bindings or []:
            self.register(binding)

    @classmethod
    def from_binding_set(cls, binding_set: FaultModelBindingSet) -> "FaultModelBindingRegistry":
        return cls(binding_set.bindings)

    @classmethod
    def from_json(cls, path: str | Path) -> "FaultModelBindingRegistry":
        return cls.from_binding_set(FaultModelBindingSet.from_json(path))

    def register(self, binding: FaultModelBinding, *, replace: bool = False) -> None:
        if binding.binding_id in self._bindings and not replace:
            raise ValueError(f"fault-model binding already registered: {binding.binding_id}")
        self._bindings[binding.binding_id] = binding

    def bindings(self) -> list[FaultModelBinding]:
        return [self._bindings[key] for key in sorted(self._bindings)]

    @staticmethod
    def _manifest_is_eligible(manifest: Any, request: ModelSelectionRequest) -> bool:
        if manifest.feature_schema_ids and request.feature_schema_id not in set(manifest.feature_schema_ids):
            return False
        required = set(request.required_capabilities)
        if not required.issubset(set(manifest.capabilities)):
            return False
        return True

    def select(
        self,
        request: ModelSelectionRequest,
        models: DataDrivenModelRegistry,
    ) -> ModelSelectionDecision:
        requested_faults = set(request.candidate_fault_ids)
        grouped: dict[tuple[str, str], list[FaultModelBinding]] = defaultdict(list)
        rejected_reasons: list[str] = []

        for binding in self.bindings():
            if not binding.enabled or binding.fault_id not in requested_faults:
                continue
            try:
                provider = models.get(binding.model_id, binding.model_version)
            except KeyError:
                rejected_reasons.append(
                    f"BOUND_MODEL_NOT_REGISTERED:{binding.model_id}@{binding.model_version or 'latest'}"
                )
                continue
            manifest = provider.manifest
            if binding.fault_id not in set(manifest.supported_fault_ids):
                rejected_reasons.append(
                    f"BINDING_FAULT_NOT_SUPPORTED:{binding.binding_id}:{binding.fault_id}"
                )
                continue
            if binding.feature_schema_ids and request.feature_schema_id not in set(binding.feature_schema_ids):
                continue
            if not self._manifest_is_eligible(manifest, request):
                continue
            grouped[(manifest.model_id, manifest.model_version)].append(binding)

        if request.allow_unbound_fallback:
            for manifest in models.manifests():
                key = (manifest.model_id, manifest.model_version)
                if key in grouped or not self._manifest_is_eligible(manifest, request):
                    continue
                if requested_faults.intersection(manifest.supported_fault_ids):
                    grouped[key] = []

        candidates: list[ModelSelectionCandidate] = []
        for (model_id, model_version), bindings in grouped.items():
            manifest = models.get(model_id, model_version).manifest
            bound_faults = {item.fault_id for item in bindings}
            supported = requested_faults.intersection(manifest.supported_fault_ids)
            matched = sorted(bound_faults or supported)
            missing = sorted(requested_faults - set(matched))
            if missing and not request.allow_partial_fault_coverage:
                continue
            coverage = len(matched) / len(requested_faults)
            priority_score = sum(item.priority for item in bindings)
            runtime_rank = (
                request.preferred_runtimes.index(manifest.runtime)
                if manifest.runtime in request.preferred_runtimes
                else len(request.preferred_runtimes)
            )
            # Coverage is dominant, explicit bindings outrank generic fallbacks, then
            # configured priority and runtime preference provide deterministic refinement.
            selection_score = (
                coverage * 10_000.0
                + (1_000.0 if bindings else 0.0)
                + float(priority_score)
                - float(runtime_rank)
            )
            candidates.append(
                ModelSelectionCandidate(
                    manifest_id=manifest.manifest_id,
                    model_id=manifest.model_id,
                    model_version=manifest.model_version,
                    runtime=manifest.runtime,
                    matched_fault_ids=matched,
                    missing_fault_ids=missing,
                    binding_ids=sorted(item.binding_id for item in bindings),
                    coverage=coverage,
                    priority_score=priority_score,
                    runtime_preference_rank=runtime_rank,
                    selection_score=selection_score,
                    used_unbound_fallback=not bindings,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.selection_score,
                -item.coverage,
                -item.priority_score,
                item.runtime_preference_rank,
                item.manifest_id,
            )
        )
        selection_id = stable_id(
            "MODEL-SELECT",
            {
                "request": request.model_dump(mode="json"),
                "candidates": [item.model_dump(mode="json") for item in candidates],
            },
        )
        if not candidates:
            return ModelSelectionDecision(
                selection_id=selection_id,
                status=ModelSelectionStatus.NO_MATCH,
                reasons=["NO_ELIGIBLE_REGISTERED_MODEL", *sorted(set(rejected_reasons))],
                requires_human_review=True,
            )

        top = candidates[0]
        tied = [item for item in candidates if item.selection_score == top.selection_score]
        ambiguous = len(tied) > 1
        reasons = [
            f"MATCHED_FAULTS:{','.join(top.matched_fault_ids)}",
            f"COVERAGE:{top.coverage:.3f}",
        ]
        if top.used_unbound_fallback:
            reasons.append("UNBOUND_GENERIC_FALLBACK_USED")
        reasons.extend(sorted(set(rejected_reasons)))
        if ambiguous:
            reasons.append("MULTIPLE_MODELS_SHARE_TOP_SELECTION_SCORE")
        return ModelSelectionDecision(
            selection_id=selection_id,
            status=ModelSelectionStatus.AMBIGUOUS if ambiguous else ModelSelectionStatus.SELECTED,
            selected_model_id=top.model_id,
            selected_model_version=top.model_version,
            candidates=candidates,
            reasons=reasons,
            requires_human_review=ambiguous or top.used_unbound_fallback or bool(top.missing_fault_ids),
        )
