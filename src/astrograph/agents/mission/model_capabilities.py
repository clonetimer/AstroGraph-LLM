"""Model capability discovery for Mission inputs.

This registry answers *whether* a model contract is compatible with a requested
channel set.  It does not authorize or execute a model; execution remains behind
existing model/provider admission boundaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskHead = Literal["detection", "localization", "classification"]


class _Strict(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)


class ModelCapabilityEntry(_Strict):
    capability_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    tasks: tuple[TaskHead, ...] = Field(min_length=1)
    runtime_kind: Literal["BUILTIN_GENERIC", "OAG_NATIVE", "PHM_RETROSPECTIVE_BUNDLE", "EXTERNAL_ADMITTED"]
    channel_match: Literal["ANY_NUMERIC", "REQUIRED_SUBSET", "EXACT", "HOST_BOUND_ONLY"] = "REQUIRED_SUBSET"
    required_channels: tuple[str, ...] = ()
    units: dict[str, str] = Field(default_factory=dict)
    artifact_manifest: str | None = None
    online_eligible: bool = True
    retrospective_only: bool = False
    priority: int = Field(default=0, ge=0, le=1000)
    claim_boundary: str = Field(min_length=1)

    @field_validator("tasks", "required_channels")
    @classmethod
    def unique(cls, value):
        if len(value)!=len(set(value)): raise ValueError("model capability values must be unique")
        return value

    @model_validator(mode="after")
    def safe_scope(self):
        if self.retrospective_only and self.online_eligible:
            raise ValueError("retrospective-only model cannot be online eligible")
        if self.channel_match in {"REQUIRED_SUBSET","EXACT"} and not self.required_channels and not self.artifact_manifest:
            raise ValueError("channel-bound model needs required_channels or artifact_manifest")
        return self


class ModelCapabilityCatalog(_Strict):
    schema_version: Literal["astrograph-model-capability-catalog.v1"] = "astrograph-model-capability-catalog.v1"
    entries: tuple[ModelCapabilityEntry, ...]


class ModelCapabilityQuery(_Strict):
    domain: str
    channel_names: tuple[str, ...] = ()
    units: dict[str, str] = Field(default_factory=dict)
    tasks: tuple[TaskHead, ...]
    online_required: bool = True
    allow_retrospective: bool = False
    bound_artifact_contract: bool = False


class ModelCapabilityRegistry:
    def __init__(self, catalog: ModelCapabilityCatalog, *, project_root: str | Path):
        self.project_root=Path(project_root).resolve()
        self.catalog=catalog
        self.entries=tuple(self._expand(x) for x in catalog.entries)

    @classmethod
    def from_json(cls, path: str | Path, *, project_root: str | Path):
        p=Path(path)
        if not p.is_absolute(): p=Path(project_root)/p
        return cls(ModelCapabilityCatalog.model_validate_json(p.read_text(encoding="utf-8")),project_root=project_root)

    def _expand(self, entry: ModelCapabilityEntry) -> ModelCapabilityEntry:
        if not entry.artifact_manifest or entry.required_channels:
            return entry
        p=Path(entry.artifact_manifest)
        if not p.is_absolute(): p=self.project_root/p
        payload=json.loads(p.read_text(encoding="utf-8"))
        columns=[]
        feature_columns=payload.get("feature_columns",{})
        for values in feature_columns.values(): columns.extend(str(x) for x in values)
        channels=sorted({x.rsplit("__",1)[0] for x in columns if "__" in x})
        return entry.model_copy(update={"required_channels":tuple(channels)})

    def query(self, query: ModelCapabilityQuery) -> dict[str, Any]:
        channels=set(query.channel_names); tasks=set(query.tasks); rows=[]
        for entry in self.entries:
            reasons=[]
            if entry.domain not in {query.domain,"*"}: reasons.append("DOMAIN_MISMATCH")
            if not tasks.issubset(set(entry.tasks)): reasons.append("TASK_NOT_SUPPORTED")
            if query.online_required and not entry.online_eligible: reasons.append("NOT_ONLINE_ELIGIBLE")
            if entry.retrospective_only and not query.allow_retrospective: reasons.append("RETROSPECTIVE_ONLY")
            required=set(entry.required_channels)
            if not query.bound_artifact_contract:
                if entry.channel_match=="HOST_BOUND_ONLY": reasons.append("HOST_BOUND_MODEL_NOT_AUTO_SELECTED")
                elif entry.channel_match=="REQUIRED_SUBSET" and not required.issubset(channels): reasons.append("REQUIRED_CHANNELS_MISSING")
                elif entry.channel_match=="EXACT" and required!=channels: reasons.append("CHANNEL_SET_NOT_EXACT")
            for name,unit in entry.units.items():
                if name in channels and query.units.get(name) not in {None,unit}: reasons.append("UNIT_MISMATCH:"+name)
            rows.append({"capability_id":entry.capability_id,"model_id":entry.model_id,"model_version":entry.model_version,
                         "tasks":list(entry.tasks),"runtime_kind":entry.runtime_kind,"compatible":not reasons,
                         "reasons":reasons,"online_eligible":entry.online_eligible,
                         "retrospective_only":entry.retrospective_only,"priority":entry.priority,
                         "required_channel_count":len(entry.required_channels),"claim_boundary":entry.claim_boundary})
        compatible=sorted((x for x in rows if x["compatible"]),key=lambda x:(-x["priority"],x["capability_id"]))
        return {"schema_version":"astrograph-model-capability-query-result.v1","query":query.model_dump(mode="json"),
                "compatible":compatible,"all_candidates":rows,
                "execution_authorized":False,
                "claim_boundary":"Compatibility discovery only. The result neither authorizes execution nor establishes model accuracy."}

    def profile(self, query: ModelCapabilityQuery) -> dict[str, Any]:
        """Return a per-head capability profile without authorizing execution.

        The profile is intentionally descriptive: a compatible model still has to
        pass its existing provider/admission path.  Host-bound and retrospective
        candidates are surfaced as next-step hints rather than being silently
        treated as generic online models.
        """
        head_results={
            head:self.query(query.model_copy(update={"tasks":(head,)}))
            for head in ("detection","localization","classification")
        }
        available=[head for head,result in head_results.items() if result["compatible"]]
        missing=[head for head in ("detection","localization","classification") if head not in available]
        if set(available)=={"detection","localization","classification"}:
            tier="FULL_THREE_TASK"
        elif set(available)>={"detection","classification"}:
            tier="DETECTION_AND_CLASSIFICATION"
        elif "detection" in available:
            tier="DETECTION_ONLY"
        else:
            tier="NO_COMPATIBLE_MODEL"
        shared=None
        if len(available)==3:
            ids=[{x["model_id"] for x in head_results[h]["compatible"]} for h in available]
            common=set.intersection(*ids) if ids else set()
            shared=sorted(common)[0] if common else None
        blocked=[]
        seen=set()
        for result in head_results.values():
            for row in result["all_candidates"]:
                if row["compatible"]: continue
                key=(row["capability_id"],tuple(row["reasons"]))
                if key in seen: continue
                seen.add(key);blocked.append({"capability_id":row["capability_id"],"model_id":row["model_id"],"reasons":row["reasons"]})
        next_step="TRAIN_OR_REGISTER_COMPATIBLE_MODEL"
        if tier=="DETECTION_ONLY": next_step="USE_GENERIC_DETECTION_AND_KNOWLEDGE_OR_REGISTER_TASK_MODELS"
        if query.domain=="adcs" and not query.bound_artifact_contract:
            if any("HOST_BOUND_MODEL_NOT_AUTO_SELECTED" in x["reasons"] for x in blocked): next_step="USE_ADCS_NATIVE_ADAPTER_WITH_APPROVED_MODEL_CONTRACT"
        if query.domain=="phm" and not query.allow_retrospective:
            if any("RETROSPECTIVE_ONLY" in x["reasons"] or "NOT_ONLINE_ELIGIBLE" in x["reasons"] for x in blocked): next_step="USE_REPLAY_TASK_FOR_RETROSPECTIVE_PHM_OR_TRAIN_ONLINE_CAUSAL_MODEL"
        if tier=="FULL_THREE_TASK":
            next_step="USE_EXISTING_ADMISSION_PATH_FOR_SHARED_MODEL" if shared else "USE_EXISTING_ADMISSION_PATHS_PER_HEAD"
        return {
            "schema_version":"astrograph-model-capability-profile.v1",
            "query":query.model_dump(mode="json"),
            "coverage_tier":tier,
            "available_heads":available,
            "missing_heads":missing,
            "shared_three_task_model_id":shared,
            "heads":head_results,
            "blocked_candidates":blocked,
            "recommended_next_step":next_step,
            "execution_authorized":False,
            "claim_boundary":"Capability profiling is descriptive only; execution still requires the model's existing admission path.",
        }
