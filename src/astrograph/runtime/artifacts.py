from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

_VOLATILE_TIME_KEYS = {"created_at", "generated_at"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _semantic_value(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, dict):
        return {
            str(k): _semantic_value(v)
            for k, v in value.items()
            if str(k) not in _VOLATILE_TIME_KEYS
        }
    if isinstance(value, list):
        return [_semantic_value(v) for v in value]
    return value


def canonical_json_sha256(value: Any) -> str:
    data = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def canonical_semantic_sha256(value: Any) -> str:
    return canonical_json_sha256(_semantic_value(value))


class RuntimeArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    producer_stage_id: str = Field(min_length=1)
    producer_stage_version: str = Field(min_length=1)
    execution_scope: str = Field(min_length=1)
    upstream_artifact_sha256: dict[str, str] = Field(default_factory=dict)


class RuntimeStageArtifact(BaseModel):
    """Content-addressed, request-scoped output from one RuntimeStage.

    `content_sha256` is exact and therefore detects any persisted-byte-level semantic
    payload mutation. `semantic_sha256` intentionally ignores generated/created clocks
    so a deterministic replay can be compared without pretending timestamps are domain
    evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-runtime-stage-artifact.v1"
    artifact_type: str = Field(min_length=1)
    artifact_version: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: RuntimeArtifactProvenance
    content: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_hashes(self) -> "RuntimeStageArtifact":
        exact = canonical_json_sha256(self.content)
        if exact != self.content_sha256:
            raise ValueError("runtime stage artifact content hash mismatch")
        semantic = canonical_semantic_sha256(self.content)
        if semantic != self.semantic_sha256:
            raise ValueError("runtime stage artifact semantic hash mismatch")
        return self


def build_runtime_stage_artifact(
    *,
    artifact_type: str,
    artifact_version: str,
    producer_stage_id: str,
    producer_stage_version: str,
    execution_scope: str,
    content: dict[str, Any],
    upstream_artifacts: Mapping[str, RuntimeStageArtifact] | None = None,
) -> RuntimeStageArtifact:
    normalized = _jsonable(content)
    upstream = {
        stage_id: artifact.content_sha256
        for stage_id, artifact in sorted((upstream_artifacts or {}).items())
    }
    return RuntimeStageArtifact(
        artifact_type=artifact_type,
        artifact_version=artifact_version,
        content_sha256=canonical_json_sha256(normalized),
        semantic_sha256=canonical_semantic_sha256(normalized),
        provenance=RuntimeArtifactProvenance(
            producer_stage_id=producer_stage_id,
            producer_stage_version=producer_stage_version,
            execution_scope=execution_scope,
            upstream_artifact_sha256=upstream,
        ),
        content=normalized,
    )


class RuntimeArtifactStore:
    """Request-scoped artifact store used by the canonical Runtime DAG."""

    def __init__(self, *, execution_scope: str) -> None:
        if not execution_scope:
            raise ValueError("execution_scope must be nonempty")
        self.execution_scope = execution_scope
        self._items: dict[str, RuntimeStageArtifact] = {}

    def put(self, stage_id: str, artifact: RuntimeStageArtifact) -> None:
        if artifact.provenance.execution_scope != self.execution_scope:
            raise ValueError(
                "runtime stage artifact execution_scope mismatch: "
                f"expected {self.execution_scope}, got {artifact.provenance.execution_scope}"
            )
        if artifact.provenance.producer_stage_id != stage_id:
            raise ValueError(
                "runtime stage artifact producer mismatch: "
                f"slot {stage_id}, producer {artifact.provenance.producer_stage_id}"
            )
        self._items[stage_id] = artifact

    def preload(self, artifacts: Mapping[str, RuntimeStageArtifact]) -> None:
        for stage_id, artifact in artifacts.items():
            self.put(stage_id, artifact)

    def get(self, stage_id: str) -> RuntimeStageArtifact:
        try:
            return self._items[stage_id]
        except KeyError as exc:
            raise KeyError(f"runtime artifact not found for stage {stage_id}") from exc

    def get_optional(self, stage_id: str) -> RuntimeStageArtifact | None:
        return self._items.get(stage_id)

    def snapshot(self) -> dict[str, RuntimeStageArtifact]:
        return dict(self._items)
