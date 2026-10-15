"""Governed Agent access to the canonical deterministic AstroGraph runtime.

The bridge deliberately exposes the runtime as one atomic capability.  Agent hosts
may choose a pipeline and provide a typed ``RuntimeRequest``; they may not invoke or
replace individual scientific stages, select policies, or bypass runtime receipts.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from astrograph.agentization.artifacts import resolve_scoped
from astrograph.agentization.protocol import (
    CapabilityProtocolError,
    sha256_json,
)
from astrograph.audit import NullAuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import (
    CsvKnowledgeRepository,
    csv_repository_from_release,
)
from astrograph.runtime import (
    DIAGNOSIS_PIPELINE_ID,
    UNIFIED_DIAGNOSIS_PIPELINE_ID,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    RuntimeRequest,
    build_application_runtime,
)


AGENT_EXECUTABLE_PIPELINES = frozenset(
    {
        DIAGNOSIS_PIPELINE_ID,
        UNIFIED_DIAGNOSIS_PIPELINE_ID,
        WHOLE_SPACECRAFT_PIPELINE_ID,
    }
)

_PATH_KEYS = {
    "artifact_output_dir",
    "catalog_path",
    "catalog_model_path",
    "manifest_path",
    "model_path",
    "nominal_path",
    "nominal_run_dir",
    "open_set_model_path",
    "protocol_root",
    "run_dir",
    "telemetry_path",
}


def _is_path_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in _PATH_KEYS
        or normalized == "path"
        or normalized.endswith("_path")
        or normalized.endswith("_dir")
        or normalized.endswith("_root")
    )


def _normalize_scoped_paths(
    value: Any,
    *,
    project_root: Path,
    allowed_roots: tuple[Path, ...],
    key: str = "",
) -> Any:
    """Resolve path-like request fields and reject paths outside configured roots."""

    if isinstance(value, Mapping):
        return {
            str(child_key): _normalize_scoped_paths(
                child_value,
                project_root=project_root,
                allowed_roots=allowed_roots,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_scoped_paths(
                child,
                project_root=project_root,
                allowed_roots=allowed_roots,
                key=key,
            )
            for child in value
        ]
    if _is_path_key(key) and isinstance(value, str) and value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return str(resolve_scoped(str(candidate), allowed_roots))
    return value


class AgentRuntimeBridge:
    """Lazy, request-scoped bridge to ``AstroGraphRuntime``."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        self._runtime_instance: Any | None = None

    def _build_default_runtime(self) -> Any:
        root = self.environment.project_root
        release_path = root / "data" / "knowledge_releases" / "active_release.json"
        if release_path.is_file():
            repository = csv_repository_from_release(root, release_path)
        else:
            repository = CsvKnowledgeRepository(
                root / "data" / "knowledge_seed" / "causes.csv"
            )
        return build_application_runtime(
            repository=repository,
            policy=DiagnosisPolicy.from_json(root / "configs" / "diagnosis.json"),
            audit_store=NullAuditStore(),
            project_root=root,
        )

    @property
    def runtime(self) -> Any:
        if self._runtime_instance is None:
            factory = self.environment.runtime_factory
            self._runtime_instance = (
                factory() if factory is not None else self._build_default_runtime()
            )
        return self._runtime_instance

    def describe(self) -> dict[str, Any]:
        description = self.runtime.describe()
        return {
            **description,
            "agent_execution_boundary": "ATOMIC_RUNTIME_REQUEST",
            "agent_executable_pipeline_ids": sorted(AGENT_EXECUTABLE_PIPELINES),
            "individual_stage_invocation_allowed": False,
            "policy_selection_allowed": False,
            "ground_truth_access_allowed": False,
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_request = arguments.get("request")
        if not isinstance(raw_request, Mapping):
            raise CapabilityProtocolError(
                "RUNTIME_REQUEST_REQUIRED",
                "runtime.diagnosis.execute requires a RuntimeRequest object",
            )
        normalized = deepcopy(dict(raw_request))
        normalized["payload"] = _normalize_scoped_paths(
            normalized.get("payload", {}),
            project_root=self.environment.project_root,
            allowed_roots=self.environment.allowed_roots,
        )
        normalized["metadata"] = _normalize_scoped_paths(
            normalized.get("metadata", {}),
            project_root=self.environment.project_root,
            allowed_roots=self.environment.allowed_roots,
        )
        try:
            request = RuntimeRequest.model_validate(normalized)
        except ValidationError as exc:
            raise CapabilityProtocolError(
                "RUNTIME_REQUEST_INVALID",
                "request does not satisfy astrograph-runtime-request.v1",
                {"errors": exc.errors(include_url=False)},
            ) from exc
        if request.pipeline_id not in AGENT_EXECUTABLE_PIPELINES:
            raise CapabilityProtocolError(
                "RUNTIME_PIPELINE_NOT_AGENT_ADMITTED",
                f"pipeline {request.pipeline_id!r} is not admitted for Agent execution",
                {"allowed_pipeline_ids": sorted(AGENT_EXECUTABLE_PIPELINES)},
            )
        request = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "entrypoint": "astrograph.agentization.runtime.diagnosis.execute",
                    "agent_execution_boundary": "ATOMIC_RUNTIME_REQUEST",
                }
            }
        )
        result = self.runtime.execute(request)
        result_payload = result.model_dump(mode="json")
        return {
            "schema_version": "astrograph-agent-runtime-execution.v1",
            "execution": "PERFORMED_BY_DETERMINISTIC_ASTROGRAPH_RUNTIME",
            "execution_authority": "ASTROGRAPH_RUNTIME_ONLY",
            "request_sha256": sha256_json(request.model_dump(mode="json")),
            "result_sha256": sha256_json(result_payload),
            "runtime_status": result_payload["status"],
            "pipeline_id": result_payload["pipeline_id"],
            "stage_count": len(result_payload["stages"]),
            "runtime_result": result_payload,
        }


RUNTIME_EXECUTE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["request"],
    "properties": {
        "request": {
            "type": "object",
            "required": ["request_id", "pipeline_id", "payload"],
            "properties": {
                "schema_version": {"type": "string"},
                "request_id": {"type": "string", "minLength": 1},
                "pipeline_id": {
                    "type": "string",
                    "enum": sorted(AGENT_EXECUTABLE_PIPELINES),
                },
                "payload": {"type": "object"},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        }
    },
    "additionalProperties": False,
}


__all__ = [
    "AGENT_EXECUTABLE_PIPELINES",
    "AgentRuntimeBridge",
    "RUNTIME_EXECUTE_INPUT_SCHEMA",
]
