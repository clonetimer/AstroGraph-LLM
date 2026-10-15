from __future__ import annotations

import os
import hmac
import hashlib
import json
import re
import secrets
import shutil
from importlib.resources import files as package_files
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, Response, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.anomaly import AnomalyEpisode, IncrementalAnomalyObservation
from astrograph.audit.store import sha256_json
from astrograph.data import SimulationInterfaceSnapshot, assess_simulation_interface_change
from astrograph.errors import (
    AstroGraphError,
    InvalidOperationError,
    ResourceNotFoundError,
    StateConflictError,
)
from astrograph.experiments import ExperimentRecord, VersionCombination
from astrograph.schemas import AnomalyEvidence, WorkflowInput
from astrograph.security import ApiAuthConfig, AuthIdentity, auth_error_details, authenticate_request, required_role
from astrograph.service import AstroGraphService, build_service_from_environment
from astrograph.runtime import RuntimeRequest
from astrograph.compat import build_compat_router
from astrograph.integrations.upload_report import (
    PlatformJobCreateRequest,
    UploadReportJobStore,
    PLATFORM_UI_ASSET_VERSION,
    platform_asset,
    platform_html,
)
from astrograph.integrations.platform_state import PlatformStateRepository
from astrograph.integrations.platform_secrets import PlatformSecretStore
from astrograph.integrations.material_engine_settings import (
    MaterialEngineRegistry, MaterialEngineSettings, EngineUpdateRequest,
    EngineRollbackRequest, engine_probe,
)
from astrograph.integrations.maintenance_input_admission import SupplementalFeatureInputCapsule
from astrograph.integrations.maintenance_production_rerun import ProductionRawInputPatchCapsule
from astrograph.integrations.material_workbench_sessions import MaterialReviewerSessions, COOKIE_NAME, TTL_SECONDS
from astrograph import __version__




class AccountBootstrap(BaseModel):
    username: str = Field(min_length=3, max_length=60)
    display_name: str = Field(default="", max_length=80)
    password: str = Field(min_length=12, max_length=1024)
    setup_secret: str = Field(min_length=24)


class AccountLogin(BaseModel):
    username: str
    password: str


class AccountCreate(BaseModel):
    username: str
    display_name: str = ""
    password: str
    role: Literal["viewer", "operator", "reviewer", "admin"]


class AccountUpdate(BaseModel):
    role: Literal["viewer", "operator", "reviewer", "admin"] | None = None
    active: bool | None = None
    password: str | None = None


class MaterialReviewLogin(BaseModel):
    model_config = ConfigDict(extra='forbid')
    credential: str = Field(min_length=1, max_length=4096)


class MaterialReasoningGapResolveRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    resolution_ref: str = Field(min_length=8, max_length=500)


class MaterialGraphReleaseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    equipment_scope: str = Field(min_length=2, max_length=80)
    snapshot: int | None = Field(default=None, ge=0)


class MaterialGraphPublishRequest(MaterialGraphReleaseRequest):
    confirm_release_id: str = Field(min_length=16, max_length=128)
    confirm_write: bool = False


class ManualReportReviewDecision(BaseModel):
    """An expert opinion tied to an immutable material-analysis report."""
    model_config = ConfigDict(extra='forbid')
    report_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    fault_id: str = Field(min_length=1, max_length=200)
    decision: Literal['FOLLOW_UP_HYPOTHESIS', 'NOT_SUPPORTED', 'EVIDENCE_INSUFFICIENT']
    evidence_ref: str = Field(default='', max_length=500)
    note: str = Field(min_length=8, max_length=2000)
    expected_revision: int = Field(ge=0)


class MaterialCaseReviewDecision(BaseModel):
    """Explicit independent review; cannot be submitted by the analyzer."""
    model_config = ConfigDict(extra='forbid')
    decision: Literal['CONFIRM','REJECT']
    truth_source_ref: str = Field(min_length=3,max_length=500)
    verified_fault_id: str | None = Field(default=None,max_length=160)
    equipment_scope: str | None = Field(default=None,max_length=80)
    approved_actions: list[str] = Field(default_factory=list,max_length=12)
    expected_snapshot: int | None = Field(default=None,ge=0)


class MaterialDocumentReviewDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: Literal['CONFIRM','REJECT']
    truth_source_ref: str = Field(min_length=3,max_length=500)
    equipment_scope: str | None = Field(default=None,max_length=80)
    expected_snapshot: int | None = Field(default=None,ge=0)


class MaterialGraphRetractionRequest(BaseModel):
    """Independent correction of a previously approved reference graph fact."""
    model_config = ConfigDict(extra='forbid')
    reason: str = Field(min_length=8,max_length=1000)
    truth_source_ref: str = Field(min_length=3,max_length=500)
    expected_snapshot: int | None = Field(default=None,ge=0)


class PlatformLLMSettingsRequest(BaseModel):
    """Process-scoped configuration for the upload platform explanation LLM.

    The secret is write-only: it is accepted here only to update the server process
    environment and is never returned by the settings endpoint. This does not modify
    Mission Agent administrator policy or a frozen diagnosis.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    provider: str = Field(default="local-openai-compatible", min_length=1, max_length=120)
    base_url: str | None = Field(default=None, max_length=2000)
    model: str | None = Field(default=None, max_length=300)
    api_key: str | None = Field(default=None, max_length=4096)
    timeout_s: float = Field(default=120.0, gt=0.0, le=3600.0)
    response_format_strategy: Literal["auto", "json_schema", "json_object", "plain_json"] = "auto"
    structured_max_tokens: int = Field(default=1200, ge=64, le=8192)

    @model_validator(mode="after")
    def validate_enabled_configuration(self) -> "PlatformLLMSettingsRequest":
        if not self.enabled:
            return self
        base_url = (self.base_url or "").strip()
        model = (self.model or "").strip()
        if not base_url or not model:
            raise ValueError("enabled LLM configuration requires base_url and model")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        return self


class PlatformLLMProfileSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    make_default: bool = True
    activate: bool = True
    settings: PlatformLLMSettingsRequest


class PlatformNeo4jSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    uri: str | None = Field(default=None, max_length=2000)
    user: str = Field(default="neo4j", min_length=1, max_length=200)
    password: str | None = Field(default=None, max_length=4096)
    database: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_enabled_configuration(self) -> "PlatformNeo4jSettingsRequest":
        if not self.enabled:
            return self
        uri = (self.uri or "").strip()
        if not uri:
            raise ValueError("enabled Neo4j configuration requires uri")
        parsed = urlparse(uri)
        if parsed.scheme not in {"bolt", "neo4j", "bolt+s", "neo4j+s", "bolt+ssc", "neo4j+ssc"}:
            raise ValueError("Neo4j uri must use bolt:// or neo4j:// family")
        return self


class PlatformNeo4jProfileSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    make_default: bool = True
    activate: bool = True
    settings: PlatformNeo4jSettingsRequest


class PlatformToolProviderSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$", min_length=2, max_length=120)
    provider_type: Literal["mcp"] = "mcp"
    name: str = Field(min_length=1, max_length=120)
    url: str = Field(min_length=8, max_length=2000)
    enabled: bool = True
    read_only: bool = True
    tool_allowlist: list[str] = Field(default_factory=list, max_length=100)
    timeout_s: float = Field(default=30.0, gt=0, le=300)


class PlatformMaintenanceAnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotation_type: Literal["MANUAL_FIRST_DETECTION", "REVIEW_NOTE"]
    time_s: float | None = Field(default=None, ge=0)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_annotation(self) -> "PlatformMaintenanceAnnotationRequest":
        if self.annotation_type == "MANUAL_FIRST_DETECTION" and self.time_s is None:
            raise ValueError("manual first-detection annotation requires time_s")
        return self


class PlatformMaintenanceProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    probe_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Z0-9_.:-]+$")


class PlatformMaintenanceRerunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: bool = False


class PlatformToolPluginInstallRequest(BaseModel):
    """Admin-only managed install/update into the configured trusted plugin root."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=4, max_length=180, pattern=r"^[A-Za-z0-9_.-]+\.py$")
    source: str = Field(min_length=1, max_length=1_000_000)



class EpisodeDiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    episodes: list[AnomalyEpisode] = Field(min_length=1)
    operating_condition: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def one_run_per_request(self) -> "EpisodeDiagnosisRequest":
        run_ids = {item.run_id for item in self.episodes}
        dataset_ids = {item.dataset_id for item in self.episodes}
        if len(run_ids) != 1:
            raise ValueError("all anomaly episodes in one diagnosis request must belong to the same run_id")
        if len(dataset_ids) != 1:
            raise ValueError("all anomaly episodes in one diagnosis request must belong to the same dataset_id")
        return self

    def to_workflow_input(self) -> WorkflowInput:
        grouped: dict[str, list[AnomalyEpisode]] = {}
        for episode in self.episodes:
            grouped.setdefault(episode.anomaly_id, []).append(episode)

        anomalies: list[AnomalyEvidence] = []
        for anomaly_id, episodes in sorted(grouped.items()):
            primary = max(episodes, key=lambda item: item.peak_confidence)
            signals = sorted({signal for item in episodes for signal in item.signals})
            condition_ids = sorted({condition for item in episodes for condition in item.condition_ids})
            first_times = [item.first_observed_at_s for item in episodes if item.first_observed_at_s is not None]
            anomalies.append(
                AnomalyEvidence(
                    anomaly_id=anomaly_id,
                    name=primary.name,
                    signal=signals[0] if signals else "episode",
                    confidence=max(item.peak_confidence for item in episodes),
                    confidence_type="synthetic_episode_peak_not_calibrated",
                    observed_at_s=min(first_times) if first_times else None,
                    metadata={
                        "episode_ids": [item.episode_id for item in episodes],
                        "run_id": primary.run_id,
                        "case_id": primary.case_id,
                        "start_s": min(item.start_s for item in episodes),
                        "end_s": max(item.end_s for item in episodes),
                        "signals": signals,
                        "condition_ids": condition_ids,
                        "episode_count": len(episodes),
                    },
                )
            )

        operating_condition = dict(self.operating_condition)
        if not operating_condition.get("condition_ids"):
            operating_condition["condition_ids"] = sorted(
                {condition for item in self.episodes for condition in item.condition_ids}
            )
        first = self.episodes[0]
        detector_versions = sorted({version for item in self.episodes for version in item.detector_versions})
        metadata = {
            **self.metadata,
            "source": self.metadata.get("source", "anomaly_episode_api"),
            "episode_ids": [item.episode_id for item in self.episodes],
            "run_id": first.run_id,
            "dataset_id": first.dataset_id,
            "paired_nominal_run_id": first.paired_nominal_run_id,
            "synthetic_case_id": first.case_id,
            "detector_versions": detector_versions,
        }
        return WorkflowInput(
            request_id=self.request_id,
            case_id=self.case_id,
            anomalies=anomalies,
            operating_condition=operating_condition,
            metadata=metadata,
        )


class SimulationCompatibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current: SimulationInterfaceSnapshot
    candidate: SimulationInterfaceSnapshot


class CloseEpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    end_s: float = Field(ge=0.0)
    reason: str = Field(default="MANUAL_CLOSE", min_length=1)


class KnowledgeDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    description: str = ""
    candidate_release_path: str = Field(min_length=1)
    created_by: str = Field(min_length=1)


class KnowledgeActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(min_length=1)
    comment: str = ""


class KnowledgeReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1)
    approve: bool
    comment: str = Field(min_length=1)


class ExperimentRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(min_length=1)
    combination: VersionCombination
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_record(self) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=ExperimentRecord.create_id(self.purpose, self.combination),
            purpose=self.purpose,
            combination=self.combination,
            metadata=self.metadata,
        )


def error_payload(code: str, message: str, *, request_id: str = "", details: Any = None) -> dict[str, Any]:
    return {
        "schema_version": "astrograph-api-error.v1",
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": {} if details is None else details,
        },
    }


def create_app(service: AstroGraphService | None = None) -> FastAPI:
    app = FastAPI(
        title="AstroGraph API",
        version=__version__,
        description="Satellite PHM service with deterministic diagnosis, controlled Agent investigation, and policy-gated release.",
    )
    app.state.astrograph_service = service
    app.state.auth_config = ApiAuthConfig.from_environment()
    app.state.upload_report_store = None
    app.state.platform_state_repository = None
    app.state.platform_secret_store = None
    app.state.material_review_sessions = MaterialReviewerSessions()
    app.state.platform_accounts = None
    app.state.mission_probe = {"status": "NOT_PROBED", "time": None, "reason": None}

    def accounts():
        if app.state.platform_accounts is None:
            from astrograph.integrations.platform_accounts import PlatformAccounts
            root = Path(os.environ.get("ASTROGRAPH_PLATFORM_ROOT") or
                        (Path.home() / ".astrograph" / "platform")).expanduser().resolve()
            app.state.platform_accounts = PlatformAccounts(root / "access_control")
        return app.state.platform_accounts

    def account_identity(request: Request):
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer":
            return None
        session = accounts().get_session(token.strip())
        if not session: return None
        return AuthIdentity(subject=session["username"],roles=frozenset({session["role"]}),mode="accounts")

    @app.middleware("http")
    async def authorization_middleware(request: Request, call_next):
        # Fail before multipart body parsing on this newly added expert-only route.
        # Starlette/FastAPI would otherwise parse uploaded data before entering the handler.
        if app.state.auth_config.mode != 'accounts' and request.method == 'POST' and request.url.path == '/v1/platform/material-knowledge/graph/build':
            try:
                secret, _reviewer, _reviewers = app.state.material_review_sessions.config()
                header = request.headers.get('x-material-review-token', '')
                if not (header and hmac.compare_digest(header, secret)):
                    entry = app.state.material_review_sessions.authenticate(request, require_csrf=True)
                    if entry is None:
                        raise HTTPException(status_code=403, detail='Expert login required')
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail},
                                    headers={'Cache-Control': 'no-store'})
        role = required_role(request.method, request.url.path)
        if role is None:
            return await call_next(request)
        identity = (account_identity(request) if app.state.auth_config.mode == "accounts"
                    else authenticate_request(request, app.state.auth_config))
        if identity is None:
            return JSONResponse(
                status_code=401,
                content=error_payload(
                    "ASTROGRAPH_AUTHENTICATION_REQUIRED",
                    "Authentication is required for this endpoint.",
                    request_id=request.headers.get("X-Request-ID", ""),
                    details=auth_error_details(app.state.auth_config, role),
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not identity.permits(role):
            return JSONResponse(
                status_code=403,
                content=error_payload(
                    "ASTROGRAPH_AUTHORIZATION_DENIED",
                    "The authenticated identity does not have the required role.",
                    request_id=request.headers.get("X-Request-ID", ""),
                    details=auth_error_details(app.state.auth_config, role),
                ),
            )
        request.state.auth_identity = identity
        response = await call_next(request)
        if request.url.path.startswith('/v1/platform/settings/'):
            response.headers['Cache-Control'] = 'no-store'
        response.headers["X-AstroGraph-Auth-Mode"] = identity.mode
        return response

    def authenticated_actor(http_request: Request, supplied: str) -> str:
        identity = getattr(http_request.state, "auth_identity", None)
        if identity is None or identity.mode == "disabled":
            return supplied
        return identity.subject

    def get_service() -> AstroGraphService:
        current = app.state.astrograph_service
        if current is None:
            current = build_service_from_environment()
            app.state.astrograph_service = current
        return current

    def get_platform_state() -> PlatformStateRepository:
        current = app.state.platform_state_repository
        if current is None:
            root = Path(os.environ.get("ASTROGRAPH_PLATFORM_ROOT") or (Path.home() / ".astrograph" / "platform")).expanduser().resolve()
            current = PlatformStateRepository.from_environment(platform_root=root)
            app.state.platform_state_repository = current
        return current

    def get_platform_secret_store() -> PlatformSecretStore:
        current = app.state.platform_secret_store
        if current is None:
            current = PlatformSecretStore()
            app.state.platform_secret_store = current
        return current

    def get_material_engine_registry() -> MaterialEngineRegistry:
        root = Path(os.environ.get("ASTROGRAPH_PLATFORM_ROOT") or (Path.home() / ".astrograph" / "platform"))
        return MaterialEngineRegistry(root, secret_store=get_platform_secret_store())

    def _apply_llm_env(config: dict[str, Any], *, secret: str | None = None) -> None:
        enabled = bool(config.get("enabled", True))
        names = (
            "ASTROGRAPH_PLATFORM_LLM_BASE_URL", "ASTROGRAPH_PLATFORM_LLM_MODEL",
            "ASTROGRAPH_PLATFORM_LLM_PROVIDER", "ASTROGRAPH_PLATFORM_LLM_TIMEOUT_S",
            "ASTROGRAPH_PLATFORM_LLM_RESPONSE_FORMAT_STRATEGY", "ASTROGRAPH_PLATFORM_LLM_MAX_TOKENS",
        )
        if not enabled:
            for name in names:
                os.environ.pop(name, None)
            return
        os.environ["ASTROGRAPH_PLATFORM_LLM_BASE_URL"] = str(config.get("base_url") or "").strip().rstrip("/")
        os.environ["ASTROGRAPH_PLATFORM_LLM_MODEL"] = str(config.get("model") or "").strip()
        os.environ["ASTROGRAPH_PLATFORM_LLM_PROVIDER"] = str(config.get("provider") or "local-openai-compatible").strip()
        os.environ["ASTROGRAPH_PLATFORM_LLM_TIMEOUT_S"] = str(config.get("timeout_s", 120.0))
        os.environ["ASTROGRAPH_PLATFORM_LLM_RESPONSE_FORMAT_STRATEGY"] = str(config.get("response_format_strategy") or "auto")
        os.environ["ASTROGRAPH_PLATFORM_LLM_MAX_TOKENS"] = str(config.get("structured_max_tokens", 1200))
        if secret:
            os.environ["ASTROGRAPH_LLM_API_KEY"] = secret

    def _ensure_default_llm_profile_loaded() -> None:
        if os.environ.get("ASTROGRAPH_PLATFORM_LLM_BASE_URL") and os.environ.get("ASTROGRAPH_PLATFORM_LLM_MODEL"):
            return
        profile = get_platform_state().default_service_profile("llm")
        if not profile:
            return
        secret = None
        if profile.get("secret_ref"):
            secret = get_platform_secret_store().get(str(profile["secret_ref"]))
        _apply_llm_env(dict(profile.get("config") or {}), secret=secret)
        os.environ["ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID"] = str(profile["profile_id"])

    def _apply_neo4j_env(config: dict[str, Any], *, secret: str | None = None) -> None:
        if not bool(config.get("enabled", True)):
            for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
                os.environ.pop(name, None)
            return
        os.environ["NEO4J_URI"] = str(config.get("uri") or "").strip()
        os.environ["NEO4J_USER"] = str(config.get("user") or "neo4j").strip()
        if config.get("database"):
            os.environ["NEO4J_DATABASE"] = str(config["database"]).strip()
        else:
            os.environ.pop("NEO4J_DATABASE", None)
        if secret:
            os.environ["NEO4J_PASSWORD"] = secret

    def _ensure_default_neo4j_profile_loaded() -> None:
        if os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_USER"):
            return
        profile = get_platform_state().default_service_profile("neo4j")
        if not profile:
            return
        secret = None
        if profile.get("secret_ref"):
            secret = get_platform_secret_store().get(str(profile["secret_ref"]))
        _apply_neo4j_env(dict(profile.get("config") or {}), secret=secret)
        os.environ["ASTROGRAPH_PLATFORM_NEO4J_ACTIVE_PROFILE_ID"] = str(profile["profile_id"])

    def get_upload_report_store() -> UploadReportJobStore:
        _ensure_default_llm_profile_loaded()
        _ensure_default_neo4j_profile_loaded()
        get_material_engine_registry().activate()
        current = app.state.upload_report_store
        if current is None:
            root = Path(os.environ.get("ASTROGRAPH_PLATFORM_ROOT") or (Path.home() / ".astrograph" / "platform")).expanduser().resolve()
            current = UploadReportJobStore(root, state_repository=get_platform_state())
            app.state.upload_report_store = current
        return current


    def platform_llm_runtime_settings() -> dict[str, Any]:
        _ensure_default_llm_profile_loaded()
        base_url = os.environ.get("ASTROGRAPH_PLATFORM_LLM_BASE_URL", "").strip()
        model = os.environ.get("ASTROGRAPH_PLATFORM_LLM_MODEL", "").strip()
        provider = os.environ.get("ASTROGRAPH_PLATFORM_LLM_PROVIDER", "local-openai-compatible").strip() or "local-openai-compatible"
        try:
            timeout_s = float(os.environ.get("ASTROGRAPH_PLATFORM_LLM_TIMEOUT_S", "120"))
        except ValueError:
            timeout_s = 120.0
        key_configured = bool(os.environ.get("ASTROGRAPH_LLM_API_KEY", "").strip())
        response_format_strategy = os.environ.get("ASTROGRAPH_PLATFORM_LLM_RESPONSE_FORMAT_STRATEGY", "auto").strip() or "auto"
        try:
            structured_max_tokens = int(os.environ.get("ASTROGRAPH_PLATFORM_LLM_MAX_TOKENS", "1200"))
        except ValueError:
            structured_max_tokens = 1200
        active_profile_id = os.environ.get("ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID") or None
        active_profile = get_platform_state().get_service_profile(active_profile_id) if active_profile_id else None
        return {
            "schema_version": "astrograph-platform-llm-settings.v3",
            "enabled": bool(base_url and model),
            "configured": bool(base_url and model),
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "timeout_s": timeout_s,
            "response_format_strategy": response_format_strategy,
            "structured_max_tokens": structured_max_tokens,
            "api_key_configured": key_configured,
            "secret_returned": False,
            "scope": ("SQLITE_PROFILE_PLUS_PROCESS_RUNTIME" if active_profile else "PROCESS_RUNTIME_ONLY"),
            "persists_across_restart": bool(active_profile),
            "active_profile_id": active_profile_id,
            "active_profile_name": (active_profile.get("name") if active_profile else None),
            "diagnostic_vote_allowed": False,
            "applies_to": "POST_DIAGNOSIS_KNOWLEDGE_AND_AGENT_INVESTIGATION",
            "mission_agent_policy_changed": False,
        }

    def apply_platform_llm_runtime_settings(request: PlatformLLMSettingsRequest) -> dict[str, Any]:
        config = request.model_dump(exclude={"api_key"})
        secret = request.api_key.strip() if request.api_key and request.api_key.strip() else None
        _apply_llm_env(config, secret=secret)
        if not request.enabled:
            os.environ.pop("ASTROGRAPH_LLM_API_KEY", None)
            os.environ.pop("ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID", None)
        elif secret is None:
            # keep an already configured process secret, if any
            pass
        app.state.upload_report_store = None
        return platform_llm_runtime_settings()

    @app.exception_handler(AstroGraphError)
    async def astrograph_error_handler(request: Request, exc: AstroGraphError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=error_payload(exc.code, exc.message, request_id=request.headers.get("X-Request-ID", ""), details=exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_payload(
                "ASTROGRAPH_REQUEST_VALIDATION_ERROR",
                "Request payload validation failed.",
                request_id=request.headers.get("X-Request-ID", ""),
                # Sensitive configuration payloads must not be echoed into
                # validation errors (including pydantic ctx.error objects).
                details=([{"loc":err.get("loc"),"type":err.get("type"),"msg":err.get("msg")}
                    for err in exc.errors()] if request.url.path.startswith(
                        "/v1/platform/settings/knowledge-engine") else exc.errors()),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "ASTROGRAPH_INTERNAL_ERROR",
                "An unexpected server error occurred.",
                request_id=request.headers.get("X-Request-ID", ""),
                details={"exception_type": type(exc).__name__},
            ),
        )

    @app.get("/live")
    def live() -> dict:
        # Liveness does not construct repositories or touch external dependencies.
        return {"status": "PASS", "service": "astrograph", "check": "liveness"}

    @app.get("/ready")
    def ready(response: Response) -> dict:
        report = get_service().health().model_dump(mode="json")
        if report["status"] != "PASS":
            response.status_code = 503
        return report

    @app.get("/health")
    def health() -> dict:
        # Backward-compatible aggregate health endpoint. Prefer /live and /ready for
        # orchestrator probes.
        return get_service().health().model_dump(mode="json")

    @app.get("/v1/runtime")
    def runtime_description() -> dict:
        return get_service().runtime.describe()

    @app.post("/v1/runtime/execute")
    def runtime_execute(request: RuntimeRequest, response: Response) -> dict:
        result = get_service().execute_runtime(request)
        if result.get("status") != "SUCCEEDED":
            response.status_code = 422
        return result

    @app.post("/v1/diagnose")
    def diagnose(request: WorkflowInput, response: Response) -> dict:
        payload, replayed = get_service().diagnose_idempotent(request)
        response.headers["X-AstroGraph-Idempotent-Replay"] = "true" if replayed else "false"
        return payload

    @app.post("/v1/diagnose/episodes")
    def diagnose_episodes(request: EpisodeDiagnosisRequest, response: Response) -> dict:
        workflow_input = request.to_workflow_input()
        temporal_analysis = get_service().evaluate_episode_temporal(request.episodes)
        workflow_input.metadata["temporal_constraint_catalog_version"] = temporal_analysis.get("catalog_version", "")
        workflow_input.metadata["temporal_constraint_evaluations"] = temporal_analysis.get("evaluations_by_fault", {})
        payload, replayed = get_service().diagnose_idempotent(workflow_input)
        episode_ids = [item.episode_id for item in request.episodes]
        episode_hash = sha256_json([item.model_dump(mode="json") for item in request.episodes])
        detector_versions = sorted({version for item in request.episodes for version in item.detector_versions})
        detector_id = detector_versions[0].split("|", 1)[0] if detector_versions else "episode_detector"
        rule_version = detector_versions[0].split("|", 1)[1] if detector_versions and "|" in detector_versions[0] else ""
        case_record = get_service().register_episode_case(
            request=workflow_input,
            response=payload,
            episode_ids=episode_ids,
            episode_hash=episode_hash,
            detector_id=detector_id,
            rule_set_version=rule_version,
        )
        payload = dict(payload)
        payload["schema_version"] = "astrograph-episode-diagnosis-api-response.v2"
        payload["episode_count"] = len(request.episodes)
        payload["episode_ids"] = episode_ids
        payload["episode_hash"] = episode_hash
        payload["diagnostic_case"] = case_record
        payload["temporal_analysis"] = temporal_analysis
        payload["idempotent_replay"] = replayed
        response.headers["X-AstroGraph-Idempotent-Replay"] = "true" if replayed else "false"
        return payload

    # Incremental episode lifecycle -----------------------------------------------------
    @app.post("/v1/episodes/observations")
    def apply_episode_observation(observation: IncrementalAnomalyObservation) -> dict:
        try:
            return get_service().apply_episode_observation(observation)
        except ValueError as exc:
            raise InvalidOperationError(str(exc)) from exc

    @app.get("/v1/episodes")
    def list_episodes(run_id: str | None = None, status: str | None = Query(default=None)) -> dict:
        try:
            records = get_service().episodes(run_id=run_id, status=status)
        except ValueError as exc:
            raise InvalidOperationError(str(exc)) from exc
        return {"schema_version": "astrograph-online-episode-list.v1", "count": len(records), "records": records}

    @app.get("/v1/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict:
        record = get_service().episode(episode_id)
        if record is None:
            raise ResourceNotFoundError("Episode not found.", details={"episode_id": episode_id})
        return record

    @app.post("/v1/episodes/{episode_id}/close")
    def close_episode(episode_id: str, request: CloseEpisodeRequest) -> dict:
        try:
            return get_service().close_episode(episode_id, end_s=request.end_s, reason=request.reason)
        except KeyError as exc:
            raise ResourceNotFoundError("Episode not found.", details={"episode_id": episode_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"episode_id": episode_id}) from exc

    @app.get("/v1/requests/{request_id}")
    def request_lifecycle(request_id: str) -> dict:
        record = get_service().request_lifecycle(request_id)
        if record is None:
            raise ResourceNotFoundError("Request lifecycle record not found.", details={"request_id": request_id})
        return record

    @app.get("/v1/requests/{request_id}/timeline")
    def request_timeline(request_id: str) -> dict:
        record = get_service().request_timeline(request_id)
        if record is None:
            raise ResourceNotFoundError("Request lifecycle timeline not found.", details={"request_id": request_id})
        return record

    @app.get("/v1/admin/persistence/status")
    def persistence_status() -> dict:
        return get_service().persistence_status()

    # Knowledge governance --------------------------------------------------------------
    @app.get("/v1/knowledge/changes")
    def knowledge_changes() -> dict:
        records = get_service().knowledge_changes()
        return {"schema_version": "astrograph-knowledge-change-list.v1", "count": len(records), "records": records}

    @app.get("/v1/knowledge/changes/{change_id}")
    def knowledge_change(change_id: str) -> dict:
        record = get_service().knowledge_change(change_id)
        if record is None:
            raise ResourceNotFoundError("Knowledge change not found.", details={"change_id": change_id})
        return record

    @app.post("/v1/knowledge/drafts")
    def create_knowledge_draft(request: KnowledgeDraftRequest, http_request: Request) -> dict:
        try:
            payload = request.model_dump()
            payload["created_by"] = authenticated_actor(http_request, request.created_by)
            return get_service().create_knowledge_draft(**payload)
        except (ValueError, FileNotFoundError) as exc:
            raise InvalidOperationError(str(exc)) from exc

    @app.post("/v1/knowledge/changes/{change_id}/submit")
    def submit_knowledge_change(change_id: str, request: KnowledgeActionRequest, http_request: Request) -> dict:
        try:
            return get_service().submit_knowledge_change(
                change_id, actor=authenticated_actor(http_request, request.actor), comment=request.comment
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Knowledge change not found.", details={"change_id": change_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"change_id": change_id}) from exc

    @app.post("/v1/knowledge/changes/{change_id}/review")
    def review_knowledge_change(change_id: str, request: KnowledgeReviewRequest, http_request: Request) -> dict:
        try:
            return get_service().review_knowledge_change(
                change_id,
                reviewer=authenticated_actor(http_request, request.reviewer),
                approve=request.approve,
                comment=request.comment,
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Knowledge change not found.", details={"change_id": change_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"change_id": change_id}) from exc

    @app.post("/v1/knowledge/changes/{change_id}/publish")
    def publish_knowledge_change(change_id: str, request: KnowledgeActionRequest, http_request: Request) -> dict:
        try:
            return get_service().publish_knowledge_change(
                change_id, actor=authenticated_actor(http_request, request.actor), comment=request.comment
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Knowledge change not found.", details={"change_id": change_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"change_id": change_id}) from exc

    @app.post("/v1/knowledge/changes/{change_id}/rollback")
    def rollback_knowledge_change(change_id: str, request: KnowledgeActionRequest, http_request: Request) -> dict:
        try:
            return get_service().rollback_knowledge_change(
                change_id, actor=authenticated_actor(http_request, request.actor), comment=request.comment
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Knowledge change not found.", details={"change_id": change_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"change_id": change_id}) from exc

    # Version combination experiments --------------------------------------------------
    @app.post("/v1/experiments")
    def register_experiment(request: ExperimentRegisterRequest) -> dict:
        return get_service().register_experiment(request.to_record())

    @app.get("/v1/experiments")
    def list_experiments() -> dict:
        records = get_service().experiments()
        return {"schema_version": "astrograph-experiment-list.v1", "count": len(records), "records": records}

    @app.get("/v1/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict:
        record = get_service().experiment(experiment_id)
        if record is None:
            raise ResourceNotFoundError("Experiment not found.", details={"experiment_id": experiment_id})
        return record

    @app.post("/v1/compatibility/simulation")
    def simulation_compatibility(request: SimulationCompatibilityRequest) -> dict:
        return assess_simulation_interface_change(request.current, request.candidate).model_dump(mode="json")

    @app.get("/v1/knowledge/active-release")
    def active_release() -> dict:
        return get_service().active_knowledge_release()

    @app.get("/v1/evidence/sources")
    def evidence_sources() -> dict:
        records = get_service().evidence_sources()
        return {"schema_version": "astrograph-evidence-source-list.v1", "count": len(records), "records": records}

    @app.get("/v1/cases")
    def diagnostic_cases() -> dict:
        records = get_service().diagnostic_cases()
        return {"schema_version": "astrograph-diagnostic-case-list.v1", "count": len(records), "records": records}

    @app.get("/v1/cases/{case_record_id}")
    def diagnostic_case(case_record_id: str) -> dict:
        record = get_service().diagnostic_case(case_record_id)
        if record is None:
            raise ResourceNotFoundError("Diagnostic case not found.", details={"case_record_id": case_record_id})
        return record

    @app.delete("/v1/cases/{case_record_id}")
    def delete_diagnostic_case(case_record_id: str) -> dict:
        deleted = get_service().delete_diagnostic_case(case_record_id)
        if not deleted:
            raise ResourceNotFoundError("Diagnostic case not found.", details={"case_record_id": case_record_id})
        return {
            "schema_version": "astrograph-diagnostic-case-deletion.v1",
            "case_record_id": case_record_id,
            "deleted": True,
        }


    # Upload-to-report platform ----------------------------------------------------------
    @app.get("/platform", response_class=HTMLResponse, include_in_schema=False)
    def upload_report_ui() -> HTMLResponse:
        # One canonical experience: legacy is explicitly opt-in for engineering users.
        return HTMLResponse("""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>正在进入 AstroGraph</title><meta http-equiv="refresh" content="0;url=/platform/mission"><script>location.replace('/platform/mission'+location.hash)</script><a href="/platform/mission">进入智能工作台</a></html>""",headers={"Cache-Control":"no-store"})

    @app.get("/platform/legacy", response_class=HTMLResponse, include_in_schema=False)
    def legacy_upload_report_ui() -> HTMLResponse:
        return HTMLResponse(platform_html(), headers={"Cache-Control": "no-cache, max-age=0"})

    @app.get("/v1/platform/knowledge/public-seed")
    def public_spacecraft_seed_graph() -> Response:
        """Read-only bundled public reference graph, never an approved flight KG.

        Package data makes the same graph available for editable installs and wheels.
        Do not mingle it with the approved equipment-specific graph endpoints.
        """
        resource = package_files("astrograph").joinpath("resources/public_spacecraft_seed.v1.json")
        data = json.loads(resource.read_text(encoding="utf-8"))
        if data.get("graph_kind") != "UNREVIEWED_PUBLIC_REFERENCE_SEED" or data.get("is_flight_qualified") is not False:
            raise HTTPException(status_code=503, detail="Public reference graph validation failed")
        return JSONResponse(
            content=data,
            headers={"Cache-Control": "public, max-age=600", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/platform/home", response_class=HTMLResponse, include_in_schema=False)
    def mission_home_ui() -> HTMLResponse:
        content = package_files("astrograph").joinpath("resources/home.html").read_text(encoding="utf-8")
        return HTMLResponse(content, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/platform/mission", response_class=HTMLResponse, include_in_schema=False)
    def mission_console_ui() -> HTMLResponse:
        """Production-connected Chinese mission UI; existing /platform remains available."""
        content = package_files("astrograph").joinpath("resources/mission_console.html").read_text(encoding="utf-8")
        return HTMLResponse(content, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/platform/mission/assets/{asset_name}", include_in_schema=False)
    def mission_console_asset(asset_name: str) -> Response:
        allowed = {
            "mission_console.css": "text/css",
            "mission_console.js": "application/javascript",
            "mission_demo_cases.json": "application/json",
            "home.css": "text/css",
            "home.js": "application/javascript",
            "home_scene.js": "application/javascript",
            "home_three.js": "application/javascript",
        }
        media = allowed.get(asset_name)
        if media is None:
            raise HTTPException(status_code=404, detail="Unknown mission UI resource")
        content = package_files("astrograph").joinpath("resources", asset_name).read_text(encoding="utf-8")
        return Response(content=content, media_type=media, headers={
            "Cache-Control": "no-cache, max-age=0", "X-Content-Type-Options": "nosniff"
        })

    @app.get('/v1/platform/auth/status')
    def account_status() -> dict:
        if app.state.auth_config.mode != 'accounts':
            return {'mode':app.state.auth_config.mode, 'accounts_enabled':False}
        return {'mode':'accounts','accounts_enabled':True,'bootstrap_required':not accounts().initialized()}

    @app.post('/v1/platform/auth/bootstrap')
    def bootstrap_account(body: AccountBootstrap, request: Request) -> dict:
        if app.state.auth_config.mode != 'accounts':
            raise HTTPException(status_code=404,detail='账户模式未启用')
        if request.client and request.client.host not in {'127.0.0.1','::1','localhost','testclient'}:
            raise HTTPException(status_code=403,detail='初始化仅允许本机请求')
        try:
            return accounts().bootstrap(body.username,body.password,body.display_name,
                                        init_secret=body.setup_secret)
        except PermissionError as exc:raise HTTPException(status_code=403,detail=str(exc)) from exc
        except ValueError as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post('/v1/platform/auth/login')
    def login_account(body: AccountLogin, response: Response) -> dict:
        if app.state.auth_config.mode != 'accounts':
            raise HTTPException(status_code=404,detail='账户模式未启用')
        try:
            token,user=accounts().authenticate(body.username,body.password)
        except PermissionError as exc:raise HTTPException(status_code=401,detail=str(exc)) from exc
        response.headers['Cache-Control']='no-store'
        return {'token':token, 'user':user, 'expires_in_seconds':28800}

    @app.post('/v1/platform/auth/logout')
    def logout_account(request: Request,response: Response) -> dict:
        if app.state.auth_config.mode != 'accounts':
            raise HTTPException(status_code=404,detail='账户模式未启用')
        header=request.headers.get('Authorization','')
        if header.lower().startswith('bearer '): accounts().logout(header[7:].strip())
        response.headers['Cache-Control']='no-store'
        return {'ok':True}

    @app.get('/v1/platform/accounts')
    def list_platform_accounts() -> dict:
        return {'users':accounts().list_users()}

    @app.post('/v1/platform/accounts')
    def add_platform_account(body: AccountCreate,request: Request) -> dict:
        try:
            return accounts().add_user(actor=request.state.auth_identity.subject,**body.model_dump())
        except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.patch('/v1/platform/accounts/{user_id}')
    def patch_platform_account(user_id:int,body:AccountUpdate,request:Request) -> dict:
        try:
            return accounts().modify_user(user_id,actor=request.state.auth_identity.subject,
                                          **body.model_dump(exclude_unset=True))
        except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.get('/v1/platform/mission/services')
    def mission_services_status(request: Request) -> dict:
        # Status describes configuration separately from the last live probe.
        settings = platform_llm_runtime_settings()
        enabled = bool(settings.get('enabled'))
        return {'llm':{'configured':enabled,'model':str(settings.get('model') or ''),
                       'provider':str(settings.get('provider') or ''),
                       **app.state.mission_probe}}

    @app.post('/v1/platform/mission/services/probe')
    def mission_services_probe() -> dict:
        config=platform_llm_runtime_settings()
        result={'status':'NOT_CONFIGURED','time':datetime.now(timezone.utc).isoformat(),
                'reason':'请先配置大模型服务'}
        if config.get('enabled') and config.get('model') and config.get('base_url'):
            from astrograph.integrations.platform_llm_probe import probe_openai_compatible_service
            try:
                probe = probe_openai_compatible_service(
                    base_url=str(config['base_url']),model=str(config['model']),
                    api_key=os.environ.get('ASTROGRAPH_LLM_API_KEY','local-no-secret'),
                    timeout_s=min(float(config.get('timeout_s',8)),10),
                    provider=str(config.get('provider','local-openai-compatible')),
                    response_format_strategy=str(config.get('response_format_strategy','auto')))
                result={'status':'AVAILABLE' if probe.get('ok') or probe.get('status') in {'PASS','AVAILABLE','READY'} else 'DEGRADED',
                        'time':datetime.now(timezone.utc).isoformat(),
                        'reason':str(probe.get('message') or probe.get('error') or '')[:200]}
            except Exception as exc:
                result={'status':'FAILED','time':datetime.now(timezone.utc).isoformat(),
                        'reason':type(exc).__name__} # do not leak backend secrets
        app.state.mission_probe=result
        return {'llm':{**result,'configured':bool(config.get('enabled')),'model':str(config.get('model') or '')}}

    @app.get("/v1/platform/mission/identity")
    def mission_identity(request: Request) -> dict:
        """Reflect actual server-side API role, never accept browser-selected authority."""
        identity = getattr(request.state, "auth_identity", None)
        if identity is None:
            # Read operations are already guarded by authorization_middleware.
            raise HTTPException(status_code=401, detail="Authentication required")
        return {
            "schema_version": "astrograph-mission-console-identity.v1",
            "subject": identity.subject,
            "roles": sorted(identity.roles),
            "mode": identity.mode,
            "permissions": {
                "read": identity.permits("viewer"),
                "analyze": identity.permits("operator"),
                "settings": identity.permits("admin"),
            },
        }

    @app.get("/platform/assets/{version}/{asset_name}", include_in_schema=False)
    def upload_report_ui_asset(version: str, asset_name: str) -> Response:
        if version != PLATFORM_UI_ASSET_VERSION:
            raise ResourceNotFoundError(
                "Platform UI asset version not found.",
                details={"version": version, "expected_version": PLATFORM_UI_ASSET_VERSION},
            )
        try:
            content, media_type = platform_asset(asset_name)
        except ValueError as exc:
            raise ResourceNotFoundError(
                "Platform UI asset not found.", details={"asset_name": asset_name}
            ) from exc
        return Response(
            content=content,
            media_type=media_type.split(";", 1)[0],
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/v1/platform/settings/knowledge-engine")
    def knowledge_engine_settings() -> dict:
        return get_material_engine_registry().current()

    @app.get("/v1/platform/settings/knowledge-engine/revisions")
    def knowledge_engine_revisions() -> dict:
        return {"schema_version":"astrograph-material-engine-revisions.v1",
                "revisions":get_material_engine_registry().revisions()}

    @app.put("/v1/platform/settings/knowledge-engine")
    def update_knowledge_engine(body: EngineUpdateRequest) -> dict:
        try:
            return get_material_engine_registry().save(body.settings, expected_revision=body.expected_revision)
        except RuntimeError as exc:
            if str(exc) == 'ENGINE_REVISION_CONFLICT':
                raise HTTPException(status_code=409, detail='Engine configuration changed: refresh before saving') from exc
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/platform/settings/knowledge-engine/rollback")
    def rollback_knowledge_engine(body: EngineRollbackRequest) -> dict:
        try:
            return get_material_engine_registry().rollback(
                expected_revision=body.expected_revision, target_revision=body.target_revision)
        except RuntimeError as exc:
            if str(exc) == 'ENGINE_REVISION_CONFLICT':
                raise HTTPException(status_code=409, detail='Engine configuration changed: refresh before rollback') from exc
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/platform/settings/knowledge-engine/probe")
    def probe_knowledge_engine(body: MaterialEngineSettings, network: bool = Query(default=False)) -> dict:
        registry = get_material_engine_registry()
        current = registry.current()
        # Never send a saved key to a newly entered endpoint: only the exact
        # active URL may reuse its existing credential for a connectivity test.
        key = body.qdrant_api_key
        if not key and body.qdrant_url == current['settings'].get('qdrant_url'):
            key = registry.active_qdrant_key()
        return engine_probe(body, network=network,
            api_key_configured=bool(key),api_key=key)

    @app.get("/v1/platform/settings/llm")
    def get_platform_llm_settings() -> dict:
        return platform_llm_runtime_settings()

    @app.put("/v1/platform/settings/llm")
    def set_platform_llm_settings(request: PlatformLLMSettingsRequest) -> dict:
        return apply_platform_llm_runtime_settings(request)

    @app.post("/v1/platform/settings/llm/test")
    def test_platform_llm_settings(request: PlatformLLMSettingsRequest) -> dict:
        if not request.enabled:
            raise InvalidOperationError("Enable and configure the LLM service before testing it.")
        from astrograph.integrations.platform_llm_probe import probe_openai_compatible_service

        secret = (request.api_key or "").strip() or os.environ.get("ASTROGRAPH_LLM_API_KEY", "local-no-secret")
        return probe_openai_compatible_service(
            base_url=(request.base_url or "").strip().rstrip("/"),
            model=(request.model or "").strip(),
            api_key=secret,
            timeout_s=request.timeout_s,
            provider=request.provider.strip(),
            response_format_strategy=request.response_format_strategy,
        )

    @app.get("/v1/platform/settings/llm/profiles")
    def list_platform_llm_profiles() -> dict:
        rows = get_platform_state().list_service_profiles("llm")
        for row in rows:
            row["secret_configured"] = bool(row.pop("secret_ref", None))
        return {"schema_version":"astrograph-platform-llm-profiles.v1","profiles":rows,"secret_backend_available":get_platform_secret_store().available()}

    @app.get("/v1/platform/analysis-engines")
    def list_analysis_engines() -> dict:
        """Sanitized per-task engine catalogue for operators and reviewers."""
        return get_upload_report_store().analysis_engines()

    @app.post("/v1/platform/settings/llm/profiles")
    def save_platform_llm_profile(request: PlatformLLMProfileSaveRequest) -> dict:
        settings = request.settings
        probe = None
        if settings.enabled:
            from astrograph.integrations.platform_llm_probe import probe_openai_compatible_service
            secret = (settings.api_key or "").strip() or os.environ.get("ASTROGRAPH_LLM_API_KEY", "local-no-secret")
            probe = probe_openai_compatible_service(
                base_url=(settings.base_url or "").strip().rstrip("/"), model=(settings.model or "").strip(),
                api_key=secret, timeout_s=settings.timeout_s, provider=settings.provider.strip(),
                response_format_strategy=settings.response_format_strategy,
            )
            if probe.get("status") != "READY":
                raise InvalidOperationError("Only a configuration that passes the service probe can be saved as a reusable profile.", details={"probe":probe})
        config = settings.model_dump(exclude={"api_key"})
        first = get_platform_state().save_service_profile(
            service_kind="llm", name=request.name, config=config, validation_status=("PASS" if probe else "DISABLED"),
            verified_at=datetime.now(timezone.utc).isoformat() if probe else None, make_default=request.make_default,
        )
        secret_ref = None
        secret_persisted = False
        if settings.api_key and settings.api_key.strip():
            secret_ref = get_platform_secret_store().make_ref("llm", str(first["profile_id"]))
            secret_persisted = get_platform_secret_store().set(secret_ref, settings.api_key.strip())
            if not secret_persisted:
                secret_ref = None
        saved = get_platform_state().save_service_profile(
            service_kind="llm", name=request.name, config=config, secret_ref=secret_ref,
            validation_status=("PASS" if probe else "DISABLED"), verified_at=datetime.now(timezone.utc).isoformat() if probe else None,
            make_default=request.make_default, profile_id=str(first["profile_id"]),
        )
        if request.activate:
            effective_secret_ref = saved.get("secret_ref")
            _apply_llm_env(config, secret=(settings.api_key.strip() if settings.api_key and settings.api_key.strip() else get_platform_secret_store().get(effective_secret_ref)))
            os.environ["ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID"] = str(saved["profile_id"])
            app.state.upload_report_store = None
        saved.pop("secret_ref", None)
        return {"profile":saved,"probe":probe,"secret_persisted":secret_persisted or not bool(settings.api_key and settings.api_key.strip()),"active":request.activate}

    @app.post("/v1/platform/settings/llm/profiles/{profile_id}/activate")
    def activate_platform_llm_profile(profile_id: str) -> dict:
        profile = get_platform_state().get_service_profile(profile_id)
        if not profile or profile.get("service_kind") != "llm":
            raise ResourceNotFoundError("LLM profile not found.", details={"profile_id":profile_id})
        secret = get_platform_secret_store().get(profile.get("secret_ref"))
        _apply_llm_env(dict(profile.get("config") or {}), secret=secret)
        get_platform_state().set_default_service_profile(profile_id)
        os.environ["ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID"] = profile_id
        app.state.upload_report_store = None
        return platform_llm_runtime_settings()

    @app.delete("/v1/platform/settings/llm/profiles/{profile_id}")
    def delete_platform_llm_profile(profile_id: str) -> dict:
        profile = get_platform_state().get_service_profile(profile_id)
        if not profile or profile.get("service_kind") != "llm":
            raise ResourceNotFoundError("LLM profile not found.", details={"profile_id":profile_id})
        get_platform_secret_store().delete(profile.get("secret_ref"))
        deleted = get_platform_state().delete_service_profile(profile_id)
        if os.environ.get("ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID") == profile_id:
            os.environ.pop("ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID", None)
        return {"profile_id":profile_id,"deleted":deleted}

    @app.get("/v1/platform/settings/neo4j")
    def get_platform_neo4j_settings() -> dict:
        _ensure_default_neo4j_profile_loaded()
        active = os.environ.get("ASTROGRAPH_PLATFORM_NEO4J_ACTIVE_PROFILE_ID") or None
        profile = get_platform_state().get_service_profile(active) if active else None
        return {
            "schema_version":"astrograph-platform-neo4j-settings.v1",
            "enabled":bool(os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_USER")),
            "uri":os.environ.get("NEO4J_URI", ""), "user":os.environ.get("NEO4J_USER", "neo4j"),
            "database":os.environ.get("NEO4J_DATABASE", ""), "password_configured":bool(os.environ.get("NEO4J_PASSWORD")),
            "active_profile_id":active, "active_profile_name":profile.get("name") if profile else None,
            "secret_returned":False, "read_only_tool_policy":True,
        }

    @app.put("/v1/platform/settings/neo4j")
    def set_platform_neo4j_settings(request: PlatformNeo4jSettingsRequest) -> dict:
        _apply_neo4j_env(request.model_dump(exclude={"password"}), secret=(request.password.strip() if request.password and request.password.strip() else None))
        os.environ.pop("ASTROGRAPH_PLATFORM_NEO4J_ACTIVE_PROFILE_ID", None)
        return get_platform_neo4j_settings()

    @app.post("/v1/platform/settings/neo4j/test")
    def test_platform_neo4j_settings(request: PlatformNeo4jSettingsRequest) -> dict:
        previous = {name:os.environ.get(name) for name in ("NEO4J_URI","NEO4J_USER","NEO4J_PASSWORD","NEO4J_DATABASE")}
        try:
            _apply_neo4j_env(request.model_dump(exclude={"password"}), secret=(request.password.strip() if request.password and request.password.strip() else os.environ.get("NEO4J_PASSWORD")))
            from astrograph.tools.neo4j_maintenance import neo4j_health
            return neo4j_health()
        finally:
            for name,value in previous.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name]=value

    @app.get("/v1/platform/settings/neo4j/profiles")
    def list_platform_neo4j_profiles() -> dict:
        rows=get_platform_state().list_service_profiles("neo4j")
        for row in rows: row["secret_configured"]=bool(row.pop("secret_ref",None))
        return {"schema_version":"astrograph-platform-neo4j-profiles.v1","profiles":rows,"secret_backend_available":get_platform_secret_store().available()}

    @app.post("/v1/platform/settings/neo4j/profiles")
    def save_platform_neo4j_profile(request: PlatformNeo4jProfileSaveRequest) -> dict:
        settings=request.settings
        # Verify before marking the profile reusable.
        probe=test_platform_neo4j_settings(settings) if settings.enabled else {"status":"DISABLED"}
        if settings.enabled and probe.get("status") != "PASS":
            raise InvalidOperationError("Neo4j configuration did not pass connectivity validation.", details={"probe":probe})
        config=settings.model_dump(exclude={"password"})
        first=get_platform_state().save_service_profile(service_kind="neo4j",name=request.name,config=config,validation_status=("PASS" if settings.enabled else "DISABLED"),verified_at=datetime.now(timezone.utc).isoformat() if settings.enabled else None,make_default=request.make_default)
        secret_ref=None; secret_persisted=False
        if settings.password and settings.password.strip():
            secret_ref=get_platform_secret_store().make_ref("neo4j",str(first["profile_id"]))
            secret_persisted=get_platform_secret_store().set(secret_ref,settings.password.strip())
            if not secret_persisted: secret_ref=None
        saved=get_platform_state().save_service_profile(service_kind="neo4j",name=request.name,config=config,secret_ref=secret_ref,validation_status=("PASS" if settings.enabled else "DISABLED"),verified_at=datetime.now(timezone.utc).isoformat() if settings.enabled else None,make_default=request.make_default,profile_id=str(first["profile_id"]))
        if request.activate:
            effective_secret_ref=saved.get("secret_ref")
            _apply_neo4j_env(config,secret=(settings.password.strip() if settings.password and settings.password.strip() else get_platform_secret_store().get(effective_secret_ref)))
            os.environ["ASTROGRAPH_PLATFORM_NEO4J_ACTIVE_PROFILE_ID"]=str(saved["profile_id"])
        saved.pop("secret_ref",None)
        return {"profile":saved,"probe":probe,"secret_persisted":secret_persisted or not bool(settings.password and settings.password.strip()),"active":request.activate}

    @app.post("/v1/platform/settings/neo4j/profiles/{profile_id}/activate")
    def activate_platform_neo4j_profile(profile_id: str) -> dict:
        profile=get_platform_state().get_service_profile(profile_id)
        if not profile or profile.get("service_kind") != "neo4j":
            raise ResourceNotFoundError("Neo4j profile not found.",details={"profile_id":profile_id})
        _apply_neo4j_env(dict(profile.get("config") or {}),secret=get_platform_secret_store().get(profile.get("secret_ref")))
        get_platform_state().set_default_service_profile(profile_id)
        os.environ["ASTROGRAPH_PLATFORM_NEO4J_ACTIVE_PROFILE_ID"]=profile_id
        return get_platform_neo4j_settings()

    @app.get("/v1/platform/agent-capabilities")
    def platform_agent_capabilities() -> dict:
        from astrograph.agents.mission.modes import MODE_OBJECTIVES, allowed_tools
        from astrograph.integrations.platform_agent import platform_agent_settings

        llm_available = get_upload_report_store().knowledge_service.llm_backend is not None
        settings = platform_agent_settings(llm_available=llm_available)
        objective_zh = {
            "normal": "使用既定证据融合诊断链，在预算内执行必要且合格的检查。",
            "thinking": "调查竞争假设、适用性与反证；没有新增证据时停止，不强行升级结论。",
            "creative": "保持已验证诊断不变；独立提出可证伪假设与离线实验方案。",
        }
        display_name = {"normal": "标准诊断", "thinking": "深入复核", "creative": "扩展排查"}
        modes = []
        for mode in settings.allowed_work_modes:
            budget_override = settings.work_mode_budgets.get(mode) if mode in {"thinking", "creative"} else None
            modes.append({
                "id": mode,
                "display_name": display_name[mode],
                "objective": MODE_OBJECTIVES[mode],
                "objective_zh": objective_zh[mode],
                "allowed_diagnose_tools": sorted(allowed_tools("diagnose", mode)),
                "budgets": {
                    "max_checks": (budget_override.max_checks if budget_override and budget_override.max_checks is not None else settings.max_checks),
                    "max_optional_actions": (budget_override.max_optional_actions if budget_override and budget_override.max_optional_actions is not None else settings.max_optional_actions),
                    "max_llm_calls": (budget_override.max_llm_calls if budget_override and budget_override.max_llm_calls is not None else settings.max_llm_calls),
                    "max_proposals": (budget_override.max_proposals if budget_override else None),
                },
            })
        return {
            "schema_version": "astrograph-platform-agent-capabilities.v1",
            "agent": "Mission Agent",
            "planner": {
                "default": settings.planner,
                "alternatives": ["deterministic", "llm_assisted"],
                "boundary_zh": "规划器只选择允许的能力与检查步骤，不生成证据、诊断标签或权限。",
            },
            "memory": {
                "configured": bool(settings.memory_db and settings.memory_policy),
                "boundary_zh": "记忆为宿主治理的已确认病例记忆；探索性内容不能自动升级为已确认记忆。",
            },
            "tools": {
                "controlled_registry": True,
                "boundary_zh": "工具由允许列表、权限、预算和审计共同约束；模型不能任意执行生成的动作。",
            },
            "execution": {
                "process_supervised": True,
                "task_timeout_s": settings.task_timeout_s,
                "subprocess_timeout_s": settings.subprocess_timeout_s,
                "boundary_zh": "执行受进程监督、超时和取消边界控制。",
            },
            "modes": modes,
            "shared_boundaries": {
                "shared_diagnostic_kernel": True,
                "mode_changes_diagnostic_thresholds": False,
                "mode_grants_permissions": False,
                "automatic_mode_escalation": False,
                "exploration_memory_policy": "TASK_LOCAL_UNVERIFIED_NEVER_CONFIRM_AUTOMATICALLY",
            },
            "platform_binding": {
                "upload_job_path_uses_existing_workflow": True,
                "mission_agent_mode_selector_changes_upload_execution": True,
                "mission_agent_mode_selector_changes_upload_diagnosis": False,
                "boundary_zh": "工作模式已接入上传任务执行：思考模式会执行额外的有界证据调查，创造模式可进一步执行离线可证伪探索；三种模式都不会修改已冻结候选、诊断阈值或 BiRRF 投票。",
            },
        }

    @app.get("/v1/platform/tools")
    def platform_tools_catalog() -> dict:
        from astrograph.integrations.platform_tools import platform_tool_catalog
        return platform_tool_catalog()

    @app.get("/v1/platform/agent-tool-specification")
    def platform_agent_tool_specification() -> dict:
        """Return the R30 governance specification; this endpoint grants no permissions."""
        from astrograph.governance.agent_tool_specification import build_agent_tool_governance_catalog
        return build_agent_tool_governance_catalog().model_dump(mode="json")

    @app.get("/v1/platform/tool-plugins")
    def platform_tool_plugins() -> dict:
        from astrograph.tools.plugins import get_trusted_agent_tool_plugin_manager
        manager = get_trusted_agent_tool_plugin_manager()
        manager.ensure_loaded()
        return manager.status()

    @app.post("/v1/platform/tool-plugins/reload")
    def reload_platform_tool_plugins() -> dict:
        from astrograph.tools.plugins import get_trusted_agent_tool_plugin_manager
        return get_trusted_agent_tool_plugin_manager().reload()

    @app.post("/v1/platform/tool-plugins/install")
    def install_platform_tool_plugin(request: PlatformToolPluginInstallRequest) -> dict:
        from astrograph.tools.plugins import get_trusted_agent_tool_plugin_manager
        try:
            return get_trusted_agent_tool_plugin_manager().install_source(
                filename=request.filename, source=request.source
            )
        except Exception as exc:
            raise InvalidOperationError(
                f"Plugin install/update failed: {type(exc).__name__}: {exc}",
                details={"filename": request.filename, "rollback": "ATTEMPTED"},
            ) from exc

    @app.delete("/v1/platform/tool-plugins/{plugin_id}")
    def uninstall_platform_tool_plugin(plugin_id: str, force: bool = Query(default=False)) -> dict:
        from astrograph.tools.plugins import get_trusted_agent_tool_plugin_manager
        try:
            return get_trusted_agent_tool_plugin_manager().uninstall(plugin_id, force=force)
        except Exception as exc:
            raise InvalidOperationError(
                f"Plugin uninstall failed: {type(exc).__name__}: {exc}",
                details={"plugin_id": plugin_id, "force": force},
            ) from exc

    @app.get("/v1/platform/tool-providers")
    def list_platform_tool_providers(probe: bool = Query(default=False)) -> dict:
        rows = get_platform_state().list_tool_providers()
        if not probe:
            return {"schema_version":"astrograph-tool-providers.v1","providers":rows}
        from astrograph.tools.providers import ToolProviderManager
        return {"schema_version":"astrograph-tool-providers.v1","providers":rows,"catalog":ToolProviderManager(get_platform_state()).catalog()}

    @app.post("/v1/platform/tool-providers")
    def save_platform_tool_provider(request: PlatformToolProviderSaveRequest) -> dict:
        config={
            "transport":"streamable_http", "url":request.url, "read_only":request.read_only,
            "tool_allowlist":request.tool_allowlist, "timeout_s":request.timeout_s,
        }
        return get_platform_state().save_tool_provider(
            provider_id=request.provider_id, provider_type=request.provider_type, name=request.name,
            config=config, enabled=request.enabled, validation_status="UNKNOWN",
        )

    @app.post("/v1/platform/tool-providers/{provider_id}/test")
    def test_platform_tool_provider(provider_id: str) -> dict:
        item=get_platform_state().get_tool_provider(provider_id)
        if not item:
            raise ResourceNotFoundError("Tool provider not found.",details={"provider_id":provider_id})
        if item.get("provider_type") != "mcp":
            raise InvalidOperationError("Unsupported provider type.")
        from astrograph.tools.providers import MCPProviderConfig, MCPToolProvider
        cfg=MCPProviderConfig.model_validate({**item.get("config",{}),"provider_id":item["provider_id"],"name":item["name"],"enabled":item["enabled"]})
        result=MCPToolProvider(cfg).health()
        get_platform_state().save_tool_provider(
            provider_id=item["provider_id"],provider_type=item["provider_type"],name=item["name"],
            config=item["config"],enabled=item["enabled"],validation_status=("PASS" if result.get("status")=="PASS" else "FAIL"),
            verified_at=datetime.now(timezone.utc).isoformat(),
        )
        return result

    @app.delete("/v1/platform/tool-providers/{provider_id}")
    def delete_platform_tool_provider(provider_id: str) -> dict:
        if not get_platform_state().get_tool_provider(provider_id):
            raise ResourceNotFoundError("Tool provider not found.",details={"provider_id":provider_id})
        return {"provider_id":provider_id,"deleted":get_platform_state().delete_tool_provider(provider_id)}

    @app.get("/v1/platform/capabilities")
    def upload_report_capabilities() -> dict:
        return {
            "schema_version": "astrograph-upload-report-capabilities.v1",
            "supported_artifact_sets": [
                {
                    "kind": "RESULT_LOG_REVIEW",
                    "required": ["analysis_input.json (astrograph-result-log-review.v1)",
                                 "diagnosis_results", "log_reports", "references"],
                    "diagnosis": "material-first root cause hypotheses, log corroboration and cited handling advice; no telemetry required",
                    "formal_release_granted": False,
                },
                {
                    "kind": "PUBLIC_CASE",
                    "required": ["manifest.json", "manifest-declared telemetry CSV"],
                    "diagnosis": "truth-blind UnifiedDiagnosisService",
                },
                {
                    "kind": "PIPELINE_RUNTIME",
                    "required": [
                        "sat-sim.diagnostic-pipeline-runtime.v1 JSON",
                        "sat-sim.astrograph-diagnostic-mapping-bundle.v1 JSON",
                    ],
                    "diagnosis": "engineering screening + native BiRRF review report",
                },
            ],
            "knowledge_documents": [".txt", ".md", ".docx", ".pdf"],
            "result_log_review": {
                "schema_version": "astrograph-result-log-review.v1",
                "accepted_roles": {
                    "diagnosis_results": [".json", ".csv", ".txt", ".md", ".docx", ".pdf"],
                    "log_reports": [".log", ".json", ".csv", ".txt", ".md", ".docx", ".pdf"],
                    "references": [".txt", ".md", ".docx", ".pdf"],
                },
                "preflight_required": True, "raw_telemetry_required": False,
                "automatic_control_authorized": False,
            },
            "oag_diagnostic_input": {
                "schema_version": "astrograph-oag-input.v1",
                "three_task_bridge": "astrograph-oag prepare-input",
                "harness": "NATIVE_RUNTIME_WITH_DURABLE_ROLE_AND_FIDELITY_RECEIPTS",
                "structured_evidence_review": True,
            },
            "material_knowledge_fusion": {
                "candidate_specific_rag":True,"offline_bm25_and_char_ngram":True,
                "optional_local_embedding_model":bool(os.environ.get('ASTROGRAPH_MATERIAL_EMBEDDING_MODEL')),
                "bound_readonly_graph_configured":bool(os.environ.get('ASTROGRAPH_MATERIAL_KG_BINDING_JSON')),
                "confirmed_case_memory":True,"review_gated_reference_library":True,
                "review_api_enabled":bool(os.environ.get('ASTROGRAPH_MATERIAL_REVIEW_TOKEN')),
                "auto_active_kg_write":False,"manual_approval_required":True,
            },
            "knowledge_services": get_upload_report_store().knowledge_service.capabilities(),
            "zip_supported": True,
            "preflight": {
                "schema_version": "astrograph-platform-preflight.v1",
                "endpoint_template": "/v1/platform/jobs/{job_id}/preflight",
                "server_revalidated_on_start": True,
                "formal_release_granted": False,
            },
            "maintenance_production_rerun": {
                "public_case_raw_binding_schema_version": "sat-sim.astrograph-maintenance-raw-input-binding.v1",
                "public_case_rerun_mode": "PUBLIC_CASE_RAW_TELEMETRY_PATCH_RERUN",
                "pipeline_runtime_adapter_contract_schema_version": "astrograph-upstream-recompute-request.v1",
                "pipeline_runtime_in_place_patch_allowed": False,
                "source_job_mutated": False,
                "formal_release_granted": False,
            },
            "upstream_recompute": get_upload_report_store().upstream_provider_capabilities(),
            "claim_boundary": (
                "Upload automation does not bypass evidence, truth-isolation, or review gates. "
                "Knowledge enrichment is post-diagnosis and cannot change BiRRF or the frozen candidate."
            ),
        }

    def _material_review_memory(request: Request, *, write: bool = False, publisher: bool = False):
        """Credential-header compatibility or an HttpOnly, CSRF-protected expert session."""
        from astrograph.integrations.material_case_memory import MaterialCaseMemory
        if app.state.auth_config.mode == 'accounts':
            identity = getattr(request.state, 'auth_identity', None)
            if identity is None or not identity.permits('reviewer'):
                raise HTTPException(status_code=403,detail='Reviewer role required')
            if publisher and not identity.permits('admin'):
                raise HTTPException(status_code=403,detail='Administrator role required')
            from astrograph.integrations.material_case_memory import MaterialCaseMemory
            return MaterialCaseMemory(get_upload_report_store().root/'material_knowledge'/'cases.sqlite3',
                                      reviewers={identity.subject}), identity.subject
        secret, reviewer, reviewers = app.state.material_review_sessions.config()
        supplied = request.headers.get('x-material-review-token', '')
        if not (supplied and hmac.compare_digest(supplied, secret)):
            session = app.state.material_review_sessions.authenticate(request, require_csrf=write)
            if session is None:
                raise HTTPException(status_code=403, detail='Expert login required')
            reviewer = session.actor
        if publisher:
            publishers = {x.strip() for x in os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_PUBLISHERS', '').split(',') if x.strip()}
            if reviewer not in publishers:
                raise HTTPException(status_code=403, detail='Graph publication requires a designated administrator')
        return MaterialCaseMemory(get_upload_report_store().root/'material_knowledge'/'cases.sqlite3', reviewers=reviewers), reviewer

    @app.get('/v1/platform/material-knowledge/review-session')
    def material_review_session(request: Request, response: Response) -> dict:
        """Session state only. Never disclose the server's bootstrap secret."""
        response.headers['Cache-Control'] = 'no-store'
        if app.state.auth_config.mode == 'accounts':
            identity = getattr(request.state,'auth_identity',None)
            approved = bool(identity and identity.permits('reviewer'))
            return {'configured': True, 'authenticated': approved,
                    'reviewer':identity.subject if approved else None,
                    'publisher':bool(identity and identity.permits('admin')),
                    'csrf':None, 'graph_model_configured':bool(os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_LLM_MODEL')),
                    'release_signing_configured':len(os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','')) >= 32,
                    'neo4j_publication_enabled':os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED')=='1',
                    'session_ttl_seconds':0}
        configured = True
        try:
            _secret, reviewer, _reviewers = app.state.material_review_sessions.config()
        except HTTPException:
            configured = False
            reviewer = ''
        session = app.state.material_review_sessions.authenticate(request) if configured else None
        publishers = {x.strip() for x in os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_PUBLISHERS', '').split(',') if x.strip()}
        return {'configured':configured,'authenticated':bool(session),
                'reviewer':reviewer if session else None,
                'publisher':bool(session and reviewer in publishers),
                'csrf':session.csrf if session else None,
                'graph_model_configured':bool((os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_LLM_BASE_URL') or os.environ.get('ASTROGRAPH_PLATFORM_LLM_BASE_URL')) and
                                               (os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_LLM_MODEL') or os.environ.get('ASTROGRAPH_PLATFORM_LLM_MODEL'))),
                'release_signing_configured':len(os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY',''))>=32,
                'neo4j_publication_enabled':os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED')=='1',
                'session_ttl_seconds':TTL_SECONDS}

    @app.post('/v1/platform/material-knowledge/review-session')
    def material_review_login(credentials: MaterialReviewLogin, request: Request, response: Response) -> dict:
        if app.state.auth_config.mode == 'accounts':
            raise HTTPException(status_code=403,detail='请使用平台账户登录')
        token, session = app.state.material_review_sessions.login(request,credentials.credential)
        response.set_cookie(COOKIE_NAME,token,httponly=True,samesite='strict',
            secure=app.state.material_review_sessions.secure_transport(request),path='/v1/platform/material-knowledge',
            max_age=TTL_SECONDS)
        response.headers['Cache-Control']='no-store'
        return {'authenticated':True,'reviewer':session.actor,'csrf':session.csrf}

    @app.delete('/v1/platform/material-knowledge/review-session')
    def material_review_logout(request: Request, response: Response) -> dict:
        # A logout also changes server state: require a valid CSRF token.
        _material_review_memory(request,write=True)
        app.state.material_review_sessions.logout(request)
        response.delete_cookie(COOKIE_NAME,path='/v1/platform/material-knowledge',samesite='strict')
        response.headers['Cache-Control']='no-store'
        return {'authenticated':False}

    def _report_review_context(job_id: str):
        """Resolve existing, completed result-log review; never trust client candidate IDs."""
        job_store = get_upload_report_store()
        try:
            report = job_store.report(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='Job not found') from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail='Report not ready for review') from exc
        if report.get('artifact_kind') != 'RESULT_LOG_REVIEW':
            raise HTTPException(status_code=422, detail='Not a material analysis report')
        from astrograph.integrations.manual_report_review import ManualReportReviewStore
        reviews = ManualReportReviewStore(job_store.root/'material_knowledge'/'manual_report_reviews.sqlite3')
        return report, reviews

    @app.get('/v1/platform/jobs/{job_id}/manual-review/status')
    def manual_report_review_status(job_id: str, response: Response) -> dict:
        """Operator-visible summary; does not expose reviewer comments or audit records."""
        response.headers['Cache-Control']='no-store'
        report, reviews = _report_review_context(job_id)
        return reviews.detail(job_id, report)

    @app.get('/v1/platform/material-knowledge/report-reviews')
    def material_report_review_queue(request: Request, response: Response,
                                     limit: int = Query(default=80, ge=1, le=200)) -> dict:
        _material_review_memory(request)
        response.headers['Cache-Control']='no-store'
        store=get_upload_report_store()
        jobs=store.overview(limit=limit).get('recent_jobs', [])
        items=[]
        for job in jobs:
            if job.get('artifact_kind') != 'RESULT_LOG_REVIEW' or job.get('status')!='SUCCEEDED':
                continue
            try:
                report, review_store=_report_review_context(str(job['job_id']))
                item=review_store.detail(str(job['job_id']), report)
                items.append({**item, 'completed_at':job.get('completed_at')})
            except (KeyError, ValueError, HTTPException):
                continue
        return {'schema_version':'astrograph-manual-report-review-queue.v1', 'items':items}

    @app.get('/v1/platform/material-knowledge/report-reviews/{job_id}')
    def material_report_review_detail(job_id: str, request: Request, response: Response) -> dict:
        _material_review_memory(request)
        response.headers['Cache-Control']='no-store'
        report, reviews = _report_review_context(job_id)
        return reviews.detail(job_id, report, include_history=True)

    @app.post('/v1/platform/material-knowledge/report-reviews/{job_id}/decisions')
    def material_report_review_decide(job_id: str, decision: ManualReportReviewDecision,
                                     request: Request, response: Response) -> dict:
        _store, actor=_material_review_memory(request, write=True)
        response.headers['Cache-Control']='no-store'
        report, reviews = _report_review_context(job_id)
        from astrograph.integrations.manual_report_review import ManualReviewConflict
        try:
            return reviews.record(job_id=job_id, report=report,
                report_sha256=decision.report_sha256, fault_id=decision.fault_id,
                decision=decision.decision, evidence_ref=decision.evidence_ref,
                note=decision.note, actor=actor, expected_revision=decision.expected_revision)
        except ManualReviewConflict as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.get('/v1/platform/material-knowledge/pending')
    def pending_material_reviews(request: Request, limit: int = Query(default=30,ge=1,le=100)) -> dict:
        store, _actor = _material_review_memory(request)
        return {'snapshot':store.snapshot(),'cases':store.pending(limit=limit),
                'documents':store.pending_documents(limit=limit),
                'graph_proposals':store.pending_graph(limit=limit),
                'reasoning_gaps':store.pending_reasoning_gaps(limit=limit)}

    @app.get('/v1/platform/material-knowledge/reasoning/gaps')
    def pending_material_reasoning_gaps(request: Request,
                                      equipment_scope: str | None = None,
                                      limit: int = Query(default=40,ge=1,le=100)) -> dict:
        store, _actor = _material_review_memory(request)
        try:
            return {'status':'EXPERT_REVIEW_QUEUE', 'snapshot':store.snapshot(),
                    'knowledge_gaps':store.pending_reasoning_gaps(
                        equipment_scope=equipment_scope,limit=limit),
                    'graph_promoted':False}
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/reasoning/gaps/{gap_id}/resolve')
    def resolve_material_reasoning_gap(gap_id: str,
                body: MaterialReasoningGapResolveRequest,request: Request) -> dict:
        store, actor = _material_review_memory(request,write=True)
        try:
            return store.resolve_reasoning_gap(
                gap_id=gap_id,reviewer=actor,resolution_ref=body.resolution_ref)
        except KeyError as exc:
            raise HTTPException(status_code=404,detail='Knowledge gap not found') from exc
        except (ValueError,PermissionError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.get('/v1/platform/material-knowledge/graph/status')
    def get_material_graph_status(request: Request, equipment_scope: str | None = None) -> dict:
        store,_actor=_material_review_memory(request)
        try:
            return store.graph_status(equipment_scope=equipment_scope)
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/cases/{case_id}/decision')
    def decide_material_case(case_id: str, decision: MaterialCaseReviewDecision,request: Request) -> dict:
        store, actor = _material_review_memory(request,write=True)
        try:
            return store.decide(case_id=case_id,reviewer=actor,decision=decision.decision,
                  truth_source_ref=decision.truth_source_ref,verified_fault_id=decision.verified_fault_id,
                  equipment_scope=decision.equipment_scope,approved_actions=decision.approved_actions,
                  expected_snapshot=decision.expected_snapshot)
        except KeyError as exc:
            raise HTTPException(status_code=404,detail='Unknown pending material case') from exc
        except (ValueError,PermissionError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/documents/{document_id}/decision')
    def decide_material_document(document_id: str, decision: MaterialDocumentReviewDecision,request: Request) -> dict:
        store, actor = _material_review_memory(request,write=True)
        try:
            return store.decide_document(document_id=document_id,reviewer=actor,
                  decision=decision.decision,truth_source_ref=decision.truth_source_ref,
                  equipment_scope=decision.equipment_scope,expected_snapshot=decision.expected_snapshot)
        except KeyError as exc:
            raise HTTPException(status_code=404,detail='Unknown pending reference document') from exc
        except (ValueError,PermissionError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/graph/{proposal_id}/decision')
    def decide_material_graph(proposal_id: str, decision: MaterialDocumentReviewDecision,request: Request) -> dict:
        store,actor=_material_review_memory(request,write=True)
        try:
            return store.decide_graph(proposal_id=proposal_id,reviewer=actor,
                 decision=decision.decision,truth_source_ref=decision.truth_source_ref,
                 equipment_scope=decision.equipment_scope,expected_snapshot=decision.expected_snapshot)
        except KeyError as exc:
            raise HTTPException(status_code=404,detail='Unknown graph proposal') from exc
        except (ValueError,PermissionError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/graph/{proposal_id}/retract')
    def retract_material_graph(proposal_id: str, correction: MaterialGraphRetractionRequest,
                               request: Request) -> dict:
        store,actor=_material_review_memory(request,write=True)
        try:
            return store.retract_graph(proposal_id=proposal_id,reviewer=actor,
                reason=correction.reason,truth_source_ref=correction.truth_source_ref,
                expected_snapshot=correction.expected_snapshot)
        except KeyError as exc:
            raise HTTPException(status_code=404,detail='Approved graph relation not found') from exc
        except (ValueError,PermissionError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/graph/build')
    def build_graph_from_references(request: Request,
                                    reference_files: list[UploadFile] = File(...),
                                    equipment_scope: str = Form(...),
                                    max_chunks: int = Form(default=8,ge=1,le=12)) -> dict:
        """Expert-only graph construction from bounded uploaded reference documents.

        This action stages untrusted LLM suggestions; it NEVER approves or publishes.
        Source files are retained under an isolated, generated source-group ID for audit.
        """
        store, actor = _material_review_memory(request,write=True)
        from astrograph.integrations.material_graph_builder import build_material_graph
        from astrograph.integrations.platform_knowledge import UploadedDocumentKnowledgeIndex
        from astrograph.integrations.material_case_memory import MaterialCaseMemory
        from astrograph.integrations.upload_report import _SAFE_FILENAME
        try:
            scope=MaterialCaseMemory._scope(equipment_scope)
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc
        if not 1 <= len(reference_files) <= 8:
            raise HTTPException(status_code=422,detail='Select 1-8 reference documents')
        names=[f.filename or '' for f in reference_files]
        if (len({name.casefold() for name in names})!=len(names) or
            any(not _SAFE_FILENAME.fullmatch(name) or name in {'.','..'} or
                Path(name).name!=name or Path(name).suffix.lower() not in {'.pdf','.docx','.txt','.md'} or
                any(ord(c)<32 for c in name) for name in names)):
            raise HTTPException(status_code=422,detail='Unsafe, unsupported or duplicate document names')
        group='JOB-'+secrets.token_hex(12).upper()
        root=get_upload_report_store().root/'material_knowledge'/'graph_sources'/group
        root.mkdir(parents=True,exist_ok=False)
        receipts=[]
        total=0
        try:
            for file in reference_files:
                data=file.file.read(16*1024*1024+1)
                if len(data)>16*1024*1024 or not data:
                    raise ValueError('Reference file is empty or exceeds 16 MiB')
                total+=len(data)
                if total>64*1024*1024:
                    raise ValueError('Total reference size exceeds 64 MiB')
                (root/file.filename).write_bytes(data)
                receipts.append({'filename':file.filename,'bytes':len(data),
                                 'sha256':hashlib.sha256(data).hexdigest()})
            (root/'source_manifest.json').write_text(json.dumps({'source_group_id':group,
                 'equipment_scope_requested':scope,'uploader_role':'AUTHENTICATED_REVIEWER',
                 'files':receipts},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        except (OSError,ValueError) as exc:
            shutil.rmtree(root,ignore_errors=True)
            raise HTTPException(status_code=422,detail=str(exc)) from exc
        # Model exceptions are surfaced as bounded status/error types by the
        # underlying extract-and-stage service. No false "success" claims.
        chunks,ingestion=UploadedDocumentKnowledgeIndex(top_k=8).build([root/name for name in names])
        result=build_material_graph(chunks=chunks,source_group_id=group,memory=store,
                                    enabled=True,max_chunks=max_chunks)
        result['source_group_id']=group
        result['equipment_scope_requested']=scope
        result['document_ingestion']=ingestion
        result['source_receipts']=receipts
        (root/'build_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return result

    @app.get('/v1/platform/material-knowledge/graph/sources/{source_group_id}/{filename}')
    def get_material_graph_reference(source_group_id: str,filename: str,request: Request):
        _material_review_memory(request)
        if not re.fullmatch(r'JOB-[0-9A-F]{24}',source_group_id) or Path(filename).name!=filename:
            raise HTTPException(status_code=404,detail='Reference unavailable')
        root=get_upload_report_store().root/'material_knowledge'/'graph_sources'/source_group_id
        manifest=root/'source_manifest.json'
        if not manifest.is_file():
            raise HTTPException(status_code=404,detail='Reference unavailable')
        refs=json.loads(manifest.read_text(encoding='utf-8')).get('files',[])
        item=next((f for f in refs if f.get('filename')==filename),None)
        target=root/filename
        if not item or not target.is_file() or target.resolve().parent!=root.resolve() or (target.stat().st_size!=item['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']):
            raise HTTPException(status_code=404,detail='Reference unavailable or checksum mismatch')
        return FileResponse(target,filename=filename,headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})

    @app.get('/v1/platform/material-knowledge/graph/reviewed')
    def list_reviewed_material_graph(request: Request,equipment_scope: str,
                                      limit: int = Query(default=80,ge=1,le=200)) -> dict:
        store,_actor=_material_review_memory(request)
        try:
            scope=store._scope(equipment_scope)
            snapshot=store.snapshot()
            claims=store._approved_graph_at(equipment_scope=scope,snapshot=snapshot)
            return {'equipment_scope':scope,'snapshot':snapshot,'total':len(claims),
                    'relations':claims[:limit], 'truncated':len(claims)>limit,
                    'retractions':store.graph_retraction_audit(equipment_scope=scope,snapshot=snapshot)[:limit]}
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/graph/export')
    def export_material_graph(request: Request, release: MaterialGraphReleaseRequest) -> dict:
        store,_actor=_material_review_memory(request,write=True,publisher=True)
        from astrograph.integrations.material_graph_release import verify_graph_release
        try:
            scope=store._scope(release.equipment_scope)
            snapshot=store.snapshot() if release.snapshot is None else release.snapshot
            out=get_upload_report_store().root/'material_knowledge'/'graph_releases'
            path=out/f'{scope}-SNAPSHOT-{snapshot:08d}'
            if path.exists():
                manifest,_=verify_graph_release(path)
                return {**manifest,'already_exported':True}
            return store.export_graph_release(equipment_scope=scope,snapshot=snapshot,output_root=out)
        except (ValueError,FileExistsError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post('/v1/platform/material-knowledge/graph/publish')
    def publish_material_graph(request: Request, publish: MaterialGraphPublishRequest) -> dict:
        store, _actor = _material_review_memory(request,write=True,publisher=True)
        from astrograph.integrations.material_graph_release import publish_release_neo4j
        from astrograph.integrations.material_case_memory import MaterialCaseMemory
        if not publish.confirm_write:
            raise HTTPException(status_code=422,detail='Explicit Neo4j write confirmation required')
        try:
            scope=MaterialCaseMemory._scope(publish.equipment_scope)
            if publish.snapshot is None:
                raise ValueError('Publish must pin an explicit snapshot')
            if publish.snapshot != store.snapshot():
                raise ValueError('Selected graph snapshot is stale; reload reviewed knowledge before publishing')
            release_id=f'{scope}-SNAPSHOT-{publish.snapshot:08d}'
            if publish.confirm_release_id!=release_id:
                raise ValueError('Release ID confirmation does not match selected snapshot')
            path=get_upload_report_store().root/'material_knowledge'/'graph_releases'/release_id
            return publish_release_neo4j(path,confirm_write=True)
        except (ValueError,PermissionError,FileNotFoundError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @app.post("/v1/platform/jobs")
    def create_upload_report_job(request: PlatformJobCreateRequest) -> dict:
        return get_upload_report_store().create(
            request.task, request.work_mode, request.analysis_engine_id,
        )

    @app.get("/v1/platform/overview")
    def get_platform_overview(limit: int = Query(default=50, ge=1, le=200)) -> dict:
        return get_upload_report_store().overview(limit=limit)

    @app.put("/v1/platform/jobs/{job_id}/files/{filename}")
    async def upload_report_file(job_id: str, filename: str, request: Request) -> dict:
        data = await request.body()
        try:
            return get_upload_report_store().put_file(job_id, filename, data)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "filename": filename}) from exc

    @app.post("/v1/platform/jobs/{job_id}/preflight")
    def preflight_upload_report_job(job_id: str) -> dict:
        try:
            return get_upload_report_store().preflight(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/run")
    def run_upload_report_job(job_id: str, response: Response) -> dict:
        try:
            result = get_upload_report_store().run(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc
        if result.get("status") != "SUCCEEDED":
            response.status_code = 422
        return result

    @app.post("/v1/platform/jobs/{job_id}/start", status_code=202)
    def start_upload_report_job(job_id: str) -> dict:
        try:
            return get_upload_report_store().start(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}")
    def get_upload_report_job(job_id: str) -> dict:
        try:
            return get_upload_report_store().get(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/diagnostics")
    def get_upload_report_job_diagnostics(job_id: str) -> dict:
        try:
            return get_upload_report_store().diagnostics(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/report")
    def get_upload_report(job_id: str) -> dict:
        try:
            return get_upload_report_store().report(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/visualization")
    def get_upload_report_visualization(job_id: str) -> dict:
        try:
            return get_upload_report_store().visualization(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/temporal-localization")
    def get_platform_temporal_localization(job_id: str) -> dict:
        try:
            return get_upload_report_store().temporal_localization(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.",details={"job_id":job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc),details={"job_id":job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/multidimensional-assessment")
    def get_platform_multidimensional_assessment(job_id: str) -> dict:
        try:
            return get_upload_report_store().multidimensional_assessment(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.",details={"job_id":job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc),details={"job_id":job_id}) from exc

    @app.get("/v1/platform/diagnostic-history")
    def get_platform_diagnostic_history(fault_id: str | None = Query(default=None, max_length=200), limit: int = Query(default=200, ge=1, le=2000)) -> dict:
        return get_upload_report_store().diagnostic_history(fault_id=fault_id, limit=limit)

    @app.get("/v1/platform/jobs/{job_id}/maintenance")
    def get_platform_maintenance(job_id: str) -> dict:
        try:
            payload=get_upload_report_store().maintenance(job_id)
            payload["annotations"]=get_upload_report_store().maintenance_annotations(job_id)
            return payload
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.",details={"job_id":job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc),details={"job_id":job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/diagnostic-workspace")
    def get_platform_diagnostic_workspace(job_id: str) -> dict:
        try:
            return get_upload_report_store().diagnostic_workspace(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/input-contracts")
    def get_platform_maintenance_input_contracts(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_input_contracts(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/input-contracts/{contract_id}/template")
    def get_platform_maintenance_input_contract_template(job_id: str, contract_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_input_contract_template(job_id, contract_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "contract_id": contract_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/input-contracts/{contract_id}/admit-feature-input")
    def admit_platform_maintenance_feature_input(
        job_id: str, contract_id: str, capsule: SupplementalFeatureInputCapsule, http_request: Request
    ) -> dict:
        try:
            actor = authenticated_actor(http_request, "operator")
            return get_upload_report_store().admit_maintenance_feature_input(
                job_id, contract_id=contract_id, capsule_payload=capsule.model_dump(mode="json"), actor=actor
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "contract_id": contract_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/supplemental-admissions")
    def get_platform_maintenance_supplemental_admissions(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_supplemental_admissions(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/supplemental-provenance")
    def get_platform_maintenance_supplemental_provenance(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_supplemental_provenance(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/upstream/providers")
    def get_platform_upstream_providers() -> dict:
        return get_upload_report_store().upstream_provider_capabilities()

    @app.get("/v1/platform/jobs/{job_id}/maintenance/upstream-recompute")
    def get_platform_upstream_recompute_request(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_upstream_recompute_request(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/upstream-recompute/request-bundle")
    def download_platform_upstream_recompute_request_bundle(job_id: str) -> FileResponse:
        try:
            path = get_upload_report_store().maintenance_upstream_recompute_request_bundle_path(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"{job_id}-upstream-recompute-request.zip",
        )

    @app.post("/v1/platform/jobs/{job_id}/maintenance/upstream-recompute/manual-response")
    async def import_platform_upstream_recompute_manual_response(
        job_id: str, http_request: Request, start: bool = Query(default=True)
    ) -> dict:
        try:
            actor = authenticated_actor(http_request, "operator")
            data = await http_request.body()
            return get_upload_report_store().create_manual_upstream_recompute_child(
                job_id, response_bundle=data, actor=actor, start=start
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/upstream-recompute/history")
    def get_platform_upstream_recompute_history(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_upstream_recompute_history(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/production-rerun/raw-input-contracts")
    def get_platform_maintenance_production_raw_input_contracts(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_production_raw_input_contracts(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/production-rerun/raw-input-contracts/{contract_id}/template")
    def get_platform_maintenance_production_raw_input_template(job_id: str, contract_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_production_raw_input_template(job_id, contract_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "contract_id": contract_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/production-rerun/raw-input-contracts/{contract_id}/admit")
    def admit_platform_maintenance_production_raw_input(
        job_id: str, contract_id: str, capsule: ProductionRawInputPatchCapsule, http_request: Request
    ) -> dict:
        try:
            actor = authenticated_actor(http_request, "operator")
            return get_upload_report_store().admit_maintenance_production_raw_input(
                job_id, contract_id=contract_id, capsule_payload=capsule.model_dump(mode="json"), actor=actor
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "contract_id": contract_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/production-reruns")
    def get_platform_maintenance_production_reruns(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_production_reruns(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/production-rerun-readiness")
    def get_platform_maintenance_production_rerun_readiness(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_production_rerun_readiness(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/production-rerun")
    def create_platform_maintenance_production_rerun(
        job_id: str, request: PlatformMaintenanceRerunRequest, http_request: Request
    ) -> dict:
        try:
            actor = authenticated_actor(http_request, "operator")
            return get_upload_report_store().create_maintenance_production_rerun(
                job_id, actor=actor, start=request.start
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/governed-reruns")
    def get_platform_maintenance_governed_reruns(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_governed_reruns(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/maintenance/governed-rerun-readiness")
    def get_platform_maintenance_governed_rerun_readiness(job_id: str) -> dict:
        try:
            return get_upload_report_store().maintenance_governed_rerun_readiness(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/governed-rerun")
    def create_platform_maintenance_governed_rerun(
        job_id: str, request: PlatformMaintenanceRerunRequest, http_request: Request
    ) -> dict:
        try:
            actor = authenticated_actor(http_request, "operator")
            return get_upload_report_store().create_maintenance_governed_rerun(
                job_id, actor=actor, start=request.start
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/diagnostic-probe")
    def run_platform_maintenance_diagnostic_probe(
        job_id: str, request: PlatformMaintenanceProbeRequest, http_request: Request
    ) -> dict:
        try:
            actor = authenticated_actor(http_request, "operator")
            return get_upload_report_store().run_maintenance_diagnostic_probe(
                job_id, probe_id=request.probe_id, actor=actor
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "probe_id": request.probe_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/neo4j-review")
    def run_platform_neo4j_review(job_id: str, http_request: Request) -> dict:
        try:
            actor=authenticated_actor(http_request,"operator")
            result=get_upload_report_store().run_neo4j_maintenance_review(job_id,actor=actor)
            if result.get("status") == "NOT_CONFIGURED":
                raise InvalidOperationError("Neo4j is not configured for this platform instance.")
            return result
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.",details={"job_id":job_id}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc),details={"job_id":job_id}) from exc

    @app.post("/v1/platform/jobs/{job_id}/maintenance/annotations")
    def add_platform_maintenance_annotation(job_id: str, request: PlatformMaintenanceAnnotationRequest, http_request: Request) -> dict:
        try:
            actor=authenticated_actor(http_request,"operator")
            return get_upload_report_store().add_maintenance_annotation(
                job_id,annotation_type=request.annotation_type,payload={"time_s":request.time_s,"note":request.note},actor=actor
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.",details={"job_id":job_id}) from exc

    @app.get("/v1/platform/jobs/{job_id}/documents/{filename}")
    def get_upload_report_document(job_id: str, filename: str) -> FileResponse:
        try:
            path = get_upload_report_store().document_path(job_id, filename)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except FileNotFoundError as exc:
            raise ResourceNotFoundError("Knowledge document not found.", details={"job_id": job_id, "filename": filename}) from exc
        except ValueError as exc:
            raise InvalidOperationError(str(exc), details={"job_id": job_id, "filename": filename}) from exc
        media_types = {".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
        return FileResponse(path, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/v1/platform/jobs/{job_id}/report.md")
    def get_upload_report_markdown(job_id: str) -> FileResponse:
        try:
            path = get_upload_report_store().report_markdown_path(job_id)
        except KeyError as exc:
            raise ResourceNotFoundError("Platform job not found.", details={"job_id": job_id}) from exc
        except ValueError as exc:
            raise StateConflictError(str(exc), details={"job_id": job_id}) from exc
        return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=f"{job_id}-report.md")

    app.include_router(build_compat_router())
    return app


app = create_app()
