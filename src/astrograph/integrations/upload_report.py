"""Upload-to-report automation for governed Satellite Simulation and material review artifacts.

The platform layer is deliberately source-specific: it stages immutable uploads,
selects one of the existing compatibility workflows, records hashes, and produces
a reviewable report. It does not infer a fault from filenames, user prose, or
evaluation labels.

Supported artifact sets:
* a truth-blind public case package (manifest.json + the declared telemetry file);
* a diagnostic pipeline runtime JSON plus its mapping bundle.

Free-form TXT/Markdown/DOCX/PDF files are accepted as session knowledge. They are
parsed and retrieved only after the diagnosis is frozen, and therefore cannot vote
in candidate generation or BiRRF. Optional PaperSeek/Anything2Ontology/LLM services
are administrator-configured and fail explicitly rather than producing mock results.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import sqlite3
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from astrograph.audit.store import sha256_json
from astrograph.integrations.sat_sim_public_case import (
    PUBLIC_CASE_SCHEMA_VERSION,
    SIMULATION_HANDOFF_SCHEMA_VERSION,
    diagnose_public_case,
)
from astrograph.integrations.sat_sim_pipeline import (
    SatSimPipelineClosureRequest,
    SatSimPipelineClosureWorkflow,
)
from astrograph.integrations.post_diagnosis_explanation import run_frozen_post_diagnosis_explanation
from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
from astrograph.integrations.platform_secrets import PlatformSecretStore
from astrograph.integrations.platform_state import PlatformStateRepository
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.agents.mission.models import WorkMode


PLATFORM_JOB_SCHEMA_VERSION = "astrograph-upload-report-job.v1"
PLATFORM_REPORT_SCHEMA_VERSION = "astrograph-upload-report.v1"
PLATFORM_PREFLIGHT_SCHEMA_VERSION = "astrograph-platform-preflight.v1"
PIPELINE_RUNTIME_SCHEMA_VERSION = "sat-sim.diagnostic-pipeline-runtime.v1"
PIPELINE_MAPPING_SCHEMA_VERSION = "sat-sim.astrograph-diagnostic-mapping-bundle.v1"

_SAFE_FILENAME = re.compile(r"^[^/\\\x00]{1,180}$")
_TEXT_SUFFIXES = {".txt", ".md"}
_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}
_ALLOWED_SUFFIXES = {".json", ".csv", ".txt", ".md", ".docx", ".pdf", ".zip", ".yaml", ".yml", ".log"}
_MAX_FILES = 32
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 96 * 1024 * 1024
_MAX_ZIP_MEMBERS = 128
_MAX_EXPANDED_BYTES = 128 * 1024 * 1024
_MAX_TEXT_CHARS = 12000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist JSON so polling readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(raw, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            # Windows scanners/indexers may transiently hold the destination.
            # Keep replacement atomic and bound the total retry delay.
            time.sleep(0.05 * (attempt + 1))


def _json_read(path: Path) -> dict[str, Any]:
    """Read one atomically replaced JSON file across transient Windows locks."""
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            # A concurrent os.replace or scanner may briefly deny even read access.
            # Retry only the transient Windows error and keep the delay bounded.
            time.sleep(0.02 * (attempt + 1))
    raise AssertionError("unreachable JSON read retry state")


class PlatformJobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task: str = Field(default="分析诊断结果、日志和参考文献，给出有依据的故障原因和处置建议", max_length=4000)
    work_mode: WorkMode = "normal"
    analysis_engine_id: str = Field(default="auto", min_length=1, max_length=180)

    @model_validator(mode="after")
    def validate_analysis_engine(self) -> "PlatformJobCreateRequest":
        engine = self.analysis_engine_id.strip()
        if engine not in {"auto", "rules"} and not re.fullmatch(r"llm:[A-Za-z0-9_-]{1,160}", engine):
            raise ValueError("analysis_engine_id must be auto, rules, or llm:<profile_id>")
        self.analysis_engine_id = engine
        return self


class PlatformFileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    filename: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PreflightBlockedError(ValueError):
    """Raised when immutable uploaded inputs cannot enter a supported diagnostic path."""



class UploadReportJobStore:
    """Small persistent local job store with immutable post-run inputs."""

    def __init__(
        self, root: str | Path, *,
        knowledge_service: PlatformKnowledgeEnrichmentService | None = None,
        state_repository: PlatformStateRepository | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_repository = state_repository or PlatformStateRepository.from_environment(platform_root=self.root)
        self.knowledge_service = knowledge_service or PlatformKnowledgeEnrichmentService.from_environment()

    @classmethod
    def from_environment(cls) -> "UploadReportJobStore":
        root = Path(os.environ.get("ASTROGRAPH_PLATFORM_ROOT") or (Path.home() / ".astrograph" / "platform")).expanduser().resolve()
        state = PlatformStateRepository.from_environment(platform_root=root)
        return cls(root, state_repository=state)

    def _job_root(self, job_id: str) -> Path:
        if not re.fullmatch(r"JOB-[0-9A-F]{24}", job_id):
            raise KeyError("invalid job id")
        root = (self.root / job_id).resolve()
        if root.parent != self.root:
            raise KeyError("invalid job id")
        return root

    def _metadata_path(self, job_id: str) -> Path:
        return self._job_root(job_id) / "job.json"

    def create(
        self,
        task: str,
        work_mode: WorkMode = "normal",
        analysis_engine_id: str = "auto",
    ) -> dict[str, Any]:
        normalized = (task or "").strip() or "分析诊断结果、日志和参考文献，给出有依据的故障原因和处置建议"
        engine_id = (analysis_engine_id or "auto").strip()
        if work_mode not in {"normal", "thinking", "creative"}:
            raise ValueError("invalid work mode")
        if engine_id not in {"auto", "rules"} and not engine_id.startswith("llm:"):
            raise ValueError("invalid analysis engine")
        if engine_id.startswith("llm:"):
            profile = self.state_repository.get_service_profile(engine_id.split(":", 1)[1])
            if not profile or profile.get("service_kind") != "llm":
                raise ValueError("selected LLM profile does not exist")
            config = dict(profile.get("config") or {})
            if not config.get("enabled", True) or not config.get("base_url") or not config.get("model"):
                raise ValueError("selected LLM profile is not enabled")
        if len(normalized) > 4000:
            raise ValueError("task exceeds 4000 characters")
        while True:
            job_id = "JOB-" + secrets.token_hex(12).upper()
            root = self.root / job_id
            try:
                root.mkdir()
                break
            except FileExistsError:
                continue
        (root / "input").mkdir()
        (root / "output").mkdir()
        payload = {
            "schema_version": PLATFORM_JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "status": "CREATED",
            "task": normalized,
            "work_mode": work_mode,
            "analysis_engine_id": engine_id,
            "analysis_engine": self._analysis_engine_snapshot(engine_id),
            "created_at": _now(),
            "updated_at": _now(),
            "artifact_kind": None,
            "files": [],
            "preflight": None,
            "stage": "AWAITING_INPUT",
            "stage_detail": "等待上传诊断结果、日志与参考文献，或旧版诊断工件",
            "execution_backend": None,
            "llm_runtime": self._analysis_engine_snapshot(engine_id),
            "llm_call_status": "NOT_STARTED",
            "error": None,
            "report_json": None,
            "report_markdown": None,
        }
        _json_write(root / "job.json", payload)
        return payload

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._metadata_path(job_id)
        if not path.is_file():
            raise KeyError(job_id)
        return _json_read(path)

    def _save(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_at"] = _now()
        _json_write(self._metadata_path(str(payload["job_id"])), payload)
        # SQLite is the queryable control plane; immutable artifacts remain file-backed.
        # A control-plane indexing failure must not corrupt or discard the governed job artifact.
        try:
            self.state_repository.upsert_job(payload, report_path=payload.get("report_json"))
        except Exception:
            pass

    def _append_event(self, job_id: str, message: str) -> None:
        root = self._job_root(job_id)
        line = f"{_now()} {message.strip()}\n"
        with (root / "events.log").open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(line)

    def _set_stage(self, job_id: str, stage: str, detail: str = "", **updates: Any) -> dict[str, Any]:
        job = self.get(job_id)
        job["stage"] = stage
        job["stage_detail"] = detail
        for key, value in updates.items():
            job[key] = value
        self._save(job)
        self._append_event(job_id, f"[{stage}] {detail}" if detail else f"[{stage}]")
        return job

    @staticmethod
    def _llm_runtime_snapshot() -> dict[str, Any]:
        return {
            "configured": bool(os.environ.get("ASTROGRAPH_PLATFORM_LLM_BASE_URL") and os.environ.get("ASTROGRAPH_PLATFORM_LLM_MODEL")),
            "provider": os.environ.get("ASTROGRAPH_PLATFORM_LLM_PROVIDER", "local-openai-compatible"),
            "base_url": os.environ.get("ASTROGRAPH_PLATFORM_LLM_BASE_URL") or None,
            "model": os.environ.get("ASTROGRAPH_PLATFORM_LLM_MODEL") or None,
            "timeout_s": os.environ.get("ASTROGRAPH_PLATFORM_LLM_TIMEOUT_S", "120"),
            "secret_returned": False,
            "diagnostic_vote_allowed": False,
        }

    def analysis_engines(self) -> dict[str, Any]:
        """Return operator-safe engine choices without endpoints or credentials."""
        runtime = self._llm_runtime_snapshot()
        engines: list[dict[str, Any]] = [{
            "engine_id": "rules",
            "kind": "rules",
            "name": "规则引擎",
            "description": "不调用大模型，使用确定性证据规则生成结果",
            "available": True,
            "is_default": not runtime["configured"],
        }]
        active_profile_id = os.environ.get("ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID", "").strip()
        profile_ids: set[str] = set()
        for profile in self.state_repository.list_service_profiles("llm"):
            config = dict(profile.get("config") or {})
            profile_id = str(profile.get("profile_id") or "")
            if not profile_id or not config.get("enabled", True) or not config.get("base_url") or not config.get("model"):
                continue
            profile_ids.add(profile_id)
            engines.append({
                "engine_id": f"llm:{profile_id}",
                "kind": "llm",
                "name": str(profile.get("name") or config.get("model") or "大模型"),
                "provider": str(config.get("provider") or "local-openai-compatible"),
                "model": str(config.get("model") or ""),
                "description": "综合诊断结果、上传文献与内置已审核图谱生成处置建议",
                "available": str(profile.get("validation_status") or "").upper() == "PASS",
                "is_default": bool(profile.get("is_default")) or profile_id == active_profile_id,
            })
        if runtime["configured"] and active_profile_id not in profile_ids:
            engines.append({
                "engine_id": "auto",
                "kind": "llm",
                "name": str(runtime.get("model") or "当前大模型"),
                "provider": str(runtime.get("provider") or "local-openai-compatible"),
                "model": str(runtime.get("model") or ""),
                "description": "当前进程配置；重启后可能需要重新配置",
                "available": True,
                "is_default": True,
            })
        default = next((item["engine_id"] for item in engines if item.get("is_default") and item.get("available")), "rules")
        return {
            "schema_version": "astrograph-analysis-engines.v1",
            "engines": engines,
            "default_engine_id": default,
            "knowledge_sources": ["UPLOADED_REFERENCES", "REVIEWED_INTERNAL_GRAPH"],
        }

    def _analysis_engine_snapshot(self, engine_id: str) -> dict[str, Any]:
        if engine_id == "rules":
            return {
                "engine_id": "rules", "kind": "rules", "name": "规则引擎",
                "configured": True, "model": None, "provider": None,
                "diagnostic_vote_allowed": False, "secret_returned": False,
            }
        if engine_id.startswith("llm:"):
            profile_id = engine_id.split(":", 1)[1]
            profile = self.state_repository.get_service_profile(profile_id)
            if not profile or profile.get("service_kind") != "llm":
                raise ValueError("selected LLM profile does not exist")
            config = dict(profile.get("config") or {})
            return {
                "engine_id": engine_id, "kind": "llm", "profile_id": profile_id,
                "name": str(profile.get("name") or config.get("model") or "大模型"),
                "configured": bool(config.get("base_url") and config.get("model")),
                "provider": str(config.get("provider") or "local-openai-compatible"),
                "model": str(config.get("model") or ""),
                "diagnostic_vote_allowed": False, "secret_returned": False,
            }
        runtime = self._llm_runtime_snapshot()
        return {"engine_id": "auto", "kind": "llm" if runtime["configured"] else "rules", "name": runtime.get("model") or "规则引擎", **runtime}

    def _knowledge_service_for_job(self, job: dict[str, Any]) -> PlatformKnowledgeEnrichmentService:
        engine_id = str(job.get("analysis_engine_id") or "auto")
        if engine_id == "auto":
            return self.knowledge_service
        base = PlatformKnowledgeConfig.from_environment()
        if engine_id == "rules":
            return PlatformKnowledgeEnrichmentService(base.model_copy(update={
                "llm_base_url": None, "llm_model": None,
            }))
        profile_id = engine_id.split(":", 1)[1]
        profile = self.state_repository.get_service_profile(profile_id)
        if not profile or profile.get("service_kind") != "llm":
            raise ValueError("selected LLM profile is unavailable")
        selected = dict(profile.get("config") or {})
        config = base.model_copy(update={
            "llm_base_url": str(selected.get("base_url") or "").rstrip("/"),
            "llm_model": str(selected.get("model") or ""),
            "llm_provider": str(selected.get("provider") or "local-openai-compatible"),
            "llm_timeout_s": float(selected.get("timeout_s", 120.0)),
            "llm_response_format_strategy": str(selected.get("response_format_strategy") or "auto"),
            "llm_max_tokens": int(selected.get("structured_max_tokens", 1200)),
        })
        secret = PlatformSecretStore().get(profile.get("secret_ref")) or "local-no-secret"
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            provider=config.llm_provider,
            base_url=config.llm_base_url or "",
            api_key=SecretStr(secret),
            model=config.llm_model or "",
            temperature=0.0,
            timeout_s=config.llm_timeout_s,
            max_tokens=config.llm_max_tokens,
            response_format_strategy=config.llm_response_format_strategy,
            tokenizer_mode="disabled",
        ))
        return PlatformKnowledgeEnrichmentService(config, llm_backend=backend)

    @staticmethod
    def _job_executor_mode() -> str:
        requested = os.environ.get("ASTROGRAPH_PLATFORM_JOB_EXECUTOR", "auto").strip().casefold()
        if requested not in {"auto", "thread", "process"}:
            raise ValueError("ASTROGRAPH_PLATFORM_JOB_EXECUTOR must be auto, thread, or process")
        if requested != "auto":
            return requested
        # Local/desktop operation benefits from sharing the exact runtime LLM configuration
        # and avoids Windows GUI/service stdio inheritance problems. Server deployments may
        # opt into an isolated process with ASTROGRAPH_PLATFORM_JOB_EXECUTOR=process.
        if os.environ.get("ASTROGRAPH_AUTH_MODE", "local").strip().casefold() == "local":
            return "thread"
        return "process"

    def put_file(self, job_id: str, filename: str, data: bytes) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] not in {"CREATED", "UPLOADING"}:
            raise ValueError("job inputs are immutable after analysis starts")
        if not _SAFE_FILENAME.fullmatch(filename) or Path(filename).name != filename:
            raise ValueError("filename must be one basename without path separators")
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"unsupported upload suffix: {suffix or '<none>'}")
        if len(data) > _MAX_FILE_BYTES:
            raise ValueError("single upload exceeds file size limit")
        input_dir = self._job_root(job_id) / "input"
        existing = [p for p in input_dir.iterdir() if p.is_file()]
        target = input_dir / filename
        if target.exists():
            raise ValueError("duplicate filename in one job")
        if len(existing) >= _MAX_FILES:
            raise ValueError("too many uploaded files")
        total = sum(p.stat().st_size for p in existing) + len(data)
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("job upload exceeds total size limit")
        target.write_bytes(data)
        receipt = PlatformFileReceipt(
            filename=filename, size_bytes=len(data), sha256=_sha256_bytes(data)
        ).model_dump(mode="json")
        job["files"] = [*job["files"], receipt]
        job["preflight"] = None
        job["status"] = "UPLOADING"
        job["stage"] = "UPLOADING"
        job["stage_detail"] = f"已接收 {len(job['files'])} 个输入文件"
        self._save(job)
        return receipt

    def _safe_expand(self, job_id: str) -> Path:
        root = self._job_root(job_id)
        input_dir = root / "input"
        job = self.get(job_id)
        expected = {str(item.get("filename")): item for item in (job.get("files") or [])}
        observed_names = sorted(path.name for path in input_dir.iterdir() if path.is_file())
        if observed_names != sorted(expected):
            raise ValueError("stored job input file set no longer matches immutable upload receipts")
        for name in observed_names:
            path = input_dir / name
            receipt = expected[name]
            if path.stat().st_size != int(receipt.get("size_bytes") or -1):
                raise ValueError(f"stored job input size drift detected: {name}")
            if _sha256_path(path) != str(receipt.get("sha256") or ""):
                raise ValueError(f"stored job input sha256 drift detected: {name}")
        expanded = root / "expanded"
        if expanded.exists():
            shutil.rmtree(expanded)
        expanded.mkdir()
        expanded_bytes = 0
        member_count = 0
        for source in sorted(input_dir.iterdir()):
            if not source.is_file():
                continue
            if source.suffix.lower() != ".zip":
                shutil.copy2(source, expanded / source.name)
                expanded_bytes += source.stat().st_size
                continue
            zip_root = expanded / ("zip-" + source.stem)
            zip_root.mkdir()
            with zipfile.ZipFile(source) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    member_count += 1
                    if member_count > _MAX_ZIP_MEMBERS:
                        raise ValueError("archive member count exceeds limit")
                    relative = Path(info.filename)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or not relative.parts
                        or any(part in {"", "."} for part in relative.parts)
                    ):
                        raise ValueError("archive contains an unsafe path")
                    # Unix symlink bit in the upper mode field.
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == 0o120000:
                        raise ValueError("archive symlinks are forbidden")
                    expanded_bytes += int(info.file_size)
                    if expanded_bytes > _MAX_EXPANDED_BYTES:
                        raise ValueError("expanded archive exceeds size limit")
                    target = (zip_root / relative).resolve()
                    if zip_root not in target.parents:
                        raise ValueError("archive member escapes extraction root")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
        return expanded

    @staticmethod
    def _load_json_candidates(root: Path) -> list[tuple[Path, dict[str, Any]]]:
        records: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(root.rglob("*.json")):
            if path.stat().st_size > 16 * 1024 * 1024:
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                records.append((path, value))
        return records

    def _classify(self, expanded: Path) -> tuple[str, dict[str, Path]]:
        json_records = self._load_json_candidates(expanded)
        review_manifests = [path for path, record in json_records
                            if path.name == "analysis_input.json"
                            and record.get("schema_version") == "astrograph-result-log-review.v1"]
        if review_manifests:
            if len(review_manifests) != 1:
                raise ValueError("one task requires exactly one analysis_input.json")
            from astrograph.integrations.result_log_review import load_manifest
            load_manifest(expanded, review_manifests[0])
            return "RESULT_LOG_REVIEW", {"manifest": review_manifests[0]}
        oag_inputs = [(path, value) for path, value in json_records
                      if value.get("schema_version") == "astrograph-oag-input.v1"]
        public_manifests = [
            (path, value)
            for path, value in json_records
            if path.name == "manifest.json"
            and value.get("schema_version") in {PUBLIC_CASE_SCHEMA_VERSION, SIMULATION_HANDOFF_SCHEMA_VERSION}
        ]
        runtimes = [
            (path, value) for path, value in json_records
            if value.get("schema_version") == PIPELINE_RUNTIME_SCHEMA_VERSION
        ]
        mappings = [
            (path, value) for path, value in json_records
            if value.get("schema_version") == PIPELINE_MAPPING_SCHEMA_VERSION
        ]
        if oag_inputs:
            if len(oag_inputs) != 1 or public_manifests or runtimes or mappings:
                raise ValueError("ambiguous upload: provide exactly one OAG diagnostic input")
            return "OAG_DIAGNOSTIC", {"input": oag_inputs[0][0]}
        if public_manifests and (runtimes or mappings):
            raise ValueError("ambiguous upload contains two executable artifact families")
        if public_manifests:
            if len(public_manifests) != 1:
                raise ValueError("one analysis job must contain exactly one public case manifest")
            return "PUBLIC_CASE", {"case_root": public_manifests[0][0].parent}
        if runtimes or mappings:
            if len(runtimes) != 1 or len(mappings) != 1:
                raise ValueError("pipeline analysis requires exactly one runtime and one mapping bundle")
            return "PIPELINE_RUNTIME", {"runtime": runtimes[0][0], "mapping": mappings[0][0]}
        raise ValueError(
            "no supported artifact set found; upload a public manifest+telemetry or a pipeline runtime+mapping bundle"
        )

    @staticmethod
    def _preflight_capability_matrix(artifact_kind: str) -> list[dict[str, str]]:
        matrices = {
            "PUBLIC_CASE": [
                ("DETECTION", "故障检测", "AVAILABLE", "真值隔离公开遥测可进入统一检测链路"),
                ("TEMPORAL_LOCALIZATION", "时序定位", "DEGRADED", "可用程度取决于病例遥测字段和诊断后端"),
                ("CLASSIFICATION", "故障分类", "DEGRADED", "可形成注册候选，但该上传路径不具备正式在轨发布资格"),
                ("ROOT_CAUSE", "根因分析", "DEGRADED", "仅提供候选机理/证据解释，不等于正式根因确认"),
                ("KNOWLEDGE_EXPLANATION", "知识解释", "AVAILABLE", "仅在核心诊断冻结后参与解释，不参与投票"),
            ],
            "PIPELINE_RUNTIME": [
                ("DETECTION", "故障检测", "AVAILABLE", "可执行既有工程筛查与受治理闭环"),
                ("TEMPORAL_LOCALIZATION", "时序定位", "DEGRADED", "依赖 runtime 中已有 episode/时序产物"),
                ("CLASSIFICATION", "故障分类", "DEGRADED", "BiRRF/模型结果属于工程筛查，非校准飞行概率"),
                ("ROOT_CAUSE", "根因分析", "DEGRADED", "可生成因果链与复核项，正式放行仍受治理门约束"),
                ("KNOWLEDGE_EXPLANATION", "知识解释", "AVAILABLE", "冻结候选后可追加受约束文档解释"),
            ],
            "RESULT_LOG_REVIEW": [
                ("UPSTREAM_DIAGNOSIS", "诊断结果读取", "AVAILABLE", "使用上游结论作为待核对候选，不重新运行遥测检测模型"),
                ("LOG_CORRELATION", "日志交叉复核", "AVAILABLE", "核对已有诊断候选的日志关联现象，不将关键词当成根因证明"),
                ("REFERENCE_GROUNDING", "参考文献定位", "AVAILABLE", "抽取参考材料并保留文件哈希、原文定位与引用状态"),
                ("ROOT_CAUSE", "故障原因", "DEGRADED", "输出假设、证据缺口及验证项，不声称正式确认根因"),
                ("HANDLING", "处置建议", "DEGRADED", "生成只读核查和需人工批准的措施，不直接发出设备控制命令"),
            ],
            "OAG_DIAGNOSTIC": [
                ("DETECTION", "故障检测", "DEGRADED", "输入为结构化诊断工件，不替代原始遥测检测链"),
                ("TEMPORAL_LOCALIZATION", "时序定位", "DEGRADED", "仅消费 OAG 输入中已有时序/证据语义"),
                ("CLASSIFICATION", "故障分类", "DEGRADED", "执行结构化证据复核，正式发布资格保持关闭"),
                ("ROOT_CAUSE", "根因分析", "DEGRADED", "可做证据化根因复核，不将模型解释当作新传感器证据"),
                ("KNOWLEDGE_EXPLANATION", "知识解释", "AVAILABLE", "知识增强只在诊断冻结后运行"),
            ],
        }
        return [
            {"capability_id": cid, "label": label, "status": status, "reason": reason}
            for cid, label, status, reason in matrices.get(artifact_kind, [])
        ]

    @staticmethod
    def _preflight_paths(expanded: Path, paths: dict[str, Path]) -> list[str]:
        values: list[str] = []
        for path in paths.values():
            resolved = path.resolve()
            if resolved.is_dir():
                continue
            try:
                values.append(str(resolved.relative_to(expanded.resolve())))
            except ValueError:
                values.append(path.name)
        return sorted(values)

    def _validate_preflight_artifact(
        self, artifact_kind: str, paths: dict[str, Path], expanded: Path
    ) -> dict[str, Any]:
        detected_schemas: list[str] = []
        model_binding: dict[str, Any] = {
            "status": "NOT_EVALUATED_AT_PREFLIGHT",
            "formal_binding_verified": False,
            "reason": "正式模型绑定/放行资格在核心诊断治理链中重新校验",
        }
        details: dict[str, Any] = {}
        executable_paths = [path.resolve() for path in paths.values() if path.resolve().is_file()]
        if artifact_kind == "RESULT_LOG_REVIEW":
            from astrograph.integrations.result_log_review import validate_review_inputs
            review = validate_review_inputs(expanded, paths["manifest"])
            detected_schemas.extend(review["detected_schema_versions"])
            executable_paths = [paths["manifest"].resolve(), *[
                (expanded / name).resolve() for name in review["executable_files"]
                if name != "analysis_input.json"
            ]]
            model_binding = review["model_binding"]
            details.update({"role_counts": review["role_counts"], "source_receipts": review["files"],
                            "material_review_only": True})
        elif artifact_kind == "PUBLIC_CASE":
            from astrograph.integrations.sat_sim_public_case import _load_public_case
            manifest, telemetry = _load_public_case(paths["case_root"])
            detected_schemas.append(str(manifest.get("schema_version")))
            executable_paths = [(paths["case_root"] / "manifest.json").resolve(), telemetry.resolve()]
            details.update({
                "case_id": str(manifest.get("case_id") or ""),
                "telemetry_file": telemetry.name,
                "telemetry_sha256_verified": True,
                "truth_isolation_checked": True,
            })
        elif artifact_kind == "PIPELINE_RUNTIME":
            runtime = json.loads(paths["runtime"].read_text(encoding="utf-8"))
            mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
            SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping, persist_dir=None)
            detected_schemas.extend([
                str(runtime.get("schema_version") or ""),
                str(mapping.get("schema_version") or ""),
            ])
            details.update({
                "fault_id": str(runtime.get("fault_id") or ""),
                "cross_project_identity_checked": True,
            })
            model_binding.update({
                "status": "DEFERRED_TO_CORE_DIAGNOSIS",
                "reason": "runtime/mapping 身份合同已通过；模型 artifact/profile 绑定在闭环执行时复核",
            })
        elif artifact_kind == "OAG_DIAGNOSTIC":
            from astrograph.oag.contracts import OAGInput
            raw = json.loads(paths["input"].read_text(encoding="utf-8"))
            OAGInput.model_validate(raw)
            detected_schemas.append(str(raw.get("schema_version") or ""))
            details.update({"structured_evidence_contract_checked": True})
        else:
            raise ValueError(f"unsupported preflight artifact kind: {artifact_kind}")

        executable = set(executable_paths)
        knowledge = (
            [(expanded / name).resolve() for name in review["knowledge_files"]]
            if artifact_kind == "RESULT_LOG_REVIEW" else self._knowledge_document_paths(expanded, executable)
        )
        executable_files: list[str] = []
        for path in executable_paths:
            try:
                executable_files.append(str(path.relative_to(expanded.resolve())))
            except ValueError:
                executable_files.append(path.name)
        return {
            "detected_schema_versions": sorted(set(filter(None, detected_schemas))),
            "executable_files": sorted(executable_files),
            "knowledge_files": [
                str(path.relative_to(expanded.resolve())) if expanded.resolve() in path.parents else path.name
                for path in knowledge
            ],
            "model_binding": model_binding,
            "details": details,
        }

    def preflight(self, job_id: str, *, persist: bool = True) -> dict[str, Any]:
        job = self.get(job_id)
        blockers: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []
        artifact_kind: str | None = None
        validation: dict[str, Any] = {
            "detected_schema_versions": [],
            "executable_files": [],
            "knowledge_files": [],
            "model_binding": {
                "status": "NOT_EVALUATED",
                "formal_binding_verified": False,
            },
            "details": {},
        }
        expanded_file_count = 0
        if not job.get("files"):
            blockers.append({"code": "NO_UPLOADED_FILES", "message": "尚未上传任何文件。"})
        else:
            try:
                expanded = self._safe_expand(job_id)
                expanded_file_count = sum(1 for path in expanded.rglob("*") if path.is_file())
                artifact_kind, paths = self._classify(expanded)
                validation = self._validate_preflight_artifact(artifact_kind, paths, expanded)
                if validation["knowledge_files"] and artifact_kind != "RESULT_LOG_REVIEW":
                    warnings.append({
                        "code": "KNOWLEDGE_POST_DIAGNOSIS_ONLY",
                        "message": "参考文档只在核心诊断冻结后用于检索/解释，不参与候选生成或 BiRRF。",
                    })
                if artifact_kind == "RESULT_LOG_REVIEW":
                    warnings.append({
                        "code": "MATERIALS_ARE_NOT_RAW_TELEMETRY",
                        "message": "此路径只对上游诊断结果进行材料交叉复核，输出故障原因假设与人工处置建议，不构成在轨确诊。",
                    })
            except Exception as exc:
                blockers.append({
                    "code": "INPUT_CONTRACT_REJECTED",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                })

        can_start = not blockers and artifact_kind is not None
        status = "BLOCKED" if blockers else ("WARNING" if warnings else "PASS")
        payload = {
            "schema_version": PLATFORM_PREFLIGHT_SCHEMA_VERSION,
            "job_id": job_id,
            "status": status,
            "can_start": can_start,
            "checked_at": _now(),
            "artifact_kind": artifact_kind,
            "input_contract": {
                "uploaded_file_count": len(job.get("files") or []),
                "expanded_file_count": expanded_file_count,
                "allowed_upload_suffixes": sorted(_ALLOWED_SUFFIXES),
                "supported_executable_families": ["RESULT_LOG_REVIEW", "PUBLIC_CASE", "PIPELINE_RUNTIME", "OAG_DIAGNOSTIC"],
                "detected_schema_versions": validation["detected_schema_versions"],
                "executable_files": validation["executable_files"],
                "knowledge_files": validation["knowledge_files"],
                "details": validation["details"],
            },
            "capability_matrix": self._preflight_capability_matrix(artifact_kind or ""),
            "model_binding": validation["model_binding"],
            "blockers": blockers,
            "warnings": warnings,
            "claim_boundary": (
                "Preflight proves only that the uploaded package can enter a supported diagnostic path. "
                "It does not predict the diagnosis and does not grant formal release qualification."
            ),
        }
        if persist:
            latest = self.get(job_id)
            latest["preflight"] = payload
            if artifact_kind is not None:
                latest["artifact_kind"] = artifact_kind
            self._save(latest)
            self._append_event(job_id, f"[PREFLIGHT] status={status} artifact_kind={artifact_kind or 'NONE'}") if hasattr(self, "_append_event") else None
        return payload

    @staticmethod
    def _supplemental_material(expanded: Path, executable_paths: set[Path]) -> list[dict[str, Any]]:
        materials: list[dict[str, Any]] = []
        for path in sorted(p for p in expanded.rglob("*") if p.is_file()):
            if path.resolve() in executable_paths:
                continue
            suffix = path.suffix.lower()
            if suffix not in _TEXT_SUFFIXES:
                continue
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                materials.append({
                    "filename": path.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "status": "NOT_UTF8",
                    "used_for_diagnosis": False,
                })
                continue
            materials.append({
                "filename": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": "INDEXED_SUPPLEMENTAL",
                "used_for_diagnosis": False,
                "excerpt": text[:_MAX_TEXT_CHARS],
                "claim_boundary": "Supplemental user text is recorded for review and does not alter diagnosis.",
            })
        return materials

    @staticmethod
    def _knowledge_document_paths(expanded: Path, executable_paths: set[Path]) -> list[Path]:
        return [
            path.resolve()
            for path in sorted(p for p in expanded.rglob("*") if p.is_file())
            if path.resolve() not in executable_paths and path.suffix.lower() in _KNOWLEDGE_SUFFIXES
        ]

    @staticmethod
    def _public_diagnosis_snapshot(closure: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
        diag = closure["diagnosis"]
        candidates = [str(item) for item in (diag.get("classification_candidates") or [])]
        evidence = []
        for index, item in enumerate(diagnosis.get("diagnostic_evidence") or [], start=1):
            evidence.append({
                "evidence_id": str(item.get("evidence_id") or f"PUBLIC-EVID-{index:04d}"),
                "field": str(item.get("field") or ""),
                "statistic": str(item.get("statistic") or ""),
                "interpretation": str(item.get("interpretation") or ""),
                "observed_value": item.get("observed_value", item.get("value")),
                "unit": str(item.get("unit") or item.get("observed_unit") or ""),
            })
        return {
            "schema_version": "astrograph-platform-frozen-diagnosis.v1",
            "diagnosis_mode": "PUBLIC_CASE",
            "status": str(diag.get("status") or "UNKNOWN"),
            "detection": str(diag.get("detection") or "UNKNOWN"),
            "candidate_fault_id": candidates[0] if candidates else None,
            "candidate_fault_ids": candidates,
            "location_candidates": [str(item) for item in (diag.get("location_candidates") or [])],
            "evidence": evidence,
            "formal_diagnosis_publishable": False,
            "labels_used": bool(closure["diagnosis_provenance"]["labels_used"]),
            "truth_used": bool(closure["diagnosis_provenance"]["truth_used"]),
        }

    @staticmethod
    def _pipeline_diagnosis_snapshot(result: dict[str, Any]) -> dict[str, Any]:
        fusion = result.get("fusion") or {}
        candidates = [
            str(item.get("fault_id"))
            for item in (fusion.get("candidates") or [])
            if isinstance(item, dict) and item.get("fault_id")
        ]
        candidate = fusion.get("top_fault_id") or (candidates[0] if candidates else None)
        evidence_ids = []
        prediction = result.get("data_driven_prediction") or {}
        metadata = prediction.get("metadata") or {}
        lineage = metadata.get("evidence_lineage_by_fault") or {}
        if candidate and isinstance(lineage.get(str(candidate)), dict):
            evidence_ids = list(lineage[str(candidate)].get("source_evidence_ids") or [])
        evidence = [
            {
                "evidence_id": str(evidence_id),
                "field": "",
                "statistic": "",
                "interpretation": "Upstream screening evidence retained by the original pipeline.",
            }
            for evidence_id in evidence_ids
        ]
        return {
            "schema_version": "astrograph-platform-frozen-diagnosis.v1",
            "diagnosis_mode": "PIPELINE_RUNTIME",
            "status": str(result.get("disposition") or "UNKNOWN"),
            "detection": "SCREENING_CANDIDATE" if candidate else "NO_CANDIDATE",
            "candidate_fault_id": str(candidate) if candidate else None,
            "candidate_fault_ids": candidates,
            "location_candidates": [],
            "evidence": evidence,
            "formal_diagnosis_publishable": bool(result.get("formal_diagnosis_publishable", False)),
            "labels_used": False,
            "truth_used": False,
        }

    @staticmethod
    def _oag_diagnosis_snapshot(result: dict[str, Any]) -> dict[str, Any]:
        output = result.get("output") or {}
        summary = output.get("summary") or {}
        assessment = output.get("assessment") or {}
        decision = summary.get("diagnostic_decision") or {}
        candidate = summary.get("suspected_fault_id")
        candidates = [str(item) for item in (decision.get("candidate_fault_ids") or []) if item]
        if candidate and str(candidate) not in candidates:
            candidates.insert(0, str(candidate))
        evidence_ids: list[str] = []
        for hypothesis in assessment.get("hypotheses") or []:
            if not isinstance(hypothesis, dict):
                continue
            if candidate and hypothesis.get("fault_id") != candidate:
                continue
            evidence_ids.extend(str(item) for item in hypothesis.get("supporting_evidence_ids") or [])
            evidence_ids.extend(str(item) for item in hypothesis.get("contradicting_evidence_ids") or [])
        structured_evidence = {
            str(item.get("evidence_id")): item
            for item in (summary.get("evidence") or [])
            if isinstance(item, dict) and item.get("evidence_id")
        }
        evidence = []
        for evidence_id in dict.fromkeys(evidence_ids):
            item = structured_evidence.get(str(evidence_id), {})
            evidence.append({
                "evidence_id": str(evidence_id),
                "field": str(item.get("feature_field") or item.get("field") or ""),
                "statistic": str(item.get("statistic") or item.get("status") or ""),
                "interpretation": str(
                    item.get("interpretation")
                    or "Structured OAG evidence retained from the frozen diagnostic review."
                ),
                "observed_value": item.get("observed_value", item.get("value")),
                "unit": str(item.get("unit") or item.get("observed_unit") or ""),
            })
        return {
            "schema_version": "astrograph-platform-frozen-diagnosis.v1",
            "diagnosis_mode": "OAG_DIAGNOSTIC",
            "status": str(summary.get("status") or "REVIEW"),
            "detection": str(decision.get("status") or ("CANDIDATE" if candidate else "NO_CANDIDATE")),
            "candidate_fault_id": str(candidate) if candidate else None,
            "candidate_fault_ids": candidates,
            "location_candidates": [],
            "evidence": evidence,
            "formal_diagnosis_publishable": False,
            "labels_used": False,
            "truth_used": False,
        }

    @staticmethod
    def _render_public_report(
        *, job: dict[str, Any], closure: dict[str, Any], diagnosis: dict[str, Any],
        materials: list[dict[str, Any]], knowledge: dict[str, Any] | None = None,
    ) -> str:
        diag = closure["diagnosis"]
        provenance = closure["diagnosis_provenance"]
        evidence = diagnosis.get("diagnostic_evidence") or []
        lines = [
            "# AstroGraph 自动诊断报告",
            "",
            f"- Job：`{job['job_id']}`",
            f"- 用户任务：{job['task']}",
            "- 工件类型：`PUBLIC_CASE`（真值隔离）",
            f"- 案例：`{closure['case_id']}`",
            f"- 诊断状态：**{diag['status']}**",
            f"- 检测结论：`{diag['detection']}`",
            f"- 后端：`{provenance['backend']}`",
            f"- labels_used：`{str(provenance['labels_used']).lower()}`",
            f"- truth_used：`{str(provenance['truth_used']).lower()}`",
            "- 正式在轨发布资格：`false`",
            "",
            "## 诊断候选",
            "",
        ]
        classes = diag.get("classification_candidates") or []
        lines.extend([f"- `{item}`" for item in classes] or ["- 无可发布候选；保留当前拒识/正常/数据状态。"])
        lines += ["", "## 定位候选", ""]
        lines.extend([f"- `{item}`" for item in diag.get("location_candidates") or []] or ["- 无"])
        lines += ["", "## 诊断证据", ""]
        if evidence:
            lines += ["| Evidence ID | Field | Statistic | Interpretation |", "|---|---|---|---|"]
            for item in evidence:
                lines.append(
                    f"| `{item.get('evidence_id','')}` | `{item.get('field','')}` | "
                    f"`{item.get('statistic','')}` | {item.get('interpretation','')} |"
                )
        else:
            lines.append("- 当前后端未返回可列举的诊断证据。")
        lines += ["", "## 补充文本材料", ""]
        if materials:
            for item in materials:
                lines.append(
                    f"- `{item['filename']}` sha256=`{item['sha256']}`，"
                    f"used_for_diagnosis=`false`，status=`{item['status']}`"
                )
        else:
            lines.append("- 未上传补充文本。")
        if knowledge is not None:
            lines += ["", "## 知识增强", "", f"- 状态：`{knowledge.get('status', 'UNKNOWN')}`"]
            hits = ((knowledge.get("local_retrieval") or {}).get("hits") or [])
            if hits:
                lines += ["", "### 上传文档相关片段", ""]
                for item in hits:
                    lines.append(
                        f"- `{item['knowledge_id']}` `{item['filename']}` score=`{item['score']}`："
                        + str(item.get("quote") or "").replace("\n", " ")[:500]
                    )
            quality = ((knowledge.get("local_retrieval") or {}).get("quality_receipt") or {})
            if quality:
                lines += [
                    f"- 本地检索锚定：`{quality.get('grounding_status', 'UNKNOWN')}`，"
                    f"hits=`{quality.get('retrieved_hit_count', 0)}`，sources=`{quality.get('unique_retrieved_source_count', 0)}`"
                ]
            paper = knowledge.get("paperseek") or {}
            lines += ["", f"- PaperSeek：`{paper.get('status', 'UNKNOWN')}`"]
            lines += [f"- Anything2Ontology：`{(knowledge.get('anything2ontology') or {}).get('status', 'UNKNOWN')}`"]
            generation = knowledge.get("generation") or {}
            lines += [f"- 大模型服务：`{generation.get('status', 'UNKNOWN')}`"]
            final = knowledge.get("final_explanation") or {}
            lines += [f"- 最终解释来源：`{final.get('source', 'NONE')}`"]
            explanation = final.get("explanation")
            if isinstance(explanation, dict):
                lines += ["", "### 受约束解释（证据候选不可改写）", "", str(explanation.get("summary") or "")]
                claims = explanation.get("claims") or []
                if claims:
                    lines += ["", "### 引用主张", ""]
                    for claim in claims[:8]:
                        citations = ", ".join(str(item) for item in claim.get("citation_ids") or [])
                        lines.append(f"- `{claim.get('kind','UNKNOWN')}` {claim.get('text','')}  引用：`{citations}`")
        lines += [
            "",
            "## 结论边界",
            "",
            "本报告来自已注册的 AstroGraph 公共案例诊断入口。用户任务文本和补充文档不会改变诊断算法。",
            "若输入包含答案代理、真值字段、哈希篡改或不符合 schema，平台会失败关闭而不是生成诊断。",
            "本报告不是飞行资格、因果证明或控制授权。",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_pipeline_report(
        *, job: dict[str, Any], result: dict[str, Any], materials: list[dict[str, Any]],
        knowledge: dict[str, Any] | None = None,
    ) -> str:
        lines = [
            "# AstroGraph 自动化工程诊断报告",
            "",
            f"- Job：`{job['job_id']}`",
            f"- 用户任务：{job['task']}",
            "- 工件类型：`PIPELINE_RUNTIME`",
            "- 注意：该运行时带有仿真实验身份，只允许工程筛查/审核用途。",
            "",
            result["report_markdown"].rstrip(),
            "",
            "## 上传补充文本",
            "",
        ]
        if materials:
            lines.extend(
                f"- `{item['filename']}` sha256=`{item['sha256']}`，used_for_diagnosis=`false`"
                for item in materials
            )
        else:
            lines.append("- 未上传补充文本。")
        if knowledge is not None:
            lines += ["", "## 知识增强", "", f"- 状态：`{knowledge.get('status', 'UNKNOWN')}`"]
            for item in ((knowledge.get("local_retrieval") or {}).get("hits") or []):
                lines.append(
                    f"- `{item['knowledge_id']}` `{item['filename']}` score=`{item['score']}`："
                    + str(item.get("quote") or "").replace("\n", " ")[:500]
                )
            quality = ((knowledge.get("local_retrieval") or {}).get("quality_receipt") or {})
            if quality:
                lines.append(
                    f"- 本地检索锚定：`{quality.get('grounding_status', 'UNKNOWN')}`，"
                    f"hits=`{quality.get('retrieved_hit_count', 0)}`，sources=`{quality.get('unique_retrieved_source_count', 0)}`"
                )
            lines.append(f"- PaperSeek：`{(knowledge.get('paperseek') or {}).get('status', 'UNKNOWN')}`")
            lines.append(f"- Anything2Ontology：`{(knowledge.get('anything2ontology') or {}).get('status', 'UNKNOWN')}`")
            generation = knowledge.get("generation") or {}
            lines.append(f"- 大模型服务：`{generation.get('status', 'UNKNOWN')}`")
            final = knowledge.get("final_explanation") or {}
            lines.append(f"- 最终解释来源：`{final.get('source', 'NONE')}`")
            explanation = final.get("explanation")
            if isinstance(explanation, dict):
                lines += ["", "### 受约束解释（不改变工程筛查候选）", "", str(explanation.get("summary") or "")]
                claims = explanation.get("claims") or []
                if claims:
                    lines += ["", "### 引用主张", ""]
                    for claim in claims[:8]:
                        citations = ", ".join(str(item) for item in claim.get("citation_ids") or [])
                        lines.append(f"- `{claim.get('kind','UNKNOWN')}` {claim.get('text','')}  引用：`{citations}`")
        lines += [
            "",
            "## 平台附加边界",
            "",
            "任务文本与补充文档没有进入候选生成或 BiRRF。label-conditioned KG projection 继续保持 audit-only。",
            "",
        ]
        return "\n".join(lines)

    def start(self, job_id: str) -> dict[str, Any]:
        """Start a background job without tying the HTTP request to diagnosis latency.

        Local/desktop mode defaults to a daemon thread.  The structured LLM HTTP call
        remains isolated by AstroGraph's bounded HTTP worker, while the job shares the
        exact process-scoped LM Studio configuration that the administrator just tested.
        Production deployments can select ``process`` explicitly.
        """
        import subprocess
        import sys

        self.get(job_id)
        root = self._job_root(job_id)
        launch_lock = root / ".launch.lock"
        try:
            with launch_lock.open("x"):
                pass
        except FileExistsError:
            return self.get(job_id)
        try:
            job = self.get(job_id)
            if job["status"] in {"QUEUED", "RUNNING", "SUCCEEDED"}:
                return job
            if not job["files"]:
                raise ValueError("job has no uploaded files")
            preflight = self.preflight(job_id)
            if not preflight["can_start"]:
                detail = "; ".join(item["message"] for item in preflight["blockers"])
                raise PreflightBlockedError(f"preflight blocked: {detail}")

            executor = self._job_executor_mode()
            job["status"] = "QUEUED"
            job["stage"] = "QUEUED"
            job["stage_detail"] = "等待后台诊断执行"
            job["execution_backend"] = executor.upper()
            job["llm_runtime"] = self._analysis_engine_snapshot(str(job.get("analysis_engine_id") or "auto"))
            job["llm_call_status"] = "NOT_STARTED"
            job["error"] = None
            self._save(job)
            self._append_event(job_id, f"[QUEUED] execution_backend={executor.upper()}")

            if executor == "thread":
                def runner() -> None:
                    try:
                        self.run(job_id)
                    except Exception as exc:  # defensive: run() normally records failures itself
                        failed = self.get(job_id)
                        failed.update(
                            status="FAILED",
                            stage="FAILED",
                            stage_detail="后台线程异常终止",
                            completed_at=_now(),
                            error={"type": type(exc).__name__, "message": str(exc)},
                        )
                        self._save(failed)
                        self._append_event(job_id, f"[FAILED] {type(exc).__name__}: {exc}")

                thread = threading.Thread(
                    target=runner,
                    name=f"astrograph-platform-{job_id[-8:]}",
                    daemon=True,
                )
                thread.start()
                return self.get(job_id)

            with (root / "worker.log").open("ab", buffering=0) as log:
                source_root = Path(__file__).resolve().parents[2]
                env = os.environ.copy()
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(source_root) + (os.pathsep + existing if existing else "")
                popen_kwargs: dict[str, Any] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": log,
                    "stderr": subprocess.STDOUT,
                    "close_fds": True,
                    "env": env,
                    # Keep the worker detached from the caller's console/session.
                    # On Windows CREATE_NEW_PROCESS_GROUP supplies the native
                    # boundary; retaining this flag also keeps the launch receipt
                    # platform-independent.
                    "start_new_session": True,
                }
                if os.name == "nt":
                    flags = 0
                    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
                    popen_kwargs["creationflags"] = flags
                subprocess.Popen(
                    [sys.executable, "-m", "astrograph.integrations.platform_worker",
                     "--root", str(self.root), "--job-id", job_id],
                    **popen_kwargs,
                )
            return self.get(job_id)
        except PreflightBlockedError:
            raise
        except Exception as exc:
            job = self.get(job_id)
            job.update(
                status="FAILED",
                stage="FAILED",
                stage_detail="后台执行器启动失败",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self._save(job)
            self._append_event(job_id, f"[FAILED] launch: {type(exc).__name__}: {exc}")
            raise
        finally:
            launch_lock.unlink(missing_ok=True)

    def _run_result_log_review(
        self, job_id: str, *, expanded: Path, manifest_path: Path,
        knowledge_service: PlatformKnowledgeEnrichmentService,
    ) -> dict[str, Any]:
        """New material-first route, deliberately separate from raw-telemetry engines."""
        from astrograph.integrations.result_log_review import (
            analyze_result_log_review, load_manifest, render_result_log_markdown,
        )

        self._set_stage(
            job_id, "CORE_DIAGNOSIS", "复核上游诊断、日志与参考材料（不重新检测遥测）",
            artifact_kind="RESULT_LOG_REVIEW",
        )
        output = self._job_root(job_id) / "output"
        review = analyze_result_log_review(expanded, manifest_path)
        _json_write(output / "result_log_review.json", review)
        # Material-first evidence fusion runs before optional legacy post-diagnosis
        # explanation. It never changes the supplied candidate IDs or Active KG.
        self._set_stage(job_id, "MATERIAL_KNOWLEDGE_FUSION",
                        "RAG / 图谱 / 案例关联；可选大模型构建待审核知识图谱")
        from astrograph.integrations.material_case_memory import MaterialCaseMemory
        from astrograph.integrations.material_reasoning import analyze_material_knowledge
        reviewers = tuple(x.strip() for x in os.environ.get('ASTROGRAPH_MATERIAL_REVIEWERS','').split(',') if x.strip())
        memory = MaterialCaseMemory(self.root / 'material_knowledge' / 'cases.sqlite3', reviewers=reviewers)
        reasoning = analyze_material_knowledge(
            expanded=expanded, manifest_path=manifest_path, review=review,
            memory=memory, source_group_id=job_id,
        )
        # The read-only current-task reasoning snapshot is saved before staging.
        # Unreviewed outputs may be archived but NEVER reused as confirmed cases.
        for candidate in review['candidates']:
            linked = next((x for x in reasoning['candidates'] if x['fault_id']==candidate['fault_id']),None)
            if linked:
                candidate['knowledge_candidate_relations'] = linked['knowledge_candidate_relations']
        try:
            proposals = memory.stage_review(source_group_id=job_id,review=review)
            reasoning['case_learning'] = {'status':'PENDING_EXPERT_REVIEW', 'proposals':proposals,
                                          'active_kg_modified':False}
        except Exception as exc:
            reasoning['case_learning'] = {'status':'STAGING_FAILED',
                'error_type':type(exc).__name__, 'message':str(exc)[:250],
                'proposals':[], 'active_kg_modified':False}
        _json_write(output / 'material_reasoning.json', reasoning)
        _json_write(output / 'material_graph_construction.json', reasoning['graph_construction'])
        job = self.get(job_id)
        _manifest, roles = load_manifest(expanded, manifest_path)
        snapshot = {
            "schema_version": "astrograph-platform-frozen-diagnosis.v1",
            "diagnosis_mode": "UPSTREAM_RESULT_REVIEW",
            "status": review["status"],
            "candidate_fault_id": review["top_fault_id"],
            "candidate_fault_ids": [x["fault_id"] for x in review["candidates"]],
            "material_source_group_id": job_id,
            "location_candidates": [],
            "evidence": [
                {"evidence_id": x["evidence_id"], "field": x["filename"],
                 "interpretation": x["excerpt"][:320]}
                for x in review["evidence"] if x["role"] in {"diagnosis_results", "log_reports"}
            ],
            "formal_diagnosis_publishable": False,
            "labels_used": False, "truth_used": False,
        }
        self._set_stage(
            job_id, "POST_DIAGNOSIS_KNOWLEDGE", "可选文献检索与大模型解释（故障候选不更改）",
            llm_call_status=("PENDING" if knowledge_service.llm_backend else "NOT_CONFIGURED"),
        )
        knowledge: dict[str, Any]
        try:
            knowledge = run_frozen_post_diagnosis_explanation(
                source_kind="RESULT_LOG_REVIEW", source_id=job_id,
                knowledge_service=knowledge_service,
                task=job["task"], diagnosis_snapshot=snapshot,
                document_paths=roles["references"], output_root=output / "knowledge",
                material_review_context=reasoning,
            )
        except Exception as exc:
            knowledge = {
                "status": "FAILED", "generation": {"status": "FAILED"},
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "claim_boundary": "Optional knowledge enrichment failed; core material review remains available.",
            }
        # Citation-ID checks alone do not prove the semantics of LLM claims.
        # In material-review mode, keep the raw generation receipt for expert
        # audit but surface only the deterministic, evidence-gated answer.
        from astrograph.integrations.material_multihop import apply_material_answer_guard
        knowledge = apply_material_answer_guard(knowledge, reasoning)
        self._set_stage(
            job_id, "REPORTING", "保存故障原因、处置建议和逐条引用",
            llm_call_status=str((knowledge.get("generation") or {}).get("status") or "UNKNOWN"),
        )
        agent_execution = {
            "work_mode": job.get("work_mode", "normal"),
            "status": "NOT_EXECUTED_MATERIAL_REVIEW_ONLY",
            "executed_actions": [], "diagnostic_candidate_modified": False,
            "limitations": ["当前材料分析路径不执行遥测探针或航天器控制工具。"],
        }
        markdown = render_result_log_markdown(job, review, knowledge, reasoning)
        report = {
            "schema_version": PLATFORM_REPORT_SCHEMA_VERSION,
            "job_id": job_id, "status": "SUCCEEDED", "task": job["task"],
            "work_mode": job.get("work_mode", "normal"),
            "analysis_engine": job.get("analysis_engine") or job.get("llm_runtime"),
            "agent_execution": agent_execution,
            "artifact_kind": "RESULT_LOG_REVIEW", "completed_at": _now(),
            "input_files": job["files"],
            "analysis": {
                "artifact_kind": "RESULT_LOG_REVIEW",
                "result_log_review": review,
                "material_reasoning": reasoning,
                "knowledge_engine_revision": os.environ.get("ASTROGRAPH_MATERIAL_ENGINE_REVISION"),
                "knowledge_enrichment": knowledge,
                "effect_guard": {
                    "upstream_result_is_ground_truth": False,
                    "reference_promoted_to_observation": False,
                    "raw_telemetry_detection_executed": False,
                    "automatic_control_authorized": False,
                    "active_kg_modified": False,
                },
            },
            "formal_diagnosis_publishable": False,
            "claim_boundary": "Material-only root cause review. No raw telemetry detection, confirmed physical cause or spacecraft control authorization.",
            "visualization": {"status": "NOT_APPLICABLE_DOCUMENT_ONLY", "channel_count": 0},
        }
        _json_write(output / "report.json", report)
        (output / "report.md").write_text(markdown, encoding="utf-8")
        _json_write(output / "visualization.json", {
            "schema_version": "astrograph-platform-visualization.v1", "status": "NOT_APPLICABLE_DOCUMENT_ONLY",
            "channels": [], "truth_or_labels_used": False,
        })
        job.update({
            "status": "SUCCEEDED", "stage": "COMPLETED", "stage_detail": "材料复核与处置建议报告已生成",
            "artifact_kind": "RESULT_LOG_REVIEW", "completed_at": report["completed_at"],
            "report_json": "output/report.json", "report_markdown": "output/report.md",
        })
        self._save(job)
        self._append_event(job_id, "[COMPLETED] 诊断结果-日志-文献辅助研判报告生成完成")
        return report

    def run(self, job_id: str) -> dict[str, Any]:
        """Enforce one writer per job across threads and worker processes."""
        # Worker/CLI processes do not share the API process environment.
        # Apply only the trusted persisted revision; absent a saved config,
        # existing environment-based deployments retain their behavior.
        from astrograph.integrations.material_engine_settings import load_active_engine_environment
        load_active_engine_environment(self.root)
        self.get(job_id)
        lock = self._job_root(job_id) / ".run.lock"
        try:
            with lock.open("x"):
                pass
        except FileExistsError as exc:
            raise ValueError("job is already running") from exc
        try:
            return self._run_unlocked(job_id)
        finally:
            lock.unlink(missing_ok=True)

    def _run_unlocked(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        knowledge_service = self._knowledge_service_for_job(job)
        if job["status"] == "SUCCEEDED":
            return self.report(job_id)
        if job["status"] == "RUNNING":
            raise ValueError("job is already running")
        if not job["files"]:
            raise ValueError("job has no uploaded files")
        job["status"] = "RUNNING"
        job["stage"] = "EXPANDING_INPUTS"
        job["stage_detail"] = "展开并校验上传工件"
        job["error"] = None
        job["started_at"] = _now()
        self._save(job)
        self._append_event(job_id, "[RUNNING] 展开并校验上传工件")
        try:
            preflight = self.preflight(job_id)
            if not preflight["can_start"]:
                detail = "; ".join(item["message"] for item in preflight["blockers"])
                raise PreflightBlockedError(f"preflight blocked: {detail}")
            expanded = self._safe_expand(job_id)
            self._set_stage(job_id, "CLASSIFYING_ARTIFACT", "识别可执行诊断工件类型")
            artifact_kind, paths = self._classify(expanded)
            if artifact_kind == "RESULT_LOG_REVIEW":
                return self._run_result_log_review(
                    job_id, expanded=expanded, manifest_path=paths["manifest"],
                    knowledge_service=knowledge_service,
                )
            self._set_stage(
                job_id, "CORE_DIAGNOSIS",
                f"执行 {artifact_kind} 核心诊断；此阶段不调用结构化大模型",
                artifact_kind=artifact_kind,
            )
            output = self._job_root(job_id) / "output"
            executable_paths: set[Path] = set()
            knowledge_paths: list[Path] = []
            diagnosis_snapshot: dict[str, Any]
            if artifact_kind == "OAG_DIAGNOSTIC":
                from astrograph.oag.platform import run_uploaded_oag, render_oag_report
                executable_paths = {paths["input"].resolve()}
                knowledge_paths = self._knowledge_document_paths(expanded, executable_paths)
                self._set_stage(
                    job_id, "CORE_DIAGNOSIS",
                    "执行 OAG 结构化证据诊断；知识增强在冻结候选后进行",
                )
                result = run_uploaded_oag(paths["input"], document_paths=knowledge_paths,
                                          output=output / "oag", job_id=job_id,
                                          service=knowledge_service)
                diagnosis_snapshot = self._oag_diagnosis_snapshot(result)
                oag_knowledge = result.get("knowledge_enrichment") or result.get("knowledge") or {}
                self._set_stage(
                    job_id, "POST_DIAGNOSIS_KNOWLEDGE",
                    "OAG 诊断与知识增强完成；准备执行 Agent 工作模式",
                    llm_call_status=str((oag_knowledge.get("generation") or {}).get("status") or "UNKNOWN"),
                )
                markdown = render_oag_report(result)
                analysis = {"artifact_kind": artifact_kind, "oag_result": result,
                            "effect_guard": {"task_text_used_for_diagnosis": False,
                                             "supplemental_text_used_for_diagnosis": False,
                                             "active_kg_modified": False}}
            elif artifact_kind == "PUBLIC_CASE":
                case_root = paths["case_root"]
                manifest = json.loads((case_root / "manifest.json").read_text(encoding="utf-8"))
                telemetry_name = str((manifest.get("telemetry") or {}).get("file") or "")
                executable_paths = {(case_root / "manifest.json").resolve(), (case_root / telemetry_name).resolve()}
                diagnosis_root = output / "diagnosis"
                closure = diagnose_public_case(case_root, output_root=diagnosis_root)
                diagnosis = json.loads((diagnosis_root / "astrograph_diagnosis.json").read_text(encoding="utf-8"))
                materials = self._supplemental_material(expanded, executable_paths)
                knowledge_paths = self._knowledge_document_paths(expanded, executable_paths)
                diagnosis_snapshot = self._public_diagnosis_snapshot(closure, diagnosis)
                self._set_stage(
                    job_id, "POST_DIAGNOSIS_KNOWLEDGE",
                    "核心候选已冻结；开始文档检索与结构化大模型解释",
                    llm_call_status=("PENDING" if knowledge_service.llm_backend is not None else "NOT_CONFIGURED"),
                )
                knowledge = run_frozen_post_diagnosis_explanation(
                    source_kind="PUBLIC_CASE", source_id=job_id,
                    knowledge_service=knowledge_service,
                    task=job["task"], diagnosis_snapshot=diagnosis_snapshot,
                    document_paths=knowledge_paths, output_root=output / "knowledge",
                )
                self._set_stage(
                    job_id, "POST_DIAGNOSIS_KNOWLEDGE",
                    "知识增强完成；准备执行 Agent 工作模式",
                    llm_call_status=str((knowledge.get("generation") or {}).get("status") or "UNKNOWN"),
                )
                markdown = self._render_public_report(
                    job=job, closure=closure, diagnosis=diagnosis, materials=materials, knowledge=knowledge
                )
                analysis = {
                    "artifact_kind": artifact_kind,
                    "closure": closure,
                    "diagnosis": diagnosis,
                    "supplemental_materials": materials,
                    "knowledge_enrichment": knowledge,
                    "effect_guard": {
                        "labels_used": closure["diagnosis_provenance"]["labels_used"],
                        "truth_used": closure["diagnosis_provenance"]["truth_used"],
                        "task_text_used_for_diagnosis": False,
                        "supplemental_text_used_for_diagnosis": False,
                    },
                }
            else:
                runtime_path, mapping_path = paths["runtime"], paths["mapping"]
                executable_paths = {runtime_path.resolve(), mapping_path.resolve()}
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
                persist = output / "pipeline"
                # Compute the diagnostic closure independently from persistence so
                # a host/CRT file-descriptor compatibility issue cannot be
                # misreported as an algorithmic CORE_DIAGNOSIS failure.
                result_model = SatSimPipelineClosureWorkflow().invoke(
                    SatSimPipelineClosureRequest(
                        runtime=runtime,
                        mapping_bundle=mapping,
                        persist_dir=None,
                    )
                )
                result = result_model.model_dump(mode="json")
                self._set_stage(
                    job_id, "CORE_DIAGNOSIS_PERSIST",
                    "核心诊断已完成；持久化受治理审计工件",
                )
                from astrograph.assurance.artifact_paths import ArtifactRoot
                source_root = Path(__file__).resolve().parents[3]
                artifacts = ArtifactRoot(persist, source_root=source_root)
                artifacts.write_json("sat_sim_pipeline_closure.json", result)
                artifacts.write_text("sat_sim_pipeline_report.md", result_model.report_markdown)
                artifacts.write_json(
                    "sat_sim_external_stage_outputs.json",
                    result_model.external_stage_outputs,
                )
                self._append_event(
                    job_id,
                    "[CORE_DIAGNOSIS_PERSIST] 核心诊断审计工件已发布",
                )
                materials = self._supplemental_material(expanded, executable_paths)
                knowledge_paths = self._knowledge_document_paths(expanded, executable_paths)
                diagnosis_snapshot = self._pipeline_diagnosis_snapshot(result)
                self._set_stage(
                    job_id, "POST_DIAGNOSIS_KNOWLEDGE",
                    "核心候选已冻结；开始文档检索与结构化大模型解释",
                    llm_call_status=("PENDING" if knowledge_service.llm_backend is not None else "NOT_CONFIGURED"),
                )
                knowledge = run_frozen_post_diagnosis_explanation(
                    source_kind="SAT_SIM_PIPELINE", source_id=job_id,
                    knowledge_service=knowledge_service,
                    task=job["task"], diagnosis_snapshot=diagnosis_snapshot,
                    document_paths=knowledge_paths, output_root=output / "knowledge",
                )
                self._set_stage(
                    job_id, "POST_DIAGNOSIS_KNOWLEDGE",
                    "知识增强完成；准备执行 Agent 工作模式",
                    llm_call_status=str((knowledge.get("generation") or {}).get("status") or "UNKNOWN"),
                )
                markdown = self._render_pipeline_report(
                    job=job, result=result, materials=materials, knowledge=knowledge
                )
                analysis = {
                    "artifact_kind": artifact_kind,
                    "pipeline_result": result,
                    "supplemental_materials": materials,
                    "knowledge_enrichment": knowledge,
                    "effect_guard": {
                        "formal_diagnosis_publishable": result["formal_diagnosis_publishable"],
                        "label_conditioned_kg_contributes_to_fusion": result["kg_projection"]["contributes_to_fusion"],
                        "task_text_used_for_diagnosis": False,
                        "supplemental_text_used_for_diagnosis": False,
                    },
                }

            self._set_stage(
                job_id, "AGENT_INVESTIGATION",
                f"执行 {job.get('work_mode', 'normal')} Agent 模式；不会修改冻结诊断",
            )
            from astrograph.integrations.platform_agent import execute_platform_agent_mode
            agent_execution = execute_platform_agent_mode(
                job_id=job_id,
                task=job["task"],
                work_mode=job.get("work_mode", "normal"),
                diagnosis_snapshot=diagnosis_snapshot,
                document_paths=knowledge_paths,
                knowledge_service=knowledge_service,
                output_root=output / "agent",
            )
            analysis["agent_execution"] = agent_execution
            markdown += (
                "\n\n## Agent 工作模式执行\n\n"
                f"- 模式：`{agent_execution['work_mode']}`\n"
                f"- 执行动作：`{len(agent_execution.get('executed_actions') or [])}`\n"
                "- 诊断候选被 Agent 模式修改：`false`\n"
                "- 探索内容参与诊断投票：`false`\n"
            )

            report = {
                "schema_version": PLATFORM_REPORT_SCHEMA_VERSION,
                "job_id": job_id,
                "status": "SUCCEEDED",
                "task": job["task"],
                "work_mode": job.get("work_mode", "normal"),
                "analysis_engine": job.get("analysis_engine") or job.get("llm_runtime"),
                "agent_execution": agent_execution,
                "artifact_kind": artifact_kind,
                "completed_at": _now(),
                "input_files": job["files"],
                "analysis": analysis,
                "rerun_provenance": job.get("rerun_provenance"),
                "production_rerun_provenance": job.get("production_rerun_provenance"),
                "formal_diagnosis_publishable": False,
                "claim_boundary": (
                    "Automated laboratory/simulation diagnosis report. Upload automation does not "
                    "promote screening or compatibility results into flight or control authorization."
                ),
            }
            self._set_stage(job_id, "VISUALIZATION", "生成遥测曲线与文献证据联动数据")
            try:
                from astrograph.integrations.platform_visualization import build_analysis_series, build_visualization
                visualization = build_visualization(expanded=expanded, report=report)
                analysis_series = build_analysis_series(expanded=expanded, report=report)
            except Exception as exc:
                visualization = {
                    "schema_version": "astrograph-platform-visualization.v1",
                    "status": "VISUALIZATION_FAILED",
                    "message": f"{type(exc).__name__}: {exc}",
                    "truth_or_labels_used": False,
                }
                analysis_series = visualization
            _json_write(output / "visualization.json", visualization)
            report["visualization"] = {
                "status": visualization.get("status"),
                "channel_count": len(visualization.get("channels") or []),
                "endpoint": f"/v1/platform/jobs/{job_id}/visualization",
            }

            # Maintenance temporal localization is intentionally computed only from
            # truth-safe telemetry already admitted into the visualization contract.
            # Simulator onset/labels in PIPELINE_RUNTIME are never consumed here.
            from astrograph.integrations.temporal_localization import localize_from_job_telemetry
            from astrograph.integrations.maintenance_presentation import build_maintenance_presentation
            temporal = localize_from_job_telemetry(
                expanded=expanded, report=report, visualization=visualization, analysis_series=analysis_series,
                diagnostic_fault_id=(diagnosis_snapshot.get("candidate_fault_id") or diagnosis_snapshot.get("fault_id") or diagnosis_snapshot.get("selected_fault_id"))
            )
            _json_write(output / "temporal_localization.json", temporal)
            # Promote the truth-safe first detection marker into the visualization contract so
            # GUI and headless clients see the same timeline semantics. It remains explicitly
            # named FIRST_DETECTION and never claims simulator/physical fault onset.
            if (visualization.get("timeline") or {}).get("onset_s") in (None, "") and temporal.get("first_detection_s") is not None:
                visualization.setdefault("timeline", {})["onset_s"] = temporal.get("first_detection_s")
                visualization["timeline"]["onset_source"] = "FIRST_DETECTION"
                _json_write(output / "visualization.json", visualization)
            try:
                self.state_repository.save_temporal_localization(job_id, temporal)
            except Exception:
                pass
            report["temporal_localization"] = {
                "status": temporal.get("status"),
                "first_detection_s": temporal.get("first_detection_s"),
                "confirmed_detection_s": temporal.get("confirmed_detection_s"),
                "endpoint": f"/v1/platform/jobs/{job_id}/temporal-localization",
                "truth_or_labels_used": False,
            }

            from astrograph.integrations.multidimensional_diagnostics import assess_multidimensional_telemetry
            multidimensional = assess_multidimensional_telemetry(analysis_series, temporal=temporal)
            _json_write(output / "multidimensional_assessment.json", multidimensional)
            try:
                self.state_repository.save_multidimensional_assessment(job_id, multidimensional)
                self.state_repository.index_diagnostic_history(
                    job_id=job_id,
                    fault_id=(diagnosis_snapshot.get("candidate_fault_id") or diagnosis_snapshot.get("fault_id") or diagnosis_snapshot.get("selected_fault_id")),
                    temporal=temporal,
                    multidimensional=multidimensional,
                )
            except Exception:
                pass
            report["multidimensional_assessment"] = {
                "status": multidimensional.get("status"),
                "endpoint": f"/v1/platform/jobs/{job_id}/multidimensional-assessment",
                "diagnostic_vote_allowed": False,
                "truth_or_labels_used": False,
            }

            maintenance = build_maintenance_presentation(
                report=report, visualization=visualization, temporal=temporal, multidimensional=multidimensional
            )
            _json_write(output / "maintenance.json", maintenance)
            try:
                self.state_repository.save_maintenance_presentation(job_id, maintenance)
            except Exception:
                pass
            report["maintenance_presentation"] = {
                "schema_version": maintenance.get("schema_version"),
                "endpoint": f"/v1/platform/jobs/{job_id}/maintenance",
            }

            self._set_stage(job_id, "REPORTING", "写入结构化诊断报告")
            report_path = output / "report.json"
            markdown_path = output / "report.md"
            _json_write(report_path, report)
            markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
            self._finalize_governed_maintenance_rerun(
                job_id, report=report, artifact_kind=artifact_kind, paths=paths
            )
            self._finalize_maintenance_production_rerun(
                job_id, report=report, artifact_kind=artifact_kind, paths=paths
            )
            self._finalize_manual_upstream_recompute_child(
                job_id, report=report, artifact_kind=artifact_kind, paths=paths
            )
            job = self.get(job_id)
            job.update({
                "status": "SUCCEEDED",
                "stage": "COMPLETED",
                "stage_detail": "诊断、Agent 后处理与报告生成完成",
                "artifact_kind": artifact_kind,
                "completed_at": report["completed_at"],
                "report_json": str(report_path.relative_to(self._job_root(job_id))),
                "report_markdown": str(markdown_path.relative_to(self._job_root(job_id))),
            })
            self._save(job)
            try:
                self.state_repository.upsert_job(
                    {**report, "created_at": job.get("created_at"), "updated_at": job.get("updated_at")},
                    report_path=str(report_path.relative_to(self._job_root(job_id))),
                )
            except Exception:
                pass
            self._append_event(job_id, "[COMPLETED] 诊断报告生成完成")
            return report
        except Exception as exc:
            job = self.get(job_id)
            failed_stage = str(job.get("stage") or "UNKNOWN")
            trace = traceback.format_exc()
            trace_path = self._job_root(job_id) / "error.traceback.log"
            try:
                trace_path.write_text(trace, encoding="utf-8")
            except OSError:
                pass
            job["status"] = "FAILED"
            job["stage"] = "FAILED"
            job["stage_detail"] = f"失败于 {failed_stage}"
            job["completed_at"] = _now()
            job["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "failed_stage": failed_stage,
                "traceback_tail": trace[-8000:],
            }
            self._save(job)
            self._append_event(job_id, f"[FAILED:{failed_stage}] {type(exc).__name__}: {exc}")
            return {
                "schema_version": PLATFORM_REPORT_SCHEMA_VERSION,
                "job_id": job_id,
                "status": "FAILED",
                "task": job["task"],
                "work_mode": job.get("work_mode", "normal"),
                "artifact_kind": None,
                "error": job["error"],
                "formal_diagnosis_publishable": False,
            }

    def diagnostics(self, job_id: str, *, tail_chars: int = 12000) -> dict[str, Any]:
        job = self.get(job_id)
        root = self._job_root(job_id)
        chunks: list[str] = []
        for name in ("events.log", "worker.log", "error.traceback.log"):
            path = root / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if text.strip():
                chunks.append(f"--- {name} ---\n" + text[-tail_chars:])
        return {
            "schema_version": "astrograph-platform-job-diagnostics.v1",
            "job_id": job_id,
            "status": job.get("status"),
            "stage": job.get("stage"),
            "stage_detail": job.get("stage_detail"),
            "execution_backend": job.get("execution_backend"),
            "artifact_kind": job.get("artifact_kind"),
            "work_mode": job.get("work_mode"),
            "llm_runtime": job.get("llm_runtime") or self._llm_runtime_snapshot(),
            "llm_call_status": job.get("llm_call_status", "UNKNOWN"),
            "error": job.get("error"),
            "log_tail": "\n".join(chunks)[-tail_chars:],
            "secret_returned": False,
        }

    def report(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] != "SUCCEEDED" or not job.get("report_json"):
            raise ValueError("job has no successful report")
        path = self._job_root(job_id) / str(job["report_json"])
        return json.loads(path.read_text(encoding="utf-8"))

    def visualization(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] != "SUCCEEDED":
            raise ValueError("job has no successful visualization")
        path = self._job_root(job_id) / "output" / "visualization.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        from astrograph.integrations.platform_visualization import build_visualization
        expanded = self._job_root(job_id) / "expanded"
        result = build_visualization(expanded=expanded, report=self.report(job_id))
        _json_write(path, result)
        return result

    def _maintenance_analysis_series(self, job_id: str) -> dict[str, Any]:
        from astrograph.integrations.platform_visualization import build_analysis_series
        return build_analysis_series(
            expanded=self._job_root(job_id) / "expanded", report=self.report(job_id)
        )

    def temporal_localization(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        path = self._job_root(job_id) / "output" / "temporal_localization.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        cached = self.state_repository.get_temporal_localization(job_id)
        if cached is not None:
            return cached
        visualization = self.visualization(job_id)
        from astrograph.integrations.temporal_localization import localize_from_job_telemetry
        report = self.report(job_id)
        fault_id = self._overview_report_summary(report).get("candidate")
        result = localize_from_job_telemetry(
            expanded=self._job_root(job_id) / "expanded", report=report, visualization=visualization,
            analysis_series=self._maintenance_analysis_series(job_id), diagnostic_fault_id=fault_id
        )
        _json_write(path, result)
        self.state_repository.save_temporal_localization(job_id, result)
        return result

    def multidimensional_assessment(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        path = self._job_root(job_id) / "output" / "multidimensional_assessment.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        cached = self.state_repository.get_multidimensional_assessment(job_id)
        if cached is not None:
            return cached
        from astrograph.integrations.multidimensional_diagnostics import assess_multidimensional_telemetry
        payload = assess_multidimensional_telemetry(
            self._maintenance_analysis_series(job_id), temporal=self.temporal_localization(job_id)
        )
        _json_write(path, payload)
        self.state_repository.save_multidimensional_assessment(job_id, payload)
        return payload

    def diagnostic_history(self, *, fault_id: str | None = None, limit: int = 200) -> dict[str, Any]:
        return self.state_repository.diagnostic_history_summary(fault_id=fault_id, limit=limit)

    def _diagnostic_probe_reviews(self, job_id: str) -> list[dict[str, Any]]:
        probe_dir = self._job_root(job_id) / "output" / "diagnostic_probes"
        if not probe_dir.is_dir():
            return []
        reviews: list[dict[str, Any]] = []
        for path in sorted(probe_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("probe_id") and payload.get("result_sha256"):
                reviews.append(payload)
        reviews.sort(key=lambda item: (str(item.get("reviewed_at") or ""), str(item.get("result_sha256") or "")), reverse=True)
        return reviews[:50]


    def _maintenance_supplemental_root(self, job_id: str) -> Path:
        return self._job_root(job_id) / "output" / "maintenance_supplemental"

    def _maintenance_supplemental_overlay(self, job_id: str):
        path = self._maintenance_supplemental_root(job_id) / "current_overlay.json"
        if not path.is_file():
            return None, None, []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            from astrograph.evidence import EpisodeEvidenceBundle
            bundle = EpisodeEvidenceBundle.model_validate(payload.get("bundle"))
            return bundle, payload.get("latest_receipt_sha256"), list(payload.get("admission_receipt_sha256s") or [])
        except Exception as exc:
            raise ValueError(f"maintenance supplemental overlay is invalid: {type(exc).__name__}:{exc}") from exc

    def maintenance(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        if self.get(job_id).get("artifact_kind") == "RESULT_LOG_REVIEW":
            review = (self.report(job_id).get("analysis") or {}).get("result_log_review") or {}
            return {
                "schema_version": "astrograph-material-maintenance.v1", "job_id": job_id,
                "diagnosis_disposition": review.get("status"), "requires_human_review": True,
                "release_status": "BLOCKED", "material_review": review,
                "claim_boundary": "Evidence-only review; no telemetry detection or control authorization.",
            }
        from astrograph.integrations.maintenance_presentation import (
            MAINTENANCE_SCHEMA_VERSION, build_maintenance_presentation,
        )
        overlay_bundle, overlay_receipt_sha256, _ = self._maintenance_supplemental_overlay(job_id)
        path = self._job_root(job_id) / "output" / "maintenance.json"
        def cache_matches(payload: dict[str, Any]) -> bool:
            if payload.get("schema_version") != MAINTENANCE_SCHEMA_VERSION:
                return False
            observed = (payload.get("evidence_summary") or {}).get("supplemental_overlay_receipt_sha256")
            return observed == overlay_receipt_sha256
        if path.is_file():
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if cache_matches(persisted):
                return persisted
        cached = self.state_repository.get_maintenance_presentation(job_id)
        if cached is not None and cache_matches(cached):
            return cached
        payload = build_maintenance_presentation(
            report=self.report(job_id),
            visualization=self.visualization(job_id),
            temporal=self.temporal_localization(job_id),
            multidimensional=self.multidimensional_assessment(job_id),
            probe_reviews=self._diagnostic_probe_reviews(job_id),
            supplemental_evidence_bundle=overlay_bundle,
            supplemental_overlay_receipt_sha256=overlay_receipt_sha256,
        )
        _json_write(path, payload)
        self.state_repository.save_maintenance_presentation(job_id, payload)
        return payload

    def _diagnostic_workspace_source_path(self, job_id: str) -> Path:
        return self._job_root(job_id) / "output" / "diagnostic_workspace_source_bundle.json"

    def retain_diagnostic_workspace_sources(
        self, job_id: str, *, system_model_graph: Any | None = None,
        simulation_sources: list[Any] | tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        """Retain already-produced graph/trace bytes for the read-only workspace.

        This is an internal producer handoff, not a browser/API write surface.  It
        validates exact R23/R29/R31 contracts and never invokes a model or provider.
        """
        job = self.get(job_id)
        if job.get("status") != "SUCCEEDED":
            raise ValueError("diagnostic workspace sources require a successful frozen job")
        from astrograph.integrations.diagnostic_workspace_presentation import (
            WorkspaceSimulationSource, authority_snapshot, build_source_bundle,
            candidate_context_identity,
        )
        from astrograph.knowledge.system_model_graph import SystemModelGraph
        graph = SystemModelGraph.model_validate(system_model_graph) if system_model_graph is not None else None
        sources = tuple(WorkspaceSimulationSource.model_validate(item) for item in simulation_sources)
        maintenance = self.maintenance(job_id)
        authority = authority_snapshot(self.report(job_id), maintenance)
        bundle = build_source_bundle(
            job_id=job_id, authority=authority,
            candidate_context_sha256=sha256_json(candidate_context_identity(maintenance)),
            system_model_graph=graph, simulation_sources=sources,
        )
        payload = bundle.model_dump(mode="json")
        path = self._diagnostic_workspace_source_path(job_id)
        # Publish complete bytes with a same-filesystem hard-link claim.  Unlike
        # exists-then-replace, this gives concurrent first retention strict
        # first-writer-wins semantics and never exposes a partial JSON file.
        raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp"
        )
        try:
            temporary.write_text(raw, encoding="utf-8", newline="\n")
            try:
                os.link(temporary, path)
                return payload
            except FileExistsError:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != payload:
                    raise ValueError("diagnostic workspace source bundle is immutable after retention")
                return existing
        finally:
            temporary.unlink(missing_ok=True)

    def diagnostic_workspace(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        from astrograph.integrations.diagnostic_workspace_presentation import (
            DiagnosticWorkspaceSourceBundle, build_diagnostic_workspace_presentation,
        )
        path = self._diagnostic_workspace_source_path(job_id)
        source_bundle = None
        if path.is_file():
            try:
                source_bundle = DiagnosticWorkspaceSourceBundle.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise ValueError(f"diagnostic workspace source bundle is invalid: {type(exc).__name__}:{exc}") from exc
        return build_diagnostic_workspace_presentation(
            report=self.report(job_id), maintenance=self.maintenance(job_id),
            visualization=self.visualization(job_id), source_bundle=source_bundle,
        )

    def maintenance_input_contracts(self, job_id: str) -> dict[str, Any]:
        maintenance = self.maintenance(job_id)
        contracts = list(maintenance.get("evidence_gap_input_contracts") or [])
        return {
            "schema_version": "astrograph-maintenance-input-contracts.v1",
            "job_id": job_id,
            "maintenance_schema_version": maintenance.get("schema_version"),
            "diagnosis_status": maintenance.get("diagnosis_status"),
            "evidence_summary_status": (maintenance.get("evidence_summary") or {}).get("status"),
            "contract_count": len(contracts),
            "contracts": contracts,
            "next_input_contract_id": (maintenance.get("next_diagnostic_check") or {}).get("input_contract_id"),
            "claim_boundary": (
                "Machine-readable maintenance input requirements only. Consumers must still pass the formal telemetry/feature admission pipeline; "
                "this endpoint does not upload data, recompute FeatureResult, or create EvidenceFact records."
            ),
        }

    def maintenance_input_contract_template(self, job_id: str, contract_id: str) -> dict[str, Any]:
        if not contract_id or len(contract_id) > 240:
            raise ValueError("invalid maintenance input contract id")
        maintenance = self.maintenance(job_id)
        contract = next(
            (item for item in (maintenance.get("evidence_gap_input_contracts") or []) if item.get("contract_id") == contract_id),
            None,
        )
        if contract is None:
            raise ValueError("maintenance input contract is not current")
        from astrograph.integrations.maintenance_input_admission import build_supplemental_input_template
        return build_supplemental_input_template(contract)

    def maintenance_supplemental_admissions(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        root = self._maintenance_supplemental_root(job_id) / "admissions"
        receipts: list[dict[str, Any]] = []
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                try:
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    receipt = artifact.get("receipt") if isinstance(artifact, dict) else None
                    if isinstance(receipt, dict) and receipt.get("receipt_sha256"):
                        receipts.append(receipt)
                except (OSError, json.JSONDecodeError):
                    continue
        receipts.sort(key=lambda item: str(item.get("admitted_at") or ""), reverse=True)
        return {
            "schema_version": "astrograph-maintenance-supplemental-admissions.v1",
            "job_id": job_id,
            "count": len(receipts),
            "admissions": receipts[:100],
            "claim_boundary": (
                "These are maintenance supplemental reanalysis admissions. They do not mutate the original frozen report or release status; "
                "full production rerun remains required before release evidence changes."
            ),
        }

    def admit_maintenance_feature_input(
        self, job_id: str, *, contract_id: str, capsule_payload: dict[str, Any], actor: str = "operator"
    ) -> dict[str, Any]:
        self.get(job_id)
        maintenance = self.maintenance(job_id)
        contract = next(
            (item for item in (maintenance.get("evidence_gap_input_contracts") or []) if item.get("contract_id") == contract_id),
            None,
        )
        if contract is None:
            raise ValueError("maintenance input contract is not current")
        if contract.get("supplemental_admission_status") != "SUPPORTED":
            reasons = ",".join(contract.get("supplemental_admission_reason_codes") or [])
            raise ValueError(f"current maintenance input contract is not eligible for direct supplemental admission: {reasons}")

        from astrograph.integrations.maintenance_input_admission import (
            OVERLAY_SCHEMA_VERSION, SupplementalFeatureInputCapsule, admit_supplemental_feature_input,
        )
        from astrograph.integrations.maintenance_presentation import build_maintenance_presentation, extract_formal_evidence_bundle
        capsule = SupplementalFeatureInputCapsule.model_validate(capsule_payload)
        if capsule.contract_id != contract_id:
            raise ValueError("capsule contract_id does not match URL contract_id")
        source_report = self.report(job_id)
        original_bundle = extract_formal_evidence_bundle(source_report)
        current_overlay, _, receipt_chain = self._maintenance_supplemental_overlay(job_id)
        base_bundle = current_overlay or original_bundle
        if base_bundle is None or original_bundle is None:
            raise ValueError("current report has no formal EpisodeEvidenceBundle for supplemental reanalysis")
        receipt, overlay_bundle = admit_supplemental_feature_input(
            bundle=base_bundle, contract=contract, capsule=capsule, actor=actor
        )
        receipt["job_id"] = job_id
        receipt["admitted_at"] = _now()
        # Bind the final persisted receipt after server-owned job/time fields are added.
        receipt_core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        receipt["receipt_sha256"] = _sha256_bytes(
            json.dumps(receipt_core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        from astrograph.governance.supplemental_provenance import build_supplemental_input_provenance_receipt
        provenance = build_supplemental_input_provenance_receipt(
            source_job_id=job_id,
            source_report=source_report,
            original_evidence_bundle=original_bundle.model_dump(mode="json"),
            contract=contract,
            capsule=capsule.model_dump(mode="json"),
            admission_receipt_sha256=receipt["receipt_sha256"],
            parent_admission_receipt_sha256s=receipt_chain,
            actor=actor,
            admitted_at=receipt["admitted_at"],
        ).model_dump(mode="json")
        root = self._maintenance_supplemental_root(job_id)
        admissions_dir = root / "admissions"
        artifact = {
            "schema_version": "astrograph-maintenance-supplemental-admission-artifact.v2",
            "receipt": receipt,
            "capsule": capsule.model_dump(mode="json"),
            "provenance_receipt": provenance,
        }
        _json_write(admissions_dir / f"{receipt['receipt_sha256']}.json", artifact)
        _json_write(root / "provenance" / f"{provenance['receipt_sha256']}.json", provenance)
        next_chain = [*receipt_chain, receipt["receipt_sha256"]]
        overlay_payload = {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "updated_at": receipt["admitted_at"],
            "latest_receipt_sha256": receipt["receipt_sha256"],
            "admission_receipt_sha256s": next_chain,
            "bundle": overlay_bundle.model_dump(mode="json"),
            "original_report_mutated": False,
            "formal_release_eligible": False,
        }
        _json_write(root / "current_overlay.json", overlay_payload)

        refreshed = build_maintenance_presentation(
            report=self.report(job_id), visualization=self.visualization(job_id),
            temporal=self.temporal_localization(job_id),
            multidimensional=self.multidimensional_assessment(job_id),
            probe_reviews=self._diagnostic_probe_reviews(job_id),
            supplemental_evidence_bundle=overlay_bundle,
            supplemental_overlay_receipt_sha256=receipt["receipt_sha256"],
        )
        maintenance_path = self._job_root(job_id) / "output" / "maintenance.json"
        _json_write(maintenance_path, refreshed)
        self.state_repository.save_maintenance_presentation(job_id, refreshed)
        self.state_repository.add_annotation(
            job_id=job_id, annotation_type="SUPPLEMENTAL_EVIDENCE_ADMISSION",
            payload={
                "contract_id": contract_id, "receipt_sha256": receipt["receipt_sha256"],
                "evidence_definition_id": receipt.get("evidence_definition_id"),
                "hypothesis_status_before": receipt.get("hypothesis_status_before"),
                "hypothesis_status_after": receipt.get("hypothesis_status_after"),
                "original_report_mutated": False, "formal_release_eligible": False,
            }, actor=actor,
        )
        return {
            **receipt,
            "provenance_receipt": provenance,
            "maintenance_schema_version": refreshed.get("schema_version"),
            "evidence_summary_after": refreshed.get("evidence_summary"),
            "next_diagnostic_check_after": refreshed.get("next_diagnostic_check"),
        }

    def maintenance_supplemental_provenance(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        root = self._maintenance_supplemental_root(job_id) / "provenance"
        receipts: list[dict[str, Any]] = []
        if root.is_dir():
            from astrograph.governance.supplemental_provenance import verify_supplemental_input_provenance_receipt
            for path in sorted(root.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    receipts.append(
                        verify_supplemental_input_provenance_receipt(payload).model_dump(mode="json")
                    )
                except Exception:
                    continue
        receipts.sort(key=lambda item: str(item.get("admitted_at") or ""), reverse=True)
        return {
            "schema_version": "astrograph-maintenance-supplemental-provenance-index.v1",
            "job_id": job_id,
            "count": len(receipts),
            "receipts": receipts[:100],
            "claim_boundary": (
                "Each receipt content-binds one admitted maintenance capsule to the frozen source report, evidence bundle and input contract. "
                "These receipts provide provenance only and do not grant release authority."
            ),
        }

    def _maintenance_rerun_source_context(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        source_report = self.report(job_id)
        overlay_bundle, latest_overlay_receipt_sha256, receipt_chain = self._maintenance_supplemental_overlay(job_id)
        expanded = self._safe_expand(job_id)
        try:
            artifact_kind, paths = self._classify(expanded)
        except Exception:
            artifact_kind, paths = str(job.get("artifact_kind") or "UNKNOWN"), {}
        original_input = None
        source_oag_input_sha256 = None
        if artifact_kind == "OAG_DIAGNOSTIC" and paths.get("input") is not None:
            from astrograph.oag.contracts import OAGInput
            source_path = paths["input"]
            original_input = OAGInput.model_validate_json(source_path.read_text(encoding="utf-8"))
            source_oag_input_sha256 = _sha256_path(source_path)
        provenance_index = self.maintenance_supplemental_provenance(job_id)
        return {
            "job": job,
            "source_report": source_report,
            "overlay_bundle": overlay_bundle,
            "latest_overlay_receipt_sha256": latest_overlay_receipt_sha256,
            "receipt_chain": receipt_chain,
            "expanded": expanded,
            "artifact_kind": artifact_kind,
            "paths": paths,
            "original_input": original_input,
            "source_oag_input_sha256": source_oag_input_sha256,
            "provenance_receipts": provenance_index.get("receipts") or [],
        }

    def maintenance_governed_reruns(self, job_id: str) -> dict[str, Any]:
        """Return verified governed rerun lineage for one frozen source job."""
        self.get(job_id)
        root = self._maintenance_supplemental_root(job_id) / "reruns"
        from astrograph.integrations.maintenance_rerun import (
            verify_governed_rerun_completion_receipt,
            verify_governed_rerun_receipt,
        )
        entries: list[dict[str, Any]] = []
        if root.is_dir():
            prepared_paths = sorted(
                path for path in root.glob("*.json")
                if not path.name.endswith(".completion.json")
            )
            for path in prepared_paths:
                try:
                    prepared = verify_governed_rerun_receipt(
                        json.loads(path.read_text(encoding="utf-8"))
                    ).model_dump(mode="json")
                except Exception:
                    continue
                child_id = str(prepared.get("child_job_id") or "")
                completion = None
                completion_path = root / f"{child_id}.completion.json"
                if completion_path.is_file():
                    try:
                        completion = verify_governed_rerun_completion_receipt(
                            json.loads(completion_path.read_text(encoding="utf-8"))
                        ).model_dump(mode="json")
                    except Exception:
                        completion = None
                try:
                    child = self.get(child_id)
                    child_status = str(child.get("status") or "UNKNOWN")
                except KeyError:
                    child_status = "MISSING"
                entries.append({
                    "child_job_id": child_id,
                    "child_status": child_status,
                    "prepared_receipt": prepared,
                    "completion_receipt": completion,
                    "completed": completion is not None,
                })
        entries.sort(
            key=lambda item: str((item.get("prepared_receipt") or {}).get("created_at") or ""),
            reverse=True,
        )
        return {
            "schema_version": "astrograph-maintenance-governed-rerun-index.v1",
            "source_job_id": job_id,
            "count": len(entries),
            "reruns": entries[:100],
            "claim_boundary": (
                "This index exposes evidence-branch rerun lineage only. Child results remain review-only and "
                "cannot change the frozen source report or formal release state."
            ),
        }

    @staticmethod
    def _equivalent_active_governed_rerun(
        history: dict[str, Any], readiness: Any, provenance_sha256s: list[str]
    ) -> dict[str, Any] | None:
        for entry in history.get("reruns") or []:
            prepared = entry.get("prepared_receipt") or {}
            if str(entry.get("child_status") or "") in {"FAILED", "CANCELLED", "MISSING"}:
                continue
            if (
                prepared.get("source_report_sha256") == readiness.source_report_sha256
                and prepared.get("source_oag_input_sha256") == readiness.source_oag_input_sha256
                and prepared.get("rerun_input_sha256") == readiness.rerun_input_sha256
                and list(prepared.get("supplemental_receipt_sha256s") or [])
                == list(readiness.supplemental_receipt_sha256s)
                and list(prepared.get("supplemental_provenance_receipt_sha256s") or [])
                == list(provenance_sha256s)
            ):
                return entry
        return None

    @staticmethod
    def _governed_rerun_idempotency_key(readiness: Any, provenance_sha256s: list[str]) -> str:
        from astrograph.integrations.maintenance_rerun import canonical_sha256
        return canonical_sha256({
            "source_report_sha256": readiness.source_report_sha256,
            "source_oag_input_sha256": readiness.source_oag_input_sha256,
            "rerun_input_sha256": readiness.rerun_input_sha256,
            "supplemental_receipt_sha256s": list(readiness.supplemental_receipt_sha256s),
            "supplemental_provenance_receipt_sha256s": list(provenance_sha256s),
        })

    def _claim_governed_rerun_creation(
        self, job_id: str, readiness: Any, provenance_sha256s: list[str], *, actor: str
    ) -> tuple[Path, str]:
        key = self._governed_rerun_idempotency_key(readiness, provenance_sha256s)
        claim_dir = self._maintenance_supplemental_root(job_id) / "reruns" / "idempotency"
        claim_dir.mkdir(parents=True, exist_ok=True)
        claim_path = claim_dir / f"{key}.json"
        if claim_path.exists():
            try:
                existing = json.loads(claim_path.read_text(encoding="utf-8"))
            except Exception:
                raise ValueError("equivalent governed rerun creation claim is unreadable")
            child_id = str(existing.get("child_job_id") or "")
            if child_id:
                try:
                    child_status = str(self.get(child_id).get("status") or "UNKNOWN")
                except KeyError:
                    child_status = "MISSING"
                if child_status in {"FAILED", "CANCELLED", "MISSING"}:
                    claim_path.unlink(missing_ok=True)
                else:
                    raise ValueError(f"EQUIVALENT_GOVERNED_RERUN_EXISTS:{child_id}:{child_status}")
            else:
                raise ValueError("EQUIVALENT_GOVERNED_RERUN_CREATION_IN_PROGRESS")
        claim = {
            "schema_version": "astrograph-maintenance-governed-rerun-idempotency-claim.v1",
            "identity_sha256": key,
            "source_job_id": job_id,
            "child_job_id": None,
            "status": "CLAIMED",
            "actor": actor,
            "created_at": _now(),
            "formal_release_eligible": False,
        }
        try:
            fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError("EQUIVALENT_GOVERNED_RERUN_CREATION_IN_PROGRESS") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return claim_path, key

    def maintenance_governed_rerun_readiness(self, job_id: str) -> dict[str, Any]:
        context = self._maintenance_rerun_source_context(job_id)
        from astrograph.integrations.maintenance_rerun import assess_governed_oag_rerun
        readiness = assess_governed_oag_rerun(
            source_job_id=job_id,
            source_artifact_kind=str(context["artifact_kind"] or "UNKNOWN"),
            original_input=context["original_input"],
            overlay_bundle=context["overlay_bundle"],
            source_report=context["source_report"],
            source_oag_input_sha256=context["source_oag_input_sha256"],
            supplemental_receipt_sha256s=list(context["receipt_chain"] or []),
            supplemental_provenance_receipts=list(context["provenance_receipts"] or []),
        )
        if context["job"].get("status") != "SUCCEEDED":
            readiness = readiness.model_copy(update={
                "status": "BLOCKED",
                "blockers": sorted(set([*readiness.blockers, "SOURCE_JOB_NOT_SUCCEEDED"])),
            })
        provenance_sha256s = [
            str(item.get("receipt_sha256")) for item in context["provenance_receipts"] if item.get("receipt_sha256")
        ]
        equivalent = None
        if readiness.status == "READY":
            equivalent = self._equivalent_active_governed_rerun(
                self.maintenance_governed_reruns(job_id), readiness, provenance_sha256s
            )
            if equivalent is not None:
                readiness = readiness.model_copy(update={
                    "status": "BLOCKED",
                    "blockers": sorted(set([*readiness.blockers, "EQUIVALENT_GOVERNED_RERUN_EXISTS"])),
                })
        payload = readiness.model_dump(mode="json")
        payload["source_job_id"] = job_id
        payload["latest_overlay_receipt_sha256"] = context["latest_overlay_receipt_sha256"]
        payload["supplemental_provenance_receipt_sha256s"] = provenance_sha256s
        payload["equivalent_child_job_id"] = (equivalent or {}).get("child_job_id")
        payload["equivalent_child_status"] = (equivalent or {}).get("child_status")
        return payload

    def create_maintenance_governed_rerun(
        self, job_id: str, *, actor: str = "operator", start: bool = False
    ) -> dict[str, Any]:
        context = self._maintenance_rerun_source_context(job_id)
        from astrograph.integrations.maintenance_rerun import (
            MaintenanceGovernedRerunReadiness,
            build_governed_oag_rerun_input,
            build_governed_rerun_receipt,
        )
        readiness_payload = self.maintenance_governed_rerun_readiness(job_id)
        readiness = MaintenanceGovernedRerunReadiness.model_validate({
            key: readiness_payload[key]
            for key in MaintenanceGovernedRerunReadiness.model_fields
        })
        if readiness.status != "READY":
            raise ValueError("governed maintenance rerun is blocked: " + ",".join(readiness.blockers))
        if context["original_input"] is None or context["overlay_bundle"] is None:
            raise ValueError("governed maintenance rerun source context is incomplete")

        provenance_sha256s = [
            str(item.get("receipt_sha256")) for item in context["provenance_receipts"] if item.get("receipt_sha256")
        ]
        claim_path, claim_key = self._claim_governed_rerun_creation(
            job_id, readiness, provenance_sha256s, actor=actor
        )
        rerun_input = build_governed_oag_rerun_input(
            original_input=context["original_input"], overlay_bundle=context["overlay_bundle"]
        )
        rerun_input_bytes = rerun_input.model_dump_json(indent=2).encode("utf-8")
        source_job = context["job"]
        child_id = ""
        try:
            child = self.create(
                f"受治理维护证据重跑：{str(source_job.get('task') or '')}"[:4000],
                source_job.get("work_mode", "normal"),
            )
            child_id = str(child["job_id"])
            _json_write(claim_path, {
                "schema_version": "astrograph-maintenance-governed-rerun-idempotency-claim.v1",
                "identity_sha256": claim_key, "source_job_id": job_id, "child_job_id": child_id,
                "status": "CHILD_CREATED", "actor": actor, "created_at": _now(),
                "formal_release_eligible": False,
            })
            self.put_file(child_id, "maintenance_rerun_oag.json", rerun_input_bytes)
            executable = {context["paths"]["input"].resolve()} if context["paths"].get("input") else set()
            knowledge_paths = self._knowledge_document_paths(context["expanded"], executable)
            if knowledge_paths:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for path in knowledge_paths:
                        relative = path.resolve().relative_to(context["expanded"].resolve()).as_posix()
                        archive.writestr(relative, path.read_bytes())
                self.put_file(child_id, "maintenance_rerun_knowledge.zip", buffer.getvalue())

            child_job = self.get(child_id)
            child_job["parent_job_id"] = job_id
            child_job["rerun_provenance"] = {
                "schema_version": "astrograph-maintenance-rerun-job-link.v1",
                "rerun_mode": "OAG_EVIDENCE_BRANCH_REPLAY",
                "source_job_id": job_id,
                "source_overlay_receipt_sha256": context["latest_overlay_receipt_sha256"],
                "source_admission_receipt_sha256s": list(context["receipt_chain"] or []),
                "source_provenance_receipt_sha256s": [
                    str(item.get("receipt_sha256")) for item in context["provenance_receipts"] if item.get("receipt_sha256")
                ],
                "raw_telemetry_rerun_performed": False,
                "formal_release_eligible": False,
            }
            self._save(child_job)
            preflight = self.preflight(child_id)
            if not preflight.get("can_start"):
                raise ValueError("generated governed rerun child failed preflight")

            receipt = build_governed_rerun_receipt(
                source_job_id=job_id,
                child_job_id=child_id,
                actor=actor,
                created_at=_now(),
                readiness=readiness,
                child_preflight=preflight,
                supplemental_provenance_receipt_sha256s=[
                    str(item.get("receipt_sha256")) for item in context["provenance_receipts"] if item.get("receipt_sha256")
                ],
            )
            source_rerun_dir = self._maintenance_supplemental_root(job_id) / "reruns"
            _json_write(source_rerun_dir / f"{child_id}.json", receipt)
            _json_write(self._job_root(child_id) / "output" / "maintenance_rerun_receipt.json", receipt)
            linked = self.get(child_id)
            linked["rerun_provenance"] = {**(linked.get("rerun_provenance") or {}), "receipt_sha256": receipt["receipt_sha256"]}
            self._save(linked)
            self.state_repository.add_annotation(
                job_id=job_id,
                annotation_type="MAINTENANCE_GOVERNED_RERUN_CREATED",
                payload={
                    "child_job_id": child_id,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "rerun_mode": receipt["rerun_mode"],
                    "raw_telemetry_rerun_performed": False,
                    "formal_release_eligible": False,
                },
                actor=actor,
            )
            child_state = self.start(child_id) if start else self.get(child_id)
            return {
                "schema_version": "astrograph-maintenance-governed-rerun-created.v1",
                "source_job_id": job_id,
                "child_job_id": child_id,
                "started": bool(start),
                "child_job": child_state,
                "preflight": preflight,
                "receipt": receipt,
                "claim_boundary": receipt["claim_boundary"],
            }
        except Exception:
            if child_id:
                try:
                    failed = self.get(child_id)
                except KeyError:
                    failed = None
                if failed is not None and failed.get("status") in {"CREATED", "UPLOADING"}:
                    failed["status"] = "FAILED"
                    failed["stage"] = "FAILED"
                    failed["stage_detail"] = "受治理维护重跑准备失败"
                    self._save(failed)
            claim_path.unlink(missing_ok=True)
            raise

    def _maintenance_production_root(self, job_id: str) -> Path:
        return self._job_root(job_id) / "output" / "maintenance_production_rerun"

    def maintenance_production_raw_input_contracts(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("status") != "SUCCEEDED":
            raise ValueError("production raw-input contracts require a completed source job")
        expanded = self._safe_expand(job_id)
        artifact_kind, paths = self._classify(expanded)
        if artifact_kind == "PUBLIC_CASE":
            from astrograph.integrations.maintenance_production_rerun import build_public_case_raw_input_contracts
            payload = build_public_case_raw_input_contracts(paths["case_root"])
            payload["job_id"] = job_id
            return payload
        blockers = (
            ["UPSTREAM_RECOMPUTE_ADAPTER_REQUIRED"]
            if artifact_kind == "PIPELINE_RUNTIME"
            else ["USE_EXISTING_OAG_GOVERNED_RERUN_WORKFLOW"]
            if artifact_kind == "OAG_DIAGNOSTIC"
            else ["SOURCE_ARTIFACT_KIND_UNSUPPORTED"]
        )
        upstream_request = None
        if artifact_kind == "PIPELINE_RUNTIME":
            upstream_view = self.maintenance_upstream_recompute_request(job_id)
            upstream_request = upstream_view["request"]
        return {
            "schema_version": "astrograph-maintenance-production-raw-input-contracts.v1",
            "status": "BLOCKED",
            "job_id": job_id,
            "source_artifact_kind": artifact_kind,
            "contract_count": 0,
            "contracts": [],
            "blockers": blockers,
            "warnings": [],
            "required_public_binding_schema_version": (
                "sat-sim.astrograph-maintenance-raw-input-binding.v1" if artifact_kind == "PUBLIC_CASE" else None
            ),
            "upstream_recompute_request": upstream_request,
            "upstream_provider_capabilities": (self.upstream_provider_capabilities() if artifact_kind == "PIPELINE_RUNTIME" else None),
            "claim_boundary": (
                "PIPELINE_RUNTIME is an already-derived upstream runtime artifact and cannot be reconstructed from a maintenance capsule. "
                "OAG evidence-branch replay uses the separate governed-rerun workflow."
            ),
        }

    def maintenance_production_raw_input_template(self, job_id: str, contract_id: str) -> dict[str, Any]:
        contracts = self.maintenance_production_raw_input_contracts(job_id)
        contract = next((item for item in contracts.get("contracts") or [] if item.get("contract_id") == contract_id), None)
        if contract is None:
            raise ValueError("production raw-input contract is not current")
        expanded = self._safe_expand(job_id)
        artifact_kind, paths = self._classify(expanded)
        if artifact_kind != "PUBLIC_CASE":
            raise ValueError("production raw-input templates are currently supported only for PUBLIC_CASE")
        from astrograph.integrations.maintenance_production_rerun import build_public_case_raw_patch_template
        return build_public_case_raw_patch_template(contract, paths["case_root"])

    def _maintenance_production_admission_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        root = self._maintenance_production_root(job_id) / "admissions"
        if not root.is_dir():
            return []
        from astrograph.integrations.maintenance_production_rerun import verify_public_case_raw_patch_admission
        artifacts: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.json")):
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
                receipt = verify_public_case_raw_patch_admission(dict(artifact.get("receipt") or {}))
                if str(receipt.get("receipt_sha256") or "") != path.stem:
                    continue
                artifacts.append(artifact)
            except Exception:
                continue
        artifacts.sort(key=lambda item: str((item.get("receipt") or {}).get("admitted_at") or ""))
        return artifacts

    def admit_maintenance_production_raw_input(
        self, job_id: str, *, contract_id: str, capsule_payload: dict[str, Any], actor: str = "operator"
    ) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("status") != "SUCCEEDED":
            raise ValueError("production raw-input admission requires a completed source job")
        contracts = self.maintenance_production_raw_input_contracts(job_id)
        contract = next((item for item in contracts.get("contracts") or [] if item.get("contract_id") == contract_id), None)
        if contract is None:
            raise ValueError("production raw-input contract is not current")
        expanded = self._safe_expand(job_id)
        artifact_kind, paths = self._classify(expanded)
        if artifact_kind != "PUBLIC_CASE":
            raise ValueError("production raw-input admission is currently supported only for PUBLIC_CASE")
        from astrograph.integrations.maintenance_production_rerun import (
            ProductionRawInputPatchCapsule,
            admit_public_case_raw_patch,
            canonical_sha256,
        )
        capsule = ProductionRawInputPatchCapsule.model_validate(capsule_payload)
        if capsule.contract_id != contract_id:
            raise ValueError("raw-input capsule contract_id does not match URL contract_id")
        capsule_identity = canonical_sha256(capsule.model_dump(mode="json", exclude={"source_note"}))
        for existing in self._maintenance_production_admission_artifacts(job_id):
            receipt = existing.get("receipt") or {}
            if receipt.get("contract_id") == contract_id and receipt.get("capsule_sha256") == capsule_identity:
                raise ValueError("identical production raw-input patch was already admitted for this contract")
            if receipt.get("contract_id") == contract_id:
                raise ValueError("one production raw-input patch per contract is allowed; create a new source job to revise it")
        admitted_at = _now()
        receipt = admit_public_case_raw_patch(
            case_root=paths["case_root"], contract=contract, capsule=capsule, actor=actor, admitted_at=admitted_at
        )
        artifact = {
            "schema_version": "astrograph-maintenance-production-raw-input-admission-artifact.v1",
            "contract": contract,
            "capsule": capsule.model_dump(mode="json"),
            "receipt": receipt,
        }
        root = self._maintenance_production_root(job_id) / "admissions"
        _json_write(root / f"{receipt['receipt_sha256']}.json", artifact)
        self.state_repository.add_annotation(
            job_id=job_id,
            annotation_type="MAINTENANCE_PRODUCTION_RAW_INPUT_ADMITTED",
            payload={
                "contract_id": contract_id,
                "field": receipt["field"],
                "receipt_sha256": receipt["receipt_sha256"],
                "raw_input_bound": True,
                "formal_release_eligible": False,
            },
            actor=actor,
        )
        return {
            "schema_version": "astrograph-maintenance-production-raw-input-admitted.v1",
            "job_id": job_id,
            "receipt": receipt,
            "production_rerun_readiness": self.maintenance_production_rerun_readiness(job_id),
        }

    def maintenance_production_reruns(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        root = self._maintenance_production_root(job_id) / "reruns"
        records: list[dict[str, Any]] = []
        if root.is_dir():
            from astrograph.integrations.maintenance_production_rerun import (
                verify_production_rerun_completion_receipt,
                verify_production_rerun_prepared_receipt,
            )
            for path in sorted(root.glob("*.prepared.json")):
                try:
                    prepared = verify_production_rerun_prepared_receipt(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
                child_id = str(prepared.get("child_job_id") or "")
                try:
                    child = self.get(child_id)
                    child_status = child.get("status")
                except KeyError:
                    child_status = "MISSING"
                completion = None
                completion_path = root / f"{child_id}.completion.json"
                if completion_path.is_file():
                    try:
                        completion = verify_production_rerun_completion_receipt(json.loads(completion_path.read_text(encoding="utf-8")))
                    except Exception:
                        completion = None
                records.append({
                    "child_job_id": child_id,
                    "child_status": child_status,
                    "patch_identity_sha256": prepared.get("patch_identity_sha256"),
                    "prepared_receipt": prepared,
                    "completion_receipt": completion,
                })
        records.sort(key=lambda item: str((item.get("prepared_receipt") or {}).get("created_at") or ""), reverse=True)
        return {
            "schema_version": "astrograph-maintenance-production-rerun-index.v1",
            "source_job_id": job_id,
            "count": len(records),
            "reruns": records[:100],
            "claim_boundary": "Only raw-input-bound PUBLIC_CASE children appear here; OAG evidence branches use the separate governed-rerun index.",
        }

    def _production_rerun_materialized_patch(self, job_id: str) -> tuple[dict[str, Any] | None, str, dict[str, Any], dict[str, Path]]:
        source_report = self.report(job_id)
        expanded = self._safe_expand(job_id)
        artifact_kind, paths = self._classify(expanded)
        materialized = None
        if artifact_kind == "PUBLIC_CASE":
            artifacts = self._maintenance_production_admission_artifacts(job_id)
            if artifacts:
                from astrograph.integrations.maintenance_production_rerun import materialize_public_case_raw_patch
                materialized = materialize_public_case_raw_patch(
                    case_root=paths["case_root"], admissions=artifacts, source_job_id=job_id
                )
        return materialized, artifact_kind, source_report, paths

    def maintenance_production_rerun_readiness(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        materialized, artifact_kind, source_report, _paths = self._production_rerun_materialized_patch(job_id)
        raw_contracts = self.maintenance_production_raw_input_contracts(job_id)
        from astrograph.integrations.maintenance_production_rerun import assess_maintenance_production_rerun
        readiness = assess_maintenance_production_rerun(
            source_artifact_kind=artifact_kind,
            source_report=source_report,
            materialized_patch=materialized,
            raw_contract_status=raw_contracts.get("status"),
            raw_contract_blockers=list(raw_contracts.get("blockers") or []),
        )
        blockers = list(readiness.blockers)
        if job.get("status") != "SUCCEEDED":
            blockers.append("SOURCE_JOB_NOT_SUCCEEDED")
        existing_child = None
        if readiness.status == "READY" and readiness.patch_identity_sha256:
            for record in self.maintenance_production_reruns(job_id).get("reruns") or []:
                if record.get("patch_identity_sha256") != readiness.patch_identity_sha256:
                    continue
                if record.get("child_status") not in {"FAILED", "CANCELLED", "MISSING"}:
                    existing_child = record
                    blockers.append("EQUIVALENT_PRODUCTION_RERUN_EXISTS")
                    break
        payload = readiness.model_dump(mode="json")
        if blockers:
            payload["status"] = "BLOCKED"
            payload["blockers"] = sorted(set(blockers))
            payload["raw_telemetry_rerun_planned"] = False
        payload["source_job_id"] = job_id
        payload["equivalent_child_job_id"] = (existing_child or {}).get("child_job_id")
        payload["equivalent_child_status"] = (existing_child or {}).get("child_status")
        return payload

    def _claim_production_rerun_creation(self, job_id: str, patch_identity_sha256: str, *, actor: str) -> Path:
        root = self._maintenance_production_root(job_id) / "claims"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{patch_identity_sha256}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                raise ValueError("equivalent production rerun creation claim is unreadable")
            child_id = str(existing.get("child_job_id") or "")
            if child_id:
                try:
                    child_status = str(self.get(child_id).get("status") or "UNKNOWN")
                except KeyError:
                    child_status = "MISSING"
                if child_status in {"FAILED", "CANCELLED", "MISSING"}:
                    path.unlink(missing_ok=True)
                else:
                    raise ValueError(f"EQUIVALENT_PRODUCTION_RERUN_EXISTS:{child_id}:{child_status}")
            else:
                raise ValueError("EQUIVALENT_PRODUCTION_RERUN_CREATION_IN_PROGRESS")
        claim = {
            "schema_version": "astrograph-maintenance-production-rerun-claim.v1",
            "source_job_id": job_id,
            "patch_identity_sha256": patch_identity_sha256,
            "child_job_id": None,
            "status": "CLAIMED",
            "actor": actor,
            "created_at": _now(),
            "formal_release_eligible": False,
        }
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError("EQUIVALENT_PRODUCTION_RERUN_CREATION_IN_PROGRESS") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path

    def create_maintenance_production_rerun(
        self, job_id: str, *, actor: str = "operator", start: bool = False
    ) -> dict[str, Any]:
        readiness_payload = self.maintenance_production_rerun_readiness(job_id)
        from astrograph.integrations.maintenance_production_rerun import (
            MaintenanceProductionRerunReadiness,
            build_production_rerun_prepared_receipt,
        )
        readiness = MaintenanceProductionRerunReadiness.model_validate({
            key: readiness_payload[key] for key in MaintenanceProductionRerunReadiness.model_fields
        })
        if readiness.status != "READY" or not readiness.patch_identity_sha256:
            raise ValueError("production maintenance rerun is blocked: " + ",".join(readiness.blockers))
        materialized, artifact_kind, _source_report, paths = self._production_rerun_materialized_patch(job_id)
        if artifact_kind != "PUBLIC_CASE" or materialized is None:
            raise ValueError("production maintenance rerun has no materialized PUBLIC_CASE raw patch")
        claim_path = self._claim_production_rerun_creation(job_id, readiness.patch_identity_sha256, actor=actor)
        child_id = ""
        source_job = self.get(job_id)
        try:
            child = self.create(
                f"受治理原始遥测重跑：{str(source_job.get('task') or '')}"[:4000],
                source_job.get("work_mode", "normal"),
            )
            child_id = str(child["job_id"])
            self.put_file(child_id, materialized["manifest_filename"], materialized["manifest_bytes"])
            self.put_file(child_id, materialized["telemetry_filename"], materialized["telemetry_bytes"])
            executable = {(paths["case_root"] / "manifest.json").resolve()}
            try:
                manifest = json.loads((paths["case_root"] / "manifest.json").read_text(encoding="utf-8"))
                telemetry_name = str((manifest.get("telemetry") or {}).get("file") or "")
                executable.add((paths["case_root"] / telemetry_name).resolve())
            except Exception:
                pass
            knowledge_paths = self._knowledge_document_paths(self._safe_expand(job_id), executable)
            if knowledge_paths:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for path in knowledge_paths:
                        relative = path.name
                        archive.writestr(relative, path.read_bytes())
                self.put_file(child_id, "maintenance_production_rerun_knowledge.zip", buffer.getvalue())
            linked = self.get(child_id)
            linked["parent_job_id"] = job_id
            linked["production_rerun_provenance"] = {
                "schema_version": "astrograph-maintenance-production-rerun-job-link.v1",
                "rerun_mode": "PUBLIC_CASE_RAW_TELEMETRY_PATCH_RERUN",
                "source_job_id": job_id,
                "patch_identity_sha256": readiness.patch_identity_sha256,
                "raw_telemetry_rerun_performed": True,
                "formal_release_eligible": False,
            }
            self._save(linked)
            preflight = self.preflight(child_id)
            if not preflight.get("can_start") or preflight.get("artifact_kind") != "PUBLIC_CASE":
                raise ValueError("generated production rerun child failed PUBLIC_CASE preflight")
            receipt = build_production_rerun_prepared_receipt(
                source_job_id=job_id,
                child_job_id=child_id,
                actor=actor,
                created_at=_now(),
                readiness=readiness,
                child_preflight=preflight,
            )
            rerun_dir = self._maintenance_production_root(job_id) / "reruns"
            _json_write(rerun_dir / f"{child_id}.prepared.json", receipt)
            _json_write(self._job_root(child_id) / "output" / "maintenance_production_rerun_receipt.json", receipt)
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim.update({"status": "CHILD_CREATED", "child_job_id": child_id, "prepared_receipt_sha256": receipt["receipt_sha256"]})
            _json_write(claim_path, claim)
            linked = self.get(child_id)
            linked["production_rerun_provenance"] = {
                **(linked.get("production_rerun_provenance") or {}),
                "prepared_receipt_sha256": receipt["receipt_sha256"],
            }
            self._save(linked)
            self.state_repository.add_annotation(
                job_id=job_id,
                annotation_type="MAINTENANCE_PRODUCTION_RERUN_CREATED",
                payload={
                    "child_job_id": child_id,
                    "patch_identity_sha256": readiness.patch_identity_sha256,
                    "prepared_receipt_sha256": receipt["receipt_sha256"],
                    "raw_telemetry_rerun_performed": True,
                    "formal_release_eligible": False,
                },
                actor=actor,
            )
            child_state = self.start(child_id) if start else self.get(child_id)
            return {
                "schema_version": "astrograph-maintenance-production-rerun-created.v1",
                "source_job_id": job_id,
                "child_job_id": child_id,
                "started": bool(start),
                "child_job": child_state,
                "preflight": preflight,
                "receipt": receipt,
                "claim_boundary": receipt["claim_boundary"],
            }
        except Exception:
            if child_id:
                try:
                    failed = self.get(child_id)
                except KeyError:
                    failed = None
                if failed is not None and failed.get("status") in {"CREATED", "UPLOADING"}:
                    failed["status"] = "FAILED"
                    failed["stage"] = "FAILED"
                    failed["stage_detail"] = "受治理原始遥测重跑准备失败"
                    self._save(failed)
            claim_path.unlink(missing_ok=True)
            raise

    def _finalize_maintenance_production_rerun(
        self, job_id: str, *, report: dict[str, Any], artifact_kind: str, paths: dict[str, Path]
    ) -> dict[str, Any] | None:
        job = self.get(job_id)
        link = job.get("production_rerun_provenance") or {}
        if str(link.get("rerun_mode") or "") != "PUBLIC_CASE_RAW_TELEMETRY_PATCH_RERUN":
            return None
        if artifact_kind != "PUBLIC_CASE" or paths.get("case_root") is None:
            raise ValueError("production rerun child lost its PUBLIC_CASE executable input")
        prepared_path = self._job_root(job_id) / "output" / "maintenance_production_rerun_receipt.json"
        if not prepared_path.is_file():
            raise ValueError("production rerun child is missing its prepared receipt")
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        source_job_id = str(prepared.get("source_job_id") or "")
        source_report = self.report(source_job_id)
        source_expanded = self._safe_expand(source_job_id)
        source_kind, source_paths = self._classify(source_expanded)
        if source_kind != "PUBLIC_CASE":
            raise ValueError("production rerun source is no longer PUBLIC_CASE")
        from astrograph.integrations.sat_sim_public_case import _load_public_case
        _source_manifest, source_telemetry = _load_public_case(source_paths["case_root"])
        _child_manifest, child_telemetry = _load_public_case(paths["case_root"])
        from astrograph.integrations.maintenance_production_rerun import build_production_rerun_completion_receipt
        completion = build_production_rerun_completion_receipt(
            prepared=prepared,
            completed_at=str(report.get("completed_at") or _now()),
            current_source_report=source_report,
            current_source_manifest_sha256=_sha256_path(source_paths["case_root"] / "manifest.json"),
            current_source_telemetry_sha256=_sha256_path(source_telemetry),
            actual_child_manifest_sha256=_sha256_path(paths["case_root"] / "manifest.json"),
            actual_child_telemetry_sha256=_sha256_path(child_telemetry),
            child_report=report,
        )
        completion_path = self._job_root(job_id) / "output" / "maintenance_production_rerun_completion_receipt.json"
        _json_write(completion_path, completion)
        rerun_dir = self._maintenance_production_root(source_job_id) / "reruns"
        _json_write(rerun_dir / f"{job_id}.completion.json", completion)
        linked = self.get(job_id)
        linked["production_rerun_provenance"] = {
            **(linked.get("production_rerun_provenance") or {}),
            "completion_receipt_sha256": completion["receipt_sha256"],
            "completion_status": completion["status"],
        }
        self._save(linked)
        self.state_repository.add_annotation(
            job_id=source_job_id,
            annotation_type="MAINTENANCE_PRODUCTION_RERUN_COMPLETED",
            payload={
                "child_job_id": job_id,
                "prepared_receipt_sha256": completion["prepared_receipt_sha256"],
                "completion_receipt_sha256": completion["receipt_sha256"],
                "child_diagnosis_status": completion["child_diagnosis_status"],
                "raw_telemetry_rerun_performed": True,
                "formal_release_eligible": False,
            },
            actor="system",
        )
        return completion

    def _finalize_governed_maintenance_rerun(
        self, job_id: str, *, report: dict[str, Any], artifact_kind: str, paths: dict[str, Path]
    ) -> dict[str, Any] | None:
        job = self.get(job_id)
        link = job.get("rerun_provenance") or {}
        if str(link.get("rerun_mode") or "") != "OAG_EVIDENCE_BRANCH_REPLAY":
            return None
        if artifact_kind != "OAG_DIAGNOSTIC" or paths.get("input") is None:
            raise ValueError("governed rerun child lost its OAG executable input")

        prepared_path = self._job_root(job_id) / "output" / "maintenance_rerun_receipt.json"
        if not prepared_path.is_file():
            raise ValueError("governed rerun child is missing its prepared receipt")
        prepared_payload = json.loads(prepared_path.read_text(encoding="utf-8"))
        from astrograph.integrations.maintenance_rerun import (
            build_governed_rerun_completion_receipt,
            verify_governed_rerun_receipt,
        )
        prepared = verify_governed_rerun_receipt(prepared_payload)
        if prepared.child_job_id != job_id:
            raise ValueError("governed rerun prepared receipt is bound to another child job")
        source_job_id = prepared.source_job_id
        source_job = self.get(source_job_id)
        if source_job.get("status") != "SUCCEEDED":
            raise ValueError("governed rerun source job no longer has SUCCEEDED status")
        source_report = self.report(source_job_id)
        source_expanded = self._safe_expand(source_job_id)
        source_kind, source_paths = self._classify(source_expanded)
        if source_kind != "OAG_DIAGNOSTIC" or source_paths.get("input") is None:
            raise ValueError("governed rerun source OAG input is no longer available")
        source_oag_input_sha256 = _sha256_path(source_paths["input"])

        from astrograph.oag.contracts import OAGInput
        actual_rerun_input = OAGInput.model_validate_json(paths["input"].read_text(encoding="utf-8"))
        completion = build_governed_rerun_completion_receipt(
            prepared_receipt=prepared_payload,
            source_report=source_report,
            source_oag_input_sha256=source_oag_input_sha256,
            actual_rerun_input=actual_rerun_input,
            child_report=report,
            completed_at=str(report.get("completed_at") or _now()),
        )
        completion_path = self._job_root(job_id) / "output" / "maintenance_rerun_completion_receipt.json"
        _json_write(completion_path, completion)
        source_rerun_dir = self._maintenance_supplemental_root(source_job_id) / "reruns"
        _json_write(source_rerun_dir / f"{job_id}.completion.json", completion)
        linked = self.get(job_id)
        linked["rerun_provenance"] = {
            **(linked.get("rerun_provenance") or {}),
            "completion_receipt_sha256": completion["receipt_sha256"],
            "completion_status": completion["status"],
        }
        self._save(linked)
        self.state_repository.add_annotation(
            job_id=source_job_id,
            annotation_type="MAINTENANCE_GOVERNED_RERUN_COMPLETED",
            payload={
                "child_job_id": job_id,
                "prepared_receipt_sha256": prepared.receipt_sha256,
                "completion_receipt_sha256": completion["receipt_sha256"],
                "child_diagnosis_status": completion["child_diagnosis_status"],
                "child_selected_fault_id": completion.get("child_selected_fault_id"),
                "child_release_status": completion["child_release_status"],
                "formal_release_eligible": False,
            },
            actor="system",
        )
        return completion

    def run_neo4j_maintenance_review(self, job_id: str, *, actor: str = "operator") -> dict[str, Any]:
        maintenance = self.maintenance(job_id)
        fault_id = str(maintenance.get("fault_id") or "").strip()
        if not fault_id:
            raise ValueError("current maintenance diagnosis has no fault candidate for graph review")
        from astrograph.tools.neo4j_maintenance import get_fault_mechanism
        result = get_fault_mechanism({"fault_id": fault_id, "max_nodes": 40})
        if result.get("status") != "SUCCEEDED":
            return result
        maintenance_channels = [
            str(item.get("channel")) for item in (maintenance.get("key_findings") or []) if item.get("channel")
        ]
        graph_nodes = []
        for node in (result.get("nodes") or []):
            enriched = dict(node)
            if str(enriched.get("id") or "") == fault_id:
                enriched["related_channels"] = maintenance_channels[:3]
            else:
                enriched.setdefault("related_channels", [])
            graph_nodes.append(enriched)
        graph = {
            "status": "AVAILABLE",
            "graph_kind": "FORMAL_KG_EVIDENCE",
            "title": "本次知识图谱复核",
            "nodes": graph_nodes,
            "edges": [
                {
                    "from": item.get("from"), "to": item.get("to"),
                    "relation": item.get("relation"), "label": item.get("relation"),
                    "source_ref": item.get("source_ref"),
                }
                for item in (result.get("edges") or [])
            ],
            "query_backed": True,
            "contributes_to_diagnosis": False,
            "boundary_zh": "本次路径来自配置的 Neo4j 只读查询，用于维护复核；不会自动改写已冻结诊断或融合投票。",
        }
        maintenance["mechanism_graph"] = graph
        path = self._job_root(job_id) / "output" / "maintenance.json"
        _json_write(path, maintenance)
        self.state_repository.save_maintenance_presentation(job_id, maintenance)
        self.state_repository.add_annotation(
            job_id=job_id, annotation_type="NEO4J_REVIEW",
            payload={"fault_id": fault_id, "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"]), "read_only": True},
            actor=actor,
        )
        return {"status":"SUCCEEDED","fault_id":fault_id,"mechanism_graph":graph}

    def run_maintenance_diagnostic_probe(
        self, job_id: str, *, probe_id: str, actor: str = "operator"
    ) -> dict[str, Any]:
        if not probe_id or len(probe_id) > 200:
            raise ValueError("invalid probe_id")
        report = self.report(job_id)
        from astrograph.integrations.maintenance_presentation import (
            build_maintenance_presentation, extract_formal_evidence_bundle,
        )
        from astrograph.investigation.diagnostic_probes import DiagnosticProbeExecutor, build_evidence_margin_probe_catalog
        bundle = extract_formal_evidence_bundle(report)
        if bundle is None:
            raise ValueError("current report has no formal EpisodeEvidenceBundle for approved probe execution")
        catalog = build_evidence_margin_probe_catalog(bundle)
        existing_probe_reviews = self._diagnostic_probe_reviews(job_id)
        maintenance = build_maintenance_presentation(
            report=report, visualization=self.visualization(job_id),
            temporal=self.temporal_localization(job_id),
            multidimensional=self.multidimensional_assessment(job_id),
            probe_reviews=existing_probe_reviews,
        )
        approved = (maintenance.get("next_diagnostic_check") or {}).get("probe_id")
        if not approved or probe_id != approved:
            raise ValueError("probe_id is not the currently approved maintenance diagnostic check")
        result = DiagnosticProbeExecutor().execute(catalog=catalog, probe_id=probe_id).model_dump(mode="json")
        result.update({
            "reviewed_at": _now(),
            "reviewed_by": actor,
            "diagnosis_changed": False,
            "release_status_changed": False,
            "formal_evidence_changed": False,
        })
        updated_maintenance = build_maintenance_presentation(
            report=report, visualization=self.visualization(job_id),
            temporal=self.temporal_localization(job_id),
            multidimensional=self.multidimensional_assessment(job_id),
            probe_reviews=[result, *existing_probe_reviews],
        )
        review_summary = next(
            (item for item in (updated_maintenance.get("probe_review_history") or [])
             if item.get("result_sha256") == result.get("result_sha256")),
            None,
        )
        result["gap_reconciliation"] = review_summary or {
            "probe_id": probe_id,
            "evidence_definition_id": result.get("evidence_definition_id"),
            "linkage_status": "UNAVAILABLE",
            "gap_closed_by_probe": False,
            "next_action_zh": "继续按正式证据协议处理。",
        }
        result["next_diagnostic_check_after"] = updated_maintenance.get("next_diagnostic_check")
        probe_dir = self._job_root(job_id) / "output" / "diagnostic_probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        _json_write(probe_dir / f"{result['result_sha256']}.json", result)
        maintenance_path = self._job_root(job_id) / "output" / "maintenance.json"
        _json_write(maintenance_path, updated_maintenance)
        self.state_repository.save_maintenance_presentation(job_id, updated_maintenance)
        self.state_repository.add_annotation(
            job_id=job_id, annotation_type="DIAGNOSTIC_PROBE_REVIEW",
            payload={
                "probe_id": probe_id, "result_sha256": result["result_sha256"],
                "status": result["status"], "shadow_only": True,
                "formal_evidence_changed": False, "diagnosis_changed": False,
                "gap_linkage_status": (review_summary or {}).get("linkage_status"),
                "formal_gap_open_after": (review_summary or {}).get("formal_gap_open_after"),
            }, actor=actor,
        )
        return result

    def add_maintenance_annotation(
        self, job_id: str, *, annotation_type: str, payload: dict[str, Any], actor: str = "operator"
    ) -> dict[str, Any]:
        self.get(job_id)
        return self.state_repository.add_annotation(
            job_id=job_id, annotation_type=annotation_type, payload=payload, actor=actor
        )

    def maintenance_annotations(self, job_id: str) -> list[dict[str, Any]]:
        self.get(job_id)
        return self.state_repository.annotations(job_id)

    def upstream_provider_capabilities(self) -> dict[str, Any]:
        from astrograph.upstream import provider_capabilities
        return provider_capabilities()

    def _upstream_recompute_root(self, job_id: str) -> Path:
        return self._job_root(job_id) / "output" / "upstream_recompute"

    def maintenance_upstream_recompute_request(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("status") != "SUCCEEDED":
            raise ValueError("upstream recompute request requires a completed source job")
        expanded = self._safe_expand(job_id)
        artifact_kind, paths = self._classify(expanded)
        if artifact_kind != "PIPELINE_RUNTIME":
            raise ValueError("upstream recompute request is currently defined only for PIPELINE_RUNTIME")
        maintenance = self.maintenance(job_id)
        from astrograph.upstream import build_pipeline_runtime_recompute_request
        request = build_pipeline_runtime_recompute_request(
            source_job_id=job_id,
            source_runtime_sha256=_sha256_path(paths["runtime"]),
            source_mapping_bundle_sha256=_sha256_path(paths["mapping"]),
            maintenance_input_contract_ids=[
                str(item.get("contract_id")) for item in (maintenance.get("evidence_gap_input_contracts") or [])
                if item.get("contract_id")
            ],
        ).model_dump(mode="json")
        root = self._upstream_recompute_root(job_id)
        _json_write(root / "request.json", request)
        return {
            "schema_version": "astrograph-upstream-recompute-request-view.v1",
            "job_id": job_id,
            "request": request,
            "providers": self.upstream_provider_capabilities(),
            "manual_exchange_available": True,
            "claim_boundary": (
                "The request freezes what must be recomputed, not how the upstream producer must be installed. "
                "AstroGraph core can continue in manual exchange mode when all optional automatic providers are unavailable."
            ),
        }

    def maintenance_upstream_recompute_request_bundle_path(self, job_id: str) -> Path:
        view = self.maintenance_upstream_recompute_request(job_id)
        from astrograph.upstream import build_manual_request_bundle
        data = build_manual_request_bundle(view["request"])
        path = self._upstream_recompute_root(job_id) / f"{view['request']['request_id']}.request.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def maintenance_upstream_recompute_history(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        root = self._upstream_recompute_root(job_id) / "reruns"
        items: list[dict[str, Any]] = []
        if root.is_dir():
            from astrograph.upstream.manual_provider import (
                verify_manual_child_prepared_receipt,
                verify_manual_child_completion_receipt,
            )
            for prepared_path in sorted(root.glob("*.prepared.json")):
                child_id = prepared_path.name.removesuffix(".prepared.json")
                try:
                    prepared = verify_manual_child_prepared_receipt(json.loads(prepared_path.read_text(encoding="utf-8")))
                except Exception:
                    continue
                completion = None
                completion_path = root / f"{child_id}.completion.json"
                if completion_path.is_file():
                    try:
                        completion = verify_manual_child_completion_receipt(json.loads(completion_path.read_text(encoding="utf-8")))
                    except Exception:
                        completion = None
                try:
                    child = self.get(child_id)
                    child_status = child.get("status")
                except KeyError:
                    child_status = "MISSING"
                items.append({
                    "child_job_id": child_id,
                    "child_status": child_status,
                    "prepared_receipt": prepared,
                    "completion_receipt": completion,
                })
        return {
            "schema_version": "astrograph-upstream-recompute-history.v1",
            "source_job_id": job_id,
            "items": items,
        }

    def create_manual_upstream_recompute_child(
        self, job_id: str, *, response_bundle: bytes, actor: str = "operator", start: bool = True
    ) -> dict[str, Any]:
        source_job = self.get(job_id)
        if source_job.get("status") != "SUCCEEDED":
            raise ValueError("manual upstream response import requires a completed source job")
        view = self.maintenance_upstream_recompute_request(job_id)
        request = view["request"]
        from astrograph.upstream.manual_provider import (
            build_manual_child_prepared_receipt,
            verify_manual_response_bundle,
        )
        admission = verify_manual_response_bundle(
            request=request, bundle_bytes=response_bundle, admitted_at=_now(), actor=actor
        )
        from astrograph.upstream.contracts import canonical_sha256
        identity = canonical_sha256({
            "request_sha256": request["request_sha256"],
            "runtime_sha256": admission.admission_receipt["runtime_sha256"],
            "mapping_sha256": admission.admission_receipt["mapping_sha256"],
            "provider_id": "manual-import",
        })
        claims = self._upstream_recompute_root(job_id) / "claims"
        claims.mkdir(parents=True, exist_ok=True)
        claim_path = claims / f"{identity}.json"
        if claim_path.exists():
            existing = json.loads(claim_path.read_text(encoding="utf-8"))
            child_id = str(existing.get("child_job_id") or "")
            if child_id:
                try:
                    child = self.get(child_id)
                except KeyError:
                    child = None
                if child is not None and child.get("status") not in {"FAILED", "CANCELLED"}:
                    raise ValueError(f"EQUIVALENT_MANUAL_UPSTREAM_RERUN_EXISTS:{child_id}:{child.get('status')}")
            claim_path.unlink(missing_ok=True)
        fd = None
        try:
            fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, json.dumps({
                "schema_version": "astrograph-upstream-recompute-claim.v1",
                "status": "CREATION_IN_PROGRESS",
                "source_job_id": job_id,
                "identity_sha256": identity,
                "request_id": request["request_id"],
                "created_at": _now(),
            }, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        except FileExistsError as exc:
            raise ValueError("EQUIVALENT_MANUAL_UPSTREAM_RERUN_CREATION_IN_PROGRESS") from exc
        finally:
            if fd is not None:
                os.close(fd)
        child_id = ""
        try:
            child = self.create(
                f"上游重算子任务：{request['request_id']}（manual provider）",
                source_job.get("work_mode", "normal"),
            )
            child_id = str(child["job_id"])
            self.put_file(child_id, "runtime.json", admission.runtime_bytes)
            self.put_file(child_id, "mapping.json", admission.mapping_bytes)
            linked = self.get(child_id)
            linked["parent_job_id"] = job_id
            linked["upstream_recompute_provenance"] = {
                "schema_version": "astrograph-upstream-recompute-job-link.v1",
                "provider_id": "manual-import",
                "source_job_id": job_id,
                "request_id": request["request_id"],
                "request_sha256": request["request_sha256"],
                "admission_receipt_sha256": identity,
                "source_raw_input_receipt_sha256": admission.admission_receipt.get("source_raw_input_receipt_sha256"),
                "raw_telemetry_recompute_verified": False,
                "formal_release_eligible": False,
            }
            self._save(linked)
            preflight = self.preflight(child_id)
            if not preflight.get("can_start") or preflight.get("artifact_kind") != "PIPELINE_RUNTIME":
                raise ValueError("manual upstream response child failed PIPELINE_RUNTIME preflight")
            prepared = build_manual_child_prepared_receipt(
                source_job_id=job_id,
                child_job_id=child_id,
                actor=actor,
                created_at=_now(),
                request=request,
                admission_receipt=admission.admission_receipt,
                source_report_sha256=canonical_sha256(self.report(job_id)),
                child_preflight=preflight,
            )
            root = self._upstream_recompute_root(job_id)
            _json_write(root / "admissions" / f"{identity}.json", {
                "schema_version": "astrograph-upstream-recompute-admission-artifact.v1",
                "request": request,
                "response": admission.response,
                "receipt": admission.admission_receipt,
            })
            _json_write(root / "reruns" / f"{child_id}.prepared.json", prepared)
            _json_write(self._job_root(child_id) / "output" / "upstream_recompute_prepared_receipt.json", prepared)
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim.update({"status": "CHILD_CREATED", "child_job_id": child_id, "prepared_receipt_sha256": prepared["receipt_sha256"]})
            _json_write(claim_path, claim)
            linked = self.get(child_id)
            linked["upstream_recompute_provenance"] = {
                **(linked.get("upstream_recompute_provenance") or {}),
                "prepared_receipt_sha256": prepared["receipt_sha256"],
            }
            self._save(linked)
            self.state_repository.add_annotation(
                job_id=job_id,
                annotation_type="UPSTREAM_RECOMPUTE_MANUAL_CHILD_CREATED",
                payload={
                    "child_job_id": child_id,
                    "request_id": request["request_id"],
                    "admission_receipt_sha256": identity,
                    "prepared_receipt_sha256": prepared["receipt_sha256"],
                    "raw_telemetry_recompute_verified": False,
                    "formal_release_eligible": False,
                },
                actor=actor,
            )
            child_state = self.start(child_id) if start else self.get(child_id)
            return {
                "schema_version": "astrograph-upstream-recompute-child-created.v1",
                "source_job_id": job_id,
                "child_job_id": child_id,
                "started": bool(start),
                "provider_id": "manual-import",
                "child_job": child_state,
                "preflight": preflight,
                "admission_receipt": admission.admission_receipt,
                "prepared_receipt": prepared,
                "claim_boundary": prepared["claim_boundary"],
            }
        except Exception:
            if child_id:
                try:
                    failed = self.get(child_id)
                except KeyError:
                    failed = None
                if failed is not None and failed.get("status") in {"CREATED", "UPLOADING"}:
                    failed["status"] = "FAILED"
                    failed["stage"] = "FAILED"
                    failed["stage_detail"] = "手动上游重算子任务准备失败"
                    self._save(failed)
            claim_path.unlink(missing_ok=True)
            raise

    def _finalize_manual_upstream_recompute_child(
        self, job_id: str, *, report: dict[str, Any], artifact_kind: str, paths: dict[str, Path]
    ) -> dict[str, Any] | None:
        job = self.get(job_id)
        link = job.get("upstream_recompute_provenance") or {}
        if str(link.get("provider_id") or "") != "manual-import":
            return None
        if artifact_kind != "PIPELINE_RUNTIME":
            raise ValueError("manual upstream recompute child lost its PIPELINE_RUNTIME executable input")
        prepared_path = self._job_root(job_id) / "output" / "upstream_recompute_prepared_receipt.json"
        if not prepared_path.is_file():
            raise ValueError("manual upstream recompute child is missing prepared receipt")
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        source_job_id = str(prepared.get("source_job_id") or "")
        source_report = self.report(source_job_id)
        source_expanded = self._safe_expand(source_job_id)
        source_kind, source_paths = self._classify(source_expanded)
        if source_kind != "PIPELINE_RUNTIME":
            raise ValueError("manual upstream recompute source is no longer PIPELINE_RUNTIME")
        from astrograph.upstream.contracts import canonical_sha256
        from astrograph.upstream.manual_provider import build_manual_child_completion_receipt
        completion = build_manual_child_completion_receipt(
            prepared=prepared,
            completed_at=str(report.get("completed_at") or _now()),
            current_source_report_sha256=canonical_sha256(source_report),
            current_source_runtime_sha256=_sha256_path(source_paths["runtime"]),
            current_source_mapping_sha256=_sha256_path(source_paths["mapping"]),
            actual_child_runtime_sha256=_sha256_path(paths["runtime"]),
            actual_child_mapping_sha256=_sha256_path(paths["mapping"]),
            child_report=report,
        )
        _json_write(self._job_root(job_id) / "output" / "upstream_recompute_completion_receipt.json", completion)
        _json_write(self._upstream_recompute_root(source_job_id) / "reruns" / f"{job_id}.completion.json", completion)
        linked = self.get(job_id)
        linked["upstream_recompute_provenance"] = {
            **(linked.get("upstream_recompute_provenance") or {}),
            "completion_receipt_sha256": completion["receipt_sha256"],
            "completion_status": completion["status"],
        }
        self._save(linked)
        self.state_repository.add_annotation(
            job_id=source_job_id,
            annotation_type="UPSTREAM_RECOMPUTE_MANUAL_CHILD_COMPLETED",
            payload={
                "child_job_id": job_id,
                "request_id": completion["request_id"],
                "completion_receipt_sha256": completion["receipt_sha256"],
                "raw_telemetry_recompute_verified": False,
                "formal_release_eligible": False,
            },
            actor="system",
        )
        return completion

    def document_path(self, job_id: str, filename: str) -> Path:
        self.get(job_id)
        if not _SAFE_FILENAME.fullmatch(filename) or Path(filename).name != filename:
            raise ValueError("invalid document filename")
        if Path(filename).suffix.lower() not in _KNOWLEDGE_SUFFIXES:
            raise ValueError("requested file is not a knowledge document")
        expanded = (self._job_root(job_id) / "expanded").resolve()
        if not expanded.is_dir():
            raise ValueError("job inputs have not been expanded")
        matches = [p.resolve() for p in expanded.rglob(filename) if p.is_file() and p.suffix.lower() in _KNOWLEDGE_SUFFIXES]
        matches = [p for p in matches if expanded in p.parents]
        if not matches:
            raise FileNotFoundError(filename)
        if len(matches) != 1:
            raise ValueError("ambiguous document filename")
        return matches[0]

    @staticmethod
    def _overview_report_summary(report: dict[str, Any]) -> dict[str, Any]:
        analysis = report.get("analysis") or {}
        candidate = None
        disposition = report.get("diagnosis_disposition") or report.get("status")
        requires_human_review = False
        if report.get("artifact_kind") == "PUBLIC_CASE":
            closure = analysis.get("closure") or {}
            diag = closure.get("diagnosis") or {}
            candidates = diag.get("classification_candidates") or []
            candidate = candidates[0] if candidates else None
            disposition = diag.get("status") or disposition
            requires_human_review = bool(diag.get("requires_human_review"))
        elif report.get("artifact_kind") == "PIPELINE_RUNTIME":
            result = analysis.get("pipeline_result") or {}
            candidate = (result.get("fusion") or {}).get("top_fault_id")
            disposition = result.get("disposition") or disposition
            requires_human_review = bool(result.get("requires_human_review"))
        elif report.get("artifact_kind") == "OAG_DIAGNOSTIC":
            output = (analysis.get("oag_result") or {}).get("output") or {}
            summary = output.get("summary") or {}
            candidate = summary.get("suspected_fault_id")
            disposition = summary.get("status") or disposition
            requires_human_review = bool(summary.get("requires_human_review"))
        elif report.get("artifact_kind") == "RESULT_LOG_REVIEW":
            # External diagnosis + logs + references are provisional analyses,
            # not final root-cause diagnoses. Expose ABSTAIN to the event queue.
            block = analysis.get("result_log_review") or {}
            cases = block.get("candidates") or []
            first = cases[0] if cases and isinstance(cases[0], dict) else {}
            candidate = first.get("fault_id")
            policy = (analysis.get("material_reasoning") or {}).get("answer_policy") or {}
            disposition = ("ABSTAIN" if policy.get("all_candidates_abstained") is not False
                           else "REVIEW_REQUIRED")
            requires_human_review = True
        normalized = str(disposition or "UNKNOWN").upper()
        if any(token in normalized for token in ("REVIEW", "AMBIGUOUS", "INSUFFICIENT")):
            requires_human_review = True
        publishable = bool(report.get("formal_diagnosis_publishable"))
        return {
            "fault_name": (first.get("fault_name") if report.get("artifact_kind") == "RESULT_LOG_REVIEW" else None),
            "candidate": candidate,
            "diagnosis_disposition": disposition,
            "disposition": disposition,  # v2 compatibility alias
            "release_status": ("FORMAL_RELEASE_ALLOWED" if publishable else "FORMAL_RELEASE_BLOCKED"),
            "requires_human_review": requires_human_review,
        }

    def overview(self, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        jobs: list[dict[str, Any]] = []
        for root in sorted(self.root.glob("JOB-*")):
            meta = root / "job.json"
            if not meta.is_file():
                continue
            try:
                job = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            jobs.append(job)
        jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        jobs = jobs[:limit]
        status_counts: dict[str, int] = {}
        mode_counts = {"normal": 0, "thinking": 0, "creative": 0}
        artifact_counts: dict[str, int] = {}
        disposition_counts: dict[str, int] = {}
        candidate_counts: dict[str, int] = {}
        daily_counts: dict[str, int] = {}
        recent: list[dict[str, Any]] = []
        latest_completed_diagnosis = None
        review_count = 0
        release_counts: dict[str, int] = {}
        # The review store is optional; reading the queue must never approve
        # provisional material reasoning or mutate existing report contents.
        review_store = None
        review_path = self.root / "material_knowledge" / "manual_report_reviews.sqlite3"
        if review_path.is_file():
            from astrograph.integrations.manual_report_review import ManualReportReviewStore
            review_store = ManualReportReviewStore(review_path)
        for job in jobs:
            status = str(job.get("status") or "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1
            artifact_kind = str(job.get("artifact_kind") or "UNCLASSIFIED")
            artifact_counts[artifact_kind] = artifact_counts.get(artifact_kind, 0) + 1
            mode = str(job.get("work_mode") or "normal")
            if mode in mode_counts:
                mode_counts[mode] += 1
            created = str(job.get("created_at") or "")
            day = created[:10] if len(created) >= 10 else "unknown"
            daily_counts[day] = daily_counts.get(day, 0) + 1
            summary = {
                "candidate": None,
                "diagnosis_disposition": "NOT_AVAILABLE",
                "disposition": "NOT_AVAILABLE",
                "release_status": "NOT_AVAILABLE",
                "requires_human_review": False,
            }
            if status == "SUCCEEDED" and job.get("report_json"):
                try:
                    report = json.loads((self._job_root(str(job["job_id"])) / str(job["report_json"])).read_text(encoding="utf-8"))
                    summary = self._overview_report_summary(report)
                    if artifact_kind == "RESULT_LOG_REVIEW":
                        review_state = "PENDING"
                        if review_store is not None:
                            try:
                                review_state = review_store.detail(str(job['job_id']), report)["status"]
                            except (OSError, ValueError, sqlite3.Error, KeyError):
                                pass
                        summary["manual_review_status"] = review_state
                        summary["requires_human_review"] = review_state in ("PENDING", "IN_PROGRESS", "REVIEWED_WITH_GAPS")
                    candidate_value = str(summary.get("candidate") or "").strip()
                    if candidate_value:
                        candidate_counts[candidate_value] = candidate_counts.get(candidate_value, 0) + 1
                    disposition_value = str(summary.get("diagnosis_disposition") or "UNKNOWN").strip() or "UNKNOWN"
                    disposition_counts[disposition_value] = disposition_counts.get(disposition_value, 0) + 1
                    release_value = str(summary.get("release_status") or "UNKNOWN")
                    release_counts[release_value] = release_counts.get(release_value, 0) + 1
                    if latest_completed_diagnosis is None:
                        latest_completed_diagnosis = {**summary, "job_id": job["job_id"], "artifact_kind": job.get("artifact_kind"), "work_mode": mode, "completed_at": job.get("completed_at")}
                    if artifact_kind == "RESULT_LOG_REVIEW":
                        if bool(summary.get("requires_human_review")):
                            review_count += 1
                    elif bool(summary.get("requires_human_review")) or any(
                        token in disposition_value.upper() for token in ("REVIEW", "UNKNOWN", "ABSTAIN", "BLOCK", "INSUFFICIENT", "AMBIGUOUS")
                    ):
                        review_count += 1
                except Exception:
                    pass
            recent.append({
                "job_id": job.get("job_id"), "status": status, "execution_status": status, "task": job.get("task"),
                "work_mode": mode, "artifact_kind": job.get("artifact_kind"),
                "created_at": job.get("created_at"), "completed_at": job.get("completed_at"),
                **summary,
            })
        trend = [{"date": day, "count": daily_counts[day]} for day in sorted(daily_counts)[-7:]]
        top_candidates = [
            {"candidate": candidate, "count": count}
            for candidate, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ]
        return {
            "schema_version": "astrograph-platform-overview.v3",
            "recent_window_count": len(jobs),
            "status_counts": status_counts,
            "active_job_count": status_counts.get("QUEUED", 0) + status_counts.get("RUNNING", 0),
            "work_mode_counts": mode_counts,
            "artifact_kind_counts": artifact_counts,
            "diagnosis_disposition_counts": disposition_counts,
            "release_status_counts": release_counts,
            "top_fault_candidates": top_candidates,
            "review_or_attention_count": review_count,
            "daily_job_counts": trend,
            "latest_completed_diagnosis": latest_completed_diagnosis,
            "latest_successful": latest_completed_diagnosis,  # v2 compatibility alias
            "recent_jobs": recent,
            "generated_at": _now(),
        }

    def report_markdown_path(self, job_id: str) -> Path:
        job = self.get(job_id)
        if job["status"] != "SUCCEEDED" or not job.get("report_markdown"):
            raise ValueError("job has no successful report")
        path = (self._job_root(job_id) / str(job["report_markdown"])).resolve()
        if self._job_root(job_id) not in path.parents:
            raise ValueError("invalid report path")
        return path


PLATFORM_UI_ASSET_VERSION = "v0.21.1"
_PLATFORM_UI_ASSETS = {
    "platform_ui.css": "text/css; charset=utf-8",
    "platform_ui.js": "application/javascript; charset=utf-8",
    "platform_core.js": "application/javascript; charset=utf-8",
    "platform_report.js": "application/javascript; charset=utf-8",
    "platform_system.js": "application/javascript; charset=utf-8",
    "platform_engine_settings.js": "application/javascript; charset=utf-8",
    "platform_agent.js": "application/javascript; charset=utf-8",
    "platform_dashboard.js": "application/javascript; charset=utf-8",
    "platform_job.js": "application/javascript; charset=utf-8",
    "platform_agent_live.js": "application/javascript; charset=utf-8",
    "platform_knowledge.js": "application/javascript; charset=utf-8",
    "platform_shell.js": "application/javascript; charset=utf-8",
    "platform_operations.js": "application/javascript; charset=utf-8",
    "platform_diagnostic_workspace.js": "application/javascript; charset=utf-8",
    "platform_graph_workbench.js": "application/javascript; charset=utf-8",
}


def platform_html() -> str:
    """Return the packaged maintenance-oriented upload/diagnosis UI shell."""
    from importlib.resources import files

    return files("astrograph").joinpath("resources/platform_ui.html").read_text(encoding="utf-8")


def platform_asset(asset_name: str) -> tuple[str, str]:
    """Return one allow-listed versioned UI asset and its content type."""
    from importlib.resources import files

    media_type = _PLATFORM_UI_ASSETS.get(asset_name)
    if media_type is None:
        raise ValueError(f"unknown platform UI asset: {asset_name}")
    content = files("astrograph").joinpath(f"resources/{asset_name}").read_text(encoding="utf-8")
    return content, media_type
