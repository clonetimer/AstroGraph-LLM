"""Local-only browser workbench for AstroGraph's external diagnosis LLM path.

This module is distributed with a minimal copy of the canonical AstroGraph
modules. It never imports or invokes the telemetry/diagnosis engine.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit
import urllib.error
import urllib.request
import webbrowser
import zipfile

from pydantic import ValidationError
from astrograph.integrations.reviewed_context import (
    ContextStore, ContextError, context_request, write_task_context, load_task_context,
)
from astrograph.integrations.stage_execution import (
    BoundedExecutor, StageRouter, AnalysisCancelled, for_stage, check_cancelled,
)

from astrograph.integrations.external_diagnosis_analysis import (
    MAX_SOURCE_CHARACTERS, _canonical, _digest, _validate_binding, load_external_diagnosis_input,
    run_external_diagnosis_analysis,
)
from astrograph.integrations.external_diagnosis_conversation import (
    ConversationConflict, ask_follow_up, get_conversation,
)
from astrograph.integrations.external_diagnosis_report import (
    imported_diagnosis_review, render_external_diagnosis_markdown,
)
from astrograph.integrations.long_report_analysis import (
    MAX_PORTABLE_SOURCE_CHARACTERS, DEFAULT_CHUNK_CHARACTERS, coalesce_sources, run_hierarchical_analysis,
)
ONE_SHOT_SOURCE_CHARACTERS = 60_000
from astrograph.integrations.http_worker import _NoRedirect, verified_tls_context
from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.integrations.execution_observability import (
    RunJournal, run_scope, execution_snapshot, recover_interrupted, read_artifact,
    retry_share_violation,
)
from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


VERSION = "2.8.0-dev.2"
BASE = Path(__file__).resolve().parent
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_BODY_BYTES = 6 * 1024 * 1024
MAX_BACKUP_BYTES = 64 * 1024 * 1024
ROLES = {"diagnosis_results": 8, "log_reports": 12, "references": 16}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv", ".log", ".pdf", ".docx"}
DEFAULTS = {"mode": "local", "base_url": "http://127.0.0.1:1234/v1", "model": "",
            "api_key": "", "max_tokens": 4096, "timeout_s": 180,
            "response_format_strategy": "auto"}


class WorkbenchError(Exception):
    def __init__(self, code: str, message: str = "操作未完成，请检查输入或连接设置。", status: int = 400):
        super().__init__(code)
        self.code, self.message, self.status = code, message, status


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
        retry_share_violation(lambda: temporary.replace(path))
    finally:
        temporary.unlink(missing_ok=True)


def _json_read(path: Path) -> dict:
    value = json.loads(retry_share_violation(lambda: path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError("object required")
    return value


def _loopback(hostname: str | None) -> bool:
    # Use explicit loopback identities, including their no-proxy entries.
    return hostname is not None and hostname.lower() in {"localhost", "127.0.0.1", "::1"}


def _bypass_loopback_proxies() -> None:
    # Child HTTP workers inherit this process-only setting. Local diagnosis text
    # must never be sent to an enterprise/environment HTTP proxy by accident.
    existing = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
    values = [x.strip() for x in existing.split(",") if x.strip()]
    for name in ("localhost", "127.0.0.1", "::1", "[::1]"):
        if name not in values:
            values.append(name)
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(values)


def parse_connection(value: Any, *, allow_empty_model: bool = False) -> dict:
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise WorkbenchError("INVALID_CONNECTION", "模型连接设置格式不正确。")
    settings = {**DEFAULTS, **value}
    if settings["mode"] not in {"local", "online"}:
        raise WorkbenchError("INVALID_CONNECTION", "请选择本地模型或在线 API。")
    for field in ("base_url", "model", "api_key", "response_format_strategy"):
        if not isinstance(settings[field], str):
            raise WorkbenchError("INVALID_CONNECTION", "模型连接设置格式不正确。")
    settings["base_url"] = settings["base_url"].strip().rstrip("/")
    settings["model"] = settings["model"].strip()
    if ((not settings["model"] and not allow_empty_model) or len(settings["model"]) > 300
            or len(settings["api_key"]) > 4096 or len(settings["base_url"]) > 2000
            or any(c in settings["api_key"] for c in "\r\n\x00")
            or any(ord(c) < 32 for c in settings["base_url"])):
        raise WorkbenchError("INVALID_CONNECTION", "请填写模型 ID，并检查地址和 API Key 格式。")
    try:
        url = urlsplit(settings["base_url"])
        _ = url.port
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment or "\\" in settings["base_url"]):
            raise ValueError("unsupported endpoint")
        if settings["mode"] == "local" and not _loopback(url.hostname):
            raise ValueError("local mode is loopback only")
        if settings["mode"] == "online" and url.scheme != "https":
            raise ValueError("online mode requires HTTPS")
    except ValueError as exc:
        raise WorkbenchError("INVALID_ENDPOINT", "本地模式请使用 localhost 或 127.0.0.1；在线 API 请使用 HTTPS，地址不要包含账号、密码、查询参数。") from exc
    if (type(settings["max_tokens"]) is not int or not 128 <= settings["max_tokens"] <= 16384
            or type(settings["timeout_s"]) not in {int, float}
            or not 10 <= settings["timeout_s"] <= 600
            or settings["response_format_strategy"] not in {"auto", "json_schema", "json_object", "plain_json"}):
        raise WorkbenchError("INVALID_CONNECTION", "最大输出应为 128–16384 tokens，超时应为 10–600 秒。")
    return settings


def parse_phase_connections(primary: dict, overrides: Any = None) -> dict:
    """Explicit overrides only; changed endpoints never inherit primary credentials."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - {"chunk", "synthesis", "follow_up"}:
        raise WorkbenchError("INVALID_PHASE_MODELS", "只支持片段、综合和追问三个阶段的显式配置。")
    result = {"analysis": dict(primary)}
    for stage in ("chunk", "synthesis", "follow_up"):
        change = overrides.get(stage, {})
        if not isinstance(change, dict) or set(change) - set(DEFAULTS):
            raise WorkbenchError("INVALID_PHASE_MODELS", "阶段配置格式不正确。")
        settings = {**primary, **change}
        # Same identity normalization as parse_connection; mode changes also clear credentials.
        endpoint = change.get("base_url", primary["base_url"])
        if (not isinstance(endpoint, str) or endpoint.strip().rstrip("/") != primary["base_url"]
                or change.get("mode", primary["mode"]) != primary["mode"]):
            settings["api_key"] = change.get("api_key", "")
        result[stage] = parse_connection(settings)
    return result


def _plan_fingerprint(phases: dict) -> str:
    # Execution parameters are frozen, but secrets are never hashed into metadata.
    return _digest({stage: {k: v for k, v in settings.items() if k != "api_key"}
                    for stage, settings in phases.items()})


def _stage_backends(phases: dict) -> StageRouter:
    return StageRouter({stage: _backend(settings) for stage, settings in phases.items()})


def _execution_fingerprint(settings: dict) -> str:
    identity = {"mode": settings["mode"], "base_url": settings["base_url"], "model": settings["model"]}
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def _backend(settings: dict) -> OpenAICompatibleStructuredBackend:
    url = urlsplit(settings["base_url"])
    provider = "lmstudio" if settings["mode"] == "local" and url.port == 1234 else "openai-compatible"
    return OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        provider=provider, base_url=settings["base_url"], model=settings["model"],
        api_key=settings["api_key"] or "local-no-key", max_tokens=settings["max_tokens"],
        timeout_s=settings["timeout_s"], response_format_strategy=settings["response_format_strategy"],
        tokenizer_mode="disabled", tokenizer_local_files_only=True, seed=None,
        temperature=0, repair_attempts=1, max_http_attempts=6,
    ))


def _safe_filename(name: Any) -> str:
    if (not isinstance(name, str) or not name or len(name) > 120 or name != name.strip()
            or name.endswith((".", " ")) or name in {".", ".."} or name.casefold() == "analysis_input.json"
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³])(?:\..*)?", name)):
        raise WorkbenchError("INVALID_FILENAME", "文件名包含路径、Windows 保留名称或不支持的字符。请重命名后导入。")
    if Path(name).suffix.lower() not in TEXT_SUFFIXES:
        raise WorkbenchError("UNSUPPORTED_FILE_TYPE", "支持 TXT、MD、JSON、CSV、LOG、PDF 和 DOCX。")
    return name


def _decode_file(item: Any) -> tuple[str, str, bytes, str, str]:
    if (not isinstance(item, dict) or set(item) - {"filename", "role", "content_base64", "encoding"}
            or item.get("role") not in ROLES):
        raise WorkbenchError("INVALID_FILE", "请为每份材料选择诊断结果、日志或参考资料。")
    name = _safe_filename(item.get("filename"))
    role = item["role"]
    if role == "references" and Path(name).suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
        raise WorkbenchError("REFERENCE_FORMAT", "参考资料请使用 TXT、Markdown、PDF 或 DOCX。")
    # Canonical manifest accepts .log only in the log role.
    if role == "diagnosis_results" and Path(name).suffix.lower() == ".log":
        raise WorkbenchError("DIAGNOSIS_FORMAT", "LOG 文件请选择日志角色，并至少提供一份诊断结果文件。")
    encoded = item.get("content_base64")
    if not isinstance(encoded, str) or len(encoded) > (MAX_TOTAL_BYTES * 4 // 3 + 8):
        raise WorkbenchError("INPUT_TOO_LARGE", "单次导入的原始文件总大小不能超过 4 MiB。", 413)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WorkbenchError("INVALID_FILE", "文件内容传输不完整，请重新选择文件。") from exc
    suffix = Path(name).suffix.lower()
    if suffix in {".pdf", ".docx"}:
        if len(raw) < 8 or (suffix == ".pdf" and not raw.startswith(b"%PDF-")) or (suffix == ".docx" and not raw.startswith(b"PK\x03\x04")):
            raise WorkbenchError("INVALID_DOCUMENT", "文件格式与扩展名不匹配，或文件已损坏。")
        return name, role, raw, None, "binary"
    encoding = item.get("encoding", "auto")
    if encoding not in {"auto", "utf-8", "gb18030"}:
        raise WorkbenchError("INVALID_ENCODING", "请选择 UTF-8/BOM 自动识别或 GB18030。")
    if encoding == "auto":
        if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
            encoding = "utf-32"
        elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        else:
            encoding = "utf-8-sig"
    elif encoding == "utf-8":
        encoding = "utf-8-sig"
    try:
        text = raw.decode(encoding)
    except UnicodeError as exc:
        raise WorkbenchError("ENCODING_ERROR", "文件无法按所选编码读取。旧版中文文本可尝试 GB18030，或另存为 UTF-8。") from exc
    if "\x00" in text or not text.strip():
        raise WorkbenchError("EMPTY_OR_BINARY_INPUT", "文件为空或包含二进制内容，请导入可读诊断文本。")
    return name, role, raw, text, encoding


DOCUMENT_ERRORS = {
    "DOCUMENT_NEEDS_OCR": "PDF 没有可提取的文字。请先 OCR 或导出为可读文本，再导入工作台。",
    "DOCUMENT_ENCRYPTED": "PDF 已加密。请先提供已解锁的副本，再导入。",
    "DOCUMENT_PARSE_TIMEOUT": "文档解析超过 30 秒，已停止解析。请拆分文档或导出为 TXT / DOCX 后重试。",
    "DOCUMENT_STRUCTURE_TOO_LARGE": "文档结构过于复杂，超过解析容量。请按章节拆分或导出为纯文本。",
    "INPUT_PARSE_FAILED": "材料解析失败。请检查文件是否损坏、JSON 格式或 CSV 行数；PDF 和 DOCX 需包含可读文字。",
}


def _parse_bundle_bounded(task_id: str, expanded: Path) -> dict:
    """Keep document parsers out of the HTTP process and bound their runtime."""
    try:
        process = run_managed_subprocess(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--parse-worker"],
            input=_canonical({"task_id": task_id, "expanded": str(expanded)}).encode("utf-8"),
            capture_output=True, timeout=30, check=False)
        if process.returncode or len(process.stdout) > 6 * 1024 * 1024:
            raise ValueError("parser worker failed")
        result = json.loads(process.stdout)
        if result.get("status") != "SUCCEEDED":
            code = result.get("code", "INPUT_PARSE_FAILED")
            if code not in DOCUMENT_ERRORS:
                code = "INPUT_PARSE_FAILED"
            raise WorkbenchError(code, DOCUMENT_ERRORS[code])
        return _validate_binding(task_id, result["bundle"])
    except subprocess.TimeoutExpired as exc:
        raise WorkbenchError("DOCUMENT_PARSE_TIMEOUT", DOCUMENT_ERRORS["DOCUMENT_PARSE_TIMEOUT"]) from exc
    except WorkbenchError:
        raise
    except Exception as exc:
        raise WorkbenchError("INPUT_PARSE_FAILED", DOCUMENT_ERRORS["INPUT_PARSE_FAILED"]) from exc


def _parse_worker() -> None:
    # Exercise the same bundled PDF engine on Linux tests and Windows delivery.
    os.environ["ASTROGRAPH_PDF_ENGINE"] = "pypdf"
    try:
        if os.name == "posix":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        request = json.loads(sys.stdin.buffer.read(16385))
        expanded = Path(request["expanded"])
        bundle = load_external_diagnosis_input(task_id=request["task_id"], expanded=expanded,
                                               manifest_path=expanded / "analysis_input.json")
        raw = _canonical({"status": "SUCCEEDED", "bundle": bundle}).encode("utf-8")
        if len(raw) > 6 * 1024 * 1024:
            raise ValueError("DOCUMENT_STRUCTURE_TOO_LARGE")
    except Exception as exc:
        detail = str(exc).lower()
        code = ("DOCUMENT_NEEDS_OCR" if "no extractable text" in detail or "no retrievable elements" in detail
                else "DOCUMENT_ENCRYPTED" if "encrypted pdf" in detail
                else "DOCUMENT_STRUCTURE_TOO_LARGE" if "document_structure_too_large" in detail
                else "INPUT_PARSE_FAILED")
        raw = _canonical({"status": "FAILED", "code": code}).encode("utf-8")
    sys.stdout.buffer.write(raw)


class Workbench:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir).resolve()
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._operation = threading.Lock()
        self.contexts = ContextStore(self.root / "reviewed_context")
        self._queue = BoundedExecutor(max_waiting=8)
        self._active_backends: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}
        _bypass_loopback_proxies()
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = _json_read(path)
                recover_interrupted(path.parent, job["job_id"])
                if job.get("status") in {"RUNNING", "QUEUED", "CANCEL_REQUESTED"}:
                    job.update(status="INTERRUPTED", reason_code="APPLICATION_RESTARTED")
                    _json_write(path, job)
            except (OSError, ValueError):
                continue

    @contextmanager
    def _idle_operation(self):
        if not self._operation.acquire(blocking=False):
            raise WorkbenchError("MODEL_BUSY", "已有操作正在进行。", 409)
        try:
            with self._queue.idle_guard():
                yield
        except ValueError as exc:
            if str(exc) == "BUSY":
                raise WorkbenchError("MODEL_BUSY", "分析或排队任务尚未完成。", 409) from exc
            raise
        finally:
            self._operation.release()

    def context_api(self, method, path, body=None):
        # Single-user loopback UI. Platform API uses its authenticated identity instead.
        try:
            if method == "GET":
                return context_request(self.contexts, method, path, actor="local-user")
            with self._idle_operation():
                return context_request(self.contexts, method, path, body, actor="local-user")
        except (ContextError, TypeError) as exc:
            raise WorkbenchError("CONTEXT_REQUEST_FAILED", "上下文请求无效、版本已变化或范围不匹配。", 409) from exc

    def _selected_context(self, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"workspace_id", "entries"}:
            raise WorkbenchError("INVALID_CONTEXT_SELECTION", "请选择工作区及已审阅条目。")
        try:
            return self.contexts.snapshot(value["workspace_id"], value["entries"])
        except (ValueError, TypeError) as exc:
            raise WorkbenchError("INVALID_CONTEXT_SELECTION", "所选上下文已改变或尚未审阅，请刷新后重新选择。", 409) from exc

    def _frozen_context(self, job):
        try:
            snapshot = load_task_context(self._job_root(job["job_id"]))
            if (snapshot["sha256"] if snapshot else None) != job.get("context_sha256"):
                raise ContextError("TASK_CONTEXT_CHANGED")
            return snapshot
        except (ContextError, OSError) as exc:
            raise WorkbenchError("TASK_CONTEXT_CHANGED", "任务上下文快照校验失败，未发起模型请求。", 409) from exc

    def _persist_queue_state(self, job_id, status):
        job = self.get(job_id)
        job["status"] = status
        if status in {"CANCELLED", "INTERRUPTED"}:
            job.update(resumable=True, reason_code="USER_CANCELLED" if status == "CANCELLED" else "APPLICATION_RESTARTED")
        _json_write(self._job_root(job_id) / "job.json", job)

    def cancel_job(self, job_id):
        self.get(job_id)
        try:
            self._queue.cancel(job_id)
        except ValueError as exc:
            raise WorkbenchError("JOB_NOT_CANCELLABLE", "任务已结束或报告已保存，不能再取消。", 409) from exc
        return self.get(job_id)

    def _schedule(self, job, bundle, phases, allow_queue):
        backend = _stage_backends(phases)
        def execute(token):
            self._run_job(self.get(job["job_id"]), bundle, backend)
        try:
            state = self._queue.submit(job["job_id"], execute, persist=self._persist_queue_state,
                                       allow_queue=allow_queue)
        except ValueError as exc:
            code = "QUEUE_FULL" if str(exc) == "QUEUE_FULL" else "MODEL_BUSY"
            raise WorkbenchError(code, "队列已满或有任务正在运行；可明确选择排队后重试。", 409) from exc
        # Keep historical thread diagnostics compatible with the shared worker.
        self._threads[job["job_id"]] = self._queue.thread
        return {**job, "status": state}

    def _job_root(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise WorkbenchError("JOB_NOT_FOUND", "找不到这项分析任务。", 404)
        root = self.jobs_dir / job_id
        if root.is_symlink():
            raise WorkbenchError("JOB_NOT_FOUND", "找不到这项分析任务。", 404)
        return root

    def get(self, job_id: str) -> dict:
        try:
            job = _json_read(self._job_root(job_id) / "job.json")
            if job.get("job_id") != job_id:
                raise ValueError("job mismatch")
            return job
        except (OSError, ValueError) as exc:
            raise WorkbenchError("JOB_NOT_FOUND", "找不到这项分析任务，或任务记录已损坏。", 404) from exc

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                jobs.append(self.get(path.parent.name))
            except WorkbenchError:
                continue
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)[:100]

    def create_backup(self) -> bytes:
        with self._idle_operation():
            return self._create_backup()

    def _create_backup(self) -> bytes:
        """Return a self-verifying archive of local job records and source files."""
        context_bytes = (_canonical(self.contexts.export_bundle()) + "\n").encode("utf-8")
        if len(context_bytes) > 32 * 1024 * 1024:
            raise WorkbenchError("BACKUP_TOO_LARGE", "上下文备份超过 32 MiB。", 413)
        files: list[tuple[str, bytes]] = [("context_backup.json", context_bytes)]
        for job_dir in sorted(self.jobs_dir.iterdir()):
            if not re.fullmatch(r"[a-f0-9]{32}", job_dir.name) or job_dir.is_symlink() or not job_dir.is_dir():
                continue
            for path in sorted(job_dir.rglob("*")):
                if path.is_symlink():
                    raise WorkbenchError("BACKUP_SYMLINK", "任务目录包含符号链接，无法安全备份。", 409)
                if path.is_file():
                    relative = path.relative_to(self.root).as_posix()
                    raw = path.read_bytes()
                    files.append((relative, raw))
        manifest = {"schema_version": "astrograph-portable-backup.v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "files": [{"path": name, "bytes": len(raw),
                               "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files]}
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("backup.json", _canonical(manifest) + "\n")
            for name, raw in files:
                archive.writestr(name, raw)
        data = output.getvalue()
        if len(data) > MAX_BACKUP_BYTES:
            raise WorkbenchError("BACKUP_TOO_LARGE", "备份超过 64 MiB，请分批备份任务。", 413)
        return data

    def restore_backup(self, raw: bytes) -> dict[str, Any]:
        with self._idle_operation():
            return self._restore_backup(raw)

    def _restore_backup(self, raw: bytes) -> dict[str, Any]:
        """Validate a complete backup before atomically adding non-conflicting jobs."""
        if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BACKUP_BYTES:
            raise WorkbenchError("BACKUP_INVALID", "备份文件为空或超过 64 MiB。", 413)
        staging = self.root / (".restore-" + secrets.token_hex(8))
        moved: list[Path] = []
        try:
            try:
                archive = zipfile.ZipFile(io.BytesIO(raw))
            except (zipfile.BadZipFile, OSError) as exc:
                raise WorkbenchError("BACKUP_INVALID", "备份不是有效 ZIP 文件。") from exc
            with archive:
                infos = archive.infolist()
                if len(infos) > 20000 or sum(item.file_size for item in infos) > 256 * 1024 * 1024:
                    raise WorkbenchError("BACKUP_TOO_LARGE", "备份展开后超过安全容量。", 413)
                names: set[str] = set()
                portable_names: set[str] = set()
                for item in infos:
                    name = item.filename
                    parts = name.split("/")
                    mode = (item.external_attr >> 16) & 0o170000
                    if (name.casefold() in portable_names or name.startswith(("/", "\\")) or ".." in parts
                            or "\\" in name or ":" in name or mode == 0o120000
                            or any(not part or part in {".", ".."} or part.endswith((".", " "))
                                   or re.search(r'[<>:"|?*\x00-\x1f]', part)
                                   or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³])(?:\..*)?", part)
                                   for part in parts)):
                        raise WorkbenchError("BACKUP_UNSAFE_ARCHIVE", "备份包含重复、越界或链接路径。")
                    names.add(name)
                    portable_names.add(name.casefold())
                if "backup.json" not in names:
                    raise WorkbenchError("BACKUP_INVALID", "备份缺少校验清单。")
                try:
                    manifest = json.loads(archive.read("backup.json").decode("utf-8"))
                except (ValueError, UnicodeError, KeyError) as exc:
                    raise WorkbenchError("BACKUP_INVALID", "备份校验清单无法读取。") from exc
                if not isinstance(manifest, dict) or manifest.get("schema_version") != "astrograph-portable-backup.v1" or not isinstance(manifest.get("files"), list):
                    raise WorkbenchError("BACKUP_INVALID", "备份版本或清单格式不受支持。")
                listed = {}
                for item in manifest["files"]:
                    if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                            or (item["path"] != "context_backup.json" and not re.fullmatch(r"jobs/[a-f0-9]{32}/.+", item["path"]))
                            or item["path"] in listed or type(item.get("bytes")) is not int
                            or item["bytes"] < 0
                            or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", "")))):
                        raise WorkbenchError("BACKUP_INVALID", "备份文件清单存在无效条目。")
                    listed[item["path"]] = item
                if names - {"backup.json"} != set(listed):
                    raise WorkbenchError("BACKUP_INVALID", "备份清单与 ZIP 文件不一致。")
                staging.mkdir()
                job_ids: set[str] = set()
                for name, item in listed.items():
                    content = archive.read(name)
                    if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                        raise WorkbenchError("BACKUP_HASH_MISMATCH", "备份文件完整性校验失败。")
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    if name != "context_backup.json":
                        job_ids.add(name.split("/")[1])
                for job_id in job_ids:
                    job_path = staging / "jobs" / job_id / "job.json"
                    if not job_path.is_file() or _json_read(job_path).get("job_id") != job_id:
                        raise WorkbenchError("BACKUP_INVALID", "备份任务记录与目录编号不匹配。")
                    restored_job = _json_read(job_path)
                    if restored_job.get("status") in {"RUNNING", "QUEUED", "CANCEL_REQUESTED"}:
                        restored_job.update(status="INTERRUPTED", reason_code="RESTORED_INCOMPLETE_TASK")
                        _json_write(job_path, restored_job)
                    recover_interrupted(staging / "jobs" / job_id, job_id)
                    if self._job_root(job_id).exists():
                        raise WorkbenchError("BACKUP_CONFLICT", "目标数据目录已有同编号任务；未覆盖任何现有数据。", 409)
                # Publish filesystem jobs inside the context-store transaction. On any
                # later error the DB rolls back and the outer handler removes moved jobs.
                def publish():
                    for job_id in sorted(job_ids):
                        destination = self._job_root(job_id)
                        os.replace(staging / "jobs" / job_id, destination)
                        moved.append(destination)
                context_path = staging / "context_backup.json"
                if context_path.exists():
                    self.contexts.import_bundle(_json_read(context_path), before_commit=publish)
                else:
                    publish()
                return {"restored_jobs": len(moved), "job_ids": sorted(job_ids)}
        except WorkbenchError:
            for path in reversed(moved):
                shutil.rmtree(path, ignore_errors=True)
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            for path in reversed(moved):
                shutil.rmtree(path, ignore_errors=True)
            raise WorkbenchError("BACKUP_RESTORE_FAILED", "恢复失败；本次导入未保留部分任务。", 409) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def read_job(self, job_id: str) -> dict:
        job = self.get(job_id)
        report = None
        if job["status"] == "SUCCEEDED":
            try:
                report = _json_read(self._job_root(job_id) / "output" / "report.json")
            except (OSError, ValueError) as exc:
                raise WorkbenchError("REPORT_UNAVAILABLE", "报告文件无法读取，请检查 data 文件夹。", 409) from exc
        return {"job": job, "report": report, "execution": self.execution(job_id)}

    def execution(self, job_id: str) -> dict:
        self.get(job_id)
        try:
            return execution_snapshot(self._job_root(job_id), job_id)
        except (ValueError, OSError, TypeError) as exc:
            raise WorkbenchError("EXECUTION_UNAVAILABLE", "执行记录无法读取，请检查任务文件。", 409) from exc

    def export_run(self, job_id: str, run_id: str, name: str) -> tuple[bytes, str]:
        self.get(job_id)
        try:
            raw = read_artifact(self._job_root(job_id), job_id, run_id, name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WorkbenchError("EXECUTION_ARTIFACT_UNAVAILABLE", "该版本文件不存在、未完成或校验未通过。", 404) from exc
        return raw, "application/json; charset=utf-8" if name.endswith(".json") else "text/markdown; charset=utf-8"

    def create_job(self, payload: Any) -> dict:
        if (not isinstance(payload, dict) or not {"task", "files", "connection"}.issubset(payload)
                or set(payload) - {"task", "files", "connection", "context", "phases", "queue"}):
            raise WorkbenchError("INVALID_REQUEST", "分析请求不完整。")
        task = payload["task"]
        if not isinstance(task, str) or not task.strip() or len(task) > 4000:
            raise WorkbenchError("INVALID_TASK", "请填写不超过 4,000 字符的分析目标。")
        settings = parse_connection(payload["connection"])
        phases = parse_phase_connections(settings, payload.get("phases"))
        if type(payload.get("queue", False)) is not bool:
            raise WorkbenchError("INVALID_REQUEST", "排队选项必须为布尔值。")
        files = payload["files"]
        if not isinstance(files, list) or not 1 <= len(files) <= 36:
            raise WorkbenchError("INVALID_FILES", "请提供至少一份诊断结果，材料总数不超过 36 份。")
        decoded = [_decode_file(item) for item in files]
        counts = {role: sum(item[1] == role for item in decoded) for role in ROLES}
        if not counts["diagnosis_results"] or any(counts[role] > limit for role, limit in ROLES.items()):
            raise WorkbenchError("ROLE_LIMIT", "请提供 1–8 份诊断结果；日志最多 12 份，参考资料最多 16 份。")
        if len({item[0].casefold() for item in decoded}) != len(decoded):
            raise WorkbenchError("DUPLICATE_FILENAME", "材料文件名重复，请重命名后导入。")
        if sum(len(item[2]) for item in decoded) > MAX_TOTAL_BYTES:
            raise WorkbenchError("INPUT_TOO_LARGE", "单次导入的原始文件总大小不能超过 4 MiB。", 413)
        if not self._operation.acquire(blocking=False):
            raise WorkbenchError("MODEL_BUSY", "已有模型操作进行中，请等待完成。", 409)
        job_id = secrets.token_hex(16)
        root = self._job_root(job_id)
        try:
            if self._queue.busy() and not payload.get("queue", False):
                raise WorkbenchError("MODEL_BUSY", "已有分析进行中，请明确勾选排队。", 409)
            snapshot = self._selected_context(payload.get("context"))
            expanded = root / "input"
            original = root / "input_original"
            expanded.mkdir(parents=True)
            original.mkdir()
            manifest = {"schema_version": "astrograph-result-log-review.v1",
                        "analysis_mode": "EXTERNAL_DIAGNOSIS_ANALYSIS",
                        **{role: [] for role in ROLES}}
            receipts = []
            for name, role, raw, text, encoding in decoded:
                (original / name).write_bytes(raw)
                normalized = raw if text is None else text.encode("utf-8")
                (expanded / name).write_bytes(normalized)
                manifest[role].append(name)
                receipts.append({"filename": name, "role": role, "encoding": encoding,
                                 "original_sha256": hashlib.sha256(raw).hexdigest(),
                                 "imported_utf8_sha256": hashlib.sha256(normalized).hexdigest()})
            _json_write(expanded / "analysis_input.json", manifest)
            try:
                if any(Path(name).suffix.lower() in {".pdf", ".docx"} for name, *_ in decoded):
                    bundle = _parse_bundle_bounded(job_id, expanded)
                else:
                    bundle = load_external_diagnosis_input(task_id=job_id, expanded=expanded,
                                                          manifest_path=expanded / "analysis_input.json")
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError("INPUT_PARSE_FAILED", DOCUMENT_ERRORS["INPUT_PARSE_FAILED"]) from exc
            if bundle["character_count"] > MAX_PORTABLE_SOURCE_CHARACTERS:
                raise WorkbenchError("INPUT_TOO_LARGE", "解析后的材料超过 180,000 字符，请拆分材料；系统不会截断原文。", 413)
            if bundle["character_count"] > ONE_SHOT_SOURCE_CHARACTERS:
                bundle["sources"] = coalesce_sources(job_id, bundle["sources"])
                bundle["input_sha256"] = _digest(bundle["sources"])
                bundle["character_count"] = sum(len(source["text"]) for source in bundle["sources"])
            output = root / "output"
            if snapshot is not None:
                write_task_context(root, snapshot)
            _json_write(output / "external_diagnosis_input.json", bundle)
            _json_write(root / "import_receipts.json", {"files": receipts})
            job = {"job_id": job_id, "task": task.strip(),
                   "created_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
                   "model": settings["model"], "mode": settings["mode"],
                   "input_sha256": bundle["input_sha256"], "character_count": bundle["character_count"],
                   "execution_fingerprint": _plan_fingerprint(phases), "execution_plan_version": 1,
                   "context_sha256": snapshot["sha256"] if snapshot else None}
            _json_write(root / "job.json", job)
            return self._schedule(job, bundle, phases, payload.get("queue", False))
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        finally:
            self._operation.release()

    def _run_job(self, job: dict, bundle: dict, backend: Any) -> None:
        root = self._job_root(job["job_id"])
        run = None
        try:
            snapshot = self._frozen_context(job)
            receipts_path = root / "import_receipts.json"
            receipts = _json_read(receipts_path).get("files", []) if receipts_path.is_file() else []
            run = RunJournal(root, task_id=job["job_id"], kind="analysis", bundle=bundle,
                             model=job.get("model", ""), mode=job.get("mode", ""), receipts=receipts)
            job["active_run_id"] = run.run_id
            with self._queue.lock:
                check_cancelled()
                _json_write(root / "job.json", job)
            with run_scope(run):
                run.event("INPUT_VALIDATED")
                if bundle["character_count"] > ONE_SHOT_SOURCE_CHARACTERS:
                    def update_progress(value: dict) -> None:
                        with self._queue.lock:
                            check_cancelled()
                            job["progress"] = value
                            _json_write(root / "job.json", job)
                    knowledge = run_hierarchical_analysis(
                        task_id=job["job_id"], task=job["task"], input_bundle=bundle,
                        llm_backend=backend, output_root=root / "output" / "knowledge", progress=update_progress,
                        execution_fingerprint=job.get("execution_fingerprint", ""), context_snapshot=snapshot)
                else:
                    knowledge = run_external_diagnosis_analysis(
                        task_id=job["job_id"], task=job["task"], input_bundle=bundle,
                        llm_backend=for_stage(backend, "analysis"), output_root=root / "output" / "knowledge",
                        context_snapshot=snapshot)
                check_cancelled()
                if snapshot is not None:
                    knowledge["model_explanation"]["reviewed_context"] = snapshot
                _json_write(root / "output/knowledge/external_diagnosis_analysis.json", knowledge)
                review = imported_diagnosis_review(bundle)
                report = {"schema_version": "astrograph-portable-report.v1", "job_id": job["job_id"],
                          "task": job["task"], "input_sha256": bundle["input_sha256"],
                          "analysis": {"analysis_mode": "EXTERNAL_DIAGNOSIS_ANALYSIS",
                                       "imported_diagnosis_review": review, "knowledge_enrichment": knowledge},
                          "model": job["model"], "mode": job["mode"],
                          "created_at": job["created_at"], "automatic_control_authorized": False}
                if snapshot is not None:
                    report["reviewed_context"] = snapshot
                with self._queue.publication(job["job_id"]):
                    run.event("REPORT_WRITE")
                    _json_write(root / "output" / "report.json", report)
                    (root / "output" / "report.md").write_text(
                        render_external_diagnosis_markdown(job, review, knowledge), encoding="utf-8")
                    run.artifact("report.json", (root / "output" / "report.json").read_bytes())
                    run.artifact("report.md", (root / "output" / "report.md").read_bytes())
                    run.finish(knowledge["generation"]["status"], knowledge["generation"].get("reason_code"))
                    job.update(status="SUCCEEDED", generation_status=knowledge["generation"]["status"],
                               resumable=knowledge["generation"]["status"] == "FAILED",
                               reason_code=knowledge["generation"].get("reason_code"),
                               completed_at=datetime.now(timezone.utc).isoformat())
                    _json_write(root / "job.json", job)
        except AnalysisCancelled:
            if run is not None:
                run.finish("CANCELLED", "USER_CANCELLED")
            self._persist_queue_state(job["job_id"], "CANCELLED")
        except Exception:
            job.update(status="FAILED", reason_code="REPORT_WRITE_FAILED", resumable=True)
            try:
                _json_write(root / "job.json", job)
            except OSError:
                pass
        finally:
            self._threads.pop(job["job_id"], None)

    def resume_job(self, job_id: str, connection: Any, *, phases=None, queue=False) -> dict:
        settings = parse_connection(connection)
        plan = parse_phase_connections(settings, phases)
        if type(queue) is not bool:
            raise WorkbenchError("INVALID_REQUEST")
        if not self._operation.acquire(blocking=False):
            raise WorkbenchError("MODEL_BUSY", "已有操作进行中。", 409)
        try:
            job = self.get(job_id)
            if job.get("status") not in {"INTERRUPTED", "CANCELLED"} and not job.get("resumable", False):
                raise WorkbenchError("JOB_NOT_RESUMABLE", "只有中断、取消或模型分析失败的任务可以继续。", 409)
            fingerprint = _plan_fingerprint(plan) if job.get("execution_plan_version") == 1 else _execution_fingerprint(settings)
            if (not job.get("execution_fingerprint") or not secrets.compare_digest(job["execution_fingerprint"], fingerprint)
                    or (job.get("execution_plan_version") != 1 and phases)):
                raise WorkbenchError("RESUME_MODEL_MISMATCH", "继续任务必须使用原来的连接、阶段模型和执行参数。")
            root = self._job_root(job_id)
            self._frozen_context(job)
            try:
                bundle = _validate_binding(job_id, _json_read(root / "output/external_diagnosis_input.json"))
                if bundle["input_sha256"] != job.get("input_sha256"):
                    raise ValueError("hash differs")
            except Exception as exc:
                raise WorkbenchError("RESUME_INPUT_CHANGED", "原始输入完整性校验失败，不能继续。", 409) from exc
            return self._schedule(job, bundle, plan, queue)
        finally:
            self._operation.release()

    def _knowledge_service_for_job(self, job: dict) -> SimpleNamespace:
        return SimpleNamespace(llm_backend=self._active_backends.get(job["job_id"]))

    def conversation(self, job_id: str) -> dict:
        try:
            return get_conversation(self, job_id)
        except ConversationConflict as exc:
            raise WorkbenchError(str(exc), "当前报告尚不可追问，或原始材料/报告已改变，请重新分析。", 409) from exc

    def follow_up(self, job_id: str, payload: Any) -> dict:
        if (not isinstance(payload, dict)
                or not {"question", "input_sha256", "request_id", "connection"}.issubset(payload)
                or set(payload) - {"question", "input_sha256", "request_id", "connection", "phases"}):
            raise WorkbenchError("INVALID_REQUEST", "追问请求不完整。")
        settings = parse_connection(payload["connection"])
        plan = parse_phase_connections(settings, payload.get("phases"))
        self._frozen_context(self.get(job_id))
        if not self._operation.acquire(blocking=False):
            raise WorkbenchError("MODEL_BUSY", "已有模型操作进行中，请等待完成。", 409)
        try:
            with self._queue.lock:
                if self._queue.busy():
                    raise WorkbenchError("MODEL_BUSY", "分析或排队任务尚未结束。", 409)
            self._active_backends[job_id] = _backend(plan["follow_up"])
            previous = self.conversation(job_id)
            if any(turn.get("request_id") == payload["request_id"] for turn in previous["turns"]):
                return ask_follow_up(self, job_id, question=payload["question"],
                                     input_sha256=payload["input_sha256"], request_id=payload["request_id"])
            root = self._job_root(job_id)
            bundle = _validate_binding(job_id, _json_read(root / "output" / "external_diagnosis_input.json"))
            run = RunJournal(root, task_id=job_id, kind="follow_up", bundle=bundle,
                             model=plan["follow_up"]["model"], mode=plan["follow_up"]["mode"])
            with run_scope(run):
                run.event("FOLLOW_UP")
                state = ask_follow_up(self, job_id, question=payload["question"],
                                      input_sha256=payload["input_sha256"], request_id=payload["request_id"])
                turn = next(t for t in state["turns"] if t["request_id"] == payload["request_id"])
                run.artifact("conversation.json", (_canonical(state) + "\n").encode("utf-8"))
                run.finish(turn["status"], turn.get("reason_code"))
            return state
        except ConversationConflict as exc:
            raise WorkbenchError(str(exc), "无法继续这次追问：请检查原始报告、追问次数，或重新提交新任务。", 409) from exc
        except ValidationError as exc:
            raise WorkbenchError("INVALID_QUESTION", "追问须为 1–4,000 字符，且需绑定当前报告。") from exc
        finally:
            self._active_backends.pop(job_id, None)
            self._operation.release()

    def export(self, job_id: str, name: str) -> tuple[bytes, str]:
        if name == "conversation.json":
            return (_canonical(self.conversation(job_id)) + "\n").encode("utf-8"), "application/json; charset=utf-8"
        if name not in {"report.json", "report.md"} or self.get(job_id)["status"] != "SUCCEEDED":
            raise WorkbenchError("EXPORT_UNAVAILABLE", "报告尚未生成。", 404)
        try:
            raw = (self._job_root(job_id) / "output" / name).read_bytes()
        except OSError as exc:
            raise WorkbenchError("EXPORT_UNAVAILABLE", "报告文件无法读取。", 404) from exc
        return raw, "application/json; charset=utf-8" if name.endswith(".json") else "text/markdown; charset=utf-8"

    def probe(self, value: Any) -> dict:
        settings = parse_connection(value, allow_empty_model=True)
        if not self._operation.acquire(blocking=False):
            raise WorkbenchError("MODEL_BUSY", "已有模型操作进行中，请等待完成。", 409)
        try:
            process = run_managed_subprocess(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--probe-worker"],
                input=_canonical(settings).encode("utf-8"), capture_output=True,
                timeout=min(settings["timeout_s"], 20), check=False)
            if process.returncode or len(process.stdout) > 1_100_000:
                raise ValueError("probe worker failed")
            result = json.loads(process.stdout)
            if result.get("status") != "SUCCEEDED":
                code = result.get("code", "PROBE_FAILED")
                allowed = {"PROBE_FAILED", "AUTH_REJECTED", "MODELS_ENDPOINT_UNAVAILABLE", "INVALID_MODELS_RESPONSE"}
                if code not in allowed:
                    code = "PROBE_FAILED"
                raise WorkbenchError(code, "未能读取模型列表。请检查地址、服务是否启动和 API Key；有些兼容服务不提供 /models，可直接填写模型 ID 尝试分析。", 502)
            return result
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("PROBE_FAILED", "连接超时或服务未响应，请检查模型服务和地址。", 502) from exc
        finally:
            self._operation.release()


def _probe_worker() -> None:
    try:
        settings = parse_connection(json.loads(sys.stdin.buffer.read(16385)), allow_empty_model=True)
        base_url = settings["base_url"]
        if base_url.endswith("/chat/completions"):
            base_url = base_url[:-len("/chat/completions")]
        request = urllib.request.Request(base_url + "/models", headers={
            "Authorization": "Bearer " + (settings["api_key"] or "local-no-key"), "Accept": "application/json"})
        handlers: list[Any] = [_NoRedirect(), urllib.request.HTTPSHandler(context=verified_tls_context())]
        if settings["mode"] == "local":
            handlers.append(urllib.request.ProxyHandler({}))
        with urllib.request.build_opener(*handlers).open(request, timeout=min(settings["timeout_s"], 15)) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("response too large")
        value = json.loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get("data"), list):
            raise ValueError("invalid model list")
        models = [{"id": item["id"]} for item in value["data"][:200]
                  if isinstance(item, dict) and isinstance(item.get("id"), str) and 0 < len(item["id"]) <= 300]
        result = {"status": "SUCCEEDED", "models": models}
    except urllib.error.HTTPError as exc:
        result = {"status": "FAILED", "code": "AUTH_REJECTED" if exc.code in {401, 403}
                  else "MODELS_ENDPOINT_UNAVAILABLE" if exc.code in {404, 405} else "PROBE_FAILED"}
        exc.close()
    except Exception:
        result = {"status": "FAILED", "code": "PROBE_FAILED"}
    sys.stdout.buffer.write(_canonical(result).encode("utf-8"))


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        # No raw request/provider exception can be written to the console.
        pass


def make_server(store: Workbench, *, port: int = 8765) -> LocalServer:
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AstroGraphLocal/" + VERSION
        sys_version = ""

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass

        def _headers(self, status: int, content_type: str, size: int, attachment: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()

        def _send_json(self, value: Any, status: int = 200):
            raw = (_canonical(value) + "\n").encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(raw))
            self.wfile.write(raw)

        def _guard(self, *, mutation: bool = False):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host", "").lower() not in hosts:
                raise WorkbenchError("INVALID_HOST", "请通过本机启动窗口提供的网址打开界面。", 403)
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {"http://" + host for host in hosts}:
                raise WorkbenchError("INVALID_ORIGIN", "请求必须来自本机界面。", 403)
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise WorkbenchError("INVALID_ORIGIN", "请求必须来自本机界面。", 403)
            if mutation and not secrets.compare_digest(self.headers.get("X-AstroGraph-Token", ""), token):
                raise WorkbenchError("SESSION_EXPIRED", "会话已过期，请刷新页面后重试。", 403)

        def _body(self):
            if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/json":
                raise WorkbenchError("INVALID_CONTENT_TYPE", "请使用 JSON 请求。", 415)
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise WorkbenchError("INVALID_REQUEST") from exc
            if not 0 < size <= MAX_BODY_BYTES:
                raise WorkbenchError("INPUT_TOO_LARGE", "请求为空或超过大小限制。", 413)
            try:
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("incomplete body")
                return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            except (ValueError, UnicodeError) as exc:
                raise WorkbenchError("INVALID_JSON", "请求内容不是有效 JSON。") from exc

        def _route(self, *, mutation: bool = False):
            self._guard(mutation=mutation)
            path = urlsplit(self.path).path
            if mutation:
                if path == "/api/restore":
                    if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/zip":
                        raise WorkbenchError("INVALID_CONTENT_TYPE", "恢复文件必须是 ZIP 备份。", 415)
                    try:
                        size = int(self.headers.get("Content-Length", "-1"))
                    except ValueError as exc:
                        raise WorkbenchError("INVALID_REQUEST") from exc
                    if not 0 < size <= MAX_BACKUP_BYTES:
                        raise WorkbenchError("BACKUP_TOO_LARGE", "备份为空或超过 64 MiB。", 413)
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise WorkbenchError("BACKUP_INVALID", "备份上传不完整。")
                    return self._send_json(store.restore_backup(raw))
                body = self._body()
                if path == "/api/workspaces" or path.startswith("/api/workspaces/"):
                    return self._send_json(store.context_api("POST", path[len("/api/"):], body))
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", path)
                if match:
                    if body != {}:
                        raise WorkbenchError("INVALID_REQUEST")
                    return self._send_json({"job": store.cancel_job(match[1])})
                if path == "/api/jobs":
                    return self._send_json({"job": store.create_job(body)}, 202)
                if path == "/api/probe":
                    return self._send_json(store.probe(body))
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/follow-up", path)
                if match:
                    return self._send_json(store.follow_up(match[1], body))
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/resume", path)
                if match and isinstance(body, dict) and "connection" in body and not set(body) - {"connection", "phases", "queue"}:
                    return self._send_json({"job": store.resume_job(match[1], body["connection"],
                        phases=body.get("phases"), queue=body.get("queue", False))}, 202)
            else:
                if path == "/api/session":
                    sample = (BASE / "examples" / "diagnosis_sample.txt")
                    sample_text = sample.read_text("utf-8") if sample.exists() else "外部诊断结果：设备温度偏高，待人工核查。\n日志：温度告警持续 3 分钟。"
                    return self._send_json({"token": token, "version": VERSION, "defaults": DEFAULTS,
                                           "limits": {"max_files": 36, "max_total_bytes": MAX_TOTAL_BYTES,
                                                      "max_characters": MAX_SOURCE_CHARACTERS},
                                           "sample": {"filename": "diagnosis_sample.txt", "text": sample_text}})
                if path == "/api/health":
                    return self._send_json({"status": "OK", "version": VERSION, "offline_startup": True})
                if path == "/api/workspaces" or path.startswith("/api/workspaces/"):
                    return self._send_json(store.context_api("GET", path[len("/api/"):]))
                if path == "/api/jobs":
                    return self._send_json({"jobs": store.list_jobs()})
                if path == "/api/backup":
                    raw = store.create_backup()
                    self._headers(200, "application/zip", len(raw), "astrograph-portable-backup.zip")
                    self.wfile.write(raw)
                    return
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/runs/([a-f0-9]{32})/(report\.md|report\.json|conversation\.json)", path)
                if match:
                    job_id, run_id, name = match.groups()
                    raw, mime = store.export_run(job_id, run_id, name)
                    self._headers(200, mime, len(raw), f"astrograph-{run_id[:8]}-{name}")
                    return self.wfile.write(raw)
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/execution", path)
                if match:
                    return self._send_json(store.execution(match[1]))
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(conversation|report\.md|report\.json|conversation\.json))?", path)
                if match:
                    job_id, action = match.groups()
                    if action is None:
                        return self._send_json(store.read_job(job_id))
                    if action == "conversation":
                        return self._send_json(store.conversation(job_id))
                    raw, mime = store.export(job_id, action)
                    self._headers(200, mime, len(raw), f"astrograph-{job_id[:8]}-{action}")
                    return self.wfile.write(raw)
                assets = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/index.html": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/navigation.js": ("navigation.js", "text/javascript; charset=utf-8"),
                          "/execution_view.js": ("execution_view.js", "text/javascript; charset=utf-8"),
                          "/execution_view.css": ("execution_view.css", "text/css; charset=utf-8"),
                          "/workspace_view.js": ("workspace_view.js", "text/javascript; charset=utf-8"),
                          "/workspace_view.css": ("workspace_view.css", "text/css; charset=utf-8"),
                          "/app.css": ("app.css", "text/css; charset=utf-8")}
                if path in assets:
                    name, mime = assets[path]
                    raw = (BASE / "assets" / name).read_bytes()
                    self._headers(200, mime, len(raw))
                    return self.wfile.write(raw)
            raise WorkbenchError("NOT_FOUND", "没有这个页面或操作。", 404)

        def _handle(self, mutation=False):
            try:
                self._route(mutation=mutation)
            except WorkbenchError as exc:
                self._send_json({"error": {"code": exc.code, "message": exc.message}}, exc.status)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
            except Exception:
                self._send_json({"error": {"code": "LOCAL_OPERATION_FAILED", "message": "本机操作未完成，请检查文件夹权限和剩余磁盘空间。"}}, 500)

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle(mutation=True)

    return LocalServer(("127.0.0.1", port), Handler)


def self_test() -> dict:
    """Exercise real transport + canonical validation using a local scripted stub."""
    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request = json.loads(request["messages"][-1]["content"])
            sid = request["payload"]["sources"][0]["source_id"]
            if request["task"] == "ExternalDiagnosisFollowUp":
                output = {"answer": [{"text": "SELF_TEST_ONLY", "citation_ids": [sid]}], "information_gaps": []}
            else:
                output = {"summary": "SELF_TEST_ONLY", "findings": [{"text": "SELF_TEST_ONLY", "citation_ids": [sid]}],
                          "hypotheses": [], "information_gaps": [], "recommended_review_actions": []}
            raw = json.dumps({"choices": [{"message": {"content": json.dumps(output)}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="AstroGraph-check-") as directory:
            store = Workbench(Path(directory) / "中文 路径")
            settings = {**DEFAULTS, "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "model": "SELF_TEST_STUB", "response_format_strategy": "plain_json", "timeout_s": 15}
            job = store.create_job({"task": "离线运行环境自检", "connection": settings,
                                    "files": [{"filename": "诊断.txt", "role": "diagnosis_results", "encoding": "auto",
                                               "content_base64": base64.b64encode("外部诊断示例：温度偏高。".encode()).decode()}]})
            deadline = time.monotonic() + 20
            while store.get(job["job_id"])["status"] == "RUNNING" and time.monotonic() < deadline:
                time.sleep(0.05)
            report = store.read_job(job["job_id"])["report"]
            if not report or report["analysis"]["knowledge_enrichment"]["generation"]["status"] != "SUCCEEDED":
                raise RuntimeError("local HTTP worker or analysis self-test failed")
            state = store.follow_up(job["job_id"], {"question": "自检追问", "input_sha256": job["input_sha256"],
                                                   "request_id": "self-test-request-1", "connection": settings})
            if state["turns"][0]["status"] != "SUCCEEDED":
                raise RuntimeError("conversation self-test failed")
            if rb"SELF\_TEST\_ONLY" not in store.export(job["job_id"], "report.md")[0]:
                raise RuntimeError("report export self-test failed")
            forbidden = {"numpy", "pandas", "torch", "transformers", "langgraph", "astrograph.diagnosis"}
            if forbidden.intersection(sys.modules):
                raise RuntimeError("unexpected heavy module import")
            return {"status": "PASS", "python": sys.version.split()[0], "platform": sys.platform,
                    "bits": 64 if sys.maxsize > 2**32 else 32, "version": VERSION,
                    "checks": ["minimal_imports", "chinese_paths", "local_http_subprocess", "source_citations",
                               "conversation_lock_and_save", "markdown_export"],
                    "real_model_called": False, "internet_required": False}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="AstroGraph local external diagnosis LLM workbench")
    parser.add_argument("--data-dir", type=Path, default=BASE.parent / "data")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--probe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parse-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.parse_worker:
        _parse_worker()
        return 0
    if args.probe_worker:
        _probe_worker()
        return 0
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if args.self_test:
        try:
            print(json.dumps(self_test(), ensure_ascii=False, indent=2))
            print("PASS：离线环境自检通过；使用本机模拟接口，未调用真实大模型。")
            return 0
        except Exception:
            print("FAIL：离线环境自检未通过。请完整解压到可写文件夹，检查 Python/依赖是否缺失及本机回环连接是否被阻止。")
            return 1
    server = None
    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=args.data_dir) as writable:
            writable.write(b"write-check")
        if not 1 <= args.port <= 65535:
            raise ValueError("invalid port")
        # One portable folder is owned by one service process; keep its lock alive.
        from astrograph.integrations.external_diagnosis_conversation import _exclusive_call
        with _exclusive_call(args.data_dir / ".workbench.lock"):
            store = Workbench(args.data_dir)
            try:
                server = make_server(store, port=args.port)
            except OSError:
                # A different program may use the default port. Let the OS assign
                # an available loopback port and display/open that exact address.
                server = make_server(store, port=0)
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"AstroGraph LLM 本地工作台 v{VERSION}\n界面地址：{url}\n数据目录：{store.root}")
            print("请保持此窗口运行。按 Ctrl+C 停止服务。启动过程不会调用模型。", flush=True)
            if args.open_browser:
                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
            server.serve_forever(poll_interval=0.3)
    except ConversationConflict:
        print("同一数据目录已有服务运行，请使用原来的浏览器窗口，或先关闭原服务。")
        return 1
    except KeyboardInterrupt:
        print("\n服务已停止。未完成的模型任务不会自动重试。")
    except Exception:
        print("启动失败：请确认已完整解压、目录可写、磁盘空间足够，且未被安全软件隔离。可先运行 CHECK.bat。")
        return 1
    finally:
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
