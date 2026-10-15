"""Schema-checked tool boundary with real killable workers for new plugins.

Existing closures remain explicitly TRUSTED_IN_PROCESS: an elapsed deadline can
be detected there but an arbitrary Python thread cannot be killed. Require hard
isolation on new plugin registries; this is process isolation, not a security sandbox.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, TYPE_CHECKING
from jsonschema import Draft202012Validator, FormatChecker
from astrograph.integrations.process_control import _terminate_process_tree
from astrograph.orchestration.models import WorkflowContext, PermissionScope
if TYPE_CHECKING:
    from .registry import ToolDefinition


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ToolDeadlineError(ToolExecutionError, TimeoutError):
    pass


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _check_schema(schema: dict) -> None:
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("$ref", "$dynamicRef") and (not isinstance(child, str) or not child.startswith("#")):
                    raise ValueError("tool schemas must not fetch external references")
                walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(schema)
    Draft202012Validator.check_schema(schema)


def validate_tool_definition(definition: "ToolDefinition") -> None:
    _check_schema(definition.spec.input_schema)
    _check_schema(definition.spec.output_schema)
    if definition.spec.execution_mode == "ISOLATED_PROCESS":
        target = definition.worker_target
        if target is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target.function_name):
            raise ValueError("isolated tool requires an explicit module function")
        if not re.fullmatch(r"[0-9a-f]{64}", target.artifact_sha256):
            raise ValueError("isolated tool requires artifact SHA-256")
    elif definition.handler is None:
        raise ValueError("in-process tool requires a handler")


def _validate(payload: dict, schema: dict, stage: str) -> None:
    if not isinstance(payload, dict):
        raise ToolExecutionError(f"TOOL_{stage}_NOT_OBJECT")
    if next(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload), None):
        raise ToolExecutionError(f"TOOL_{stage}_SCHEMA_REJECTED")


class ToolExecutor:
    def __init__(self, *, require_hard_timeout: bool = False,
                 deployment_permissions: frozenset[PermissionScope] | None = None):
        self.require_hard_timeout = require_hard_timeout
        self.deployment_permissions = deployment_permissions
        self._receipts: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def receipts(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._receipts)

    def invoke(self, definition: "ToolDefinition", payload: dict, context: WorkflowContext,
               *, cancel: threading.Event | None = None) -> dict:
        spec = definition.spec
        started = time.monotonic()
        receipt = {"tool_id": spec.tool_id, "tool_version": spec.tool_version,
                   "run_id": context.run_id, "execution_mode": spec.execution_mode,
                   "hard_timeout_enforced": spec.execution_mode == "ISOLATED_PROCESS",
                   "retry_policy": "NO_AUTOMATIC_RETRY", "status": "FAILED"}
        try:
            required = set(spec.required_permissions)
            if not required <= set(context.permissions):
                raise PermissionError("tool task permissions denied")
            if self.deployment_permissions is not None and not required <= self.deployment_permissions:
                raise PermissionError("tool deployment permissions denied")
            pins = context.metadata.get("tool_versions")
            if pins is not None and pins.get(spec.tool_id) != spec.tool_version:
                raise ToolExecutionError("TOOL_VERSION_PIN_MISMATCH")
            if self.require_hard_timeout and spec.execution_mode != "ISOLATED_PROCESS":
                raise ToolExecutionError("TOOL_HARD_TIMEOUT_NOT_SUPPORTED")
            if cancel is not None and cancel.is_set():
                raise ToolExecutionError("TOOL_CANCELLED")
            data = _json(payload)
            if len(data) > spec.max_payload_bytes:
                raise ToolExecutionError("TOOL_INPUT_TOO_LARGE")
            # JSON round trip gives the handler a task-owned copy.
            owned = json.loads(data)
            _validate(owned, spec.input_schema, "INPUT")
            receipt["input_sha256"] = hashlib.sha256(data).hexdigest()
            if spec.execution_mode == "ISOLATED_PROCESS":
                result = self._isolated(definition, owned, cancel)
            else:
                from astrograph.integrations.execution_budget import interruptible_timeout
                from astrograph.integrations.execution_budget import ExecutionDeadlineExceeded
                try:
                    with interruptible_timeout(spec.timeout_s, "TOOL_DEADLINE_EXCEEDED") as supported:
                        receipt["python_interruptible_timeout"] = supported
                        result = definition.handler(owned)
                except ExecutionDeadlineExceeded as exc:
                    raise ToolDeadlineError("TOOL_DEADLINE_EXCEEDED") from exc
            if time.monotonic() - started > spec.timeout_s:
                raise ToolExecutionError("TOOL_DEADLINE_EXCEEDED")
            encoded = _json(result)
            if len(encoded) > spec.max_output_bytes:
                raise ToolExecutionError("TOOL_OUTPUT_TOO_LARGE")
            _validate(result, spec.output_schema, "OUTPUT")
            receipt.update(status="SUCCEEDED", output_sha256=hashlib.sha256(encoded).hexdigest())
            return json.loads(encoded)
        except Exception as exc:
            receipt["error_code"] = getattr(exc, "code", type(exc).__name__)
            raise
        finally:
            receipt["elapsed_ms"] = (time.monotonic() - started) * 1000
            with self._lock:
                self._receipts.append(receipt)

    def _isolated(self, definition: "ToolDefinition", payload: dict, cancel) -> dict:
        target, spec = definition.worker_target, definition.spec
        path = Path(target.module_file).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != target.artifact_sha256:
            raise ToolExecutionError("TOOL_ARTIFACT_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="astrograph-plugin-") as directory:
            root = Path(directory)
            request, response = root/"request.json", root/"response.json"
            request.write_bytes(_json({"module_file": str(path), "function_name": target.function_name,
                                       "artifact_sha256": target.artifact_sha256, "payload": payload,
                                       "max_output_bytes": spec.max_output_bytes}))
            command = [sys.executable, "-m", "astrograph.tools.plugin_worker", str(request), str(response)]
            env = os.environ.copy()
            source = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
            # stdout/stderr are discarded: tool errors return only structured error codes.
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, env=env, start_new_session=os.name == "posix")
            deadline = time.monotonic() + spec.timeout_s
            try:
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        raise ToolExecutionError("TOOL_CANCELLED")
                    if time.monotonic() >= deadline:
                        raise ToolExecutionError("TOOL_DEADLINE_EXCEEDED")
                    if response.exists() and response.stat().st_size > spec.max_output_bytes + 1024:
                        raise ToolExecutionError("TOOL_OUTPUT_TOO_LARGE")
                    time.sleep(0.005)
                if process.returncode:
                    raise ToolExecutionError("TOOL_WORKER_FAILED")
                if not response.is_file() or response.stat().st_size > spec.max_output_bytes + 1024:
                    raise ToolExecutionError("TOOL_WORKER_INVALID_RESPONSE")
                envelope = json.loads(response.read_text(encoding="utf-8"))
                if envelope.get("status") != "SUCCEEDED":
                    raise ToolExecutionError(envelope.get("error_code", "TOOL_WORKER_FAILED"))
                return envelope["output"]
            finally:
                # Clean up descendants even on cancellation/errors. POSIX group cleanup
                # after successful exit is handled explicitly to avoid orphan children.
                if process.poll() is None:
                    _terminate_process_tree(process)
                if os.name == "posix":
                    import signal
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                process.wait()
