from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import RuntimeStageArtifact
from .contracts import StageResult
from .execution_control import CancellationToken, RuntimeCancelledError, RuntimeDeadlineExceeded, RuntimeExecutionPolicy, StageExecutionProfile




def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def cleanup_stale_isolation_directories(*, temp_root: Path | None = None) -> list[str]:
    """Remove only isolation workdirs whose owning parent PID no longer exists."""
    root = (temp_root or Path(tempfile.gettempdir())).resolve()
    removed: list[str] = []
    prefix = "astrograph-isolated-stage-"
    for candidate in root.glob(f"{prefix}*-*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name[len(prefix):]
        pid_text = suffix.split("-", 1)[0]
        if not pid_text.isdigit():
            continue
        owner_pid = int(pid_text)
        if owner_pid == os.getpid() or _pid_is_alive(owner_pid):
            continue
        try:
            shutil.rmtree(candidate)
            removed.append(str(candidate))
        except OSError:
            continue
    return removed


class IsolatedStageInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-isolated-stage-invocation.v1"
    adapter_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    stage_version: str = Field(min_length=1)
    project_root: str
    execution_scope: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_policy: RuntimeExecutionPolicy
    stage_profile: StageExecutionProfile
    upstream_artifacts: dict[str, RuntimeStageArtifact] = Field(default_factory=dict)


class IsolatedStageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-isolated-stage-response.v1"
    worker_pid: int = Field(gt=0)
    result: StageResult


class IsolatedExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _terminate_process_tree(process: subprocess.Popen[Any], *, grace_s: float) -> str:
    """Terminate one isolated worker boundary and best-effort descendants.

    On POSIX the worker is started in a new session, so the process group can be killed.
    On Windows `terminate()` is a real TerminateProcess operation for the worker; the
    isolated worker protocol does not authorize arbitrary child-process providers unless
    they separately claim provider-enforced tree cleanup.
    """
    if process.poll() is not None:
        return "already-exited"
    mode = "terminate"
    try:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return "already-exited"
        else:
            process.terminate()
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=grace_s)
        return mode
    except subprocess.TimeoutExpired:
        mode = "kill"
    try:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return "terminate"
        else:
            process.kill()
    finally:
        try:
            process.wait(timeout=max(0.1, grace_s))
        except subprocess.TimeoutExpired:
            pass
    return mode


def execute_isolated_stage(
    invocation: IsolatedStageInvocation,
    *,
    cancellation: CancellationToken,
    termination_grace_s: float,
) -> tuple[StageResult, dict[str, Any]]:
    """Execute a reconstructable Stage in a killable Python worker process."""

    project_root = Path(invocation.project_root).resolve()
    if not project_root.is_dir():
        raise IsolatedExecutionError(
            "ASTROGRAPH_ISOLATED_PROJECT_ROOT_INVALID",
            f"isolated stage project_root is unavailable: {project_root}",
        )

    cleanup_stale_isolation_directories()
    with tempfile.TemporaryDirectory(
        prefix=f"astrograph-isolated-stage-{os.getpid()}-", ignore_cleanup_errors=True
    ) as temp:
        temp_root = Path(temp)
        request_path = temp_root / "invocation.json"
        response_path = temp_root / "response.json"
        stderr_path = temp_root / "stderr.log"
        request_path.write_text(invocation.model_dump_json(indent=None), encoding="utf-8")

        env = os.environ.copy()
        source_root = project_root / "src"
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(source_root) + (os.pathsep + existing if existing else "")
        command = [
            sys.executable,
            "-m",
            "astrograph.runtime.isolated_worker",
            str(request_path),
            str(response_path),
        ]
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        with stderr_path.open("wb") as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_stream,
                start_new_session=(os.name == "posix"),
                creationflags=creationflags,
            )

        started = time.monotonic()
        termination_mode = "natural"
        while process.poll() is None:
            try:
                cancellation.checkpoint()
            except RuntimeDeadlineExceeded as exc:
                termination_mode = _terminate_process_tree(process, grace_s=termination_grace_s)
                raise IsolatedExecutionError(
                    "ASTROGRAPH_RUNTIME_STAGE_TIMEOUT",
                    str(exc),
                    details={
                        "worker_pid": process.pid,
                        "termination_mode": termination_mode,
                        "execution_boundary": "ISOLATED_PROCESS",
                    },
                ) from exc
            except RuntimeCancelledError as exc:
                termination_mode = _terminate_process_tree(process, grace_s=termination_grace_s)
                raise IsolatedExecutionError(
                    "ASTROGRAPH_RUNTIME_STAGE_CANCELLED",
                    str(exc),
                    details={
                        "worker_pid": process.pid,
                        "termination_mode": termination_mode,
                        "execution_boundary": "ISOLATED_PROCESS",
                    },
                ) from exc
            time.sleep(0.005)

        elapsed = time.monotonic() - started
        stderr_text = ""
        try:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-8192:]
        except OSError:
            pass
        if process.returncode != 0:
            raise IsolatedExecutionError(
                "ASTROGRAPH_ISOLATED_WORKER_CRASHED",
                f"isolated stage worker exited with code {process.returncode}",
                details={
                    "worker_pid": process.pid,
                    "returncode": process.returncode,
                    "stderr_tail": stderr_text,
                },
            )
        if not response_path.is_file():
            raise IsolatedExecutionError(
                "ASTROGRAPH_ISOLATED_WORKER_NO_RESPONSE",
                "isolated stage worker exited without a response artifact",
                details={"worker_pid": process.pid, "stderr_tail": stderr_text},
            )
        try:
            response = IsolatedStageResponse.model_validate_json(response_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IsolatedExecutionError(
                "ASTROGRAPH_ISOLATED_WORKER_RESPONSE_INVALID",
                f"invalid isolated stage worker response: {type(exc).__name__}: {exc}",
                details={"worker_pid": process.pid, "stderr_tail": stderr_text},
            ) from exc
        return response.result, {
            "worker_pid": response.worker_pid,
            "worker_elapsed_ms": elapsed * 1000.0,
            "termination_mode": termination_mode,
            "execution_boundary": "ISOLATED_PROCESS",
            "adapter_id": invocation.adapter_id,
        }
