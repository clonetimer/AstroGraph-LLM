from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence


def _terminate_process_tree(process: subprocess.Popen[Any], *, grace_s: float = 0.25) -> str:
    """Best-effort tree termination for optional external CLI providers.

    POSIX uses a dedicated process group. Windows uses taskkill /T /F when available,
    then falls back to TerminateProcess. The Windows branch is implemented for portability
    but requires native-platform qualification before Stable parallel promotion.
    """
    if process.poll() is not None:
        return "already-exited"
    deadline = time.monotonic() + max(0.05, grace_s)
    mode = "terminate-tree"
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return "already-exited"
        except OSError:
            process.terminate()
    elif os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(0.05, deadline - time.monotonic()),
            )
        except Exception:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
        return mode
    except subprocess.TimeoutExpired:
        mode = "kill-tree"
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return "terminate-tree"
        except OSError:
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=0.1)
    except Exception:
        pass
    return mode


def run_managed_subprocess(
    args: Sequence[str | os.PathLike[str]],
    *,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    input: str | bytes | None = None,
    termination_grace_s: float = 0.25,
) -> subprocess.CompletedProcess[Any]:
    """`subprocess.run` compatible subset with process-tree timeout cleanup."""
    from .execution_budget import remaining_time
    timeout = remaining_time(timeout)
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        [str(item) for item in args],
        stdin=subprocess.PIPE if input is not None else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        start_new_session=(os.name == "posix"),
        creationflags=creationflags,
    )
    started = time.monotonic()
    try:
        out, err = process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process, grace_s=termination_grace_s)
        try:
            out, err = process.communicate(timeout=max(0.1, termination_grace_s))
        except Exception:
            out = exc.output
            err = exc.stderr
        raise subprocess.TimeoutExpired(
            cmd=[str(item) for item in args],
            timeout=timeout,
            output=out,
            stderr=err,
        ) from exc
    except BaseException:
        _terminate_process_tree(process, grace_s=termination_grace_s)
        process.wait()
        raise
    result = subprocess.CompletedProcess(
        args=[str(item) for item in args],
        returncode=process.returncode,
        stdout=out,
        stderr=err,
    )
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    _ = started  # retained to make future receipt timing additive without changing API
    return result
