"""Start test/validation commands detached and publish an atomic .done trigger.

Windows / Linux:
    python scripts/launch_logged_tests.py start --timeout 600 smoke -- python -m pytest -q tests/test_document_claim_fusion_adapter.py
    python scripts/launch_logged_tests.py status smoke

`start` returns immediately. `.done.json` is the completion trigger for CI,
file watchers, or a human reading the status; it is NOT a ChatGPT push alert.
The command is an argv vector (no shell / shell injection).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "validation_logs"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def job_dir(name: str) -> Path:
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("job name must contain only letters, digits, _ and -")
    return LOG_ROOT / name


def worker(name: str) -> int:
    target = job_dir(name)
    spec = json.loads((target / "job.json").read_text(encoding="utf-8"))
    started = stamp()
    atomic_json(target / "running.json", {"state": "RUNNING", "started_at": started, "pid": os.getpid()})
    return_code = 125
    error = None
    try:
        with (target / "output.log").open("wb") as log:
            try:
                proc = subprocess.run(
                    spec["argv"], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=spec["timeout_s"], check=False,
                )
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                return_code = 124
                error = "TIMEOUT"
            except Exception as exc:
                return_code = 125
                error = f"EXECUTION_ERROR:{type(exc).__name__}:{exc}"
    finally:
        atomic_json(target / "done.json", {
            "state": "PASSED" if return_code == 0 else "FAILED",
            "exit_code": return_code,
            "started_at": started,
            "completed_at": stamp(),
            "error": error,
            "log": str(target / "output.log"),
            "completion_trigger": str(target / "done.json"),
        })
        (target / "running.json").unlink(missing_ok=True)
    return return_code


def launch(name: str, timeout_s: int, argv: list[str]) -> None:
    if argv and argv[0] == "--":
        argv = argv[1:]
    if argv and argv[0].startswith("--"):
        raise ValueError("place runner options before job name: start --timeout N JOB -- COMMAND")
    if not argv:
        raise ValueError("supply a command after --")
    target = job_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "running.json").exists() or (target / "job.json").exists():
        raise ValueError("job already exists; choose a new job name to retain audit history")
    atomic_json(target / "job.json", {"argv": argv, "timeout_s": timeout_s, "submitted_at": stamp()})
    args = [sys.executable, str(Path(__file__).resolve()), "_worker", name]
    options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
               "stderr": subprocess.DEVNULL, "cwd": str(ROOT), "close_fds": True}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        options["start_new_session"] = True
    proc = subprocess.Popen(args, **options)
    print(json.dumps({"state": "SUBMITTED", "job": name, "pid": proc.pid,
                      "log": str(target / "output.log"),
                      "completion_trigger": str(target / "done.json")}, ensure_ascii=False))


def status(name: str) -> None:
    target = job_dir(name)
    for file in ("done.json", "running.json", "job.json"):
        path = target / file
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps(payload if file != "job.json" else {"state": "SUBMITTED", **payload}, ensure_ascii=False, indent=2))
            return
    raise ValueError(f"job not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start")
    start.add_argument("name")
    start.add_argument("--timeout", type=int, default=600)
    start.add_argument("command", nargs=argparse.REMAINDER)
    s = sub.add_parser("status"); s.add_argument("name")
    worker_parser = sub.add_parser("_worker"); worker_parser.add_argument("name")
    args = parser.parse_args()
    if args.action == "start":
        if args.timeout <= 0:
            parser.error("timeout must be positive")
        launch(args.name, args.timeout, args.command)
    elif args.action == "status":
        status(args.name)
    elif args.action == "_worker":
        raise SystemExit(worker(args.name))


if __name__ == "__main__":
    main()