"""Explicit dependency planning, conflict checks and auditable parallel execution.

This is a deterministic local execution backend, not a substitute labelled as
LangGraph. The optional LangGraph adapter consumes the same plan and handlers.
Handlers own their execution limits; killable tools use ToolExecutor.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath, Path
import re
import threading
import time
from typing import Any, Callable, Literal
from uuid import uuid4


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


class ChangeLedger:
    """Single-process, thread-safe, hash-chained JSONL execution ledger.

    The hash chain detects accidental edits; it is not an authenticated signature.
    A single coordinator owns this file. Use distinct files across processes.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        records = self.verify(self.path) if self.path.exists() else []
        self._previous = records[-1]["event_sha256"] if records else "0" * 64
        self._sequence = len(records)

    @staticmethod
    def verify(path: str | Path) -> list[dict[str, Any]]:
        records = []
        previous = "0" * 64
        for sequence, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            value = json.loads(line)
            digest = value.pop("event_sha256")
            if value.get("sequence") != sequence or value.get("previous_sha256") != previous:
                raise ValueError("ledger sequence/hash chain is broken")
            if hashlib.sha256(_canonical(value)).hexdigest() != digest:
                raise ValueError("ledger event checksum differs")
            value["event_sha256"] = digest
            records.append(value)
            previous = digest
        return records

    def append(self, *, run_id: str, task_id: str, event: str, **details: Any) -> dict[str, Any]:
        with self._lock:
            record = {"sequence": self._sequence + 1, "previous_sha256": self._previous,
                      "time_utc": datetime.now(timezone.utc).isoformat(),
                      "monotonic_ns": time.monotonic_ns(), "run_id": run_id,
                      "task_id": task_id, "event": event, "details": details}
            record["event_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(_canonical(record).decode("utf-8") + "\n")
                stream.flush()
            self._sequence += 1
            self._previous = record["event_sha256"]
            return record


@dataclass(frozen=True)
class TaskNode:
    task_id: str
    depends_on: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    allow_failed_dependencies: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.task_id):
            raise ValueError("invalid DAG task ID")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("duplicate dependency")
        for value in self.writes:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or value in ("", "."):
                raise ValueError("writes must be explicit relative file/directory paths")


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    status: Literal["SUCCEEDED", "FAILED", "BLOCKED"]
    output: Any = None
    error_type: str | None = None
    started_ns: int | None = None
    finished_ns: int | None = None


Handler = Callable[[dict[str, TaskOutcome]], Any]


@dataclass(frozen=True)
class DAGPlan:
    plan_id: str
    nodes: tuple[TaskNode, ...]

    def __post_init__(self) -> None:
        ids = [node.task_id for node in self.nodes]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("a DAG needs non-duplicate nodes")
        known = set(ids)
        for node in self.nodes:
            if not set(node.depends_on) <= known:
                raise ValueError(f"unknown dependency for {node.task_id}")
        ancestors = self.ancestors()  # also detects cycles
        for index, left in enumerate(self.nodes):
            for right in self.nodes[index + 1:]:
                ordered = left.task_id in ancestors[right.task_id] or right.task_id in ancestors[left.task_id]
                if not ordered and any(_overlap(a, b) for a in left.writes for b in right.writes):
                    raise ValueError(f"unordered tasks share write paths: {left.task_id}, {right.task_id}")

    def ancestors(self) -> dict[str, set[str]]:
        by_id = {node.task_id: node for node in self.nodes}
        resolved: dict[str, set[str]] = {}
        active: set[str] = set()
        def visit(task_id: str) -> set[str]:
            if task_id in active:
                raise ValueError("DAG contains a cycle")
            if task_id in resolved:
                return resolved[task_id]
            active.add(task_id)
            result: set[str] = set()
            for parent in by_id[task_id].depends_on:
                result.add(parent)
                result.update(visit(parent))
            active.remove(task_id)
            resolved[task_id] = result
            return result
        for task_id in by_id:
            visit(task_id)
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "astrograph-dag-plan.v1", "plan_id": self.plan_id,
                "nodes": [asdict(node) for node in self.nodes]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DAGPlan":
        if value.get("schema_version") != "astrograph-dag-plan.v1":
            raise ValueError("unsupported DAG plan version")
        return cls(value["plan_id"], tuple(TaskNode(
            task_id=item["task_id"], depends_on=tuple(item.get("depends_on", ())),
            writes=tuple(item.get("writes", ())),
            allow_failed_dependencies=item.get("allow_failed_dependencies", False))
            for item in value["nodes"]))

    def mermaid(self) -> str:
        lines = ["flowchart TD"]
        for node in self.nodes:
            lines.append(f"  {node.task_id}[{node.task_id}]")
            lines.extend(f"  {parent} --> {node.task_id}" for parent in node.depends_on)
        return "\n".join(lines) + "\n"


def _overlap(a: str, b: str) -> bool:
    left, right = PurePosixPath(a), PurePosixPath(b)
    return left == right or left in right.parents or right in left.parents


def execute_node(node: TaskNode, handler: Handler, dependencies: dict[str, TaskOutcome],
                 *, run_id: str, ledger: ChangeLedger | None = None) -> TaskOutcome:
    failed = [key for key, value in dependencies.items() if value.status != "SUCCEEDED"]
    if failed and not node.allow_failed_dependencies:
        outcome = TaskOutcome(node.task_id, "BLOCKED", error_type="DEPENDENCY_FAILED",
                              finished_ns=time.monotonic_ns())
        if ledger:
            ledger.append(run_id=run_id, task_id=node.task_id, event="BLOCKED", failed_dependencies=failed)
        return outcome
    start = time.monotonic_ns()
    if ledger:
        ledger.append(run_id=run_id, task_id=node.task_id, event="STARTED")
    try:
        output = handler(deepcopy(dependencies))
        outcome = TaskOutcome(node.task_id, "SUCCEEDED", output, started_ns=start,
                              finished_ns=time.monotonic_ns())
    except Exception as exc:
        # Callers can retain their own detailed local logs; avoid payload/secret echo.
        outcome = TaskOutcome(node.task_id, "FAILED", error_type=type(exc).__name__,
                              started_ns=start, finished_ns=time.monotonic_ns())
    if ledger:
        ledger.append(run_id=run_id, task_id=node.task_id, event=outcome.status,
                      error_type=outcome.error_type, started_ns=start, finished_ns=outcome.finished_ns)
    return outcome


class DAGExecutor:
    backend = "native-thread-dag"

    def __init__(self, plan: DAGPlan, handlers: dict[str, Handler], *, max_workers: int = 4,
                 ledger: ChangeLedger | None = None):
        if max_workers < 1 or set(handlers) != {node.task_id for node in plan.nodes}:
            raise ValueError("positive workers and exactly one handler per task are required")
        self.plan, self.handlers, self.max_workers, self.ledger = plan, handlers, max_workers, ledger

    def run(self, *, run_id: str | None = None) -> dict[str, TaskOutcome]:
        run_id = run_id or str(uuid4())
        pending = {node.task_id: node for node in self.plan.nodes}
        results: dict[str, TaskOutcome] = {}
        running = {}
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="astrograph-dag") as pool:
            while pending or running:
                for task_id in sorted(tuple(pending)):
                    node = pending[task_id]
                    if set(node.depends_on) <= results.keys() and len(running) < self.max_workers:
                        dependencies = {parent: results[parent] for parent in node.depends_on}
                        future = pool.submit(execute_node, node, self.handlers[task_id], dependencies,
                                             run_id=run_id, ledger=self.ledger)
                        running[future] = task_id
                        del pending[task_id]
                if not running:
                    raise RuntimeError("validated DAG could not progress")
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = running.pop(future)
                    results[task_id] = future.result()
        return {node.task_id: results[node.task_id] for node in self.plan.nodes}
