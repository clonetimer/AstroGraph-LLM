"""Bounded UTF-8 reads and immutable evaluator records."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from astrograph.agents.mission.storage import canonical, file_sha, sha

MAX_JSON_BYTES = 128_000_000


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"MISSING_OR_OVERSIZE_JSON:{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    """Never rewrite a prior trial or a failed attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("EVALUATION_ARTIFACT_ESCAPES_ROOT")
    return path


def verify_mission(root: Path) -> dict:
    """Check the committed result, artifact inventory and event hash chain.

    Hashes prove internal consistency, not a trusted signature or source truth.
    The original absolute paths inside artifacts are never opened.
    """
    root = Path(root).resolve()
    result = read_json(root / "result.json")
    expected = read_json(root / "result_receipt.json")["sha256"]
    if file_sha(root / "result.json") != expected:
        raise ValueError("RESULT_HASH_MISMATCH")
    seen = set()
    for entry in read_json(root / "artifacts.json"):
        if entry["name"] in seen:
            raise ValueError("DUPLICATE_ARTIFACT_INVENTORY_ENTRY")
        seen.add(entry["name"])
        if file_sha(safe_child(root, entry["name"])) != entry["sha256"]:
            raise ValueError("MISSION_ARTIFACT_HASH_MISMATCH:" + entry["name"])
    previous = "0" * 64
    events = []
    for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        body = dict(event)
        digest = body.pop("event_sha256")
        if body["previous_sha256"] != previous or sha(body) != digest:
            raise ValueError("MISSION_EVENT_CHAIN_MISMATCH")
        previous = digest
        events.append(event)
    if not events or events[-1]["event"] != "TASK_FINISHED":
        raise ValueError("MISSION_NOT_COMMITTED")
    return {"result": result, "events": events, "result_sha256": expected,
            "artifact_names": sorted(seen), "event_tip": previous}
