from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.anomaly import JsonEpisodeLifecycleStore
from astrograph.lifecycle import JsonRequestLifecycleStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PersistenceComponentStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    path: str
    exists: bool
    size_bytes: int = 0
    sha256: str = ""
    parse_status: str = "NOT_APPLICABLE"


class PersistenceStatusReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-persistence-status.v1"
    status: str
    generated_at: str = Field(default_factory=_now)
    components: list[PersistenceComponentStatus] = Field(default_factory=list)


class JsonStatePersistenceCoordinator:
    """Coordinates snapshots and deterministic recovery for local JSON/JSONL stores.

    This is an engineering single-node persistence layer. It is not a substitute for
    a transactional production database or distributed consensus.
    """

    def __init__(
        self,
        components: dict[str, str | Path],
        *,
        episode_store: JsonEpisodeLifecycleStore | None = None,
        request_store: JsonRequestLifecycleStore | None = None,
    ):
        self.components = {key: Path(value) for key, value in components.items()}
        self.episode_store = episode_store
        self.request_store = request_store

    def status(self) -> PersistenceStatusReport:
        records: list[PersistenceComponentStatus] = []
        failed = False
        for component_id, path in sorted(self.components.items()):
            exists = path.exists()
            parse_status = "MISSING"
            size = 0
            checksum = ""
            if exists:
                size = path.stat().st_size
                checksum = _sha256(path)
                parse_status = "PASS"
                if path.suffix == ".json":
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        parse_status = "FAIL"
                        failed = True
                elif path.suffix == ".jsonl":
                    try:
                        for line in path.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                json.loads(line)
                    except Exception:
                        parse_status = "FAIL"
                        failed = True
            records.append(
                PersistenceComponentStatus(
                    component_id=component_id,
                    path=str(path),
                    exists=exists,
                    size_bytes=size,
                    sha256=checksum,
                    parse_status=parse_status,
                )
            )
        return PersistenceStatusReport(status="FAIL" if failed else "PASS", components=records)

    def snapshot(self, output_dir: str | Path) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for component_id, source in sorted(self.components.items()):
            if not source.exists():
                entries.append({"component_id": component_id, "included": False, "source": str(source)})
                continue
            target = destination / f"{component_id}{source.suffix}"
            shutil.copy2(source, target)
            entries.append(
                {
                    "component_id": component_id,
                    "included": True,
                    "source": str(source),
                    "snapshot_file": target.name,
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        manifest = {
            "schema_version": "astrograph-state-snapshot.v1",
            "created_at": _now(),
            "entry_count": len(entries),
            "entries": entries,
            "claim_boundary": "Single-node engineering state snapshot; no distributed transaction guarantee.",
        }
        (destination / "snapshot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def recover(self) -> dict[str, Any]:
        request_report = (
            self.request_store.recover_interrupted() if self.request_store else {"status": "NOT_CONFIGURED"}
        )
        episode_report = (
            self.episode_store.rebuild_open_index() if self.episode_store else {"status": "NOT_CONFIGURED"}
        )
        status = self.status()
        return {
            "schema_version": "astrograph-service-recovery.v1",
            "status": "PASS" if status.status == "PASS" else "FAIL",
            "generated_at": _now(),
            "request_recovery": request_report,
            "episode_recovery": episode_report,
            "persistence_status": status.model_dump(mode="json"),
        }
