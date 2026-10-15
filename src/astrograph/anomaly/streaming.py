from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.anomaly.episodes import AnomalyEpisode, EpisodeEvidenceItem


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeLifecycleStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class EpisodeTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    at: str = Field(default_factory=_now)
    observed_at_s: float = Field(ge=0.0)
    revision: int = Field(ge=1)
    details: dict[str, Any] = Field(default_factory=dict)


class IncrementalAnomalyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = ""
    dataset_id: str = ""
    run_id: str = Field(min_length=16)
    paired_nominal_run_id: str | None = None
    case_id: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    observed_at_s: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    signal: str = Field(min_length=1)
    active: bool = True
    semantic_role: str = ""
    condition_ids: list[str] = Field(default_factory=list)
    detector_id: str = ""
    rule_id: str = ""
    rule_set_version: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("condition_ids")
    @classmethod
    def unique_conditions(cls, value: list[str]) -> list[str]:
        return sorted(set(item for item in value if item))


class EpisodeLifecycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-online-episode.v1"
    episode_id: str = Field(min_length=16)
    status: EpisodeLifecycleStatus
    revision: int = Field(ge=1)
    dataset_id: str = ""
    run_id: str = Field(min_length=16)
    paired_nominal_run_id: str | None = None
    case_id: str
    anomaly_id: str
    name: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    first_observed_at_s: float = Field(ge=0.0)
    last_observed_at_s: float = Field(ge=0.0)
    peak_confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    semantic_roles: list[str] = Field(default_factory=list)
    condition_ids: list[str] = Field(default_factory=list)
    detector_versions: list[str] = Field(default_factory=list)
    evidence_items: list[EpisodeEvidenceItem] = Field(default_factory=list)
    opened_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    closed_at: str | None = None
    close_reason: str = ""
    transitions: list[EpisodeTransition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("signals", "semantic_roles", "condition_ids", "detector_versions")
    @classmethod
    def unique_sorted(cls, value: list[str]) -> list[str]:
        return sorted(set(item for item in value if item))

    def to_episode(self) -> AnomalyEpisode:
        return AnomalyEpisode(
            episode_id=self.episode_id,
            dataset_id=self.dataset_id,
            run_id=self.run_id,
            paired_nominal_run_id=self.paired_nominal_run_id,
            case_id=self.case_id,
            anomaly_id=self.anomaly_id,
            name=self.name,
            start_s=self.start_s,
            end_s=self.end_s,
            first_observed_at_s=self.first_observed_at_s,
            peak_confidence=self.peak_confidence,
            signals=self.signals,
            semantic_roles=self.semantic_roles,
            condition_ids=self.condition_ids,
            evidence_items=self.evidence_items,
            detector_versions=self.detector_versions,
            metadata={
                **self.metadata,
                "lifecycle_status": self.status.value,
                "lifecycle_revision": self.revision,
                "close_reason": self.close_reason,
            },
        )


class EpisodeUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    episode: EpisodeLifecycleRecord | None = None
    closed_episode: EpisodeLifecycleRecord | None = None
    message: str = ""


def _stream_key(observation: IncrementalAnomalyObservation) -> str:
    return "|".join([observation.dataset_id, observation.run_id, observation.anomaly_id])


def _episode_id(observation: IncrementalAnomalyObservation) -> str:
    payload = (
        f"{observation.dataset_id}|{observation.run_id}|{observation.anomaly_id}|"
        f"{observation.observed_at_s:.9f}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


class JsonEpisodeLifecycleStore:
    schema_version = "astrograph-online-episode-store.v1"

    def __init__(self, path: str | Path, *, max_gap_s: float = 30.0):
        if max_gap_s < 0:
            raise ValueError("max_gap_s must be nonnegative")
        self.path = Path(path)
        self.max_gap_s = float(max_gap_s)
        self._lock = Lock()

    def _empty(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "episodes": {}, "open_index": {}}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported online episode store schema")
        payload.setdefault("episodes", {})
        payload.setdefault("open_index", {})
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _detector_version(observation: IncrementalAnomalyObservation) -> str:
        return "|".join(filter(None, [observation.detector_id, observation.rule_set_version]))

    @staticmethod
    def _evidence_item(observation: IncrementalAnomalyObservation) -> EpisodeEvidenceItem:
        return EpisodeEvidenceItem(
            signal=observation.signal,
            confidence=observation.confidence,
            observed_at_s=observation.observed_at_s,
            detector_id=observation.detector_id,
            rule_id=observation.rule_id,
            semantic_role=observation.semantic_role,
            metadata={
                "observation_id": observation.observation_id,
                "rule_set_version": observation.rule_set_version,
                **observation.metadata,
            },
        )

    def _open_record(self, observation: IncrementalAnomalyObservation) -> EpisodeLifecycleRecord:
        transition = EpisodeTransition(
            action="OPEN",
            observed_at_s=observation.observed_at_s,
            revision=1,
            details={"signal": observation.signal},
        )
        return EpisodeLifecycleRecord(
            episode_id=_episode_id(observation),
            status=EpisodeLifecycleStatus.OPEN,
            revision=1,
            dataset_id=observation.dataset_id,
            run_id=observation.run_id,
            paired_nominal_run_id=observation.paired_nominal_run_id,
            case_id=observation.case_id,
            anomaly_id=observation.anomaly_id,
            name=observation.name,
            start_s=observation.observed_at_s,
            end_s=observation.observed_at_s,
            first_observed_at_s=observation.observed_at_s,
            last_observed_at_s=observation.observed_at_s,
            peak_confidence=observation.confidence,
            signals=[observation.signal],
            semantic_roles=[observation.semantic_role],
            condition_ids=observation.condition_ids,
            detector_versions=[self._detector_version(observation)],
            evidence_items=[self._evidence_item(observation)],
            transitions=[transition],
            metadata={"confidence_type": "synthetic_online_episode_peak_not_calibrated"},
        )

    @staticmethod
    def _close_record(
        record: EpisodeLifecycleRecord,
        *,
        observed_at_s: float,
        reason: str,
        action: str = "CLOSE",
    ) -> EpisodeLifecycleRecord:
        if observed_at_s < record.last_observed_at_s:
            observed_at_s = record.last_observed_at_s
        record.status = EpisodeLifecycleStatus.CLOSED
        record.revision += 1
        record.end_s = max(record.end_s, observed_at_s)
        record.updated_at = _now()
        record.closed_at = record.updated_at
        record.close_reason = reason
        record.transitions.append(
            EpisodeTransition(
                action=action,
                observed_at_s=observed_at_s,
                revision=record.revision,
                details={"reason": reason},
            )
        )
        return record

    def apply(self, observation: IncrementalAnomalyObservation) -> EpisodeUpdateResult:
        key = _stream_key(observation)
        with self._lock:
            payload = self._load()
            open_id = payload["open_index"].get(key)
            current = None
            if open_id:
                current = EpisodeLifecycleRecord.model_validate(payload["episodes"][open_id])

            if not observation.active:
                if current is None:
                    return EpisodeUpdateResult(action="NO_OPEN_EPISODE", message="No open episode matched the inactive observation.")
                closed = self._close_record(
                    current,
                    observed_at_s=observation.observed_at_s,
                    reason="EXPLICIT_INACTIVE_OBSERVATION",
                )
                payload["episodes"][closed.episode_id] = closed.model_dump(mode="json")
                payload["open_index"].pop(key, None)
                self._save(payload)
                return EpisodeUpdateResult(action="CLOSED", episode=closed)

            closed_by_gap = None
            if current is not None and observation.observed_at_s < current.last_observed_at_s:
                raise ValueError("online observations must be monotonic within one run/anomaly stream")
            if current is not None and observation.observed_at_s - current.last_observed_at_s > self.max_gap_s:
                closed_by_gap = self._close_record(
                    current,
                    observed_at_s=current.last_observed_at_s,
                    reason="MAX_GAP_EXCEEDED",
                    action="AUTO_CLOSE",
                )
                payload["episodes"][closed_by_gap.episode_id] = closed_by_gap.model_dump(mode="json")
                payload["open_index"].pop(key, None)
                current = None

            if current is None:
                current = self._open_record(observation)
                action = "OPENED"
            else:
                current.revision += 1
                current.end_s = observation.observed_at_s
                current.last_observed_at_s = observation.observed_at_s
                current.peak_confidence = max(current.peak_confidence, observation.confidence)
                current.signals = sorted(set(current.signals + [observation.signal]))
                current.semantic_roles = sorted(set(current.semantic_roles + [observation.semantic_role]) - {""})
                current.condition_ids = sorted(set(current.condition_ids + observation.condition_ids))
                current.detector_versions = sorted(
                    set(current.detector_versions + [self._detector_version(observation)]) - {""}
                )
                current.evidence_items.append(self._evidence_item(observation))
                current.updated_at = _now()
                current.transitions.append(
                    EpisodeTransition(
                        action="UPDATE",
                        observed_at_s=observation.observed_at_s,
                        revision=current.revision,
                        details={"signal": observation.signal, "confidence": observation.confidence},
                    )
                )
                action = "UPDATED"

            payload["episodes"][current.episode_id] = current.model_dump(mode="json")
            payload["open_index"][key] = current.episode_id
            self._save(payload)
            return EpisodeUpdateResult(action=action, episode=current, closed_episode=closed_by_gap)

    def close(self, episode_id: str, *, end_s: float, reason: str = "MANUAL_CLOSE") -> EpisodeLifecycleRecord:
        with self._lock:
            payload = self._load()
            raw = payload["episodes"].get(episode_id)
            if raw is None:
                raise KeyError(episode_id)
            record = EpisodeLifecycleRecord.model_validate(raw)
            if record.status == EpisodeLifecycleStatus.CLOSED:
                return record
            record = self._close_record(record, observed_at_s=end_s, reason=reason)
            payload["episodes"][episode_id] = record.model_dump(mode="json")
            for key, value in list(payload["open_index"].items()):
                if value == episode_id:
                    payload["open_index"].pop(key, None)
            self._save(payload)
            return record

    def rebuild_open_index(self) -> dict[str, Any]:
        """Rebuild the derived open-index and close duplicate OPEN records safely."""
        with self._lock:
            payload = self._load()
            grouped: dict[str, list[EpisodeLifecycleRecord]] = {}
            for raw in payload["episodes"].values():
                record = EpisodeLifecycleRecord.model_validate(raw)
                if record.status != EpisodeLifecycleStatus.OPEN:
                    continue
                key = "|".join([record.dataset_id, record.run_id, record.anomaly_id])
                grouped.setdefault(key, []).append(record)
            rebuilt: dict[str, str] = {}
            closed_duplicates: list[str] = []
            for key, records in grouped.items():
                records.sort(key=lambda item: (item.last_observed_at_s, item.revision, item.episode_id), reverse=True)
                keep = records[0]
                rebuilt[key] = keep.episode_id
                for duplicate in records[1:]:
                    duplicate = self._close_record(
                        duplicate,
                        observed_at_s=duplicate.last_observed_at_s,
                        reason="RECOVERY_DUPLICATE_OPEN_STREAM",
                        action="RECOVERY_CLOSE",
                    )
                    payload["episodes"][duplicate.episode_id] = duplicate.model_dump(mode="json")
                    closed_duplicates.append(duplicate.episode_id)
            previous = dict(payload.get("open_index", {}))
            payload["open_index"] = rebuilt
            self._save(payload)
            return {
                "schema_version": "astrograph-episode-recovery-report.v1",
                "open_stream_count": len(rebuilt),
                "closed_duplicate_count": len(closed_duplicates),
                "closed_duplicate_episode_ids": sorted(closed_duplicates),
                "index_changed": previous != rebuilt,
            }

    def get(self, episode_id: str) -> EpisodeLifecycleRecord | None:
        payload = self._load()
        raw = payload["episodes"].get(episode_id)
        return None if raw is None else EpisodeLifecycleRecord.model_validate(raw)

    def list_records(
        self,
        *,
        run_id: str | None = None,
        status: EpisodeLifecycleStatus | None = None,
    ) -> list[EpisodeLifecycleRecord]:
        records = [EpisodeLifecycleRecord.model_validate(item) for item in self._load()["episodes"].values()]
        if run_id is not None:
            records = [item for item in records if item.run_id == run_id]
        if status is not None:
            records = [item for item in records if item.status == status]
        return sorted(records, key=lambda item: (item.run_id, item.start_s, item.anomaly_id, item.episode_id))
