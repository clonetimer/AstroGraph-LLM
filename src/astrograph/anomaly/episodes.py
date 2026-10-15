from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.schemas import AnomalyEvidence


class EpisodeEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at_s: float | None = Field(default=None, ge=0.0)
    detector_id: str = ""
    rule_id: str = ""
    semantic_role: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnomalyEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-anomaly-episode.v1"
    episode_id: str = Field(min_length=16)
    dataset_id: str = ""
    run_id: str = Field(min_length=16)
    paired_nominal_run_id: str | None = None
    case_id: str
    anomaly_id: str
    name: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    first_observed_at_s: float | None = Field(default=None, ge=0.0)
    peak_confidence: float = Field(ge=0.0, le=1.0)
    confidence_type: str = "synthetic_episode_peak_not_calibrated"
    signals: list[str] = Field(default_factory=list)
    semantic_roles: list[str] = Field(default_factory=list)
    condition_ids: list[str] = Field(default_factory=list)
    evidence_items: list[EpisodeEvidenceItem] = Field(default_factory=list)
    detector_versions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_s")
    @classmethod
    def end_must_be_finite(cls, value: float) -> float:
        if value < 0:
            raise ValueError("end_s must be nonnegative")
        return value

    @field_validator("signals", "semantic_roles", "condition_ids", "detector_versions")
    @classmethod
    def unique_sorted(cls, value: list[str]) -> list[str]:
        return sorted(set(item for item in value if item))

    def as_anomaly_evidence(self) -> AnomalyEvidence:
        signal = self.signals[0] if self.signals else "episode"
        return AnomalyEvidence(
            anomaly_id=self.anomaly_id,
            name=self.name,
            signal=signal,
            confidence=self.peak_confidence,
            confidence_type=self.confidence_type,
            observed_at_s=self.first_observed_at_s,
            metadata={
                "episode_id": self.episode_id,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "start_s": self.start_s,
                "end_s": self.end_s,
                "signals": self.signals,
                "semantic_roles": self.semantic_roles,
                "condition_ids": self.condition_ids,
                "evidence_item_count": len(self.evidence_items),
                **self.metadata,
            },
        )


def _episode_id(run_id: str, anomaly_id: str, start_s: float, end_s: float) -> str:
    payload = f"{run_id}|{anomaly_id}|{start_s:.9f}|{end_s:.9f}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _normalized_record(item: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in item.items() if key in {
        "anomaly_id", "name", "signal", "confidence", "confidence_type", "observed_at_s", "metadata"
    }}
    evidence = AnomalyEvidence.model_validate(payload)
    metadata = evidence.metadata
    start_s = float(metadata.get("active_start_s", evidence.observed_at_s or 0.0))
    end_s = float(metadata.get("active_end_s", evidence.observed_at_s or start_s))
    if end_s < start_s:
        raise ValueError(f"anomaly interval end precedes start: {evidence.anomaly_id}")
    return {
        "run_id": str(metadata.get("run_id", item.get("run_id", ""))),
        "paired_nominal_run_id": metadata.get("paired_nominal_run_id"),
        "case_id": str(metadata.get("case_id", item.get("case_id", ""))),
        "anomaly_id": evidence.anomaly_id,
        "name": evidence.name,
        "start_s": start_s,
        "end_s": end_s,
        "confidence": evidence.confidence,
        "confidence_type": evidence.confidence_type,
        "observed_at_s": evidence.observed_at_s,
        "signal": evidence.signal,
        "semantic_role": str(metadata.get("semantic_role", "")),
        "conditions": list(metadata.get("baseline_condition_ids", [])),
        "detector_id": str(metadata.get("detector_id", "")),
        "rule_id": str(metadata.get("rule_id", "")),
        "rule_set_version": str(metadata.get("rule_set_version", "")),
        "metadata": metadata,
    }


def build_anomaly_episodes(
    records: Iterable[dict[str, Any]],
    *,
    dataset_id: str = "",
    merge_gap_s: float = 0.0,
) -> list[AnomalyEpisode]:
    if merge_gap_s < 0:
        raise ValueError("merge_gap_s must be nonnegative")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        record = _normalized_record(item)
        if not record["run_id"]:
            raise ValueError("anomaly record is missing run_id")
        grouped[(record["run_id"], record["anomaly_id"])].append(record)

    episodes: list[AnomalyEpisode] = []
    for (run_id, anomaly_id), items in sorted(grouped.items()):
        items.sort(key=lambda row: (row["start_s"], row["end_s"], row["signal"]))
        clusters: list[list[dict[str, Any]]] = []
        for item in items:
            if not clusters or item["start_s"] > max(row["end_s"] for row in clusters[-1]) + merge_gap_s:
                clusters.append([item])
            else:
                clusters[-1].append(item)
        for cluster in clusters:
            start_s = min(item["start_s"] for item in cluster)
            end_s = max(item["end_s"] for item in cluster)
            observed_values = [item["observed_at_s"] for item in cluster if item["observed_at_s"] is not None]
            evidence_items = [
                EpisodeEvidenceItem(
                    signal=item["signal"],
                    confidence=item["confidence"],
                    observed_at_s=item["observed_at_s"],
                    detector_id=item["detector_id"],
                    rule_id=item["rule_id"],
                    semantic_role=item["semantic_role"],
                    metadata={
                        "confidence_type": item["confidence_type"],
                        "rule_set_version": item["rule_set_version"],
                    },
                )
                for item in cluster
            ]
            episodes.append(
                AnomalyEpisode(
                    episode_id=_episode_id(run_id, anomaly_id, start_s, end_s),
                    dataset_id=dataset_id,
                    run_id=run_id,
                    paired_nominal_run_id=cluster[0]["paired_nominal_run_id"],
                    case_id=cluster[0]["case_id"],
                    anomaly_id=anomaly_id,
                    name=cluster[0]["name"],
                    start_s=start_s,
                    end_s=end_s,
                    first_observed_at_s=min(observed_values) if observed_values else None,
                    peak_confidence=max(item["confidence"] for item in cluster),
                    confidence_type="synthetic_episode_peak_not_calibrated",
                    signals=[item["signal"] for item in cluster],
                    semantic_roles=[item["semantic_role"] for item in cluster],
                    condition_ids=[condition for item in cluster for condition in item["conditions"]],
                    evidence_items=evidence_items,
                    detector_versions=[
                        "|".join(filter(None, [item["detector_id"], item["rule_set_version"]]))
                        for item in cluster
                    ],
                    metadata={
                        "merge_gap_s": merge_gap_s,
                        "source_record_count": len(cluster),
                        "calibration_status": "synthetic_uncalibrated",
                    },
                )
            )
    return sorted(episodes, key=lambda item: (item.run_id, item.start_s, item.anomaly_id, item.episode_id))


def load_anomaly_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return records


def write_episode_jsonl(path: str | Path, episodes: list[AnomalyEpisode]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) for item in episodes)
    output.write_text(text + ("\n" if text else ""), encoding="utf-8")
