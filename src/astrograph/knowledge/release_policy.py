from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrograph.knowledge.release import KnowledgeRelease, sha256_file


D3_KNOWLEDGE_VERSION = "astrograph-ontology-v1-e0-d3rw-v1"
D3_RELEASE_ID = "KR-20260718-E0-D3RW-002"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_d3_rw_candidate_release(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    seed = root / "data" / "knowledge_seed"
    extension = root / "data" / "knowledge_extensions" / "d3_rw"
    release_dir = root / "data" / "knowledge_releases" / "candidates" / D3_RELEASE_ID
    release_dir.mkdir(parents=True, exist_ok=True)

    fault_rows = _read_csv(seed / "fault_modes.csv")
    existing_fault_ids = {row["fault_id"] for row in fault_rows}
    for row in _read_csv(extension / "fault_modes.csv"):
        if row["fault_id"] in existing_fault_ids:
            continue
        fault_rows.append(
            {
                "fault_id": row["fault_id"],
                "name": row["name"],
                "subsystem": row.get("subsystem", "ADCS"),
                "component": row.get("component", "ReactionWheel"),
                "model_role": row.get("mechanism_model", "d3_rw_engineering_mechanism"),
                "evidence_level": row.get("evidence_level", "E0"),
            }
        )
        existing_fault_ids.add(row["fault_id"])

    anomaly_rows = _read_csv(seed / "anomalies.csv")
    existing_anomaly_ids = {row["anomaly_id"] for row in anomaly_rows}
    for row in _read_csv(extension / "anomalies.csv"):
        if row["anomaly_id"] in existing_anomaly_ids:
            continue
        anomaly_rows.append(
            {
                "anomaly_id": row["anomaly_id"],
                "name": row["name"],
                "signal": row.get("channel_pattern", ""),
                "direction": "deviation",
            }
        )
        existing_anomaly_ids.add(row["anomaly_id"])
    if "ANOM_RW_DRIVER_AVAILABLE_CONFIRMED" not in existing_anomaly_ids:
        anomaly_rows.append(
            {
                "anomaly_id": "ANOM_RW_DRIVER_AVAILABLE_CONFIRMED",
                "name": "反作用轮驱动器可用状态确认",
                "signal": "status.adcs.rw.available_{wheel}",
                "direction": "normal_confirmation",
            }
        )

    edge_rows = _read_csv(seed / "causes.csv")
    normalized_edges: list[dict[str, Any]] = []
    for row in edge_rows:
        item = dict(row)
        item["knowledge_version"] = D3_KNOWLEDGE_VERSION
        normalized_edges.append(item)

    fault_names = {row["fault_id"]: row["name"] for row in fault_rows}
    anomaly_names = {row["anomaly_id"]: row["name"] for row in anomaly_rows}
    role_map = {
        "direct": "direct",
        "component_response": "indirect",
        "subsystem_propagation": "cascade",
    }
    existing_keys = {(row["fault_id"], row["anomaly_id"]) for row in normalized_edges}
    for row in _read_csv(extension / "causes.csv"):
        key = (row["fault_id"], row["anomaly_id"])
        if key in existing_keys:
            continue
        original_role = row.get("edge_role", "direct")
        normalized_edges.append(
            {
                "fault_id": row["fault_id"],
                "fault_name": fault_names[row["fault_id"]],
                "anomaly_id": row["anomaly_id"],
                "anomaly_name": anomaly_names[row["anomaly_id"]],
                "weight": row.get("weight", "1.0"),
                "edge_role": role_map.get(original_role, "indirect"),
                "evidence_level": row.get("evidence_level", "E0"),
                "evidence_source": "PHM_D3_RW_PRODUCTION_E0",
                "source_ref": f"D3 fault protocol; original_role={original_role}",
                "review_status": row.get("review_status", "provisional"),
                "provisional": row.get("provisional", "true"),
                "knowledge_version": D3_KNOWLEDGE_VERSION,
                "applicable_condition_ids": "",
                "excluded_condition_ids": "",
            }
        )
        existing_keys.add(key)

    normalized_edges.sort(key=lambda row: (row["fault_id"], row["anomaly_id"]))
    fault_rows.sort(key=lambda row: row["fault_id"])
    anomaly_rows.sort(key=lambda row: row["anomaly_id"])

    causes_path = release_dir / "causes.csv"
    faults_path = release_dir / "fault_modes.csv"
    anomalies_path = release_dir / "anomalies.csv"
    _write_csv(
        causes_path,
        normalized_edges,
        [
            "fault_id", "fault_name", "anomaly_id", "anomaly_name", "weight", "edge_role",
            "evidence_level", "evidence_source", "source_ref", "review_status", "provisional",
            "knowledge_version", "applicable_condition_ids", "excluded_condition_ids",
        ],
    )
    _write_csv(
        faults_path,
        fault_rows,
        ["fault_id", "name", "subsystem", "component", "model_role", "evidence_level"],
    )
    _write_csv(anomalies_path, anomaly_rows, ["anomaly_id", "name", "signal", "direction"])

    release = KnowledgeRelease(
        release_id=D3_RELEASE_ID,
        knowledge_version=D3_KNOWLEDGE_VERSION,
        ontology_version="astrograph-phm-ontology-v1",
        release_status="engineering_released",
        created_at=datetime.now(timezone.utc).isoformat(),
        edges_file=str(causes_path.relative_to(root)).replace("\\", "/"),
        edges_sha256=sha256_file(causes_path),
        fault_modes_file=str(faults_path.relative_to(root)).replace("\\", "/"),
        anomalies_file=str(anomalies_path.relative_to(root)).replace("\\", "/"),
        evidence_levels=["E0"],
        review_statuses=["provisional"],
        edge_count=len(normalized_edges),
        claim_boundary=(
            "Merged D0/D1 E0 provisional knowledge plus D3 reaction-wheel engineering fault protocols; "
            "not flight evidence, expert probability, or hardware root-cause certification."
        ),
    )
    release_path = release_dir / "release.json"
    release_path.write_text(json.dumps(release.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "schema_version": "astrograph-d3-candidate-release-build.v1",
        "status": "PASS",
        "release_id": release.release_id,
        "knowledge_version": release.knowledge_version,
        "release_path": str(release_path.relative_to(root)).replace("\\", "/"),
        "edge_count": release.edge_count,
        "fault_count": len(fault_rows),
        "anomaly_count": len(anomaly_rows),
        "edges_sha256": release.edges_sha256,
    }
