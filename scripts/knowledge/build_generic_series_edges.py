#!/usr/bin/env python3
"""Build the committed provisional generic-series knowledge edge snapshot.

Only exact channel matches are admitted. CONTEXT-only channel occurrences are not
converted into causal/support edges. The source fault catalog is already marked E0
and expert-review-required; generated edges preserve that boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "fault_id","fault_name","anomaly_id","anomaly_name","weight","edge_role",
    "evidence_level","evidence_source","source_ref","review_status","provisional",
    "knowledge_version","applicable_condition_ids","excluded_condition_ids",
]
ROLE_TO_EDGE = {"PRIMARY":"direct", "SAFETY":"direct", "SUPPORTING":"indirect"}


def load_rows(root: Path) -> list[dict[str,str]]:
    rules = json.loads((root/"configs/anomaly_rules.json").read_text(encoding="utf-8"))["rules"]
    faults = json.loads((root/"configs/fault_modes/lab_fault_catalog.json").read_text(encoding="utf-8"))["fault_modes"]
    anomaly_by_channel = {
        str(x["channel_id"]):(str(x["anomaly_id"]), str(x.get("anomaly_name_zh") or x["anomaly_id"]))
        for x in rules if x.get("enabled", True)
    }
    rows: list[dict[str,str]] = []
    # Preserve the historical seed edge as an explicit compatibility/provenance row.
    seed = root/"configs/knowledge/generic_series_seed_edges.csv"
    with seed.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({field:str(row.get(field,"")) for field in FIELDS})
    for fault in faults:
        for channel in fault.get("telemetry_channels", []):
            source_channel = str(channel.get("channel", ""))
            role = str(channel.get("role", ""))
            if source_channel not in anomaly_by_channel or role not in ROLE_TO_EDGE:
                continue
            anomaly_id, anomaly_name = anomaly_by_channel[source_channel]
            rows.append({
                "fault_id":str(fault["fault_id"]),
                "fault_name":str(fault.get("display_name") or fault["fault_id"]),
                "anomaly_id":anomaly_id,
                "anomaly_name":anomaly_name,
                "weight":"1.0",
                "edge_role":ROLE_TO_EDGE[role],
                "evidence_level":"E0",
                "evidence_source":"astrograph_lab_fault_catalog_exact_channel_match",
                "source_ref":f"configs/fault_modes/lab_fault_catalog.json#{fault['fault_id']}:{source_channel}:{role}",
                "review_status":"provisional",
                "provisional":"true",
                "knowledge_version":"astrograph-generic-series-e0.2",
                "applicable_condition_ids":"",
                "excluded_condition_ids":"",
            })
    dedup: dict[tuple[str,str],dict[str,str]] = {}
    for row in rows:
        key=(row["fault_id"],row["anomaly_id"])
        if key in dedup:
            raise ValueError(f"duplicate generated knowledge edge {key}")
        dedup[key]=row
    return [dedup[key] for key in sorted(dedup)]


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--output", default="configs/knowledge/generic_series_fault_catalog_edges.csv")
    args=parser.parse_args()
    root=Path(args.project_root).resolve()
    output=Path(args.output)
    if not output.is_absolute(): output=root/output
    rows=load_rows(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"output":str(output),"edges":len(rows)},ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
